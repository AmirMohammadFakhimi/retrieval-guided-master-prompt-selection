import copy
import gc
import random
import warnings
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

import lancedb
import numpy as np
import torch
from lancedb.table import LanceTable
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM as AutoCausalLanguageModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from dataset import (
    Column,
    TARGET_TO_AUDIT_COLUMN,
    display_column_name,
)

RETRIEVAL_METHODS = frozenset({'semantic', 'balanced_semantic'})
EXAMPLE_ORDERS = frozenset({
    'most_similar_first',
    'most_similar_last',
    'shuffle',
})
LANCEDB_INGEST_BATCH_SIZE = 2048
TORCH_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}
LANGUAGE_MODEL_DTYPES = frozenset({*TORCH_DTYPES, 'auto'})
GENERATED_OUTPUT_MAX_NEW_TOKENS = 32
UNBOUNDED_TOKENIZER_LENGTH = 1_000_000_000


class SemanticResource(TypedDict):
    """LanceDB table and matching evaluation embeddings for one embedding model."""

    table: LanceTable
    evaluation_vectors: np.ndarray
    training_filter: str


def _warn_about_embedding_input_truncation(
        encoder: SentenceTransformer,
        rows: list[dict[str, Any]],
        embedding_model_id: str,
        max_sequence_length: int,
        input_kind: str,
        prompt: str = '',
) -> None:
    """Report embedding inputs that SentenceTransformers will truncate."""

    if not rows:
        return

    texts = [f'{prompt}{row[Column.HARD_TEXT]}' for row in rows]
    tokenized = encoder.tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        verbose=False,
    )

    overlength_inputs = []
    for row, token_ids in zip(rows, tokenized['input_ids'], strict=True):
        token_count = len(token_ids)
        if token_count > max_sequence_length:
            overlength_inputs.append(f'{row[Column.ID]!r} ({token_count} tokens)')

    if overlength_inputs:
        warnings.warn(
            f'{embedding_model_id} will truncate {len(overlength_inputs)} {input_kind} input(s) '
            f'to max_sequence_length={max_sequence_length}: {", ".join(overlength_inputs)}',
            stacklevel=2,
        )


def _language_model_context_length(tokenizer: PreTrainedTokenizerBase, language_model: PreTrainedModel) -> int:
    """Return the Qwen tokenizer limit or Gemma text-model limit."""

    if tokenizer.model_max_length < UNBOUNDED_TOKENIZER_LENGTH:
        return tokenizer.model_max_length

    return language_model.config.text_config.max_position_embeddings


def choose_device(requested: str = 'auto') -> str:
    """Choose CUDA, then Apple MPS, then CPU."""

    if requested not in {'auto', 'cuda', 'mps', 'cpu'}:
        raise ValueError('inference.device must be auto, cuda, mps, or cpu')

    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is not available')

    return requested


def _embedding_table_name(embedding_model_id: str) -> str:
    """Return the stable LanceDB table name for an embedding model."""

    return f'semantic_{embedding_model_id.lower().replace("/", "_")}'


def _load_embedding_encoder(embedding_model: dict[str, Any], device: str) -> SentenceTransformer:
    """Load one pinned embedding model and apply its configured sequence limit."""

    embedding_model_id = embedding_model['id']
    embedding_dimension = embedding_model['dimension']
    max_sequence_length = embedding_model['max_sequence_length']
    dtype_name = embedding_model['dtype']

    encoder = SentenceTransformer(
        embedding_model_id,
        revision=embedding_model['revision'],
        device=device,
        model_kwargs={'dtype': TORCH_DTYPES[dtype_name]},
        truncate_dim=embedding_dimension,
    )
    native_max_sequence_length = cast(int, encoder.get_max_seq_length())
    if max_sequence_length > native_max_sequence_length:
        raise ValueError(
            f'{embedding_model_id} supports at most {native_max_sequence_length} tokens, '
            f'but max_sequence_length={max_sequence_length} was configured'
        )
    encoder.max_seq_length = max_sequence_length
    return encoder


def clear_model_memory(device: str) -> None:
    """Collect released model objects and clear the active accelerator cache."""

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


def prepare_training_embedding_cache(
        training_rows: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        device: str,
        database_path: Path,
        progress: Callable[[str], None] = print,
) -> LanceTable:
    """Explicitly build or reuse all training embeddings for one model."""

    if not training_rows:
        raise ValueError('Cannot prepare an embedding cache from an empty training corpus')

    embedding_model_id = embedding_model['id']
    embedding_dimension = embedding_model['dimension']
    max_sequence_length = embedding_model['max_sequence_length']
    batch_size = embedding_model['batch_size']
    table_name = _embedding_table_name(embedding_model_id)

    database_path.mkdir(parents=True, exist_ok=True)
    database = lancedb.connect(database_path)
    if table_name in database.list_tables().tables:
        table = database.open_table(table_name)
        cached_row_count = table.count_rows()

        if cached_row_count == len(training_rows):
            progress(f'Reusing {cached_row_count} complete training embeddings from {database_path / table_name}')
            return cast(LanceTable, table)

        progress(f'Replacing incomplete LanceDB table {table_name}: {cached_row_count} of {len(training_rows)} rows')
        database.drop_table(table_name)

    length_sorted_training_rows = sorted(
        training_rows,
        key=lambda row: len(row[Column.HARD_TEXT]),
        reverse=True,
    )
    progress(f'Embedding all {len(length_sorted_training_rows)} training rows for {embedding_model_id}')
    encoder = _load_embedding_encoder(embedding_model, device)
    progress_bar = tqdm(
        total=len(length_sorted_training_rows),
        desc='Embedding all training rows for LanceDB',
        unit='row',
        leave=True,
    )
    table: LanceTable | None = None

    try:
        try:
            for start in range(0, len(length_sorted_training_rows), LANCEDB_INGEST_BATCH_SIZE):
                batch = length_sorted_training_rows[start:start + LANCEDB_INGEST_BATCH_SIZE]
                _warn_about_embedding_input_truncation(
                    encoder,
                    batch,
                    embedding_model_id,
                    max_sequence_length,
                    'training-document',
                )
                # SentenceTransformer re-sorts this bounded input internally
                # and restores its returned vectors to the batch's input order.
                vectors = np.asarray(
                    encoder.encode(
                        [row[Column.HARD_TEXT] for row in batch],
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                )

                if vectors.ndim != 2 or vectors.shape[1] != embedding_dimension:
                    raise RuntimeError(
                        f'Embedding model returned vectors with shape '
                        f'{vectors.shape}; expected (*, {embedding_dimension})'
                    )

                records = [
                    {
                        Column.ID: row[Column.ID],
                        Column.SPLIT: row[Column.SPLIT],
                        Column.HARD_TEXT: row[Column.HARD_TEXT],
                        Column.PROFESSION: row[Column.PROFESSION],
                        Column.GENDER: row[Column.GENDER],
                        Column.TRAIN_ORDER: row[Column.TRAIN_ORDER],
                        'vector': vector.tolist(),
                    }
                    for row, vector in zip(batch, vectors, strict=True)
                ]

                if table is None:
                    table = cast(LanceTable, database.create_table(table_name, data=records))
                else:
                    table.add(records)

                progress_bar.update(len(batch))
        except BaseException:
            if table_name in database.list_tables().tables:
                database.drop_table(table_name)
            raise
        finally:
            progress_bar.close()
    finally:
        del encoder
        clear_model_memory(device)

    if table is None:
        raise RuntimeError('Cannot build a LanceDB table from an empty training corpus')
    return table


def open_training_embedding_cache(
        training_rows: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        database_path: Path,
) -> LanceTable:
    """Open a table only when it contains every canonical training row."""

    if not database_path.exists():
        raise RuntimeError(
            f'Embedding cache for {embedding_model["id"]} is missing. '
            f'Run "Prepare all training embeddings" before the experiment.'
        )

    database = lancedb.connect(database_path)
    table_name = _embedding_table_name(embedding_model['id'])
    if table_name not in database.list_tables().tables:
        raise RuntimeError(
            f'Embedding cache for {embedding_model["id"]} is missing. '
            f'Run "Prepare all training embeddings" before the experiment.'
        )

    table = database.open_table(table_name)
    cached_row_count = table.count_rows()
    if cached_row_count != len(training_rows):
        raise RuntimeError(
            f'Embedding cache for {embedding_model["id"]} contains '
            f'{cached_row_count} of {len(training_rows)} training rows. '
            f'Run "Prepare all training embeddings" before the experiment.'
        )

    return cast(LanceTable, table)


def encode_embedding_queries(
        queries: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        device: str,
) -> np.ndarray:
    """Encode only the selected evaluation queries for one cached model."""

    embedding_model_id = embedding_model['id']
    embedding_dimension = embedding_model['dimension']
    max_sequence_length = embedding_model['max_sequence_length']
    batch_size = embedding_model['batch_size']
    query_prompt = embedding_model['query_prompt']
    encoder = _load_embedding_encoder(embedding_model, device)

    try:
        _warn_about_embedding_input_truncation(
            encoder,
            queries,
            embedding_model_id,
            max_sequence_length,
            'query',
            query_prompt,
        )
        query_vectors = np.asarray(
            encoder.encode(
                [row[Column.HARD_TEXT] for row in queries],
                prompt=query_prompt,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )
    finally:
        del encoder
        clear_model_memory(device)

    if query_vectors.ndim != 2 or query_vectors.shape != (len(queries), embedding_dimension):
        raise RuntimeError(
            f'Embedding model returned query vectors with shape '
            f'{query_vectors.shape}; expected '
            f'({len(queries)}, {embedding_dimension})'
        )
    return query_vectors


def build_training_filter(train: list[dict[str, Any]], train_size: int | None) -> str:
    """Describe the configured retrieval pool inside the complete master table."""

    if not train:
        raise ValueError('Cannot build a training filter from an empty pool')

    professions = sorted({str(row[Column.PROFESSION]) for row in train})
    quoted_professions = ', '.join(f"'{profession}'" for profession in professions)
    training_filter = f'{Column.PROFESSION.value} IN ({quoted_professions})'

    if train_size is not None:
        cutoff = max(int(row[Column.TRAIN_ORDER]) for row in train)
        training_filter += f' AND {Column.TRAIN_ORDER.value} <= {cutoff}'

    return training_filter


def _get_semantic_candidate_page(
        semantic_table: LanceTable,
        query_vector: np.ndarray,
        training_filter: str,
        limit: int,
        offset: int = 0,
) -> list[dict[str, Any]]:
    """Return exact cosine-nearest LanceDB rows in semantic rank order."""

    rows = (
        semantic_table.search(query_vector)
        .distance_type('cosine')
        .bypass_vector_index()
        .where(training_filter, prefilter=True)
        .select([
            Column.ID,
            Column.SPLIT,
            Column.HARD_TEXT,
            Column.PROFESSION,
            Column.GENDER,
            '_distance',
        ])
        .limit(limit)
        .offset(offset)
        .to_list()
    )

    candidates: list[dict[str, Any]] = []
    for row in rows:
        distance = float(row.pop('_distance'))
        candidates.append({**row, 'retrieval_score': 1.0 - distance})

    return candidates


def _get_balanced_semantic_candidates(
        semantic_table: LanceTable,
        query_vector: np.ndarray,
        training_filter: str,
        example_count: int,
        profession_gender_pairs: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Select nearest candidates while balancing profession, gender, and pairs."""

    if not profession_gender_pairs:
        raise ValueError('Balanced semantic retrieval requires profession-gender pairs')

    professions = tuple(sorted({profession for profession, _ in profession_gender_pairs}))
    genders = tuple(sorted({gender for _, gender in profession_gender_pairs}))

    profession_counts = Counter()
    gender_counts = Counter()
    pair_counts = Counter()

    unselected_candidates: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    row_count = semantic_table.count_rows(training_filter)
    offset = 0
    page_size = max(32, 8 * example_count)

    while len(selected) < example_count:
        minimum_profession_count = min(profession_counts[profession] for profession in professions)
        minimum_gender_count = min(gender_counts[gender] for gender in genders)
        minimum_pair_count = min(pair_counts[pair] for pair in profession_gender_pairs)

        first_eligible_index = next(
            (
                index
                for index, candidate in enumerate(unselected_candidates)
                if profession_counts[candidate[Column.PROFESSION]] == minimum_profession_count
                   and gender_counts[candidate[Column.GENDER]] == minimum_gender_count
                   and pair_counts[(candidate[Column.PROFESSION], candidate[Column.GENDER])] == minimum_pair_count
            ),
            None,
        )

        if first_eligible_index is None:
            if offset >= row_count:
                break

            page = _get_semantic_candidate_page(
                semantic_table,
                query_vector,
                training_filter,
                limit=min(page_size, row_count - offset),
                offset=offset,
            )
            if not page:
                break

            unselected_candidates.extend(page)
            offset += len(page)
            page_size *= 2
            continue

        candidate = unselected_candidates.pop(first_eligible_index)
        profession = candidate[Column.PROFESSION]
        gender = candidate[Column.GENDER]

        selected.append(candidate)
        profession_counts[profession] += 1
        gender_counts[gender] += 1
        pair_counts[(profession, gender)] += 1

    if len(selected) != example_count:
        raise RuntimeError(
            f'Could not select {example_count} balanced semantic examples after scanning {offset} of {row_count} training rows'
        )

    return selected


def retrieve_examples(
        retrieval_method: str,
        query_vector: np.ndarray,
        semantic_table: LanceTable,
        training_filter: str,
        example_count: int,
        training_profession_gender_pairs: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Retrieve exact semantic or relevance-first balanced examples."""

    if retrieval_method not in RETRIEVAL_METHODS:
        raise ValueError(f'Unknown retrieval method {retrieval_method!r}; expected one of {sorted(RETRIEVAL_METHODS)}')

    row_count = semantic_table.count_rows(training_filter)
    if example_count > row_count:
        raise ValueError(f'Cannot retrieve {example_count} examples from {row_count} training rows')

    if retrieval_method == 'semantic':
        candidates = _get_semantic_candidate_page(semantic_table, query_vector, training_filter, example_count)

        if len(candidates) != example_count:
            raise RuntimeError(f'LanceDB returned {len(candidates)} candidates; expected {example_count}')
        return candidates

    return _get_balanced_semantic_candidates(
        semantic_table,
        query_vector,
        training_filter,
        example_count,
        training_profession_gender_pairs,
    )


def order_examples(
        examples: list[dict[str, Any]],
        order: str,
        seed: int,
) -> list[dict[str, Any]]:
    """Order one already-selected demonstration set for prompt presentation."""

    if order not in EXAMPLE_ORDERS:
        raise ValueError(f'Unknown example order {order!r}; expected one of {sorted(EXAMPLE_ORDERS)}')

    if order == 'most_similar_first':
        return sorted(
            examples,
            key=lambda example: example['retrieval_score'],
            reverse=True,
        )

    if order == 'most_similar_last':
        return sorted(
            examples,
            key=lambda example: example['retrieval_score'],
        )

    if order == 'shuffle':
        random.Random(seed).shuffle(examples)
        return examples

    raise RuntimeError(f'Example order {order!r} is allowed but not implemented')


def render_input(row: dict[str, Any], target: Column) -> str:
    """Render the biography plus the audit column."""

    audit_column = TARGET_TO_AUDIT_COLUMN[target]
    return (
        f'Biography: {row[Column.HARD_TEXT]}\n'
        f'{display_column_name(audit_column)}: {row[audit_column]}'
    )


def build_prompt(
        query: dict[str, Any],
        examples: list[dict[str, Any]],
        target: Column,
        target_labels: list[str],
        master_prompt: str,
) -> list[dict[str, str]]:
    """Build a structured chat with zero or more demonstrations and a target-free query."""

    # Rows contain decoded target-label names, not raw numeric IDs.
    audit_column = TARGET_TO_AUDIT_COLUMN[target]
    target_name = display_column_name(target)
    audit_column_name = display_column_name(audit_column)
    target_labels_text = ', '.join(target_labels)
    try:
        instruction = master_prompt.format(
            target=target_name,
            audit_column=audit_column_name,
            labels=target_labels_text,
        )
    except KeyError as exc:
        raise ValueError('Prompt templates may use only {target}, {audit_column}, and {labels}') from exc

    messages = [{'role': 'system', 'content': instruction}]
    for example in examples:
        messages.extend([
            {'role': 'user', 'content': render_input(example, target)},
            {'role': 'assistant', 'content': example[target]},
        ])

    messages.append({'role': 'user', 'content': render_input(query, target)})

    return messages


def load_language_model(
        language_model_id: str,
        revision: str,
        device: str,
        dtype_name: str,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load one causal Hugging Face language model."""

    tokenizer = cast(
        PreTrainedTokenizerBase,
        cast(object, AutoTokenizer.from_pretrained(language_model_id, revision=revision)),
    )

    dtype: str | torch.dtype = 'auto' if dtype_name == 'auto' else TORCH_DTYPES[dtype_name]
    language_model = AutoCausalLanguageModel.from_pretrained(language_model_id, revision=revision, dtype=dtype)

    language_model.to(device)
    language_model.eval()

    return tokenizer, language_model


def _apply_chat_template(messages: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Render structured messages directly as one unpadded token sequence."""

    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError(f'{tokenizer.name_or_path} must provide a chat template for this experiment')

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
        add_generation_prompt=False,
        enable_thinking=False,
        preserve_thinking=False,
    )

    if not isinstance(encoded, Mapping):
        raise TypeError('The chat template must return a token dict')

    input_ids = encoded['input_ids']
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(
            f'The chat template returned input_ids with shape {tuple(input_ids.shape)}; expected (1, sequence_length)'
        )

    return input_ids[0]


def _prepare_allowed_label_token_ids(
        messages: list[dict[str, str]],
        allowed_labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        language_model: PreTrainedModel,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the common assistant prefix and exact content tokens for every label."""

    if not allowed_labels:
        raise ValueError('At least one allowed label is required')

    context_length = _language_model_context_length(tokenizer, language_model)
    empty_answer_ids = _apply_chat_template(
        [
            *messages,
            {'role': 'assistant', 'content': ''}
        ],
        tokenizer,
    )
    empty_token_ids: list[int] = empty_answer_ids.tolist()

    reference_prompt_ids: torch.Tensor | None = None
    candidate_ids_by_label: dict[str, torch.Tensor] = {}
    for label in allowed_labels:
        full_answer_ids = _apply_chat_template(
            [
                *messages,
                {'role': 'assistant', 'content': label}
            ],
            tokenizer,
        )
        full_token_ids: list[int] = full_answer_ids.tolist()

        if len(full_token_ids) <= len(empty_token_ids):
            raise RuntimeError(f'Allowed label {label} did not add tokens to the formatted chat')

        prefix_length = 0
        while prefix_length < len(empty_token_ids) and empty_token_ids[prefix_length] == full_token_ids[prefix_length]:
            prefix_length += 1

        suffix_length = 0
        while (
                suffix_length < len(empty_token_ids) - prefix_length
                and empty_token_ids[-1 - suffix_length] == full_token_ids[-1 - suffix_length]
        ):
            suffix_length += 1

        if prefix_length + suffix_length != len(empty_token_ids):
            raise RuntimeError(
                f'{tokenizer.name_or_path} does not expose an unambiguous assistant-content span through its chat template'
            )

        candidate_stop = len(full_token_ids) - suffix_length
        prompt_ids = full_answer_ids[:prefix_length]
        candidate_ids = full_answer_ids[prefix_length:candidate_stop]

        scored_sequence_length = prefix_length + candidate_ids.numel()
        if scored_sequence_length > context_length:
            raise ValueError(
                f'The formatted prompt plus candidate label {label!r} contains {scored_sequence_length} tokens, '
                f'exceeding {tokenizer.name_or_path}\'s {context_length}-token context limit. '
                f'Shorten the prompt or reduce retrieval.example_counts.'
            )

        if prompt_ids.numel() == 0:
            raise ValueError('The formatted chat produced no prompt tokens')
        if candidate_ids.numel() == 0:
            raise ValueError(f'Allowed label {label} produced no tokens')
        if reference_prompt_ids is None:
            reference_prompt_ids = prompt_ids
        elif not torch.equal(reference_prompt_ids, prompt_ids):
            raise RuntimeError('The chat template produced label-dependent prompt tokens')

        candidate_ids_by_label[label] = candidate_ids

    if reference_prompt_ids is None:
        raise RuntimeError('The chat template produced no reference prompt')

    return reference_prompt_ids, candidate_ids_by_label


def generate_allowed_label(
        messages: list[dict[str, str]],
        allowed_labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        language_model: PreTrainedModel,
        device: str,
) -> tuple[str, str]:
    """Generate one assistant response and accept only an exact configured label."""

    if not allowed_labels:
        raise ValueError('At least one allowed label is required')
    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError(f'{tokenizer.name_or_path} must provide a chat template for this experiment')

    model_inputs = cast(
        BatchEncoding,
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=False,
        ),
    ).to(device)
    prompt_length = model_inputs['input_ids'].shape[1]

    context_length = _language_model_context_length(tokenizer, language_model)
    generated_sequence_length = prompt_length + GENERATED_OUTPUT_MAX_NEW_TOKENS

    if generated_sequence_length > context_length:
        raise ValueError(
            f'The formatted prompt plus {GENERATED_OUTPUT_MAX_NEW_TOKENS} generated tokens contains '
            f'{generated_sequence_length} tokens, exceeding {tokenizer.name_or_path}\'s '
            f'{context_length}-token context limit. Shorten the prompt or reduce retrieval.example_counts.'
        )

    generation_eos_token_id = language_model.generation_config.eos_token_id
    if language_model.config.model_type == 'olmo3':
        # OLMo demonstrations end with <|im_end|>, which its final generation config omits.
        end_of_turn_token_id = tokenizer.get_added_vocab().get('<|im_end|>')
        if end_of_turn_token_id is not None:
            generation_eos_token_id = [generation_eos_token_id, end_of_turn_token_id]
        else:
            raise RuntimeError(f'{tokenizer.name_or_path} does not define the OLMo end-of-turn token <|im_end|>')


    with torch.inference_mode():
        generated_ids = language_model.generate(
            **model_inputs,
            do_sample=False,
            eos_token_id=generation_eos_token_id,
            max_new_tokens=GENERATED_OUTPUT_MAX_NEW_TOKENS,
            return_dict_in_generate=False,
        )

    new_token_ids = generated_ids[0, prompt_length:]
    raw_output = cast(
        str,
        tokenizer.decode(new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
    )

    predicted_label = raw_output.strip().lower()
    if predicted_label not in allowed_labels:
        raise ValueError(
            f'Language model generated {raw_output!r}; expected exactly one of {allowed_labels!r} '
            f'after trimming surrounding whitespace and converting to lowercase'
        )

    return predicted_label, raw_output


def score_allowed_labels(
        messages: list[dict[str, str]],
        allowed_labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        language_model: PreTrainedModel,
        device: str,
) -> tuple[str, dict[str, float]]:
    """Choose among allowed labels using mean conditional token log-probability.

    Each label is rendered as the final assistant message. Comparing that
    rendering with an empty assistant message isolates the exact
    language-model-specific assistant prefix, label tokens, and end markers.
    Mean rather than summed log-probability reduces the automatic disadvantage
    of labels that use more tokenizer tokens. The common prompt is evaluated
    once, then its inference cache is copied for each multi-token label so only
    that label's remaining tokens require additional model work.
    """

    reference_prompt_ids, candidate_ids_by_label = _prepare_allowed_label_token_ids(
        messages,
        allowed_labels,
        tokenizer,
        language_model,
    )

    text_config = getattr(language_model.config, 'text_config', language_model.config)
    if getattr(text_config, 'num_kv_shared_layers', 0) > 0:
        raise RuntimeError('Inference-cache reuse does not support language models with shared KV layers')

    prompt_input_ids = reference_prompt_ids.unsqueeze(0).to(device)

    with torch.inference_mode():
        prompt_output = language_model(
            input_ids=prompt_input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        if prompt_output.logits.shape[:2] != (1, 1):
            raise RuntimeError(
                f'Could not align language model logits for the common prompt; '
                f'received shape {tuple(prompt_output.logits.shape)}'
            )
        if prompt_output.past_key_values is None:
            raise RuntimeError('The language model did not return an inference cache for the common prompt')

        first_token_log_probabilities = torch.log_softmax(prompt_output.logits[0, 0].float(), dim=-1)

        scores: dict[str, float] = {}
        for label in allowed_labels:
            candidate_ids = candidate_ids_by_label[label]
            candidate_on_device = candidate_ids.to(device)
            selected_token_log_probabilities = first_token_log_probabilities[candidate_on_device[0]].unsqueeze(0)

            if candidate_ids.numel() > 1:
                candidate_cache = copy.deepcopy(prompt_output.past_key_values)
                continuation_ids = candidate_on_device[:-1].unsqueeze(0)
                attention_mask = torch.ones(
                    (1, reference_prompt_ids.numel() + continuation_ids.shape[1]),
                    dtype=torch.long,
                    device=device,
                )
                continuation_output = language_model(
                    input_ids=continuation_ids,
                    attention_mask=attention_mask,
                    past_key_values=candidate_cache,
                    use_cache=False,
                    logits_to_keep=continuation_ids.shape[1],
                )
                continuation_logits = continuation_output.logits[0]

                if continuation_logits.shape[0] != continuation_ids.shape[1]:
                    raise RuntimeError(f'Could not align language model logits for allowed label {label}')

                continuation_log_probabilities = torch.log_softmax(continuation_logits.float(), dim=-1)
                selected_continuation_log_probabilities = continuation_log_probabilities.gather(
                    1, candidate_on_device[1:].unsqueeze(1)
                ).squeeze(1)

                selected_token_log_probabilities = torch.cat((
                    selected_token_log_probabilities,
                    selected_continuation_log_probabilities,
                ))

                del continuation_output, candidate_cache

            score = selected_token_log_probabilities.mean().item()

            if not np.isfinite(score):
                raise RuntimeError(f'Language model returned a non-finite score for allowed label {label}')

            scores[label] = score

    predicted_label = max(allowed_labels, key=lambda label: scores[label])
    return predicted_label, scores
