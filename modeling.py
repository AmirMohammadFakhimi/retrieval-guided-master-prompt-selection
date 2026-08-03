import gc
import random
import shlex
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

import lancedb
import numpy as np
import torch
from lancedb.table import LanceTable
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from dataset import (
    Column,
    TARGET_TO_OTHER_COLUMN,
    display_column_name,
)

RETRIEVAL_METHODS = frozenset({'semantic', 'balanced_semantic'})
EXAMPLE_ORDERS = frozenset({'as_retrieved', 'reverse', 'shuffle'})
LANCEDB_INGEST_BATCH_SIZE = 2048
TORCH_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}
LLM_DTYPES = frozenset({*TORCH_DTYPES, 'auto'})


class SemanticResource(TypedDict):
    """LanceDB table and matching evaluation embeddings for one embedding model."""

    table: LanceTable
    validation_vectors: np.ndarray
    test_vectors: np.ndarray


def choose_device(requested: str = 'auto') -> str:
    """Choose CUDA, then Apple MPS, then CPU."""

    if requested not in {'auto', 'cuda', 'mps', 'cpu'}:
        raise ValueError('llm.device must be auto, cuda, mps, or cpu')

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


def prepare_semantic_retrieval(
        train: list[dict[str, Any]],
        queries: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        device: str,
        database_path: Path,
        progress: Callable[[str], None] = print,
) -> tuple[LanceTable, np.ndarray]:
    """Open or build the persistent LanceDB training table and encode queries."""

    embedding_model_id = embedding_model['id']
    embedding_dimension = embedding_model['dimension']
    max_sequence_length = embedding_model['max_sequence_length']
    batch_size = embedding_model['batch_size']
    dtype_name = embedding_model['dtype']
    query_prompt = embedding_model['query_prompt']
    database = lancedb.connect(database_path)
    table_name = f'semantic_{embedding_model_id.lower().replace("/", "_")}'

    if table_name in database.list_tables().tables:
        table = database.open_table(table_name)
        if table.count_rows() != len(train):
            raise RuntimeError(
                f'LanceDB table {table_name} does not match the configured '
                'training pool. Delete the LanceDB directory and rerun from '
                'the beginning:\n'
                f'rm -rf -- {shlex.quote(str(database_path))}'
            )
        progress(
            f'Reusing {len(train)} cached training embeddings from '
            f'{database_path / table_name}'
        )
    else:
        table = None

    encoder = SentenceTransformer(
        embedding_model_id,
        device=device,
        model_kwargs={'dtype': TORCH_DTYPES[dtype_name]},
        truncate_dim=embedding_dimension,
    )
    encoder.max_seq_length = max_sequence_length

    if table is None:
        progress(
            f'Building LanceDB table {table_name} with '
            f'{len(train)} training embeddings'
        )
        progress_bar = tqdm(
            total=len(train),
            desc='Embedding training rows for LanceDB',
            unit='row',
        )

        try:
            for start in range(0, len(train), LANCEDB_INGEST_BATCH_SIZE):
                batch = train[start:start + LANCEDB_INGEST_BATCH_SIZE]
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
                        'vector': vector.tolist(),
                    }
                    for row, vector in zip(batch, vectors, strict=True)
                ]

                if table is None:
                    table = database.create_table(table_name, data=records)
                else:
                    cast(LanceTable, table).add(records)

                progress_bar.update(len(batch))
            progress_bar.close()

        except Exception:
            progress_bar.close()
            if table_name in database.list_tables().tables:
                database.drop_table(table_name)
            raise

    if table is None:
        raise RuntimeError('Cannot build a LanceDB table from an empty training pool')

    query_vectors = np.asarray(
        encoder.encode(
            [row[Column.HARD_TEXT] for row in queries],
            prompt=query_prompt,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    )

    if query_vectors.ndim != 2 or query_vectors.shape != (len(queries), embedding_dimension):
        raise RuntimeError(
            f'Embedding model returned query vectors with shape '
            f'{query_vectors.shape}; expected '
            f'({len(queries)}, {embedding_dimension})'
        )

    del encoder
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return cast(LanceTable, table), query_vectors


def _get_semantic_candidate_page(
        semantic_table: LanceTable,
        query_vector: np.ndarray,
        limit: int,
        offset: int = 0,
) -> list[dict[str, Any]]:
    """Return exact cosine-nearest LanceDB rows in semantic rank order."""

    rows = (
        semantic_table.search(query_vector)
        .distance_type('cosine')
        .bypass_vector_index()
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
        example_count: int,
        cells: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Select the nearest candidates while keeping all group counts balanced."""

    if not cells:
        raise ValueError('Balanced semantic retrieval requires training cells')

    professions = tuple(sorted({profession for profession, _ in cells}))
    genders = tuple(sorted({gender for _, gender in cells}))

    profession_counts = Counter()
    gender_counts = Counter()
    cell_counts = Counter()

    unselected_candidates: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    row_count = semantic_table.count_rows()
    offset = 0
    page_size = max(32, 8 * example_count)

    while len(selected) < example_count:
        minimum_profession_count = min(profession_counts[profession] for profession in professions)
        minimum_gender_count = min(gender_counts[gender] for gender in genders)
        minimum_cell_count = min(cell_counts[cell] for cell in cells)

        first_eligible_index = next(
            (
                index
                for index, candidate in enumerate(unselected_candidates)
                if profession_counts[candidate[Column.PROFESSION]] == minimum_profession_count
                   and gender_counts[candidate[Column.GENDER]] == minimum_gender_count
                   and cell_counts[(candidate[Column.PROFESSION], candidate[Column.GENDER])] == minimum_cell_count
            ),
            None,
        )

        if first_eligible_index is None:
            if offset >= row_count:
                break

            page = _get_semantic_candidate_page(
                semantic_table,
                query_vector,
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
        cell_counts[(profession, gender)] += 1

    if len(selected) != example_count:
        raise RuntimeError(
            f'Could not select {example_count} balanced semantic examples after scanning {offset} of {row_count} training rows'
        )

    return selected


def retrieve_examples(
        method: str,
        query_vector: np.ndarray,
        semantic_table: LanceTable,
        example_count: int,
        cells: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Retrieve exact semantic or relevance-first balanced examples."""

    if method not in RETRIEVAL_METHODS:
        raise ValueError(f'Unknown retrieval method {method!r}; expected one of {sorted(RETRIEVAL_METHODS)}')

    row_count = semantic_table.count_rows()
    if example_count > row_count:
        raise ValueError(f'Cannot retrieve {example_count} examples from {row_count} training rows')

    if method == 'semantic':
        candidates = _get_semantic_candidate_page(semantic_table, query_vector, example_count)
        if len(candidates) != example_count:
            raise RuntimeError(f'LanceDB returned {len(candidates)} candidates; expected {example_count}')
        return candidates

    return _get_balanced_semantic_candidates(semantic_table, query_vector, example_count, cells)


def order_examples(
        examples: list[dict[str, Any]],
        order: str,
        seed: int,
) -> list[dict[str, Any]]:
    """Apply the configured demonstration order after retrieval."""

    if order not in EXAMPLE_ORDERS:
        raise ValueError(f'Unknown example order {order!r}; expected one of {sorted(EXAMPLE_ORDERS)}')

    if order == 'as_retrieved':
        return examples

    if order == 'reverse':
        return examples[::-1]

    if order == 'shuffle':
        random.Random(seed).shuffle(examples)
        return examples

    raise RuntimeError(f'Example order {order!r} is allowed but not implemented')


def render_input(row: dict[str, Any], target: Column) -> str:
    """Render the biography plus the non-target structured column."""

    other_column = TARGET_TO_OTHER_COLUMN[target]
    return (
        f'Biography: {row[Column.HARD_TEXT]}\n'
        f'{display_column_name(other_column)}: {row[other_column]}'
    )


def build_prompt(
        query: dict[str, Any],
        examples: list[dict[str, Any]],
        target: Column,
        labels: list[str],
        master_prompt: str,
) -> list[dict[str, str]]:
    """Build a structured chat with demonstrations and a target-free query."""

    # Rows must contain decoded class names from load_data, not raw numeric IDs.
    other_column = TARGET_TO_OTHER_COLUMN[target]
    target_name = display_column_name(target)
    other_column_name = display_column_name(other_column)
    labels_text = ', '.join(labels)
    try:
        instruction = master_prompt.format(
            target=target_name,
            other_column=other_column_name,
            labels=labels_text,
        )
    except KeyError as exc:
        raise ValueError(f'Prompt templates may use only {target}, {other_column}, and {labels}') from exc

    instruction_block = (
        f'{instruction}\n'
        f'Allowed values for {target_name}: {labels_text}.\n'
        'Output exactly one allowed value and no explanation.'
    )

    messages = [
        {'role': 'system', 'content': instruction_block}
    ]
    for example in examples:
        messages.extend([
            {'role': 'user', 'content': render_input(example, target)},
            {'role': 'assistant', 'content': example[target]},
        ])
    messages.append(
        {'role': 'user', 'content': render_input(query, target)}
    )
    return messages


def load_llm(
        llm_id: str,
        revision: str,
        device: str,
        dtype_name: str,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load one causal Hugging Face LLM."""

    tokenizer = cast(
        PreTrainedTokenizerBase,
        cast(object, AutoTokenizer.from_pretrained(llm_id, revision=revision)),
    )

    dtype: str | torch.dtype = 'auto' if dtype_name == 'auto' else TORCH_DTYPES[dtype_name]
    llm = AutoModelForCausalLM.from_pretrained(llm_id, revision=revision, dtype=dtype)

    llm.to(device)
    llm.eval()

    return tokenizer, llm


def clear_llm_memory(device: str) -> None:
    """Collect released LLM objects and clear the active accelerator cache."""

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


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
    )

    if not isinstance(encoded, dict):
        raise TypeError('The chat template must return a token dict')

    input_ids = encoded['input_ids']
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(
            f'The chat template returned input_ids with shape {tuple(input_ids.shape)}; expected (1, sequence_length)'
        )

    return input_ids[0]


def score_allowed_labels(
        messages: list[dict[str, str]],
        labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        llm: PreTrainedModel,
        device: str,
) -> tuple[str, dict[str, float]]:
    """Choose among allowed labels using mean conditional token log-probability.

    Each label is rendered as the final assistant message. Comparing that
    rendering with an empty assistant message isolates the exact LLM-specific
    assistant prefix, label tokens, and end markers. Mean rather than summed
    log-probability reduces the automatic disadvantage of labels that use more
    tokenizer tokens. Labels are evaluated sequentially to keep accelerator
    memory small.
    """

    if not labels:
        raise ValueError('At least one allowed label is required')

    empty_answer_ids = _apply_chat_template(
        [
            *messages,
            {'role': 'assistant', 'content': ''}
        ],
        tokenizer,
    )
    empty_token_ids: list[int] = empty_answer_ids.tolist()

    scores: dict[str, float] = {}
    reference_prompt_ids: torch.Tensor | None = None
    for label in labels:
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

        if prompt_ids.numel() == 0:
            raise ValueError('The formatted chat produced no prompt tokens')
        if candidate_ids.numel() == 0:
            raise ValueError(f'Allowed label {label} produced no tokens')
        if reference_prompt_ids is None:
            reference_prompt_ids = prompt_ids
        elif not torch.equal(reference_prompt_ids, prompt_ids):
            raise RuntimeError('The chat template produced label-dependent prompt tokens')

        scoring_ids = full_answer_ids[:candidate_stop - 1].unsqueeze(0).to(device)

        with torch.inference_mode():
            # Omitting the final candidate token leaves exactly the C logits
            # that predict the C candidate tokens.
            output = llm(input_ids=scoring_ids, use_cache=False, logits_to_keep=candidate_ids.numel())
            candidate_logits = output.logits[0]

            if candidate_logits.shape[0] != candidate_ids.numel():
                raise RuntimeError(f'Could not align LLM logits for allowed label {label}')

            token_log_probabilities = torch.log_softmax(candidate_logits.float(), dim=-1)
            candidate_on_device = candidate_ids.to(device)
            selected_token_log_probabilities = token_log_probabilities.gather(
                1, candidate_on_device.unsqueeze(1)
            ).squeeze(1)
            score = selected_token_log_probabilities.mean().item()

        if not np.isfinite(score):
            raise RuntimeError(f'LLM returned a non-finite score for allowed label {label}')
        scores[label] = score

    predicted_label = max(labels, key=lambda label: scores[label])
    return predicted_label, scores
