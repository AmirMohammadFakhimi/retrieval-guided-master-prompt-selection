import copy
import gc
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM as AutoCausalLanguageModel,
    AutoModelForImageTextToText as AutoMultimodalLanguageModel,
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

TORCH_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}
LANGUAGE_MODEL_DTYPES = frozenset({*TORCH_DTYPES, 'auto'})
GENERATED_OUTPUT_MAX_NEW_TOKENS = 32
UNBOUNDED_TOKENIZER_LENGTH = 1_000_000_000


class GeneratedOutputError(ValueError):
    """Report which item in a generated-output batch failed validation."""

    def __init__(self, batch_index: int, message: str) -> None:
        super().__init__(message)
        self.batch_index = batch_index


def _language_model_context_length(tokenizer: PreTrainedTokenizerBase, language_model: PreTrainedModel) -> int:
    """Return the tokenizer limit or the underlying text-model limit."""

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


def clear_model_memory(device: str) -> None:
    """Collect released model objects and clear the active accelerator cache."""

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


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

    model_config = AutoConfig.from_pretrained(language_model_id, revision=revision)
    model_loader = AutoCausalLanguageModel
    if model_config.model_type == 'mistral3':
        # The official checkpoint wraps its causal text model in a multimodal Mistral3 configuration.
        model_loader = AutoMultimodalLanguageModel

    tokenizer = cast(
        PreTrainedTokenizerBase,
        cast(object, AutoTokenizer.from_pretrained(language_model_id, revision=revision)),
    )

    dtype: str | torch.dtype = 'auto' if dtype_name == 'auto' else TORCH_DTYPES[dtype_name]
    language_model = model_loader.from_pretrained(language_model_id, revision=revision, dtype=dtype)

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
    if language_model.config.model_type == 'mistral3':
        # Mistral rejects an empty assistant message, but its completed answers end with EOS.
        empty_answer_ids = torch.cat((
            _apply_chat_template(messages, tokenizer),
            torch.tensor([tokenizer.eos_token_id], dtype=torch.long),
        ))
    else:
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


def generate_allowed_labels(
        message_batch: list[list[dict[str, str]]],
        allowed_labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        language_model: PreTrainedModel,
        device: str,
) -> list[tuple[str, str]]:
    """Generate batched assistant responses and accept only exact configured labels."""

    if not allowed_labels:
        raise ValueError('At least one allowed label is required')
    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError(f'{tokenizer.name_or_path} must provide a chat template for this experiment')

    model_inputs = cast(
        BatchEncoding,
        tokenizer.apply_chat_template(
            message_batch,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors='pt',
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=False,
            tokenizer_kwargs={
                'padding_side': 'left',
                'return_attention_mask': True,
            },
        ),
    )
    prompt_width = model_inputs['input_ids'].shape[1]
    prompt_lengths = model_inputs['attention_mask'].sum(dim=1).tolist()

    context_length = _language_model_context_length(tokenizer, language_model)
    for batch_index, prompt_length in enumerate(prompt_lengths):
        generated_sequence_length = int(prompt_length) + GENERATED_OUTPUT_MAX_NEW_TOKENS
        if generated_sequence_length > context_length:
            raise GeneratedOutputError(
                batch_index,
                f'The formatted prompt plus {GENERATED_OUTPUT_MAX_NEW_TOKENS} generated tokens contains '
                f'{generated_sequence_length} tokens, exceeding {tokenizer.name_or_path}\'s '
                f'{context_length}-token context limit. Shorten the prompt or reduce retrieval.example_counts.',
            )

    model_inputs = model_inputs.to(device)

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
            max_length=None,
            max_new_tokens=GENERATED_OUTPUT_MAX_NEW_TOKENS,
            return_dict_in_generate=False,
        )

    new_token_ids = generated_ids[:, prompt_width:]
    raw_outputs = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    predictions: list[tuple[str, str]] = []
    for batch_index, raw_output in enumerate(raw_outputs):
        predicted_label = raw_output.strip().lower()

        if predicted_label not in allowed_labels:
            raise GeneratedOutputError(
                batch_index,
                f'Language model generated {raw_output!r}; expected exactly one of {allowed_labels!r} '
                f'after trimming surrounding whitespace and converting to lowercase',
            )
        predictions.append((predicted_label, raw_output))

    return predictions


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
