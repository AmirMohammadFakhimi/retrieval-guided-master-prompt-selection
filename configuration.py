"""Load and validate the experiment configuration."""

import string
from pathlib import Path
from typing import Any

import yaml

import dataset
import evaluation
import modeling


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    return config


def _require_non_empty_string(value: Any, setting: str) -> str:
    """Return a stripped string or raise a setting-specific error."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{setting} must be a non-empty string')
    return value.strip()


def _require_integer(value: Any, setting: str, minimum: int) -> int:
    """Return a real integer, excluding booleans, at or above a minimum."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = 'non-negative' if minimum == 0 else 'positive'
        raise ValueError(f'{setting} must be a {qualifier} integer')
    return value


def _require_enum(value: Any, setting: str, allowed: set[str] | frozenset[str]) -> str:
    """Return a supported string value or report the exact allowed values."""

    if not isinstance(value, str) or value not in allowed:
        allowed_text = ', '.join(sorted(allowed))
        raise ValueError(f'{setting} must be one of: {allowed_text}')
    return value


def _validate_defaults(defaults: dict[str, Any]) -> None:
    """Validate run-wide target, split, output, and ranking settings."""

    _require_enum(defaults.get('target'), 'defaults.target', {'profession', 'gender'})
    _require_enum(
        defaults.get('evaluation_split'),
        'defaults.evaluation_split',
        {'validation', 'test'},
    )
    _require_enum(
        defaults.get('ranking_direction'),
        'defaults.ranking_direction',
        {'maximize', 'minimize'},
    )
    _require_integer(defaults.get('seed'), 'defaults.seed', 0)
    _require_non_empty_string(defaults.get('output_dir'), 'defaults.output_dir')
    ranking_metric = _require_non_empty_string(
        defaults.get('ranking_metric'),
        'defaults.ranking_metric',
    )
    resolved_metric = evaluation.resolve_metric_column(ranking_metric)
    if resolved_metric not in evaluation.RESULT_NUMERIC_COLUMNS:
        aliases = ', '.join(sorted(evaluation.METRIC_COLUMN_ALIASES))
        columns = ', '.join(sorted(evaluation.RESULT_NUMERIC_COLUMNS))
        raise ValueError(
            f'defaults.ranking_metric must be a numeric result column or alias. '
            f'Aliases: {aliases}. Result columns: {columns}'
        )


def _validate_dataset(config: dict[str, Any]) -> int | None:
    """Validate dataset settings and return the optional train cap."""

    dataset_settings = config['dataset']
    _require_non_empty_string(dataset_settings.get('file'), 'dataset.file')
    _require_non_empty_string(dataset_settings.get('hub_id'), 'dataset.hub_id')
    professions = dataset_settings.get('professions')
    if professions != 'all':
        if not isinstance(professions, list) or not professions:
            raise ValueError('dataset.professions must be "all" or a non-empty list')
        if any(not isinstance(value, str) or not value.strip() for value in professions):
            raise ValueError('dataset.professions list entries must be non-empty strings')
        if len(professions) != len(set(professions)):
            raise ValueError('dataset.professions cannot contain duplicates')
    _require_integer(dataset_settings.get('shuffle_seed'), 'dataset.shuffle_seed', 0)
    evaluation_per_cell = dataset_settings.get('evaluation_per_profession_gender')
    if evaluation_per_cell != 'max_balanced' and (
            isinstance(evaluation_per_cell, bool)
            or not isinstance(evaluation_per_cell, int)
            or evaluation_per_cell < 1
    ):
        raise ValueError(
            'dataset.evaluation_per_profession_gender must be a positive '
            'integer or max_balanced'
        )
    dataset.task_settings(config)
    return dataset.train_size_limit(config)


def _validate_retrieval(
        retrieval: dict[str, Any],
        train_size: int | None,
) -> None:
    """Validate retrieval conditions and embedding-model settings."""

    methods = retrieval.get('methods')
    if not isinstance(methods, list) or not methods:
        raise ValueError('retrieval.methods must be a non-empty list')
    if any(not isinstance(method, str) or not method.strip() for method in methods):
        raise ValueError('retrieval.methods entries must be non-empty strings')
    unknown_methods = sorted(set(methods) - modeling.RETRIEVAL_METHODS)
    if unknown_methods:
        raise ValueError(
            f'Unknown retrieval methods {unknown_methods}; expected values from '
            f'{sorted(modeling.RETRIEVAL_METHODS)}'
        )
    if len(methods) != len(set(methods)):
        raise ValueError('retrieval.methods cannot contain duplicates')

    example_counts = retrieval.get('example_counts')
    if not isinstance(example_counts, list) or not example_counts:
        raise ValueError('retrieval.example_counts must be a non-empty list')
    for index, value in enumerate(example_counts):
        _require_integer(value, f'retrieval.example_counts[{index}]', 1)
    if train_size is not None and max(example_counts) > train_size:
        raise ValueError('Every retrieval.example_counts entry must be <= dataset.train_size')
    if len(example_counts) != len(set(example_counts)):
        raise ValueError('retrieval.example_counts cannot contain duplicates')

    orders = retrieval.get('example_orders')
    if not isinstance(orders, list) or not orders:
        raise ValueError('retrieval.example_orders must be a non-empty list')
    if any(not isinstance(order, str) or not order.strip() for order in orders):
        raise ValueError('retrieval.example_orders entries must be non-empty strings')
    unknown_orders = sorted(set(orders) - modeling.EXAMPLE_ORDERS)
    if unknown_orders:
        raise ValueError(
            f'Unknown example orders {unknown_orders}; expected values from '
            f'{sorted(modeling.EXAMPLE_ORDERS)}'
        )
    if len(orders) != len(set(orders)):
        raise ValueError('retrieval.example_orders cannot contain duplicates')

    _require_non_empty_string(retrieval.get('lancedb_path'), 'retrieval.lancedb_path')
    embedding_models = retrieval.get('embedding_models')
    if not isinstance(embedding_models, list) or not embedding_models:
        raise ValueError('retrieval.embedding_models must contain at least one embedding model')
    required_settings = {
        'id',
        'dimension',
        'max_sequence_length',
        'batch_size',
        'dtype',
        'query_prompt',
    }
    embedding_model_ids: list[str] = []
    for index, embedding_model in enumerate(embedding_models):
        if not isinstance(embedding_model, dict):
            raise ValueError(f'retrieval.embedding_models[{index}] must be a mapping')
        missing_settings = sorted(required_settings - set(embedding_model))
        if missing_settings:
            raise ValueError(f'retrieval.embedding_models[{index}] is missing: {missing_settings}')
        embedding_model_ids.append(_require_non_empty_string(
            embedding_model['id'],
            f'retrieval.embedding_models[{index}].id',
        ))
        for setting in ('dimension', 'max_sequence_length', 'batch_size'):
            _require_integer(
                embedding_model[setting],
                f'retrieval.embedding_models[{index}].{setting}',
                1,
            )
        _require_enum(
            embedding_model['dtype'],
            f'retrieval.embedding_models[{index}].dtype',
            frozenset(modeling.TORCH_DTYPES),
        )
        _require_non_empty_string(
            embedding_model['query_prompt'],
            f'retrieval.embedding_models[{index}].query_prompt',
        )
    if len(embedding_model_ids) != len(set(embedding_model_ids)):
        raise ValueError('retrieval.embedding_models cannot contain duplicate IDs')


def _validate_prompts(templates: dict[str, Any]) -> None:
    """Validate prompt names, text, braces, and supported placeholders."""

    supported_placeholders = {'target', 'audit_column', 'labels'}
    formatter = string.Formatter()
    for name, template in templates.items():
        _require_non_empty_string(name, 'prompt template name')
        text = _require_non_empty_string(template, f'prompt_templates.{name}')
        try:
            placeholders = {
                field_name
                for _, field_name, _, _ in formatter.parse(text)
                if field_name is not None
            }
        except ValueError as exc:
            raise ValueError(f'prompt_templates.{name} has invalid braces') from exc
        unknown_placeholders = sorted(placeholders - supported_placeholders)
        if unknown_placeholders:
            raise ValueError(
                f'prompt_templates.{name} has unsupported placeholders: '
                f'{unknown_placeholders}; supported values are '
                f'{sorted(supported_placeholders)}'
            )


def _validate_inference(inference: dict[str, Any]) -> None:
    """Validate language-model entries and the requested device name."""

    language_models = inference.get('language_models')
    if not isinstance(language_models, list) or not language_models:
        raise ValueError('inference.language_models must contain at least one language model')

    language_model_ids: list[str] = []
    for index, language_model in enumerate(language_models):
        if not isinstance(language_model, dict):
            raise ValueError(f'inference.language_models[{index}] must be a mapping')
        missing_settings = sorted({'id', 'revision', 'dtype'} - set(language_model))
        if missing_settings:
            raise ValueError(f'inference.language_models[{index}] is missing: {missing_settings}')
        language_model_ids.append(_require_non_empty_string(
            language_model['id'],
            f'inference.language_models[{index}].id',
        ))
        _require_non_empty_string(
            language_model['revision'],
            f'inference.language_models[{index}].revision',
        )
        _require_enum(
            language_model['dtype'],
            f'inference.language_models[{index}].dtype',
            modeling.LANGUAGE_MODEL_DTYPES,
        )
    if len(language_model_ids) != len(set(language_model_ids)):
        raise ValueError('inference.language_models cannot contain duplicate IDs')

    _require_enum(
        inference.get('device'),
        'inference.device',
        {'auto', 'cuda', 'mps', 'cpu'},
    )


def validate_config(config: dict[str, Any]) -> None:
    """Validate every documented static configuration contract."""

    if not isinstance(config, dict):
        raise ValueError('The configuration must be a YAML mapping')
    sections = ('defaults', 'dataset', 'retrieval', 'prompt_templates', 'inference')
    for section in sections:
        if not isinstance(config.get(section), dict) or not config[section]:
            raise ValueError(f'{section} must be a non-empty mapping')

    _validate_defaults(config['defaults'])
    train_size = _validate_dataset(config)
    _validate_retrieval(config['retrieval'], train_size)
    _validate_prompts(config['prompt_templates'])
    _validate_inference(config['inference'])
