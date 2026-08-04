import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import dataset
import evaluation
import modeling

RetrievalCache = dict[tuple[str, str, str], list[dict[str, Any]]]


@dataclass
class PredictionContext:
    """Values and resources shared by every prediction condition in one run."""

    example_order_seed: int
    dataset_shuffle_seed: int
    target: dataset.Column
    audit_column: dataset.Column
    labels: list[str]
    max_example_count: int
    device: str
    semantic_resources: dict[str, modeling.SemanticResource]
    training_profession_gender_pairs: tuple[tuple[str, str], ...]
    retrieval_cache: RetrievalCache


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early for the small set of settings that would invalidate a run."""

    dataset.task_settings(config)
    train_size = dataset.train_size_limit(config)
    defaults = config['defaults']
    retrieval = config['retrieval']
    inference_settings = config['inference']

    if defaults['seed'] < 0:
        raise ValueError('defaults.seed must be non-negative')
    if defaults['ranking_direction'] not in {'maximize', 'minimize'}:
        raise ValueError('defaults.ranking_direction must be maximize or minimize')
    if config['dataset']['validation_per_profession_gender'] < 1:
        raise ValueError('dataset.validation_per_profession_gender must be at least 1')
    if config['dataset']['test_per_profession_gender'] < 1:
        raise ValueError('dataset.test_per_profession_gender must be at least 1')
    if config['dataset']['shuffle_seed'] < 0:
        raise ValueError('dataset.shuffle_seed must be non-negative')
    language_model_configs = inference_settings['language_models']
    if not isinstance(language_model_configs, list) or not language_model_configs:
        raise ValueError('inference.language_models must contain at least one language model')

    language_model_ids: list[str] = []
    for index, language_model_config in enumerate(language_model_configs):
        if not isinstance(language_model_config, dict):
            raise ValueError(f'inference.language_models[{index}] must be a dict')

        missing_settings = sorted({'id', 'revision', 'dtype'} - set(language_model_config))
        if missing_settings:
            raise ValueError(f'inference.language_models[{index}] is missing: {missing_settings}')

        language_model_id = language_model_config['id']
        if not language_model_id:
            raise ValueError(f'inference.language_models[{index}].id cannot be empty')
        language_model_ids.append(language_model_id)
        if not language_model_config['revision']:
            raise ValueError(f'inference.language_models[{index}].revision cannot be empty')
        if language_model_config['dtype'] not in modeling.LANGUAGE_MODEL_DTYPES:
            allowed_dtypes = ', '.join(sorted(modeling.LANGUAGE_MODEL_DTYPES))
            raise ValueError(f'inference.language_models[{index}].dtype must be one of: {allowed_dtypes}')
    if len(language_model_ids) != len(set(language_model_ids)):
        raise ValueError('inference.language_models cannot contain duplicate IDs')

    methods = retrieval['methods']
    if not isinstance(methods, list) or not methods:
        raise ValueError('retrieval.methods must be a non-empty list')
    unknown_methods = sorted(set(methods) - modeling.RETRIEVAL_METHODS)
    if unknown_methods:
        raise ValueError(
            f'Unknown retrieval methods {unknown_methods}; expected values from {sorted(modeling.RETRIEVAL_METHODS)}'
        )
    if len(methods) != len(set(methods)):
        raise ValueError('retrieval.methods cannot contain duplicates')
    example_counts = [value for value in retrieval['example_counts']]
    if not example_counts or any(value < 1 for value in example_counts):
        raise ValueError('retrieval.example_counts must contain positive integers')
    if train_size is not None and max(example_counts) > train_size:
        raise ValueError('Every retrieval.example_counts entry must be <= dataset.train_size')
    if len(example_counts) != len(set(example_counts)):
        raise ValueError('retrieval.example_counts cannot contain duplicates')
    if not str(retrieval['lancedb_path']):
        raise ValueError('retrieval.lancedb_path cannot be empty')

    embedding_models = retrieval['embedding_models']
    if not isinstance(embedding_models, list) or not embedding_models:
        raise ValueError('retrieval.embedding_models must contain at least one embedding model')
    required_embedding_settings = {
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
            raise ValueError(f'retrieval.embedding_models[{index}] must be a dict')
        missing_settings = sorted(required_embedding_settings - set(embedding_model))
        if missing_settings:
            raise ValueError(f'retrieval.embedding_models[{index}] is missing: {missing_settings}')
        embedding_model_id = embedding_model['id']
        if not embedding_model_id:
            raise ValueError(f'retrieval.embedding_models[{index}].id cannot be empty')
        embedding_model_ids.append(embedding_model_id)
        for setting in ('dimension', 'max_sequence_length', 'batch_size'):
            value = embedding_model[setting]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'retrieval.embedding_models[{index}].{setting} must be a positive integer')
        if embedding_model['dtype'] not in modeling.TORCH_DTYPES:
            allowed_dtypes = ', '.join(modeling.TORCH_DTYPES)
            raise ValueError(f'retrieval.embedding_models[{index}].dtype must be one of: {allowed_dtypes}')
        if not embedding_model['query_prompt']:
            raise ValueError(f'retrieval.embedding_models[{index}].query_prompt cannot be empty')
    if len(embedding_model_ids) != len(set(embedding_model_ids)):
        raise ValueError('retrieval.embedding_models cannot contain duplicate IDs')

    orders = retrieval['example_orders']
    if not isinstance(orders, list) or not orders:
        raise ValueError('retrieval.example_orders must be a non-empty list')
    unknown_orders = sorted(set(orders) - modeling.EXAMPLE_ORDERS)
    if unknown_orders:
        raise ValueError(
            f'Unknown example orders {unknown_orders}; '
            f'expected values from {sorted(modeling.EXAMPLE_ORDERS)}'
        )
    if len(orders) != len(set(orders)):
        raise ValueError('retrieval.example_orders cannot contain duplicates')

    templates = config['prompt_templates']
    if not isinstance(templates, dict) or not templates:
        raise ValueError('prompt_templates must contain at least one named prompt')
    if not all(name and text for name, text in templates.items()):
        raise ValueError('prompt template names and texts cannot be empty')


def _build_conditions(
        target: dataset.Column,
        methods: list[str],
        embedding_models: list[dict[str, Any]],
        language_model_configs: list[dict[str, Any]],
        example_counts: list[int],
        example_orders: list[str],
        prompt_templates: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the full cross-product of configured experiment conditions."""

    conditions: list[dict[str, Any]] = []
    for language_model_config in language_model_configs:
        language_model_id = language_model_config['id']
        for method in methods:
            for embedding_model in embedding_models:
                embedding_model_id = embedding_model['id']
                for example_count in example_counts:
                    for example_order in example_orders:
                        for prompt_name, master_prompt in prompt_templates.items():
                            condition_name = (
                                f'{target} | language_model={language_model_id} | '
                                f'{method} | embedding={embedding_model_id} | '
                                f'examples={example_count} | {example_order} | {prompt_name}'
                            )

                            conditions.append({
                                'condition': condition_name,
                                'language_model': language_model_id,
                                'retrieval_method': method,
                                'embedding_model': embedding_model_id,
                                'example_count': example_count,
                                'example_order': example_order,
                                'prompt_name': prompt_name,
                                'master_prompt': master_prompt,
                            })
    return conditions


def _prepare_semantic_resources(
        train: list[dict[str, Any]],
        validation: list[dict[str, Any]],
        test: list[dict[str, Any]],
        embedding_models: list[dict[str, Any]],
        device: str,
        database_path: Path,
        progress: Callable[[str], None],
) -> dict[str, modeling.SemanticResource]:
    """Prepare retrieval tables and evaluation vectors for every embedding model."""

    evaluation_rows = validation + test
    semantic_resources: dict[str, modeling.SemanticResource] = {}
    progress_bar = tqdm(
        embedding_models,
        desc='Preparing embedding models',
        unit='model',
    )
    for embedding_model in progress_bar:
        embedding_model_id = embedding_model['id']

        semantic_table, evaluation_vectors = modeling.prepare_semantic_retrieval(
            train,
            evaluation_rows,
            embedding_model,
            device,
            database_path,
            progress,
        )

        semantic_resources[embedding_model_id] = {
            'table': semantic_table,
            'validation_vectors': evaluation_vectors[:len(validation)],
            'test_vectors': evaluation_vectors[len(validation):],
        }

    return semantic_resources


def _generate_condition_predictions(
        context: PredictionContext,
        language_model_condition: dict[str, Any],
        queries: list[dict[str, Any]],
        evaluation_split: Literal['validation', 'test'],
        tokenizer: PreTrainedTokenizerBase,
        language_model: PreTrainedModel,
) -> list[dict[str, Any]]:
    """Predict every row for one prompt condition and one data split."""

    retrieval_method = language_model_condition['retrieval_method']
    embedding_model_id = language_model_condition['embedding_model']
    semantic_resource = context.semantic_resources[embedding_model_id]
    semantic_table = semantic_resource['table']
    if evaluation_split == 'validation':
        query_vectors = semantic_resource['validation_vectors']
    elif evaluation_split == 'test':
        query_vectors = semantic_resource['test_vectors']
    else:
        raise ValueError(f'Unknown evaluation split: {evaluation_split}')

    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        query_vector = query_vectors[query_index]
        retrieval_key = (retrieval_method, embedding_model_id, query[dataset.Column.ID])

        if retrieval_key not in context.retrieval_cache:
            context.retrieval_cache[retrieval_key] = modeling.retrieve_examples(
                retrieval_method,
                query_vector,
                semantic_table,
                context.max_example_count,
                context.training_profession_gender_pairs,
            )

        examples = context.retrieval_cache[retrieval_key][:language_model_condition['example_count']]
        examples = modeling.order_examples(
            examples,
            language_model_condition['example_order'],
            context.example_order_seed,
        )

        messages = modeling.build_prompt(
            query,
            examples,
            context.target,
            context.labels,
            language_model_condition['master_prompt'],
        )
        predicted_label, label_scores = modeling.score_allowed_labels(
            messages,
            context.labels,
            tokenizer,
            language_model,
            context.device,
        )

        condition_metadata = {
            'condition': language_model_condition['condition'],
            'target': context.target,
            'audit_column': context.audit_column,
            'retrieval_method': retrieval_method,
            'embedding_model': embedding_model_id,
            'example_count': language_model_condition['example_count'],
            'example_order': language_model_condition['example_order'],
            'prompt_name': language_model_condition['prompt_name'],
            'master_prompt': language_model_condition['master_prompt'],
            'language_model': language_model_condition['language_model'],
            'device': context.device,
            'example_order_seed': context.example_order_seed,
            'dataset_shuffle_seed': context.dataset_shuffle_seed,
        }
        prediction_metadata = {
            'evaluation_split': evaluation_split,
            'query_id': query[dataset.Column.ID],
            dataset.Column.HARD_TEXT: query[dataset.Column.HARD_TEXT],
            dataset.Column.PROFESSION: query[dataset.Column.PROFESSION],
            dataset.Column.GENDER: query[dataset.Column.GENDER],
            'true_label': query[context.target],
            'audit_group': query[context.audit_column],
            'predicted_label': predicted_label,
            'prompt': json.dumps(messages, ensure_ascii=False),
            'label_scores': json.dumps(label_scores, sort_keys=True),
        }
        example_metadata = {
            'examples_used': len(examples),
            'examples': json.dumps([
                {
                    'id': example[dataset.Column.ID],
                    'profession': example[dataset.Column.PROFESSION],
                    'gender': example[dataset.Column.GENDER],
                    'retrieval_score': example['retrieval_score'],
                }
                for example in examples
            ]),
        }
        rows.append({
            **condition_metadata,
            **prediction_metadata,
            **example_metadata,
        })

    return rows


def _write_run_outputs(
        root: Path,
        config: dict[str, Any],
        result_tables: dict[str, pd.DataFrame],
        selected_validation: pd.DataFrame,
        labels: list[str],
) -> dict[str, Any]:
    """Write experiment tables and reports, then return their paths and data."""

    defaults = config['defaults']
    run_dir = (
            root
            / defaults['output_dir']
            / datetime.now().strftime('run-%Y%m%d-%H%M%S-%f')
    )
    run_dir.mkdir(parents=True)

    for table_name, table in result_tables.items():
        table.to_csv(run_dir / f'{table_name}.csv', index=False)

    with (run_dir / 'config_used.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    plot_path = run_dir / 'results.png'
    evaluation.plot_results(result_tables['validation_results'], plot_path)
    best_prompts_path = run_dir / 'best_prompts.txt'
    evaluation.write_best_prompts(
        best_prompts_path,
        selected_validation,
        config['prompt_templates'],
        labels,
        defaults['ranking_metric'],
        defaults['ranking_direction'],
        result_tables['results'],
    )

    return {
        'run_dir': run_dir,
        **result_tables,
        'plot': plot_path,
        'best_prompts': best_prompts_path,
    }


def run_experiment(
        config: dict[str, Any],
        project_root: str | Path = '.',
        progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the complete experiment and return tables plus output paths."""

    validate_config(config)
    root = Path(project_root).resolve()
    defaults = config['defaults']
    ranking_metric = evaluation.resolve_metric_column(defaults['ranking_metric'])
    target, audit_column, _, labels = dataset.task_settings(config)

    inference_settings = config['inference']
    language_models_config = inference_settings['language_models']
    device = modeling.choose_device(inference_settings['device'])
    progress(f'Using device: {device}')
    progress(f'Holding out {target}; language model input is hard_text + {audit_column}')

    train, validation, test, dataset_counts = dataset.load_data(config, root)
    progress(f'Loaded {len(train)} train, {len(validation)} validation, and {len(test)} test biographies')

    embedding_models = config['retrieval']['embedding_models']
    semantic_resources = _prepare_semantic_resources(
        train,
        validation,
        test,
        embedding_models,
        device,
        root / config['retrieval']['lancedb_path'],
        progress,
    )

    retrieval_methods = config['retrieval']['methods']
    example_counts = config['retrieval']['example_counts']
    example_orders = config['retrieval']['example_orders']
    prompt_templates = config['prompt_templates']

    training_profession_gender_pairs = tuple(sorted(
        {(row[dataset.Column.PROFESSION], row[dataset.Column.GENDER]) for row in train}
    ))
    prediction_context = PredictionContext(
        example_order_seed=defaults['seed'],
        dataset_shuffle_seed=config['dataset']['shuffle_seed'],
        target=target,
        audit_column=audit_column,
        labels=labels,
        max_example_count=max(example_counts),
        device=device,
        semantic_resources=semantic_resources,
        training_profession_gender_pairs=training_profession_gender_pairs,
        retrieval_cache={},
    )
    conditions = _build_conditions(
        target,
        retrieval_methods,
        embedding_models,
        language_models_config,
        example_counts,
        example_orders,
        prompt_templates,
    )

    validation_prediction_rows: list[dict[str, Any]] = []
    language_model_progress_bar = tqdm(
        language_models_config,
        desc='Validating language models',
        unit='model',
        position=0,
        leave=True,
    )
    for language_model_config in language_model_progress_bar:
        language_model_id = language_model_config['id']
        tokenizer, language_model = modeling.load_language_model(
            language_model_id,
            language_model_config['revision'],
            device,
            language_model_config['dtype'],
        )

        try:
            language_model_conditions = [
                condition
                for condition in conditions
                if condition['language_model'] == language_model_id
            ]
            condition_progress_bar = tqdm(
                language_model_conditions,
                desc=f'Validating {language_model_id}',
                unit='condition',
                position=1,
                leave=True,
            )
            for language_model_condition in condition_progress_bar:
                validation_prediction_rows.extend(
                    _generate_condition_predictions(
                        prediction_context,
                        language_model_condition,
                        validation,
                        'validation',
                        tokenizer,
                        language_model,
                    )
                )
        finally:
            del tokenizer, language_model
            modeling.clear_language_model_memory(device)

    validation_predictions = pd.DataFrame(validation_prediction_rows)

    (
        validation_results,
        validation_class_metrics,
        validation_confusion,
        validation_group_metrics,
        validation_fairness_metrics,
    ) = evaluation.calculate_metrics(validation_predictions, labels)
    validation_results = evaluation.rank_results(
        validation_results,
        ranking_metric,
        defaults['ranking_direction'],
    )
    validation_results.insert(
        2, 'selected_for_test', validation_results['rank'].eq(1)
    )
    selected_validation = validation_results.loc[
        validation_results['selected_for_test']
    ].copy()

    conditions_by_name = {
        setting['condition']: setting for setting in conditions
    }
    selected_by_language_model = {
        row['language_model']: row
        for row in selected_validation.to_dict('records')
    }
    test_prediction_rows: list[dict[str, Any]] = []
    for language_model_number, language_model_config in enumerate(
            language_models_config, start=1
    ):
        language_model_id = language_model_config['id']
        best_validation = selected_by_language_model[language_model_id]
        selected_setting = conditions_by_name[best_validation['condition']]
        progress(
            f'Loading selected language model '
            f'[{language_model_number}/{len(language_models_config)}]: '
            f'{language_model_id}'
        )
        tokenizer, language_model = modeling.load_language_model(
            language_model_id,
            language_model_config['revision'],
            device,
            language_model_config['dtype'],
        )
        try:
            progress(f'Final test: {selected_setting['condition']}')
            test_prediction_rows.extend(
                _generate_condition_predictions(
                    prediction_context,
                    selected_setting,
                    test,
                    'test',
                    tokenizer,
                    language_model,
                )
            )
        finally:
            del tokenizer, language_model
            modeling.clear_language_model_memory(device)
    test_predictions = pd.DataFrame(test_prediction_rows)
    (
        results,
        test_class_metrics,
        test_confusion,
        test_group_metrics,
        test_fairness_metrics,
    ) = evaluation.calculate_metrics(test_predictions, labels)
    results.insert(0, 'selected_on_validation_rank', 1)
    results.insert(
        1,
        'validation_selection_score',
        results['language_model'].map(
            selected_validation.set_index('language_model')[ranking_metric]
        ),
    )

    predictions = pd.concat(
        [validation_predictions, test_predictions], ignore_index=True
    )
    class_metrics = pd.concat(
        [validation_class_metrics, test_class_metrics], ignore_index=True
    )
    confusion_matrix = pd.concat(
        [validation_confusion, test_confusion], ignore_index=True
    )
    group_metrics = pd.concat(
        [validation_group_metrics, test_group_metrics], ignore_index=True
    )
    fairness_metrics = pd.concat(
        [validation_fairness_metrics, test_fairness_metrics], ignore_index=True
    )

    result_tables = {
        'predictions': predictions,
        'validation_results': validation_results,
        'results': results,
        'class_metrics': class_metrics,
        'confusion_matrix': confusion_matrix,
        'group_metrics': group_metrics,
        'fairness_metrics': fairness_metrics,
        'dataset_counts': dataset_counts,
    }
    output = _write_run_outputs(
        root,
        config,
        result_tables,
        selected_validation,
        labels,
    )

    progress(f'Finished: {output['run_dir']}')
    return output
