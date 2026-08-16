import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from lancedb.table import LanceTable
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import configuration
import dataset
import evaluation
import modeling
import plotting

RetrievalCache = dict[tuple[str, str, str], list[dict[str, Any]]]
INCOMPLETE_RUN_DIRECTORY = 'incomplete_run'
NOT_APPLICABLE = 'not_applicable'


@dataclass
class PredictionContext:
    """Values and resources shared by every prediction condition in one run."""

    example_order_seed: int
    dataset_shuffle_seed: int
    evaluation_split: str
    target: dataset.Column
    audit_column: dataset.Column
    target_labels: list[str]
    max_example_count: int
    device: str
    semantic_resources: dict[str, modeling.SemanticResource]
    training_profession_gender_pairs: tuple[tuple[str, str], ...]
    retrieval_cache: RetrievalCache


def _incomplete_run_directory(root: Path, config: dict[str, Any]) -> Path:
    """Return the single checkpoint directory for the configured output path."""

    return root / config['defaults']['output_dir'] / INCOMPLETE_RUN_DIRECTORY


def discard_incomplete_run(config: dict[str, Any], project_root: str | Path = '.') -> bool:
    """Delete the configured output directory's incomplete run, if present."""

    checkpoint_dir = _incomplete_run_directory(Path(project_root).resolve(), config)
    if not checkpoint_dir.exists():
        return False

    shutil.rmtree(checkpoint_dir)
    return True


def _language_model_checkpoint_path(checkpoint_dir: Path, language_model_id: str) -> Path:
    """Return the stable checkpoint path for one configured language model."""

    filename = language_model_id.replace('/', '--')
    return checkpoint_dir / f'{filename}_predictions.csv'


def _write_language_model_checkpoint(path: Path, predictions: pd.DataFrame) -> None:
    """Atomically save one language model's complete raw predictions."""

    temporary_path = path.with_suffix('.tmp')
    predictions.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _build_experiment_conditions(
        target: dataset.Column,
        retrieval_methods: list[str],
        embedding_models: list[dict[str, Any]],
        language_model_configs: list[dict[str, Any]],
        example_counts: list[int],
        example_orders: list[str],
        prompt_templates: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build zero-shot conditions once and cross positive counts with retrieval controls."""

    experiment_conditions: list[dict[str, Any]] = []
    example_configurations: list[tuple[str, str, int, str]] = []
    if 0 in example_counts:
        example_configurations.append((NOT_APPLICABLE, NOT_APPLICABLE, 0, NOT_APPLICABLE))

    example_configurations.extend(
        (retrieval_method, embedding_model['id'], example_count, example_order)
        for retrieval_method in retrieval_methods
        for embedding_model in embedding_models
        for example_count in example_counts
        if example_count > 0
        for example_order in example_orders
    )

    for language_model_config in language_model_configs:
        language_model_id = language_model_config['id']

        for retrieval_method, embedding_model_id, example_count, example_order in example_configurations:
            for prompt_name, master_prompt in prompt_templates.items():
                condition_name = (
                    f'{target} | language_model={language_model_id} | '
                    f'{retrieval_method} | embedding={embedding_model_id} | '
                    f'examples={example_count} | {example_order} | {prompt_name}'
                )

                experiment_conditions.append({
                    'condition': condition_name,
                    'language_model': language_model_id,
                    'retrieval_method': retrieval_method,
                    'embedding_model': embedding_model_id,
                    'example_count': example_count,
                    'example_order': example_order,
                    'prompt_name': prompt_name,
                    'master_prompt': master_prompt,
                })

    return experiment_conditions


def _prepare_semantic_resources(
        evaluation_rows: list[dict[str, Any]],
        embedding_models: list[dict[str, Any]],
        cached_tables: dict[str, LanceTable],
        training_filter: str,
        device: str,
) -> dict[str, modeling.SemanticResource]:
    """Encode evaluation rows against validated complete training tables."""

    semantic_resources: dict[str, modeling.SemanticResource] = {}
    progress_bar = tqdm(
        embedding_models,
        desc='Preparing embedding models',
        unit='model',
    )
    for embedding_model in progress_bar:
        embedding_model_id = embedding_model['id']

        evaluation_vectors = modeling.encode_embedding_queries(evaluation_rows, embedding_model, device)

        semantic_resources[embedding_model_id] = {
            'table': cached_tables[embedding_model_id],
            'evaluation_vectors': evaluation_vectors,
            'training_filter': training_filter,
        }

    return semantic_resources


def _predict_labels_for_condition(
        context: PredictionContext,
        condition: dict[str, Any],
        evaluation_rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        language_model: PreTrainedModel,
) -> list[dict[str, Any]]:
    """Predict the held-out label for every evaluation row under one condition."""

    retrieval_method = condition['retrieval_method']
    embedding_model_id = condition['embedding_model']
    example_count = condition['example_count']
    if example_count > 0:
        semantic_resource = context.semantic_resources[embedding_model_id]
        semantic_table = semantic_resource['table']
        query_vectors = semantic_resource['evaluation_vectors']
        training_filter = semantic_resource['training_filter']

    rows: list[dict[str, Any]] = []
    query_progress_bar = tqdm(
        evaluation_rows,
        desc=f'Predicting {context.evaluation_split} rows',
        unit='row',
        position=2,
        leave=False,
    )
    for query_index, query in enumerate(query_progress_bar):
        if example_count == 0:
            examples = []
        else:
            query_vector = query_vectors[query_index]
            retrieval_key = (retrieval_method, embedding_model_id, query[dataset.Column.ID])

            if retrieval_key not in context.retrieval_cache:
                context.retrieval_cache[retrieval_key] = modeling.retrieve_examples(
                    retrieval_method,
                    query_vector,
                    semantic_table,
                    training_filter,
                    context.max_example_count,
                    context.training_profession_gender_pairs,
                )

            # Balanced retrieval order preserves balanced prefixes. Slice the
            # requested set before applying its independent prompt-presentation
            # order so smaller example counts keep those balanced memberships.
            examples = context.retrieval_cache[retrieval_key][:example_count]
            examples = modeling.order_examples(
                examples,
                condition['example_order'],
                context.example_order_seed,
            )

        messages = modeling.build_prompt(
            query,
            examples,
            context.target,
            context.target_labels,
            condition['master_prompt'],
        )
        predicted_label, label_scores = modeling.score_allowed_labels(
            messages,
            context.target_labels,
            tokenizer,
            language_model,
            context.device,
        )

        condition_metadata = {
            'condition': condition['condition'],
            'target': context.target,
            'audit_column': context.audit_column,
            'retrieval_method': retrieval_method,
            'embedding_model': embedding_model_id,
            'example_count': example_count,
            'example_order': condition['example_order'],
            'prompt_name': condition['prompt_name'],
            'master_prompt': condition['master_prompt'],
            'language_model': condition['language_model'],
            'device': context.device,
            'example_order_seed': context.example_order_seed,
            'dataset_shuffle_seed': context.dataset_shuffle_seed,
        }
        prediction_metadata = {
            'evaluation_split': context.evaluation_split,
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
        target_labels: list[str],
) -> dict[str, Any]:
    """Write experiment tables and reports, then return their paths and data."""

    defaults = config['defaults']
    evaluation_split = defaults['evaluation_split']
    run_dir = (
            root
            / defaults['output_dir']
            / datetime.now().strftime(f'{evaluation_split}-run-%Y%m%d-%H%M%S-%f')
    )
    run_dir.mkdir(parents=True)

    for table_name, table in result_tables.items():
        table.to_csv(
            run_dir / f'{evaluation_split}_{table_name}.csv',
            index=False,
        )

    with (run_dir / 'config_used.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    plots = plotting.create_metric_plots(
        result_tables,
        run_dir / 'plots',
        evaluation_split,
    )
    selected = result_tables['results'].loc[
        result_tables['results']['is_best']
    ].copy()
    best_prompts_path = run_dir / f'{evaluation_split}_best_prompts.txt'
    evaluation.write_best_prompts(
        best_prompts_path,
        selected,
        config['prompt_templates'],
        target_labels,
        defaults['ranking_metric'],
        defaults['ranking_direction'],
        evaluation_split,
    )

    return {
        'evaluation_split': evaluation_split,
        'run_dir': run_dir,
        **result_tables,
        'plots': plots,
        'best_prompts': best_prompts_path,
    }


def prepare_embedding_cache(
        config: dict[str, Any],
        project_root: str | Path = '.',
        progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Explicitly prepare all canonical training embeddings for configured models."""

    configuration.validate_config(config)
    root = Path(project_root).resolve()
    source_rows = dataset.load_source_rows(config, root)
    training_rows = [row for row in source_rows if row[dataset.Column.SPLIT] == 'train']
    if not training_rows:
        raise ValueError('The complete canonical training corpus is empty')

    device = modeling.choose_device(config['inference']['device'])
    database_path = root / config['retrieval']['lancedb_path']
    progress(f'Using device: {device}')
    progress(f'Preparing {len(training_rows)} canonical training rows')

    row_counts: dict[str, int] = {}
    embedding_models = config['retrieval']['embedding_models']
    progress_bar = tqdm(embedding_models, desc='Preparing complete embedding caches', unit='model')
    for embedding_model in progress_bar:
        table = modeling.prepare_training_embedding_cache(
            training_rows,
            embedding_model,
            device,
            database_path,
            progress,
        )
        row_counts[embedding_model['id']] = table.count_rows()

    progress(f'Prepared complete embedding caches at {database_path}')
    return row_counts


def run_experiment(
        config: dict[str, Any],
        project_root: str | Path = '.',
        progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the complete experiment and return tables plus output paths."""

    configuration.validate_config(config)
    root = Path(project_root).resolve()
    defaults = config['defaults']
    evaluation_split = defaults['evaluation_split']
    target, audit_column, _, target_labels = dataset.task_settings(config)
    checkpoint_dir = _incomplete_run_directory(root, config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress(f'Incomplete-run checkpoints: {checkpoint_dir}')

    inference_settings = config['inference']
    language_models_config = inference_settings['language_models']
    progress(f'Holding out {target}; language model input is hard_text + {audit_column}')

    source_rows = dataset.load_source_rows(config, root)
    source_splits = dataset.select_profession_splits(config, source_rows)
    embedding_models = config['retrieval']['embedding_models']

    source_dataset_counts = dataset.calculate_dataset_counts(config, source_splits)
    train, evaluation_rows, evaluation_per_cell = dataset.select_run_data(config, source_splits)
    if config['dataset']['evaluation_per_profession_gender'] == 'max_balanced':
        progress(
            f'Resolved evaluation_per_profession_gender=max_balanced to '
            f'{evaluation_per_cell} rows per profession/gender cell'
        )

    run_dataset_counts = dataset.calculate_dataset_counts(
        config,
        {'train': train, evaluation_split: evaluation_rows},
    )
    progress(
        f'Loaded {sum(map(len, source_splits.values()))} filtered source rows; '
        f'selected {len(train)} train and {len(evaluation_rows)} {evaluation_split} rows for this run'
    )

    retrieval_methods = config['retrieval']['methods']
    example_counts = config['retrieval']['example_counts']
    example_orders = config['retrieval']['example_orders']
    prompt_templates = config['prompt_templates']

    experiment_conditions = _build_experiment_conditions(
        target,
        retrieval_methods,
        embedding_models,
        language_models_config,
        example_counts,
        example_orders,
        prompt_templates,
    )

    resumed_language_models: list[str] = []
    prediction_context: PredictionContext | None = None
    condition_prediction_tables: list[pd.DataFrame] = []
    ranked_result_tables: list[pd.DataFrame] = []
    condition_target_label_metric_tables: list[pd.DataFrame] = []
    condition_confusion_matrices: list[pd.DataFrame] = []
    condition_audit_group_metric_tables: list[pd.DataFrame] = []
    condition_fairness_metric_tables: list[pd.DataFrame] = []

    language_model_progress_bar = tqdm(
        language_models_config,
        desc=f'Evaluating language models on {evaluation_split}',
        unit='model',
        position=0,
        leave=True,
    )
    for language_model_config in language_model_progress_bar:
        language_model_id = language_model_config['id']
        language_model_conditions = [
            condition
            for condition in experiment_conditions
            if condition['language_model'] == language_model_id
        ]
        checkpoint_path = _language_model_checkpoint_path(checkpoint_dir, language_model_id)

        if checkpoint_path.exists():
            progress(f'Resuming completed language model: {language_model_id}')
            model_predictions = pd.read_csv(checkpoint_path)
            resumed_language_models.append(language_model_id)
        else:
            if prediction_context is None:
                device = modeling.choose_device(inference_settings['device'])
                progress(f'Using device: {device}')
                max_example_count = int(max(example_counts))

                if max_example_count > 0:
                    complete_training_rows = [row for row in source_rows if row[dataset.Column.SPLIT] == 'train']
                    database_path = root / config['retrieval']['lancedb_path']
                    cached_tables = {
                        embedding_model['id']: modeling.open_training_embedding_cache(
                            complete_training_rows,
                            embedding_model,
                            database_path,
                        )
                        for embedding_model in embedding_models
                    }

                    training_filter = modeling.build_training_filter(train, dataset.train_size_limit(config))
                    for embedding_model_id, table in cached_tables.items():
                        filtered_row_count = table.count_rows(training_filter)
                        if filtered_row_count != len(train):
                            raise RuntimeError(
                                f'{embedding_model_id} training filter selected '
                                f'{filtered_row_count} cached rows; expected the configured '
                                f'training pool size {len(train)}'
                            )

                    semantic_resources = _prepare_semantic_resources(
                        evaluation_rows,
                        embedding_models,
                        cached_tables,
                        training_filter,
                        device,
                    )
                    training_profession_gender_pairs = tuple(sorted(
                        {(row[dataset.Column.PROFESSION], row[dataset.Column.GENDER]) for row in train}
                    ))
                else:
                    progress('Skipping semantic resources because every example count is zero')
                    semantic_resources = {}
                    training_profession_gender_pairs = ()

                prediction_context = PredictionContext(
                    example_order_seed=defaults['seed'],
                    dataset_shuffle_seed=config['dataset']['shuffle_seed'],
                    evaluation_split=evaluation_split,
                    target=target,
                    audit_column=audit_column,
                    target_labels=target_labels,
                    max_example_count=max_example_count,
                    device=device,
                    semantic_resources=semantic_resources,
                    training_profession_gender_pairs=training_profession_gender_pairs,
                    retrieval_cache={},
                )

            prediction_context: PredictionContext = cast(PredictionContext, prediction_context)
            device = prediction_context.device
            tokenizer, language_model = modeling.load_language_model(
                language_model_id,
                language_model_config['revision'],
                device,
                language_model_config['dtype'],
            )

            try:
                model_condition_prediction_tables: list[pd.DataFrame] = []
                condition_progress_bar = tqdm(
                    language_model_conditions,
                    desc=f'Evaluating {language_model_id}',
                    unit='condition',
                    position=1,
                    leave=True,
                )
                for condition in condition_progress_bar:
                    model_condition_prediction_tables.append(pd.DataFrame(
                        _predict_labels_for_condition(
                            prediction_context,
                            condition,
                            evaluation_rows,
                            tokenizer,
                            language_model,
                        )
                    ))

                model_predictions = pd.concat(
                    model_condition_prediction_tables,
                    ignore_index=True,
                )
                _write_language_model_checkpoint(checkpoint_path, model_predictions)
                progress(f'Checkpointed completed language model: {language_model_id}')
            finally:
                del tokenizer, language_model
                modeling.clear_model_memory(device)

        condition_result_tables: list[pd.DataFrame] = []
        for condition in language_model_conditions:
            condition_predictions = model_predictions.loc[
                model_predictions['condition'].eq(condition['condition'])
            ].copy()

            (
                condition_result,
                condition_target_label_metrics,
                condition_confusion_matrix,
                condition_audit_group_metrics,
                condition_fairness_metrics,
            ) = evaluation.calculate_condition_metrics(condition_predictions, target_labels)

            condition_prediction_tables.append(condition_predictions)
            condition_result_tables.append(condition_result)
            condition_target_label_metric_tables.append(condition_target_label_metrics)
            condition_confusion_matrices.append(condition_confusion_matrix)
            condition_audit_group_metric_tables.append(condition_audit_group_metrics)
            condition_fairness_metric_tables.append(condition_fairness_metrics)

        ranked_result_tables.append(
            evaluation.rank_results(
                pd.concat(condition_result_tables, ignore_index=True),
                defaults['ranking_metric'],
                defaults['ranking_direction'],
            )
        )

    results = pd.concat(ranked_result_tables, ignore_index=True)

    result_tables = {
        'predictions': pd.concat(condition_prediction_tables, ignore_index=True),
        'results': results,
        'target_label_metrics': pd.concat(condition_target_label_metric_tables, ignore_index=True),
        'confusion_matrix': pd.concat(condition_confusion_matrices, ignore_index=True),
        'audit_group_metrics': pd.concat(condition_audit_group_metric_tables, ignore_index=True),
        'fairness_metrics': pd.concat(condition_fairness_metric_tables, ignore_index=True),
        'source_dataset_counts': source_dataset_counts,
        'run_dataset_counts': run_dataset_counts,
    }

    output = _write_run_outputs(root, config, result_tables, target_labels)
    discard_incomplete_run(config, root)
    output['resumed_language_models'] = resumed_language_models

    progress(f'Finished: {output['run_dir']}')
    return output
