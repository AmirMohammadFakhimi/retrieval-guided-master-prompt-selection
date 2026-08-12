import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import configuration
import dataset
import evaluation
import modeling
import plotting

RetrievalCache = dict[tuple[str, str, str], list[dict[str, Any]]]


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
        evaluation_rows: list[dict[str, Any]],
        embedding_models: list[dict[str, Any]],
        device: str,
        database_path: Path,
        progress: Callable[[str], None],
) -> dict[str, modeling.SemanticResource]:
    """Prepare retrieval tables and evaluation vectors for every embedding model."""

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
            'evaluation_vectors': evaluation_vectors,
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
    semantic_resource = context.semantic_resources[embedding_model_id]
    semantic_table = semantic_resource['table']
    query_vectors = semantic_resource['evaluation_vectors']

    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(evaluation_rows):
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

        examples = context.retrieval_cache[retrieval_key][:condition['example_count']]
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
            'example_count': condition['example_count'],
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

    inference_settings = config['inference']
    language_models_config = inference_settings['language_models']
    device = modeling.choose_device(inference_settings['device'])
    progress(f'Using device: {device}')
    progress(f'Holding out {target}; language model input is hard_text + {audit_column}')

    source_splits = dataset.load_data(config, root)
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

    embedding_models = config['retrieval']['embedding_models']
    semantic_resources = _prepare_semantic_resources(
        train,
        evaluation_rows,
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
        evaluation_split=evaluation_split,
        target=target,
        audit_column=audit_column,
        target_labels=target_labels,
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
        condition_result_tables: list[pd.DataFrame] = []
        tokenizer, language_model = modeling.load_language_model(
            language_model_id,
            language_model_config['revision'],
            device,
            language_model_config['dtype'],
        )

        try:
            conditions_for_language_model = [
                condition
                for condition in conditions
                if condition['language_model'] == language_model_id
            ]
            condition_progress_bar = tqdm(
                conditions_for_language_model,
                desc=f'Evaluating {language_model_id}',
                unit='condition',
                position=1,
                leave=True,
            )
            for condition in condition_progress_bar:
                condition_predictions = pd.DataFrame(
                    _predict_labels_for_condition(
                        prediction_context,
                        condition,
                        evaluation_rows,
                        tokenizer,
                        language_model,
                    )
                )

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
        finally:
            del tokenizer, language_model
            modeling.clear_language_model_memory(device)

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

    progress(f'Finished: {output['run_dir']}')
    return output
