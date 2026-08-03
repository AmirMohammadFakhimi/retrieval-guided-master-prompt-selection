import json
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import dataset
import evaluation
import modeling


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
    llm_settings = config['llm']

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
    llm_configs = llm_settings['llms']
    if not isinstance(llm_configs, list) or not llm_configs:
        raise ValueError('llm.llms must contain at least one LLM')

    llm_ids: list[str] = []
    for index, llm_config in enumerate(llm_configs):
        if not isinstance(llm_config, dict):
            raise ValueError(f'llm.llms[{index}] must be a mapping')

        missing_settings = sorted({'id', 'revision', 'dtype'} - set(llm_config))
        if missing_settings:
            raise ValueError(
                f'llm.llms[{index}] is missing: '
                f'{missing_settings}'
            )

        llm_id = llm_config['id']
        if not llm_id:
            raise ValueError(
                f'llm.llms[{index}].id cannot be empty'
            )
        llm_ids.append(llm_id)
        if not llm_config['revision']:
            raise ValueError(f'llm.llms[{index}].revision cannot be empty')
        if llm_config['dtype'] not in modeling.LLM_DTYPES:
            allowed_dtypes = ', '.join(sorted(modeling.LLM_DTYPES))
            raise ValueError(
                f'llm.llms[{index}].dtype must be one of: {allowed_dtypes}'
            )
    if len(llm_ids) != len(set(llm_ids)):
        raise ValueError('llm.llms cannot contain duplicate IDs')

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
            raise ValueError(f'retrieval.embedding_models[{index}] must be a mapping')
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
        llm_configs: list[dict[str, Any]],
        example_counts: list[int],
        example_orders: list[str],
        prompt_templates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the full cross-product of configured experiment conditions."""

    conditions: list[dict[str, Any]] = []
    for llm_config in llm_configs:
        llm_id = llm_config['id']
        for method in methods:
            for embedding_model in embedding_models:
                embedding_model_id = embedding_model['id']
                for example_count in example_counts:
                    for example_order in example_orders:
                        for prompt_name, master_prompt in prompt_templates.items():
                            condition = (
                                f'{target} | llm={llm_id} | '
                                f'{method} | embedding={embedding_model_id} | '
                                f'examples={example_count} | {example_order} | {prompt_name}'
                            )
                            conditions.append(
                                {
                                    'condition': condition,
                                    'retrieval': method,
                                    'embedding_model': embedding_model_id,
                                    'example_count': example_count,
                                    'example_order': example_order,
                                    'prompt_name': prompt_name,
                                    'master_prompt': master_prompt,
                                    'llm': llm_id,
                                }
                            )
    return conditions


def run_experiment(
        config: dict[str, Any],
        project_root: str | Path = '.',
        progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the complete experiment and return tables plus output paths."""

    validate_config(config)
    root = Path(project_root).resolve()
    defaults = config['defaults']
    seed = defaults['seed']
    target, audit_column, _, labels = dataset.task_settings(config)
    np.random.seed(seed)
    torch.manual_seed(seed)

    llm_settings = config['llm']
    llm_configs = [
        dict(settings)
        for settings in llm_settings['llms']
    ]
    device = modeling.choose_device(llm_settings['device'])
    progress(f'Using device: {device}')
    progress(
        f'Holding out {target}; LLM input is hard_text + {audit_column}'
    )
    train, validation, test, dataset_counts = dataset.load_data(config, root)
    progress(
        f'Loaded {len(train)} train, {len(validation)} validation, '
        f'and {len(test)} test biographies'
    )

    methods = list(config['retrieval']['methods'])
    evaluation_rows = validation + test
    embedding_models = [
        dict(settings)
        for settings in config['retrieval']['embedding_models']
    ]
    semantic_resources: dict[str, modeling.SemanticResource] = {}
    for embedding_model_number, embedding_model in enumerate(embedding_models, start=1):
        embedding_model_id = embedding_model['id']
        progress(
            f'Preparing embedding model '
            f'[{embedding_model_number}/{len(embedding_models)}]: '
            f'{embedding_model_id}'
        )
        semantic_table, evaluation_vectors = modeling.prepare_semantic_retrieval(
            train,
            evaluation_rows,
            embedding_model,
            device,
            root / config['retrieval']['lancedb_path'],
            progress,
        )
        semantic_resources[embedding_model_id] = {
            'table': semantic_table,
            'validation_vectors': evaluation_vectors[:len(validation)],
            'test_vectors': evaluation_vectors[len(validation):],
        }

    example_counts = list(config['retrieval']['example_counts'])
    example_orders = list(config['retrieval']['example_orders'])
    prompt_templates = dict(config['prompt_templates'])
    training_cells = tuple(sorted({
        (
            row[dataset.Column.PROFESSION],
            row[dataset.Column.GENDER],
        )
        for row in train
    }))
    conditions = _build_conditions(
        target,
        methods,
        embedding_models,
        llm_configs,
        example_counts,
        example_orders,
        prompt_templates,
    )

    retrieval_cache: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = {}

    def generate_condition_predictions(
            setting: Mapping[str, Any],
            queries: list[dict[str, Any]],
            evaluation_split: str,
            tokenizer: PreTrainedTokenizerBase,
            llm: PreTrainedModel,
    ) -> list[dict[str, Any]]:
        """Predict every row for one prompt condition and one data split."""

        rows: list[dict[str, Any]] = []
        retrieval_method = setting['retrieval']
        embedding_model_id = setting['embedding_model']
        semantic_resource = semantic_resources[embedding_model_id]
        semantic_table = semantic_resource['table']
        query_vectors = semantic_resource[
            f'{evaluation_split}_vectors'
        ]
        for query_index, query in enumerate(queries):
            query_vector = query_vectors[query_index]
            retrieval_key = (
                retrieval_method,
                embedding_model_id,
                query[dataset.Column.ID],
            )
            if retrieval_key not in retrieval_cache:
                retrieval_cache[retrieval_key] = modeling.retrieve_examples(
                    retrieval_method,
                    query_vector,
                    semantic_table,
                    max(example_counts),
                    training_cells,
                )
            examples = retrieval_cache[retrieval_key][:setting['example_count']]
            examples = modeling.order_examples(
                examples,
                setting['example_order'],
                seed,
            )
            messages = modeling.build_prompt(
                query,
                examples,
                target,
                labels,
                setting['master_prompt'],
            )
            predicted_label, label_scores = modeling.score_allowed_labels(
                messages,
                labels,
                tokenizer,
                llm,
                device,
            )
            rows.append(
                {
                    'condition': setting['condition'],
                    'evaluation_split': evaluation_split,
                    'target': target,
                    'audit_column': audit_column,
                    'query_id': query[dataset.Column.ID],
                    dataset.Column.HARD_TEXT: query[dataset.Column.HARD_TEXT],
                    dataset.Column.PROFESSION: query[dataset.Column.PROFESSION],
                    dataset.Column.GENDER: query[dataset.Column.GENDER],
                    'true_label': query[target],
                    'audit_group': query[audit_column],
                    'predicted_label': predicted_label,
                    'retrieval': retrieval_method,
                    'embedding_model': embedding_model_id,
                    'example_count': setting['example_count'],
                    'examples_used': len(examples),
                    'example_order': setting['example_order'],
                    'prompt_name': setting['prompt_name'],
                    'master_prompt': setting['master_prompt'],
                    'llm': setting['llm'],
                    'device': device,
                    'seed': seed,
                    'example_ids': json.dumps(
                        [example[dataset.Column.ID] for example in examples]
                    ),
                    'example_professions': json.dumps(
                        [
                            example[dataset.Column.PROFESSION]
                            for example in examples
                        ]
                    ),
                    'example_genders': json.dumps(
                        [example[dataset.Column.GENDER] for example in examples]
                    ),
                    'example_scores': json.dumps(
                        [example['retrieval_score'] for example in examples]
                    ),
                    'prompt': json.dumps(messages, ensure_ascii=False),
                    'label_scores': json.dumps(label_scores, sort_keys=True),
                }
            )
        return rows

    validation_prediction_rows: list[dict[str, Any]] = []
    condition_number = 0
    for llm_number, llm_config in enumerate(
            llm_configs, start=1
    ):
        llm_id = llm_config['id']
        progress(
            f'Loading LLM [{llm_number}/{len(llm_configs)}]: '
            f'{llm_id}'
        )
        tokenizer, llm = modeling.load_llm(
            llm_id,
            llm_config['revision'],
            device,
            llm_config['dtype'],
        )
        try:
            llm_conditions = [
                setting
                for setting in conditions
                if setting['llm'] == llm_id
            ]
            for setting in llm_conditions:
                condition_number += 1
                progress(
                    f'[{condition_number}/{len(conditions)}] validation: '
                    f'{setting['condition']}'
                )
                validation_prediction_rows.extend(
                    generate_condition_predictions(
                        setting,
                        validation,
                        'validation',
                        tokenizer,
                        llm,
                    )
                )
        finally:
            del tokenizer, llm
            modeling.clear_llm_memory(device)

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
        defaults['ranking_metric'],
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
    selected_by_llm = {
        row['llm']: row
        for row in selected_validation.to_dict('records')
    }
    test_prediction_rows: list[dict[str, Any]] = []
    for llm_number, llm_config in enumerate(
            llm_configs, start=1
    ):
        llm_id = llm_config['id']
        best_validation = selected_by_llm[llm_id]
        selected_setting = conditions_by_name[best_validation['condition']]
        progress(
            f'Loading selected LLM '
            f'[{llm_number}/{len(llm_configs)}]: {llm_id}'
        )
        tokenizer, llm = modeling.load_llm(
            llm_id,
            llm_config['revision'],
            device,
            llm_config['dtype'],
        )
        try:
            progress(f'Final test: {selected_setting['condition']}')
            test_prediction_rows.extend(
                generate_condition_predictions(
                    selected_setting,
                    test,
                    'test',
                    tokenizer,
                    llm,
                )
            )
        finally:
            del tokenizer, llm
            modeling.clear_llm_memory(device)
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
        results['llm'].map(
            selected_validation.set_index('llm')[defaults['ranking_metric']]
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

    run_dir = (
            root
            / defaults['output_dir']
            / datetime.now().strftime('run-%Y%m%d-%H%M%S-%f')
    )
    run_dir.mkdir(parents=True)
    predictions.to_csv(run_dir / 'predictions.csv', index=False)
    validation_results.to_csv(run_dir / 'validation_results.csv', index=False)
    results.to_csv(run_dir / 'results.csv', index=False)
    class_metrics.to_csv(run_dir / 'class_metrics.csv', index=False)
    confusion_matrix.to_csv(run_dir / 'confusion_matrix.csv', index=False)
    group_metrics.to_csv(run_dir / 'group_metrics.csv', index=False)
    fairness_metrics.to_csv(run_dir / 'fairness_metrics.csv', index=False)
    dataset_counts.to_csv(run_dir / 'dataset_counts.csv', index=False)
    with (run_dir / 'config_used.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    plot_path = run_dir / 'results.png'
    evaluation.plot_results(validation_results, plot_path)
    best_prompts_path = run_dir / 'best_prompts.txt'
    evaluation.write_best_prompts(
        best_prompts_path,
        selected_validation,
        prompt_templates,
        labels,
        defaults['ranking_metric'],
        defaults['ranking_direction'],
        results,
    )

    progress(f'Finished: {run_dir}')
    return {
        'run_dir': run_dir,
        'predictions': predictions,
        'validation_results': validation_results,
        'results': results,
        'class_metrics': class_metrics,
        'confusion_matrix': confusion_matrix,
        'group_metrics': group_metrics,
        'fairness_metrics': fairness_metrics,
        'dataset_counts': dataset_counts,
        'plot': plot_path,
        'best_prompts': best_prompts_path,
    }
