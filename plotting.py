from collections.abc import Sequence
from math import ceil
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from evaluation import (
    ACCURACY_METRIC_COLUMN,
    MACRO_RECALL_METRIC_COLUMN,
    resolve_metric_column,
)

MetricSpec = tuple[str, str]

QUALITY_METRICS: tuple[MetricSpec, ...] = (
    (
        ACCURACY_METRIC_COLUMN,
        'Accuracy / micro precision / micro recall / micro F1 / weighted recall',
    ),
    ('macro_precision', 'Macro precision'),
    (MACRO_RECALL_METRIC_COLUMN, 'Macro recall / balanced accuracy'),
    ('macro_f1', 'Macro F1'),
    ('weighted_precision', 'Weighted precision'),
    ('weighted_f1', 'Weighted F1'),
    ('worst_group_accuracy', 'Worst-group accuracy'),
)

AGREEMENT_METRICS: tuple[MetricSpec, ...] = (
    ('matthews_correlation_coefficient', 'Matthews correlation coefficient'),
    ('cohen_kappa', "Cohen's kappa"),
)

MEAN_FAIRNESS_DIFFERENCES: tuple[MetricSpec, ...] = (
    ('mean_demographic_parity_difference', 'Mean demographic-parity difference'),
    ('mean_equal_opportunity_difference', 'Mean equal-opportunity difference'),
    ('mean_false_positive_rate_difference', 'Mean false-positive-rate difference'),
    ('mean_equalized_odds_difference', 'Mean equalized-odds difference'),
    ('mean_predictive_parity_difference', 'Mean predictive-parity difference'),
)

WORST_FAIRNESS_DIFFERENCES: tuple[MetricSpec, ...] = (
    ('group_accuracy_difference', 'Group-accuracy difference'),
    ('max_demographic_parity_difference', 'Maximum demographic-parity difference'),
    ('max_equal_opportunity_difference', 'Maximum equal-opportunity difference'),
    ('max_false_positive_rate_difference', 'Maximum false-positive-rate difference'),
    ('max_equalized_odds_difference', 'Maximum equalized-odds difference'),
    ('max_predictive_parity_difference', 'Maximum predictive-parity difference'),
)

FAIRNESS_RATIOS: tuple[MetricSpec, ...] = (
    ('mean_demographic_parity_ratio', 'Mean demographic-parity ratio'),
    ('min_demographic_parity_ratio', 'Minimum demographic-parity ratio'),
)

SUMMARY_SIZE_COLUMNS: tuple[MetricSpec, ...] = (
    ('sample_count', 'Evaluated rows'),
    ('n_classes', 'Target classes'),
    ('n_audit_groups', 'Audit groups'),
)

SUMMARY_COVERAGE_COLUMNS: tuple[MetricSpec, ...] = (
    ('n_precision_defined_classes', 'Classes defining precision'),
    ('n_recall_defined_classes', 'Classes defining recall'),
    ('n_f1_defined_classes', 'Classes defining F1'),
    (
        'n_demographic_parity_defined_classes',
        'Classes defining demographic-parity difference',
    ),
    (
        'n_demographic_parity_ratio_defined_classes',
        'Classes defining demographic-parity ratio',
    ),
    (
        'n_equal_opportunity_defined_classes',
        'Classes defining equal-opportunity difference',
    ),
    (
        'n_false_positive_rate_defined_classes',
        'Classes defining false-positive-rate difference',
    ),
    (
        'n_equalized_odds_defined_classes',
        'Classes defining equalized-odds difference',
    ),
    (
        'n_predictive_parity_defined_classes',
        'Classes defining predictive-parity difference',
    ),
)

CLASS_RATE_COLUMNS: tuple[MetricSpec, ...] = (
    ('precision', 'Precision'),
    ('recall', 'Recall / true-positive rate'),
    ('f1', 'F1'),
    ('specificity', 'Specificity / true-negative rate'),
    ('false_positive_rate', 'False-positive rate'),
    ('false_negative_rate', 'False-negative rate'),
    ('negative_predictive_value', 'Negative predictive value'),
)

CLASS_COUNT_COLUMNS: tuple[MetricSpec, ...] = (
    ('support', 'True support'),
    ('predicted_count', 'Predicted count'),
    ('tp', 'True positives'),
    ('fp', 'False positives'),
    ('fn', 'False negatives'),
    ('tn', 'True negatives'),
)

GROUP_RATE_COLUMNS: tuple[MetricSpec, ...] = (
    ('selection_rate', 'Selection rate'),
    ('true_positive_rate', 'True-positive rate'),
    ('false_positive_rate', 'False-positive rate'),
    ('false_negative_rate', 'False-negative rate'),
    ('positive_predictive_value', 'Positive predictive value'),
    ('specificity', 'Specificity / true-negative rate'),
    ('negative_predictive_value', 'Negative predictive value'),
)

GROUP_COUNT_COLUMNS: tuple[MetricSpec, ...] = (
    ('positive_support', 'Positive support'),
    ('negative_support', 'Negative support'),
    ('predicted_positive', 'Predicted positive'),
    ('tp', 'True positives'),
    ('fp', 'False positives'),
    ('fn', 'False negatives'),
    ('tn', 'True negatives'),
)

FAIRNESS_CLASS_COLUMNS: tuple[MetricSpec, ...] = (
    ('demographic_parity_difference', 'Demographic-parity difference'),
    ('demographic_parity_ratio', 'Demographic-parity ratio'),
    ('equal_opportunity_difference', 'Equal-opportunity difference'),
    ('false_positive_rate_difference', 'False-positive-rate difference'),
    ('equalized_odds_difference', 'Equalized-odds difference'),
    ('predictive_parity_difference', 'Predictive-parity difference'),
)

FAIRNESS_COVERAGE_COLUMNS: tuple[MetricSpec, ...] = (
    ('groups_compared', 'Groups compared'),
    ('selection_rate_groups_defined', 'Groups defining selection rate'),
    ('tpr_groups_defined', 'Groups defining true-positive rate'),
    ('fpr_groups_defined', 'Groups defining false-positive rate'),
    ('ppv_groups_defined', 'Groups defining positive predictive value'),
)

SUMMARY_METRICS = (
    QUALITY_METRICS
    + AGREEMENT_METRICS
    + MEAN_FAIRNESS_DIFFERENCES
    + WORST_FAIRNESS_DIFFERENCES
    + FAIRNESS_RATIOS
    + SUMMARY_SIZE_COLUMNS
    + SUMMARY_COVERAGE_COLUMNS
)


def _column_names(metrics: Sequence[MetricSpec]) -> set[str]:
    return {column for column, _ in metrics}


def _validate_numeric_coverage(
        table_name: str,
        frame: pd.DataFrame,
        plotted_columns: set[str],
        excluded_columns: set[str],
) -> None:
    numeric_columns = set(frame.select_dtypes(include='number').columns)
    uncovered = sorted(numeric_columns - plotted_columns - excluded_columns)
    if uncovered:
        raise ValueError(
            f'Numeric columns in {table_name} have no corresponding plot: {uncovered}'
        )


def _context_title(frame: pd.DataFrame) -> str:
    return (
        f'target={frame['target'].iloc[0]}, '
        f'audit groups={frame['audit_column'].iloc[0]}'
    )


def _condition_labels(frame: pd.DataFrame) -> list[str]:
    return [
        (
            f'{row.language_model} | {row.retrieval_method} | '
            f'{row.embedding_model} | examples={row.example_count} | '
            f'{row.prompt_name} | {row.example_order}'
        )
        for row in frame.itertuples(index=False)
    ]


def _class_labels(frame: pd.DataFrame) -> list[str]:
    return [
        f'{row.language_model} | class={row.target_class}'
        for row in frame.itertuples(index=False)
    ]


def _group_labels(frame: pd.DataFrame) -> list[str]:
    return [
        (
            f'{row.language_model} | class={row.target_class} | '
            f'group={row.audit_group}'
        )
        for row in frame.itertuples(index=False)
    ]


def _group_level_labels(frame: pd.DataFrame) -> list[str]:
    return [
        f'{row.language_model} | group={row.audit_group}'
        for row in frame.itertuples(index=False)
    ]


def _metric_axis_limits(values: np.ndarray) -> tuple[float, float]:
    defined = values[~np.isnan(values)]
    if not len(defined):
        return 0.0, 1.0
    minimum = min(0.0, float(defined.min()))
    maximum = max(0.0, float(defined.max()))
    if minimum == maximum:
        maximum = minimum + 1.0
    margin = 0.05 * (maximum - minimum)
    return minimum - margin if minimum < 0 else 0.0, maximum + margin


def _plot_metric_panels(
        frame: pd.DataFrame,
        metrics: Sequence[MetricSpec],
        row_labels: Sequence[str],
        title: str,
        output: Path,
        bounds: tuple[float, float] | None = None,
        selected: pd.Series | None = None,
) -> None:
    row_count = len(frame)
    figure_width = max(8.0, 3.25 * len(metrics))
    figure_height = max(4.8, 0.3 * row_count + 2.4)
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(figure_width, figure_height),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    y = np.arange(row_count)
    if selected is None:
        colors = np.full(row_count, '#4C78A8', dtype=object)
    else:
        colors = np.where(selected.to_numpy(dtype=bool), '#1565C0', '#90CAF9')

    for metric_number, ((column, label), axis) in enumerate(
            zip(metrics, axes, strict=True)
    ):
        values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
        defined = ~np.isnan(values)
        axis.barh(y[defined], values[defined], color=colors[defined], height=0.72)
        axis.scatter(
            np.zeros((~defined).sum()),
            y[~defined],
            marker='x',
            color='#777777',
            linewidths=1.6,
            label='Undefined',
        )
        axis.axvline(0, color='#444444', linewidth=0.7)
        axis.set_title(fill(label, width=25), fontsize=10)
        axis.grid(axis='x', alpha=0.2)
        axis.set_xlabel('Value')
        axis.set_xlim(*(bounds or _metric_axis_limits(values)))
        axis.set_yticks(y)
        if metric_number == 0:
            axis.set_yticklabels(row_labels, fontsize=7)
            axis.invert_yaxis()
        else:
            axis.tick_params(axis='y', labelleft=False)

    legend_items = [Patch(facecolor='#4C78A8', label='Result')]
    if selected is not None:
        legend_items = [
            Patch(facecolor='#1565C0', label='Selected for final test'),
            Patch(facecolor='#90CAF9', label='Other validation condition'),
        ]
    legend_items.append(
        plt.Line2D(
            [0],
            [0],
            marker='x',
            color='#777777',
            linestyle='none',
            label='Undefined',
        )
    )
    figure.suptitle(title, y=0.995, fontsize=13)
    figure.legend(
        handles=legend_items,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.92),
        ncol=len(legend_items),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(figure)


def _summary_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == 'validation':
        return frame.sort_values(
            ['language_model', 'rank'], kind='stable'
        ).reset_index(drop=True)
    return frame.sort_values('language_model', kind='stable').reset_index(drop=True)


def _create_summary_plots(
        frame: pd.DataFrame,
        split: str,
        output_dir: Path,
) -> dict[str, Path]:
    plot_frame = _summary_frame(frame, split)
    labels = (
        _condition_labels(plot_frame)
        if split == 'validation'
        else plot_frame['language_model'].tolist()
    )
    selected = (
        plot_frame['selected_for_test']
        if split == 'validation'
        else None
    )
    context = _context_title(plot_frame)
    plot_specs = (
        ('quality_rates', QUALITY_METRICS, (0.0, 1.0), 'Quality rates'),
        ('agreement_scores', AGREEMENT_METRICS, (-1.0, 1.0), 'Agreement scores'),
        (
            'mean_fairness_differences',
            MEAN_FAIRNESS_DIFFERENCES,
            (0.0, 1.0),
            'Mean classwise fairness differences',
        ),
        (
            'worst_fairness_differences',
            WORST_FAIRNESS_DIFFERENCES,
            (0.0, 1.0),
            'Worst classwise and group-accuracy differences',
        ),
        (
            'fairness_ratios',
            FAIRNESS_RATIOS,
            (0.0, 1.0),
            'Demographic-parity ratios',
        ),
        ('sample_counts', SUMMARY_SIZE_COLUMNS, None, 'Evaluation sizes'),
        (
            'defined_class_counts',
            SUMMARY_COVERAGE_COLUMNS,
            None,
            'Classes defining each aggregate metric',
        ),
    )
    plots: dict[str, Path] = {}
    for suffix, metrics, bounds, plot_title in plot_specs:
        name = f'{split}_{suffix}'
        path = output_dir / f'{name}.png'
        _plot_metric_panels(
            plot_frame,
            metrics,
            labels,
            f'{split.title()} — {plot_title}\n{context}',
            path,
            bounds=bounds,
            selected=selected,
        )
        plots[name] = path
    return plots


def _plot_selection_vs_test(
        final_results: pd.DataFrame,
        ranking_metric: str,
        output: Path,
) -> None:
    metric_column = resolve_metric_column(ranking_metric)
    frame = final_results.sort_values('language_model', kind='stable').reset_index(
        drop=True
    )
    labels = frame['language_model'].tolist()
    y = np.arange(len(frame))
    validation_values = frame['validation_selection_score'].to_numpy(dtype=float)
    test_values = frame[metric_column].to_numpy(dtype=float)
    figure, axis = plt.subplots(
        figsize=(10, max(4.8, 0.65 * len(frame) + 2.4))
    )
    axis.barh(
        y - 0.18,
        validation_values,
        height=0.36,
        label='Validation selection score',
        color='#72B7B2',
    )
    axis.barh(
        y + 0.18,
        test_values,
        height=0.36,
        label='Final-test score',
        color='#4C78A8',
    )
    undefined_validation = np.isnan(validation_values)
    undefined_test = np.isnan(test_values)
    axis.scatter(
        np.zeros(undefined_validation.sum()),
        y[undefined_validation] - 0.18,
        marker='x',
        color='#777777',
        label='Undefined',
    )
    axis.scatter(
        np.zeros(undefined_test.sum()),
        y[undefined_test] + 0.18,
        marker='x',
        color='#777777',
    )
    if metric_column in _column_names(AGREEMENT_METRICS):
        axis.set_xlim(-1.0, 1.0)
    elif metric_column in _column_names(
            SUMMARY_SIZE_COLUMNS + SUMMARY_COVERAGE_COLUMNS
    ):
        axis.set_xlim(*_metric_axis_limits(np.concatenate([
            validation_values,
            test_values,
        ])))
    else:
        axis.set_xlim(0.0, 1.0)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.grid(axis='x', alpha=0.2)
    axis.set_xlabel('Value')
    axis.set_title(
        f'Validation-selected versus final-test {ranking_metric}\n'
        f'{_context_title(frame)}'
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(figure)


def _plot_confusion_matrices(frame: pd.DataFrame, output: Path) -> None:
    language_models = frame['language_model'].drop_duplicates().tolist()
    columns = min(2, len(language_models))
    rows = ceil(len(language_models) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8 * columns, 7 * rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, language_model in zip(axes_flat, language_models, strict=False):
        model_frame = frame.loc[frame['language_model'].eq(language_model)]
        true_labels = model_frame['true_label'].drop_duplicates().tolist()
        predicted_labels = model_frame['predicted_label'].drop_duplicates().tolist()
        label_order = list(dict.fromkeys(true_labels + predicted_labels))
        matrix = model_frame.pivot_table(
            index='true_label',
            columns='predicted_label',
            values='count',
            aggfunc='sum',
            fill_value=0,
        ).reindex(index=label_order, columns=label_order, fill_value=0)
        image = axis.imshow(matrix.to_numpy(), cmap='Blues', vmin=0)
        axis.set_title(language_model)
        tick_font_size = 8 if len(label_order) <= 12 else 5
        axis.set_xticks(
            np.arange(len(label_order)),
            label_order,
            rotation=90,
            fontsize=tick_font_size,
        )
        axis.set_yticks(
            np.arange(len(label_order)),
            label_order,
            fontsize=tick_font_size,
        )
        axis.set_xlabel('Predicted label')
        axis.set_ylabel('True label')
        if len(label_order) <= 12:
            text_threshold = matrix.to_numpy().max() / 2
            for row_number in range(len(label_order)):
                for column_number in range(len(label_order)):
                    count = int(matrix.iloc[row_number, column_number])
                    axis.text(
                        column_number,
                        row_number,
                        count,
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white' if count > text_threshold else 'black',
                    )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label='Count')

    for axis in axes_flat[len(language_models):]:
        axis.remove()
    figure.suptitle(
        f'Final-test confusion matrices\n{_context_title(frame)}', fontsize=14
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(figure)


def create_metric_plots(
        result_tables: dict[str, pd.DataFrame],
        output_dir: Path,
        ranking_metric: str,
) -> dict[str, Path]:
    """Create validation summaries and final-test metric diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_results = result_tables['validation_results']
    final_results = result_tables['results']
    class_metrics = result_tables['class_metrics']
    group_metrics = result_tables['group_metrics']
    fairness_metrics = result_tables['fairness_metrics']
    confusion_matrix = result_tables['confusion_matrix']

    test_class_metrics = class_metrics.loc[
        class_metrics['evaluation_split'].eq('test')
    ].sort_values(['language_model', 'target_class'], kind='stable').reset_index(
        drop=True
    )
    test_group_metrics = group_metrics.loc[
        group_metrics['evaluation_split'].eq('test')
    ].sort_values(
        ['language_model', 'target_class', 'audit_group'], kind='stable'
    ).reset_index(drop=True)
    test_fairness_metrics = fairness_metrics.loc[
        fairness_metrics['evaluation_split'].eq('test')
    ].sort_values(['language_model', 'target_class'], kind='stable').reset_index(
        drop=True
    )
    test_confusion = confusion_matrix.loc[
        confusion_matrix['evaluation_split'].eq('test')
    ].sort_values(
        ['language_model', 'true_label', 'predicted_label'], kind='stable'
    ).reset_index(drop=True)
    test_group_level = test_group_metrics.drop_duplicates(
        ['language_model', 'condition', 'audit_group']
    ).sort_values(['language_model', 'audit_group'], kind='stable').reset_index(
        drop=True
    )

    _validate_numeric_coverage(
        'validation_results',
        validation_results,
        _column_names(SUMMARY_METRICS),
        {'example_count', 'rank'},
    )
    _validate_numeric_coverage(
        'results',
        final_results,
        _column_names(SUMMARY_METRICS) | {'validation_selection_score'},
        {'example_count', 'selected_on_validation_rank'},
    )
    _validate_numeric_coverage(
        'class_metrics',
        class_metrics,
        _column_names(CLASS_RATE_COLUMNS + CLASS_COUNT_COLUMNS),
        {'example_count'},
    )
    _validate_numeric_coverage(
        'group_metrics',
        group_metrics,
        _column_names(GROUP_RATE_COLUMNS + GROUP_COUNT_COLUMNS)
        | {'group_n', 'group_accuracy'},
        {'example_count'},
    )
    _validate_numeric_coverage(
        'fairness_metrics',
        fairness_metrics,
        _column_names(FAIRNESS_CLASS_COLUMNS + FAIRNESS_COVERAGE_COLUMNS),
        {'example_count'},
    )
    _validate_numeric_coverage(
        'confusion_matrix',
        confusion_matrix,
        {'count'},
        {'example_count'},
    )

    plots: dict[str, Path] = {}
    plots.update(_create_summary_plots(validation_results, 'validation', output_dir))
    plots.update(_create_summary_plots(final_results, 'test', output_dir))

    selection_path = output_dir / 'validation_selection_vs_final_test.png'
    _plot_selection_vs_test(final_results, ranking_metric, selection_path)
    plots['validation_selection_vs_final_test'] = selection_path

    detail_specs = (
        (
            'test_class_rates',
            test_class_metrics,
            CLASS_RATE_COLUMNS,
            _class_labels(test_class_metrics),
            'Final-test per-class rates',
            (0.0, 1.0),
        ),
        (
            'test_class_counts',
            test_class_metrics,
            CLASS_COUNT_COLUMNS,
            _class_labels(test_class_metrics),
            'Final-test per-class counts',
            None,
        ),
        (
            'test_group_accuracy',
            test_group_level,
            (('group_accuracy', 'Group accuracy'),),
            _group_level_labels(test_group_level),
            'Final-test group accuracy',
            (0.0, 1.0),
        ),
        (
            'test_group_size',
            test_group_level,
            (('group_n', 'Group size'),),
            _group_level_labels(test_group_level),
            'Final-test group sizes',
            None,
        ),
        (
            'test_group_class_rates',
            test_group_metrics,
            GROUP_RATE_COLUMNS,
            _group_labels(test_group_metrics),
            'Final-test per-class/per-group rates',
            (0.0, 1.0),
        ),
        (
            'test_group_class_counts',
            test_group_metrics,
            GROUP_COUNT_COLUMNS,
            _group_labels(test_group_metrics),
            'Final-test per-class/per-group counts',
            None,
        ),
        (
            'test_classwise_fairness',
            test_fairness_metrics,
            FAIRNESS_CLASS_COLUMNS,
            _class_labels(test_fairness_metrics),
            'Final-test classwise fairness metrics',
            (0.0, 1.0),
        ),
        (
            'test_fairness_coverage',
            test_fairness_metrics,
            FAIRNESS_COVERAGE_COLUMNS,
            _class_labels(test_fairness_metrics),
            'Final-test fairness coverage',
            None,
        ),
    )
    for name, frame, metrics, labels, title, bounds in detail_specs:
        path = output_dir / f'{name}.png'
        _plot_metric_panels(
            frame,
            metrics,
            labels,
            f'{title}\n{_context_title(frame)}',
            path,
            bounds=bounds,
        )
        plots[name] = path

    confusion_path = output_dir / 'test_confusion_matrices.png'
    _plot_confusion_matrices(test_confusion, confusion_path)
    plots['test_confusion_matrices'] = confusion_path
    return plots
