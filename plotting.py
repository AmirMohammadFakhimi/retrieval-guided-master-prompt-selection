from collections.abc import Sequence
from math import ceil
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from evaluation import ACCURACY_METRIC_COLUMN, MACRO_RECALL_METRIC_COLUMN

MetricSpec = tuple[str, str]

QUALITY_METRICS: tuple[MetricSpec, ...] = (
    (ACCURACY_METRIC_COLUMN, 'Accuracy / micro precision / micro recall / micro F1 / weighted recall'),
    ('macro_precision', 'Macro precision'),
    (MACRO_RECALL_METRIC_COLUMN, 'Macro recall / balanced accuracy'),
    ('macro_f1', 'Macro F1'),
    ('weighted_precision', 'Weighted precision'),
    ('weighted_f1', 'Weighted F1'),
    ('worst_audit_group_accuracy', 'Worst audit-group accuracy'),
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

MINIMUM_FAIRNESS_DIFFERENCES: tuple[MetricSpec, ...] = (
    ('min_demographic_parity_difference', 'Minimum demographic-parity difference'),
    ('min_equal_opportunity_difference', 'Minimum equal-opportunity difference'),
    ('min_false_positive_rate_difference', 'Minimum false-positive-rate difference'),
    ('min_equalized_odds_difference', 'Minimum equalized-odds difference'),
    ('min_predictive_parity_difference', 'Minimum predictive-parity difference'),
)

WORST_FAIRNESS_DIFFERENCES: tuple[MetricSpec, ...] = (
    ('audit_group_accuracy_difference', 'Audit-group accuracy difference'),
    ('max_demographic_parity_difference', 'Maximum demographic-parity difference'),
    ('max_equal_opportunity_difference', 'Maximum equal-opportunity difference'),
    ('max_false_positive_rate_difference', 'Maximum false-positive-rate difference'),
    ('max_equalized_odds_difference', 'Maximum equalized-odds difference'),
    ('max_predictive_parity_difference', 'Maximum predictive-parity difference'),
)

FAIRNESS_RATIOS: tuple[MetricSpec, ...] = (
    ('mean_demographic_parity_ratio', 'Mean demographic-parity ratio'),
    ('min_demographic_parity_ratio', 'Minimum demographic-parity ratio'),
    ('max_demographic_parity_ratio', 'Maximum demographic-parity ratio'),
)

SUMMARY_SIZE_COLUMNS: tuple[MetricSpec, ...] = (
    ('sample_count', 'Evaluated rows'),
    ('n_target_labels', 'Target labels'),
    ('n_audit_groups', 'Audit groups'),
)

SUMMARY_COVERAGE_COLUMNS: tuple[MetricSpec, ...] = (
    ('n_target_labels', 'Total target labels'),
    ('n_precision_defined_target_labels', 'Target labels defining precision'),
    ('n_recall_defined_target_labels', 'Target labels defining recall'),
    ('n_f1_defined_target_labels', 'Target labels defining F1'),
    ('n_demographic_parity_defined_target_labels', 'Target labels defining demographic-parity difference'),
    ('n_demographic_parity_ratio_defined_target_labels', 'Target labels defining demographic-parity ratio'),
    ('n_equal_opportunity_defined_target_labels', 'Target labels defining equal-opportunity difference'),
    ('n_false_positive_rate_defined_target_labels', 'Target labels defining false-positive-rate difference'),
    ('n_equalized_odds_defined_target_labels', 'Target labels defining equalized-odds difference'),
    ('n_predictive_parity_defined_target_labels', 'Target labels defining predictive-parity difference'),
)

TARGET_LABEL_RATE_COLUMNS: tuple[MetricSpec, ...] = (
    ('selection_rate', 'Selection rate'),
    ('precision', 'Precision / positive predictive value'),
    ('recall', 'Recall / true-positive rate'),
    ('f1', 'F1'),
    ('specificity', 'Specificity / true-negative rate'),
    ('false_positive_rate', 'False-positive rate'),
    ('false_negative_rate', 'False-negative rate'),
    ('negative_predictive_value', 'Negative predictive value'),
)

TARGET_LABEL_COUNT_COLUMNS: tuple[MetricSpec, ...] = (
    ('positive_support', 'Positive support'),
    ('negative_support', 'Negative support'),
    ('predicted_positive', 'Predicted positive'),
    ('tp', 'True positives'),
    ('fp', 'False positives'),
    ('fn', 'False negatives'),
    ('tn', 'True negatives'),
)

AUDIT_GROUP_RATE_COLUMNS = TARGET_LABEL_RATE_COLUMNS
AUDIT_GROUP_COUNT_COLUMNS = TARGET_LABEL_COUNT_COLUMNS

FAIRNESS_TARGET_LABEL_COLUMNS: tuple[MetricSpec, ...] = (
    ('demographic_parity_difference', 'Demographic-parity difference'),
    ('demographic_parity_ratio', 'Demographic-parity ratio'),
    ('equal_opportunity_difference', 'Equal-opportunity difference'),
    ('false_positive_rate_difference', 'False-positive-rate difference'),
    ('equalized_odds_difference', 'Equalized-odds difference'),
    ('predictive_parity_difference', 'Predictive-parity difference'),
)

FAIRNESS_COVERAGE_COLUMNS: tuple[MetricSpec, ...] = (
    ('n_audit_groups_compared', 'Total audit groups'),
    ('n_selection_rate_defined_audit_groups', 'Audit groups defining selection rate'),
    ('n_recall_defined_audit_groups', 'Audit groups defining recall'),
    ('n_false_positive_rate_defined_audit_groups', 'Audit groups defining false-positive rate'),
    ('n_precision_defined_audit_groups', 'Audit groups defining precision'),
)

SUMMARY_METRICS = (
        QUALITY_METRICS
        + AGREEMENT_METRICS
        + MEAN_FAIRNESS_DIFFERENCES
        + MINIMUM_FAIRNESS_DIFFERENCES
        + WORST_FAIRNESS_DIFFERENCES
        + FAIRNESS_RATIOS
        + SUMMARY_SIZE_COLUMNS
        + SUMMARY_COVERAGE_COLUMNS
)


def _column_names(metrics: Sequence[MetricSpec]) -> set[str]:
    return {column for column, _ in metrics}


def _validate_numeric_coverage(
        table_name: str,
        table: pd.DataFrame,
        plotted_columns: set[str],
        excluded_columns: set[str],
) -> None:
    numeric_columns = set(table.select_dtypes(include='number').columns)
    uncovered = sorted(numeric_columns - plotted_columns - excluded_columns)

    if uncovered:
        raise ValueError(f'Numeric columns in {table_name} have no corresponding plot: {uncovered}')


def _context_title(table: pd.DataFrame) -> str:
    return (f'target={table['target'].iloc[0]}, audit groups={table['audit_column'].iloc[0]}')


def _condition_label(row: object) -> str:
    return (
        f'{row.language_model} | {row.retrieval_method} | '
        f'{row.embedding_model} | examples={row.example_count} | '
        f'{row.prompt_name} | {row.example_order}'
    )


def _condition_labels(results: pd.DataFrame) -> list[str]:
    return [_condition_label(row) for row in results.itertuples(index=False)]


def _target_label_labels(metric_table: pd.DataFrame) -> list[str]:
    return [
        f'{_condition_label(row)} | target label={row.target_label}'
        for row in metric_table.itertuples(index=False)
    ]


def _target_label_audit_group_labels(
        audit_group_metrics: pd.DataFrame,
) -> list[str]:
    return [
        f'{_condition_label(row)} | target label={row.target_label} | '
        f'audit group={row.audit_group}'
        for row in audit_group_metrics.itertuples(index=False)
    ]


def _audit_group_labels(audit_group_metrics: pd.DataFrame) -> list[str]:
    return [
        f'{_condition_label(row)} | audit group={row.audit_group}'
        for row in audit_group_metrics.itertuples(index=False)
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
        metric_table: pd.DataFrame,
        metrics: Sequence[MetricSpec],
        row_labels: Sequence[str],
        title: str,
        output: Path,
        bounds: tuple[float, float] | None = None,
        is_selected: pd.Series | None = None,
        selection_name: str | None = None,
) -> None:
    row_count = len(metric_table)
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
    if is_selected is None:
        colors = np.full(row_count, '#4C78A8', dtype=object)
    else:
        colors = np.where(
            is_selected.to_numpy(dtype=bool),
            '#1565C0',
            '#90CAF9',
        )

    for metric_number, ((column, label), axis) in enumerate(zip(metrics, axes, strict=True)):
        values = pd.to_numeric(metric_table[column], errors='coerce').to_numpy(dtype=float)
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
    if is_selected is not None:
        if selection_name is None:
            raise ValueError('selection_name is required when is_selected is provided')
        legend_items = [
            Patch(
                facecolor='#1565C0',
                label=f'{selection_name.title()}-best condition',
            ),
            Patch(facecolor='#90CAF9', label='Other condition'),
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


def _create_summary_plots(
        results: pd.DataFrame,
        split: str,
        output_dir: Path,
        selection_name: str,
) -> dict[str, Path]:
    rank_column = f'{selection_name}_rank'
    winner_column = f'is_{selection_name}_best'
    ranked_results = results.sort_values(
        ['language_model', rank_column, 'condition'],
        kind='stable',
    ).reset_index(drop=True)
    condition_labels = _condition_labels(ranked_results)
    is_selected = ranked_results[winner_column]
    context = _context_title(ranked_results)
    plot_specs = (
        ('quality_rates', QUALITY_METRICS, (0.0, 1.0), 'Quality rates'),
        ('agreement_scores', AGREEMENT_METRICS, (-1.0, 1.0), 'Agreement scores'),
        (
            'mean_fairness_differences',
            MEAN_FAIRNESS_DIFFERENCES,
            (0.0, 1.0),
            'Mean target-label fairness differences',
        ),
        (
            'minimum_fairness_differences',
            MINIMUM_FAIRNESS_DIFFERENCES,
            (0.0, 1.0),
            'Minimum target-label fairness differences',
        ),
        (
            'worst_fairness_differences',
            WORST_FAIRNESS_DIFFERENCES,
            (0.0, 1.0),
            'Worst target-label and audit-group accuracy differences',
        ),
        (
            'fairness_ratios',
            FAIRNESS_RATIOS,
            (0.0, 1.0),
            'Demographic-parity ratios',
        ),
        ('sample_counts', SUMMARY_SIZE_COLUMNS, None, 'Evaluation sizes'),
        (
            'defined_target_label_counts',
            SUMMARY_COVERAGE_COLUMNS,
            None,
            'Target-label metric coverage: total and defined counts',
        ),
    )
    plots: dict[str, Path] = {}
    for suffix, metrics, bounds, plot_title in plot_specs:
        name = f'{split}_{selection_name}_ranked_{suffix}'
        path = output_dir / f'{name}.png'
        _plot_metric_panels(
            ranked_results,
            metrics,
            condition_labels,
            f'{split.title()} — {selection_name.title()}-ranked {plot_title}\n'
            f'{context}',
            path,
            bounds=bounds,
            is_selected=is_selected,
            selection_name=selection_name,
        )
        plots[name] = path
    return plots


def _plot_selected_confusion_matrices(
        confusion_matrix: pd.DataFrame,
        split: str,
        selection_name: str,
        output: Path,
) -> None:
    selected_conditions = confusion_matrix[
        ['language_model', 'condition']
    ].drop_duplicates()
    columns = min(2, len(selected_conditions))
    rows = ceil(len(selected_conditions) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8 * columns, 7 * rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, selected in zip(
            axes_flat,
            selected_conditions.itertuples(index=False),
            strict=False,
    ):
        condition_confusion_matrix = confusion_matrix.loc[
            confusion_matrix['language_model'].eq(selected.language_model)
            & confusion_matrix['condition'].eq(selected.condition)
        ]
        true_labels = (
            condition_confusion_matrix['true_label'].drop_duplicates().tolist()
        )
        predicted_labels = (
            condition_confusion_matrix['predicted_label']
            .drop_duplicates()
            .tolist()
        )
        label_order = list(dict.fromkeys(true_labels + predicted_labels))
        confusion_counts = condition_confusion_matrix.pivot_table(
            index='true_label',
            columns='predicted_label',
            values='count',
            aggfunc='sum',
            fill_value=0,
        ).reindex(index=label_order, columns=label_order, fill_value=0)
        image = axis.imshow(confusion_counts.to_numpy(), cmap='Blues', vmin=0)
        axis.set_title(fill(_condition_label(
            condition_confusion_matrix.iloc[0]
        ), width=70))
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
            text_threshold = confusion_counts.to_numpy().max() / 2
            for row_number in range(len(label_order)):
                for column_number in range(len(label_order)):
                    count = int(
                        confusion_counts.iloc[row_number, column_number]
                    )
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

    for axis in axes_flat[len(selected_conditions):]:
        axis.remove()
    figure.suptitle(
        f'{split.title()} {selection_name}-best-condition confusion matrices\n'
        f'{_context_title(confusion_matrix)}',
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(figure)


def _create_selected_condition_plots(
        result_tables: dict[str, pd.DataFrame],
        output_dir: Path,
        evaluation_split: str,
        selection_name: str,
) -> dict[str, Path]:
    results = result_tables['results']
    selected_conditions = set(
        results.loc[results[f'is_{selection_name}_best'], 'condition']
    )
    if not selected_conditions:
        return {}

    target_label_metrics = result_tables['target_label_metrics']
    audit_group_metrics = result_tables['audit_group_metrics']
    fairness_metrics = result_tables['fairness_metrics']
    confusion_matrix = result_tables['confusion_matrix']
    selected_target_label_metrics = target_label_metrics.loc[
        target_label_metrics['condition'].isin(selected_conditions)
    ].sort_values(
        ['language_model', 'condition', 'target_label'],
        kind='stable',
    ).reset_index(drop=True)
    selected_audit_group_metrics = audit_group_metrics.loc[
        audit_group_metrics['condition'].isin(selected_conditions)
    ].sort_values(
        ['language_model', 'condition', 'target_label', 'audit_group'],
        kind='stable',
    ).reset_index(drop=True)
    selected_fairness_metrics = fairness_metrics.loc[
        fairness_metrics['condition'].isin(selected_conditions)
    ].sort_values(
        ['language_model', 'condition', 'target_label'],
        kind='stable',
    ).reset_index(drop=True)
    selected_confusion_matrix = confusion_matrix.loc[
        confusion_matrix['condition'].isin(selected_conditions)
    ].sort_values(
        ['language_model', 'condition', 'true_label', 'predicted_label'],
        kind='stable',
    ).reset_index(drop=True)
    selected_overall_audit_group_metrics = (
        selected_audit_group_metrics.drop_duplicates(
            ['language_model', 'condition', 'audit_group']
        )
        .sort_values(
            ['language_model', 'condition', 'audit_group'],
            kind='stable',
        )
        .reset_index(drop=True)
    )

    name_prefix = f'{evaluation_split}_best_{selection_name}'
    title_prefix = (
        f'{evaluation_split.title()} {selection_name}-best-condition'
    )
    detail_specs = (
        (
            f'{name_prefix}_target_label_rates',
            selected_target_label_metrics,
            TARGET_LABEL_RATE_COLUMNS,
            _target_label_labels(selected_target_label_metrics),
            f'{title_prefix} per-target-label rates',
            (0.0, 1.0),
        ),
        (
            f'{name_prefix}_target_label_counts',
            selected_target_label_metrics,
            TARGET_LABEL_COUNT_COLUMNS,
            _target_label_labels(selected_target_label_metrics),
            f'{title_prefix} per-target-label counts',
            None,
        ),
        (
            f'{name_prefix}_audit_group_accuracy',
            selected_overall_audit_group_metrics,
            (('audit_group_accuracy', 'Audit-group accuracy'),),
            _audit_group_labels(selected_overall_audit_group_metrics),
            f'{title_prefix} audit-group accuracy',
            (0.0, 1.0),
        ),
        (
            f'{name_prefix}_audit_group_size',
            selected_overall_audit_group_metrics,
            (('audit_group_n', 'Audit-group size'),),
            _audit_group_labels(selected_overall_audit_group_metrics),
            f'{title_prefix} audit-group sizes',
            None,
        ),
        (
            f'{name_prefix}_target_label_rates_by_audit_group',
            selected_audit_group_metrics,
            AUDIT_GROUP_RATE_COLUMNS,
            _target_label_audit_group_labels(selected_audit_group_metrics),
            f'{title_prefix} target-label rates by audit group',
            (0.0, 1.0),
        ),
        (
            f'{name_prefix}_target_label_counts_by_audit_group',
            selected_audit_group_metrics,
            AUDIT_GROUP_COUNT_COLUMNS,
            _target_label_audit_group_labels(selected_audit_group_metrics),
            f'{title_prefix} target-label counts by audit group',
            None,
        ),
        (
            f'{name_prefix}_target_label_fairness',
            selected_fairness_metrics,
            FAIRNESS_TARGET_LABEL_COLUMNS,
            _target_label_labels(selected_fairness_metrics),
            f'{title_prefix} target-label fairness metrics',
            (0.0, 1.0),
        ),
        (
            f'{name_prefix}_fairness_coverage',
            selected_fairness_metrics,
            FAIRNESS_COVERAGE_COLUMNS,
            _target_label_labels(selected_fairness_metrics),
            f'{title_prefix} fairness coverage: total and defined groups',
            None,
        ),
    )
    plots: dict[str, Path] = {}
    for name, metric_table, metrics, row_labels, title, bounds in detail_specs:
        path = output_dir / f'{name}.png'
        _plot_metric_panels(
            metric_table,
            metrics,
            row_labels,
            f'{title}\n{_context_title(metric_table)}',
            path,
            bounds=bounds,
        )
        plots[name] = path

    confusion_name = f'{name_prefix}_confusion_matrices'
    confusion_path = output_dir / f'{confusion_name}.png'
    _plot_selected_confusion_matrices(
        selected_confusion_matrix,
        evaluation_split,
        selection_name,
        confusion_path,
    )
    plots[confusion_name] = confusion_path
    return plots


def create_metric_plots(
        result_tables: dict[str, pd.DataFrame],
        output_dir: Path,
        evaluation_split: str,
) -> dict[str, Path]:
    """Create complete plots for the configured evaluation split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results = result_tables['results']
    target_label_metrics = result_tables['target_label_metrics']
    audit_group_metrics = result_tables['audit_group_metrics']
    fairness_metrics = result_tables['fairness_metrics']
    confusion_matrix = result_tables['confusion_matrix']

    _validate_numeric_coverage(
        'results',
        results,
        _column_names(SUMMARY_METRICS),
        {'example_count', 'quality_rank', 'fairness_rank'},
    )
    _validate_numeric_coverage(
        'target_label_metrics',
        target_label_metrics,
        _column_names(TARGET_LABEL_RATE_COLUMNS + TARGET_LABEL_COUNT_COLUMNS),
        {'example_count'},
    )
    _validate_numeric_coverage(
        'audit_group_metrics',
        audit_group_metrics,
        _column_names(AUDIT_GROUP_RATE_COLUMNS + AUDIT_GROUP_COUNT_COLUMNS)
        | {'audit_group_n', 'audit_group_accuracy'},
        {'example_count'},
    )
    _validate_numeric_coverage(
        'fairness_metrics',
        fairness_metrics,
        _column_names(FAIRNESS_TARGET_LABEL_COLUMNS + FAIRNESS_COVERAGE_COLUMNS),
        {'example_count'},
    )
    _validate_numeric_coverage(
        'confusion_matrix',
        confusion_matrix,
        {'count'},
        {'example_count'},
    )

    plots: dict[str, Path] = {}
    for selection_name in ('quality', 'fairness'):
        plots.update(_create_summary_plots(
            results,
            evaluation_split,
            output_dir,
            selection_name,
        ))
        plots.update(_create_selected_condition_plots(
            result_tables,
            output_dir,
            evaluation_split,
            selection_name,
        ))
    return plots
