from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from dataset import display_column_name

CONDITION_COLUMNS = [
    'condition',
    'evaluation_split',
    'target',
    'audit_column',
    'retrieval',
    'embedding_model',
    'k',
    'example_order',
    'prompt_name',
    'llm',
]


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else np.nan


def _rate_range(values: list[float]) -> float:
    defined = [value for value in values if not pd.isna(value)]
    return max(defined) - min(defined) if len(defined) >= 2 else np.nan


def calculate_metrics(
        predictions: pd.DataFrame,
        labels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate hard-label classification and group-disparity metrics."""

    result_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    for _, condition_predictions in predictions.groupby(['evaluation_split', 'condition'], sort=False):
        condition_metadata = {column: condition_predictions[column].iloc[0] for column in CONDITION_COLUMNS}
        true_labels = condition_predictions['true_label']
        predicted_labels = condition_predictions['predicted_label']

        condition_class_rows: list[dict[str, Any]] = []
        for label in labels:
            true_positive = ((true_labels == label) & (predicted_labels == label)).sum()
            false_positive = ((true_labels != label) & (predicted_labels == label)).sum()
            false_negative = ((true_labels == label) & (predicted_labels != label)).sum()
            true_negative = ((true_labels != label) & (predicted_labels != label)).sum()
            support = true_positive + false_negative
            predicted_count = true_positive + false_positive
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / support
            f1 = 2 * true_positive / (
                    2 * true_positive + false_positive + false_negative
            )
            specificity = true_negative / (true_negative + false_positive)
            class_metric_row = {
                **condition_metadata,
                'target_class': label,
                'support': support,
                'predicted_count': predicted_count,
                'tp': true_positive,
                'fp': false_positive,
                'fn': false_negative,
                'tn': true_negative,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'false_positive_rate': false_positive / (false_positive + true_negative),
                'false_negative_rate': false_negative / (false_negative + true_positive),
                'negative_predictive_value': _safe_rate(true_negative, true_negative + false_negative),
            }
            condition_class_rows.append(class_metric_row)
            class_rows.append(class_metric_row)

        class_frame = pd.DataFrame(condition_class_rows)
        sample_count = len(condition_predictions)
        correct_count = (true_labels == predicted_labels).sum()
        accuracy = correct_count / sample_count
        class_weights = class_frame['support'] / sample_count

        for true_label in labels:
            for predicted_label in labels:
                confusion_rows.append(
                    {
                        **condition_metadata,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'count': ((true_labels == true_label) & (predicted_labels == predicted_label)).sum(),
                    }
                )

        audit_groups = condition_predictions['audit_group'].unique().tolist()
        group_accuracy_values: list[float] = []
        condition_fairness: list[dict[str, Any]] = []

        for target_class in labels:
            class_group_rows: list[dict[str, Any]] = []
            for audit_group in audit_groups:
                group_predictions = condition_predictions[condition_predictions['audit_group'] == audit_group]
                group_truth = group_predictions['true_label']
                group_predicted = predicted_labels.loc[group_predictions.index]
                actual_positive = group_truth == target_class
                predicted_positive = group_predicted == target_class
                tp = (actual_positive & predicted_positive).sum()
                fp = (~actual_positive & predicted_positive).sum()
                fn = (actual_positive & ~predicted_positive).sum()
                tn = (~actual_positive & ~predicted_positive).sum()
                group_n = len(group_predictions)
                group_accuracy = (group_truth == group_predicted).mean()
                group_metric_row = {
                    **condition_metadata,
                    'target_class': target_class,
                    'audit_group': audit_group,
                    'group_n': group_n,
                    'group_accuracy': group_accuracy,
                    'positive_support': tp + fn,
                    'negative_support': fp + tn,
                    'predicted_positive': tp + fp,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn,
                    'selection_rate': (tp + fp) / group_n,
                    'true_positive_rate': _safe_rate(tp, tp + fn),
                    'false_positive_rate': _safe_rate(fp, fp + tn),
                    'false_negative_rate': _safe_rate(fn, tp + fn),
                    'positive_predictive_value': _safe_rate(tp, tp + fp),
                    'specificity': _safe_rate(tn, tn + fp),
                    'negative_predictive_value': _safe_rate(tn, tn + fn),
                }
                class_group_rows.append(group_metric_row)
                group_rows.append(group_metric_row)

            group_metrics = pd.DataFrame(class_group_rows)
            selection_rates = group_metrics['selection_rate']
            demographic_parity_ratio = selection_rates.min() / selection_rates.max() if selection_rates.max() > 0 else np.nan
            equal_opportunity_difference = _rate_range(group_metrics['true_positive_rate'].tolist())
            false_positive_rate_difference = _rate_range(group_metrics['false_positive_rate'].tolist())
            equalized_odds_difference = np.maximum(equal_opportunity_difference, false_positive_rate_difference)

            fairness_row = {
                **condition_metadata,
                'target_class': target_class,
                'groups_compared': len(audit_groups),
                'selection_rate_groups_defined': (group_metrics['selection_rate'].notna().sum()),
                'tpr_groups_defined': (group_metrics['true_positive_rate'].notna().sum()),
                'fpr_groups_defined': (group_metrics['false_positive_rate'].notna().sum()),
                'ppv_groups_defined': (group_metrics['positive_predictive_value'].notna().sum()),
                'demographic_parity_difference': _rate_range(group_metrics['selection_rate'].tolist()),
                'demographic_parity_ratio': demographic_parity_ratio,
                'equal_opportunity_difference': equal_opportunity_difference,
                'false_positive_rate_difference': false_positive_rate_difference,
                'equalized_odds_difference': equalized_odds_difference,
                'predictive_parity_difference': _rate_range(group_metrics['positive_predictive_value'].tolist()),
            }
            fairness_rows.append(fairness_row)
            condition_fairness.append(fairness_row)

        # Group-wide accuracy does not depend on a target class.
        for audit_group in audit_groups:
            group_predictions = condition_predictions[condition_predictions['audit_group'] == audit_group]
            group_truth = group_predictions['true_label']
            group_predicted = predicted_labels.loc[group_predictions.index]
            group_accuracy_values.append((group_truth == group_predicted).mean())

        condition_fairness_metrics = pd.DataFrame(condition_fairness)
        result_rows.append(
            {
                **condition_metadata,
                'sample_count': sample_count,
                'n_classes': len(labels),
                'n_audit_groups': len(audit_groups),
                'accuracy': accuracy,
                'balanced_accuracy': class_frame['recall'].mean(),
                'macro_precision': class_frame['precision'].mean(),
                'macro_recall': class_frame['recall'].mean(),
                'macro_f1': class_frame['f1'].mean(),
                'micro_precision': accuracy,
                'micro_recall': accuracy,
                'micro_f1': accuracy,
                'weighted_precision': (
                        class_frame['precision'] * class_weights
                ).sum(),
                'weighted_recall': (class_frame['recall'] * class_weights).sum(),
                'weighted_f1': (class_frame['f1'] * class_weights).sum(),
                'matthews_correlation_coefficient': matthews_corrcoef(
                    true_labels, predicted_labels
                ),
                'cohen_kappa': cohen_kappa_score(true_labels, predicted_labels),
                'worst_group_accuracy': min(group_accuracy_values),
                'group_accuracy_difference': _rate_range(group_accuracy_values),
                'mean_demographic_parity_difference': condition_fairness_metrics[
                    'demographic_parity_difference'
                ].mean(),
                'max_demographic_parity_difference': condition_fairness_metrics[
                    'demographic_parity_difference'
                ].max(),
                'mean_demographic_parity_ratio': condition_fairness_metrics[
                    'demographic_parity_ratio'
                ].mean(),
                'min_demographic_parity_ratio': condition_fairness_metrics[
                    'demographic_parity_ratio'
                ].min(),
                'mean_equal_opportunity_difference': condition_fairness_metrics[
                    'equal_opportunity_difference'
                ].mean(),
                'max_equal_opportunity_difference': condition_fairness_metrics[
                    'equal_opportunity_difference'
                ].max(),
                'mean_false_positive_rate_difference': condition_fairness_metrics[
                    'false_positive_rate_difference'
                ].mean(),
                'max_false_positive_rate_difference': condition_fairness_metrics[
                    'false_positive_rate_difference'
                ].max(),
                'mean_equalized_odds_difference': condition_fairness_metrics[
                    'equalized_odds_difference'
                ].mean(),
                'max_equalized_odds_difference': condition_fairness_metrics[
                    'equalized_odds_difference'
                ].max(),
                'n_demographic_parity_defined_classes': condition_fairness_metrics[
                    'demographic_parity_difference'
                ].notna().sum(),
                'n_demographic_parity_ratio_defined_classes': condition_fairness_metrics[
                    'demographic_parity_ratio'
                ].notna().sum(),
                'n_equal_opportunity_defined_classes': condition_fairness_metrics[
                    'equal_opportunity_difference'
                ].notna().sum(),
                'n_false_positive_rate_defined_classes': condition_fairness_metrics[
                    'false_positive_rate_difference'
                ].notna().sum(),
                'n_equalized_odds_defined_classes': condition_fairness_metrics[
                    'equalized_odds_difference'
                ].notna().sum(),
                'n_predictive_parity_defined_classes': condition_fairness_metrics[
                    'predictive_parity_difference'
                ].notna().sum(),
                'mean_predictive_parity_difference': condition_fairness_metrics[
                    'predictive_parity_difference'
                ].mean(),
                'max_predictive_parity_difference': condition_fairness_metrics[
                    'predictive_parity_difference'
                ].max(),
            }
        )

    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(fairness_rows),
    )


def rank_results(
        results: pd.DataFrame, metric: str, direction: str
) -> pd.DataFrame:
    """Rank prompt configurations separately within each LLM."""

    if metric not in results.columns:
        available = ', '.join(results.select_dtypes(include='number').columns)
        raise ValueError(
            f'Unknown ranking metric {metric!r}. Numeric result columns: {available}'
        )
    if not pd.api.types.is_numeric_dtype(results[metric]):
        raise ValueError(f'Ranking metric {metric!r} must be numeric')
    if results[metric].notna().sum() == 0:
        raise ValueError(f'Ranking metric {metric!r} is undefined for every condition')
    ranked = results.sort_values(
        ['llm', metric, 'condition'],
        ascending=[True, direction == 'minimize', True],
        na_position='last',
        kind='stable',
    ).reset_index(drop=True)
    ranked.insert(
        0,
        'rank',
        ranked.groupby('llm', sort=False).cumcount() + 1,
    )
    ranked.insert(1, 'is_best', ranked['rank'].eq(1))
    return ranked


def plot_results(
        validation_results: pd.DataFrame,
        output: Path,
) -> None:
    """Plot validation comparisons; final-test rows are saved separately."""

    import matplotlib.pyplot as plt

    plot_frame = validation_results.sort_values(
        'rank', ascending=False
    ).reset_index(drop=True)
    labels = [
        (
            f'llm={row.llm}, {row.retrieval}, '
            f'embedding={row.embedding_model}, '
            f'k={row.k}, {row.prompt_name}, {row.example_order}'
        )
        for row in plot_frame.itertuples(index=False)
    ]
    y = np.arange(len(plot_frame))
    height = max(5.0, 0.46 * len(plot_frame) + 1.8)
    figure, axes = plt.subplots(1, 2, figsize=(16, height), sharey=True)

    axes[0].barh(y - 0.18, plot_frame['accuracy'], 0.36, label='Accuracy')
    axes[0].barh(y + 0.18, plot_frame['macro_f1'], 0.36, label='Macro-F1')
    axes[0].set_xlim(0, 1)
    axes[0].set_title('Prediction quality (higher is better)')
    axes[0].legend()

    axes[1].barh(
        y - 0.24,
        plot_frame['max_demographic_parity_difference'],
        0.24,
        label='Demographic parity',
    )
    axes[1].barh(
        y,
        plot_frame['max_equal_opportunity_difference'],
        0.24,
        label='Equal opportunity',
    )
    axes[1].barh(
        y + 0.24,
        plot_frame['max_equalized_odds_difference'],
        0.24,
        label='Equalized odds',
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Maximum group differences (lower is better)')
    axes[1].legend()

    axes[0].set_yticks(y, labels)
    for axis in axes:
        axis.grid(axis='x', alpha=0.2)
        axis.set_xlabel('Score')
    figure.suptitle(
        f'Validation prompt comparison — target: {plot_frame['target'].iloc[0]}, '
        f'audit groups: {plot_frame['audit_column'].iloc[0]}\n'
        f'One validation winner and one final-test row per LLM'
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(figure)


def write_best_prompts(
        path: Path,
        selected: pd.DataFrame,
        prompt_templates: Mapping[str, Any],
        labels: list[str],
        ranking_metric: str,
        ranking_direction: str,
        final_results: pd.DataFrame,
) -> None:
    """Save one validation-selected prompt and final-test score per LLM."""

    sections: list[str] = []
    final_by_llm = final_results.set_index('llm')
    for best in selected.itertuples(index=False):
        final_result = final_by_llm.loc[best.llm]
        resolved_prompt = prompt_templates[best.prompt_name].format(
            target=display_column_name(best.target),
            other_column=display_column_name(best.audit_column),
            labels=', '.join(labels),
        )
        sections.append(
            f'LLM: {best.llm}\n'
            f'Selected on validation metric: {ranking_metric} '
            f'({ranking_direction})\n'
            f'Validation score: {getattr(best, ranking_metric)}\n'
            f'Final test score: {final_result[ranking_metric]}\n'
            f'Retrieval: {best.retrieval}\n'
            f'Embedding model: {best.embedding_model}\n'
            f'k: {best.k}\n'
            f'Example order: {best.example_order}\n'
            f'Prompt name: {best.prompt_name}\n\n'
            f'{resolved_prompt}'
        )
    path.write_text('\n\n---\n\n'.join(sections) + '\n', encoding='utf-8')
