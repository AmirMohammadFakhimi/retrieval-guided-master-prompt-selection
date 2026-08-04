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
    'retrieval_method',
    'embedding_model',
    'example_count',
    'example_order',
    'prompt_name',
    'language_model',
]

ACCURACY_METRIC_COLUMN = (
    'accuracy / micro_precision / micro_recall / micro_f1 / weighted_recall'
)
MACRO_RECALL_METRIC_COLUMN = 'macro_recall / balanced_accuracy'

METRIC_COLUMN_ALIASES = {
    'accuracy': ACCURACY_METRIC_COLUMN,
    'micro_precision': ACCURACY_METRIC_COLUMN,
    'micro_recall': ACCURACY_METRIC_COLUMN,
    'micro_f1': ACCURACY_METRIC_COLUMN,
    'weighted_recall': ACCURACY_METRIC_COLUMN,
    'macro_recall': MACRO_RECALL_METRIC_COLUMN,
    'balanced_accuracy': MACRO_RECALL_METRIC_COLUMN,
}

FAIRNESS_DIFFERENCE_METRICS = {
    'demographic_parity_difference': 'n_demographic_parity_defined_classes',
    'equal_opportunity_difference': 'n_equal_opportunity_defined_classes',
    'false_positive_rate_difference': 'n_false_positive_rate_defined_classes',
    'equalized_odds_difference': 'n_equalized_odds_defined_classes',
    'predictive_parity_difference': 'n_predictive_parity_defined_classes',
}

# Metric notation used in the calculations below:
# c is a target class, g is an audit group, K is the number of target classes,
# n_c is the true support of class c, N is the number of evaluated rows, and
# N_g is the number of evaluated rows in group g.
# TP, FP, FN, and TN are one-vs-rest counts for c (and for g in group metrics).
# P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN), TNR=TN/(TN+FP),
# FPR=FP/(FP+TN), FNR=FN/(FN+TP), and NPV=TN/(TN+FN).
# D_m is the set of classes where metric m is defined. Macro(m) averages m_c
# over D_m; Weighted(m)=sum_{c in D_m}(n_c*m_c)/sum_{c in D_m}(n_c).
# When m is defined for every class, D_m contains all K classes.
# For single-label multiclass predictions, sum_c(TP_c)=T and
# sum_c(FP_c)=sum_c(FN_c)=E, where N=T+E. Therefore micro precision,
# micro recall, micro F1, and accuracy all equal T/N. Also,
# WeightedRecall=sum_c(n_c*TP_c/n_c)/N=sum_c(TP_c)/N=accuracy, while
# BalancedAccuracy averages R_c over the supported classes and equals
# MacroRecall. Each equality family is stored once.
# For confusion matrix C, s=sum_ij(C_ij), c0=trace(C), p_k=sum_i(C_ik),
# and t_k=sum_j(C_kj): MCC=(c0*s-sum_k(p_k*t_k)) /
# sqrt((s^2-sum_k(p_k^2))*(s^2-sum_k(t_k^2))). Cohen's kappa is
# (p_o-p_e)/(1-p_e), with p_o=accuracy and p_e=sum_k((t_k/s)*(p_k/s)).


def resolve_metric_column(metric: str) -> str:
    """Return the result-column name for a standard metric name or shared column."""

    return METRIC_COLUMN_ALIASES.get(metric, metric)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    """Return a rate, or NaN when its required denominator is absent."""

    return numerator / denominator if denominator else np.nan


def _rate_range(values: list[float]) -> float:
    """Return max minus min when at least two rates are defined."""

    defined = [value for value in values if not pd.isna(value)]
    return max(defined) - min(defined) if len(defined) >= 2 else np.nan


def _rate_ratio(values: list[float]) -> float:
    """Return min divided by max when at least two rates and a positive max exist."""

    defined = [value for value in values if not pd.isna(value)]
    if len(defined) < 2 or max(defined) == 0:
        return np.nan
    return min(defined) / max(defined)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Average defined values using their class supports as weights."""

    defined = values.notna() & weights.gt(0)
    if not defined.any():
        return np.nan
    return float(np.average(values.loc[defined], weights=weights.loc[defined]))


def _validate_metric_inputs(predictions: pd.DataFrame, labels: list[str]) -> None:
    """Validate the assumptions required by every metric table."""

    if predictions.empty:
        raise ValueError('predictions must contain at least one row')
    if not labels:
        raise ValueError('labels must contain at least one target class')
    if len(labels) != len(set(labels)):
        raise ValueError('labels cannot contain duplicates')

    required_columns = set(CONDITION_COLUMNS) | {
        'true_label',
        'predicted_label',
        'audit_group',
    }
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise ValueError(f'predictions is missing required columns: {missing_columns}')

    value_columns = CONDITION_COLUMNS + ['true_label', 'predicted_label', 'audit_group']
    columns_with_missing_values = [
        column for column in value_columns if predictions[column].isna().any()
    ]
    if columns_with_missing_values:
        raise ValueError(
            f'predictions contains missing values in: {columns_with_missing_values}'
        )

    observed_labels = set(predictions['true_label']) | set(predictions['predicted_label'])
    unknown_labels = sorted(observed_labels - set(labels))
    if unknown_labels:
        raise ValueError(f'predictions contains labels not configured in labels: {unknown_labels}')


def _condition_metadata(condition_predictions: pd.DataFrame) -> dict[str, Any]:
    """Return metadata after confirming it is constant within a condition."""

    inconsistent_columns = [
        column
        for column in CONDITION_COLUMNS
        if condition_predictions[column].nunique(dropna=False) != 1
    ]
    if inconsistent_columns:
        condition = condition_predictions['condition'].iloc[0]
        raise ValueError(
            f'Condition {condition!r} has inconsistent metadata columns: '
            f'{inconsistent_columns}'
        )
    return {
        column: condition_predictions[column].iloc[0]
        for column in CONDITION_COLUMNS
    }


def _calculate_class_metrics(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        labels: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Calculate one-vs-rest class rates and the multiclass confusion matrix."""

    true_labels = condition_predictions['true_label']
    predicted_labels = condition_predictions['predicted_label']
    class_rows: list[dict[str, Any]] = []

    for label in labels:
        true_positive = int(((true_labels == label) & (predicted_labels == label)).sum())
        false_positive = int(((true_labels != label) & (predicted_labels == label)).sum())
        false_negative = int(((true_labels == label) & (predicted_labels != label)).sum())
        true_negative = int(((true_labels != label) & (predicted_labels != label)).sum())
        support = true_positive + false_negative
        predicted_count = true_positive + false_positive

        class_rows.append(
            {
                **condition_metadata,
                'target_class': label,
                'support': support,
                'predicted_count': predicted_count,
                'tp': true_positive,
                'fp': false_positive,
                'fn': false_negative,
                'tn': true_negative,
                'precision': _safe_rate(true_positive, predicted_count),
                'recall': _safe_rate(true_positive, support),
                'f1': _safe_rate(
                    2 * true_positive,
                    2 * true_positive + false_positive + false_negative,
                ),
                'specificity': _safe_rate(
                    true_negative, true_negative + false_positive
                ),
                'false_positive_rate': _safe_rate(
                    false_positive, false_positive + true_negative
                ),
                'false_negative_rate': _safe_rate(
                    false_negative, false_negative + true_positive
                ),
                'negative_predictive_value': _safe_rate(
                    true_negative, true_negative + false_negative
                ),
            }
        )

    confusion_counts = pd.crosstab(true_labels, predicted_labels).reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )
    confusion_rows = [
        {
            **condition_metadata,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'count': int(confusion_counts.loc[true_label, predicted_label]),
        }
        for true_label in labels
        for predicted_label in labels
    ]
    return pd.DataFrame(class_rows), confusion_rows


def _calculate_group_metrics(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    """Calculate group rates and one-vs-rest group disparities for every class."""

    grouped_predictions = list(condition_predictions.groupby('audit_group', sort=False))
    audit_groups = [audit_group for audit_group, _ in grouped_predictions]
    group_accuracy_values = [
        float((group['true_label'] == group['predicted_label']).mean())
        for _, group in grouped_predictions
    ]
    group_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    # For class c and group g: SR=(TP+FP)/N_g, TPR=TP/(TP+FN),
    # FPR=FP/(FP+TN), PPV=TP/(TP+FP), and group accuracy=correct_g/N_g.
    for target_class in labels:
        class_group_rows: list[dict[str, Any]] = []
        for (audit_group, group_predictions), group_accuracy in zip(
                grouped_predictions, group_accuracy_values, strict=True
        ):
            group_truth = group_predictions['true_label']
            group_predicted = group_predictions['predicted_label']
            actual_positive = group_truth == target_class
            predicted_positive = group_predicted == target_class
            true_positive = int((actual_positive & predicted_positive).sum())
            false_positive = int((~actual_positive & predicted_positive).sum())
            false_negative = int((actual_positive & ~predicted_positive).sum())
            true_negative = int((~actual_positive & ~predicted_positive).sum())
            group_size = len(group_predictions)

            group_metric_row = {
                **condition_metadata,
                'target_class': target_class,
                'audit_group': audit_group,
                'group_n': group_size,
                'group_accuracy': group_accuracy,
                'positive_support': true_positive + false_negative,
                'negative_support': false_positive + true_negative,
                'predicted_positive': true_positive + false_positive,
                'tp': true_positive,
                'fp': false_positive,
                'fn': false_negative,
                'tn': true_negative,
                'selection_rate': (true_positive + false_positive) / group_size,
                'true_positive_rate': _safe_rate(
                    true_positive, true_positive + false_negative
                ),
                'false_positive_rate': _safe_rate(
                    false_positive, false_positive + true_negative
                ),
                'false_negative_rate': _safe_rate(
                    false_negative, true_positive + false_negative
                ),
                'positive_predictive_value': _safe_rate(
                    true_positive, true_positive + false_positive
                ),
                'specificity': _safe_rate(
                    true_negative, true_negative + false_positive
                ),
                'negative_predictive_value': _safe_rate(
                    true_negative, true_negative + false_negative
                ),
            }
            class_group_rows.append(group_metric_row)
            group_rows.append(group_metric_row)

        group_frame = pd.DataFrame(class_group_rows)
        selection_rates = group_frame['selection_rate'].tolist()
        true_positive_rates = group_frame['true_positive_rate'].tolist()
        false_positive_rates = group_frame['false_positive_rate'].tolist()
        positive_predictive_values = group_frame['positive_predictive_value'].tolist()
        equal_opportunity_difference = _rate_range(true_positive_rates)
        false_positive_rate_difference = _rate_range(false_positive_rates)
        equalized_odds_difference = (
            max(equal_opportunity_difference, false_positive_rate_difference)
            if not pd.isna(equal_opportunity_difference)
            and not pd.isna(false_positive_rate_difference)
            else np.nan
        )

        # Across audit groups g: DPD=max(SR_g)-min(SR_g), DPR=min(SR_g)/max(SR_g),
        # EOD=max(TPR_g)-min(TPR_g), and FPRD=max(FPR_g)-min(FPR_g).
        # Equalized-odds difference=max(EOD,FPRD); predictive-parity difference
        # is max(PPV_g)-min(PPV_g). At least two defined group rates are required.
        fairness_rows.append(
            {
                **condition_metadata,
                'target_class': target_class,
                'groups_compared': len(audit_groups),
                'selection_rate_groups_defined': int(
                    group_frame['selection_rate'].notna().sum()
                ),
                'tpr_groups_defined': int(
                    group_frame['true_positive_rate'].notna().sum()
                ),
                'fpr_groups_defined': int(
                    group_frame['false_positive_rate'].notna().sum()
                ),
                'ppv_groups_defined': int(
                    group_frame['positive_predictive_value'].notna().sum()
                ),
                'demographic_parity_difference': _rate_range(selection_rates),
                'demographic_parity_ratio': _rate_ratio(selection_rates),
                'equal_opportunity_difference': equal_opportunity_difference,
                'false_positive_rate_difference': false_positive_rate_difference,
                'equalized_odds_difference': equalized_odds_difference,
                'predictive_parity_difference': _rate_range(
                    positive_predictive_values
                ),
            }
        )

    return group_rows, fairness_rows, group_accuracy_values


def _summarize_fairness(fairness_frame: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate classwise disparities while reporting how many classes define them."""

    # If D_d contains the classes where disparity d_c is defined, the reported
    # mean is sum_{c in D_d}(d_c)/|D_d| and the worst difference is max(d_c).
    # For demographic-parity ratio, where 1 is best, the worst ratio is min(d_c).
    summary: dict[str, float | int] = {}
    for metric, defined_count_column in FAIRNESS_DIFFERENCE_METRICS.items():
        summary[f'mean_{metric}'] = fairness_frame[metric].mean()
        summary[f'max_{metric}'] = fairness_frame[metric].max()
        summary[defined_count_column] = int(fairness_frame[metric].notna().sum())

    ratio = fairness_frame['demographic_parity_ratio']
    summary['mean_demographic_parity_ratio'] = ratio.mean()
    summary['min_demographic_parity_ratio'] = ratio.min()
    summary['n_demographic_parity_ratio_defined_classes'] = int(
        ratio.notna().sum()
    )
    return summary


def _summarize_condition(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        class_frame: pd.DataFrame,
        fairness_frame: pd.DataFrame,
        group_accuracy_values: list[float],
) -> dict[str, Any]:
    """Build one overall classification-and-fairness result row."""

    true_labels = condition_predictions['true_label']
    predicted_labels = condition_predictions['predicted_label']
    sample_count = len(condition_predictions)
    accuracy = float((true_labels == predicted_labels).mean())
    class_weights = class_frame['support'] / sample_count

    # The two shared columns below store formula-identical metrics once. Macro
    # and weighted averages ignore an undefined class rate and weight only the
    # remaining defined rates; the defined-class counts expose that coverage.
    # Accuracy=sum_c(TP_c)/N, worst-group accuracy=min_g(accuracy_g), and the
    # group-accuracy difference=max_g(accuracy_g)-min_g(accuracy_g).
    return {
        **condition_metadata,
        'sample_count': sample_count,
        'n_classes': len(class_frame),
        'n_audit_groups': len(group_accuracy_values),
        ACCURACY_METRIC_COLUMN: accuracy,
        'macro_precision': class_frame['precision'].mean(),
        MACRO_RECALL_METRIC_COLUMN: class_frame['recall'].mean(),
        'macro_f1': class_frame['f1'].mean(),
        'weighted_precision': _weighted_average(
            class_frame['precision'], class_weights
        ),
        'weighted_f1': _weighted_average(class_frame['f1'], class_weights),
        'n_precision_defined_classes': int(class_frame['precision'].notna().sum()),
        'n_recall_defined_classes': int(class_frame['recall'].notna().sum()),
        'n_f1_defined_classes': int(class_frame['f1'].notna().sum()),
        'matthews_correlation_coefficient': matthews_corrcoef(
            true_labels, predicted_labels
        ),
        'cohen_kappa': cohen_kappa_score(true_labels, predicted_labels),
        'worst_group_accuracy': min(group_accuracy_values),
        'group_accuracy_difference': _rate_range(group_accuracy_values),
        **_summarize_fairness(fairness_frame),
    }


def calculate_metrics(
        predictions: pd.DataFrame,
        labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate hard-label classification and group-disparity metrics."""

    _validate_metric_inputs(predictions, labels)
    result_rows: list[dict[str, Any]] = []
    class_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    for _, condition_predictions in predictions.groupby(
            ['evaluation_split', 'condition'], sort=False
    ):
        metadata = _condition_metadata(condition_predictions)
        class_frame, condition_confusion_rows = _calculate_class_metrics(
            condition_predictions, metadata, labels
        )
        (
            condition_group_rows,
            condition_fairness_rows,
            group_accuracy_values,
        ) = _calculate_group_metrics(condition_predictions, metadata, labels)
        fairness_frame = pd.DataFrame(condition_fairness_rows)

        result_rows.append(
            _summarize_condition(
                condition_predictions,
                metadata,
                class_frame,
                fairness_frame,
                group_accuracy_values,
            )
        )
        class_frames.append(class_frame)
        confusion_rows.extend(condition_confusion_rows)
        group_rows.extend(condition_group_rows)
        fairness_rows.extend(condition_fairness_rows)

    return (
        pd.DataFrame(result_rows),
        pd.concat(class_frames, ignore_index=True),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(fairness_rows),
    )


def rank_results(
        results: pd.DataFrame,
        metric: str,
        direction: str,
) -> pd.DataFrame:
    """Rank prompt configurations separately within each language model."""

    if direction not in {'maximize', 'minimize'}:
        raise ValueError("direction must be 'maximize' or 'minimize'")

    metric_column = resolve_metric_column(metric)
    if metric_column not in results.columns:
        available = ', '.join(results.select_dtypes(include='number').columns)
        raise ValueError(
            f'Unknown ranking metric {metric!r}. Numeric result columns: {available}'
        )
    if not pd.api.types.is_numeric_dtype(results[metric_column]):
        raise ValueError(f'Ranking metric {metric!r} must be numeric')

    defined_by_language_model = results.groupby('language_model', sort=False)[
        metric_column
    ].apply(lambda values: values.notna().any())
    undefined_language_models = defined_by_language_model.index[
        ~defined_by_language_model
    ].tolist()
    if undefined_language_models:
        raise ValueError(
            f'Ranking metric {metric!r} is undefined for every condition of '
            f'language models: {undefined_language_models}'
        )

    ranked = results.sort_values(
        ['language_model', metric_column, 'condition'],
        ascending=[True, direction == 'minimize', True],
        na_position='last',
        kind='stable',
    ).reset_index(drop=True)
    ranked.insert(
        0,
        'rank',
        ranked.groupby('language_model', sort=False).cumcount() + 1,
    )
    ranked.insert(1, 'is_best', ranked['rank'].eq(1))
    return ranked


def write_best_prompts(
        path: Path,
        selected: pd.DataFrame,
        prompt_templates: dict[str, Any],
        labels: list[str],
        ranking_metric: str,
        ranking_direction: str,
        final_results: pd.DataFrame,
) -> None:
    """Save one validation-selected prompt and final-test score per language model."""

    metric_column = resolve_metric_column(ranking_metric)
    sections: list[str] = []
    final_by_language_model = final_results.set_index('language_model')
    for best in selected.to_dict('records'):
        final_result = final_by_language_model.loc[best['language_model']]
        resolved_prompt = prompt_templates[best['prompt_name']].format(
            target=display_column_name(best['target']),
            audit_column=display_column_name(best['audit_column']),
            labels=', '.join(labels),
        )
        sections.append(
            f'Language model: {best['language_model']}\n'
            f'Selected on validation metric: {ranking_metric} '
            f'({ranking_direction})\n'
            f'Validation score: {best[metric_column]}\n'
            f'Final test score: {final_result[metric_column]}\n'
            f'Retrieval method: {best['retrieval_method']}\n'
            f'Embedding model: {best['embedding_model']}\n'
            f'Examples: {best['example_count']}\n'
            f'Example order: {best['example_order']}\n'
            f'Prompt name: {best['prompt_name']}\n\n'
            f'{resolved_prompt}'
        )
    path.write_text('\n\n---\n\n'.join(sections) + '\n', encoding='utf-8')
