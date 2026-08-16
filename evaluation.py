from pathlib import Path
from typing import Any, cast

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
    'prediction_method',
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

RESULT_NUMERIC_COLUMNS = frozenset({
    'example_count',
    'sample_count',
    'n_target_labels',
    'n_audit_groups',
    ACCURACY_METRIC_COLUMN,
    'macro_precision',
    MACRO_RECALL_METRIC_COLUMN,
    'macro_f1',
    'weighted_precision',
    'weighted_f1',
    'n_precision_defined_target_labels',
    'n_recall_defined_target_labels',
    'n_f1_defined_target_labels',
    'matthews_correlation_coefficient',
    'cohen_kappa',
    'worst_audit_group_accuracy',
    'audit_group_accuracy_difference',
    'mean_demographic_parity_difference',
    'min_demographic_parity_difference',
    'max_demographic_parity_difference',
    'n_demographic_parity_defined_target_labels',
    'mean_equal_opportunity_difference',
    'min_equal_opportunity_difference',
    'max_equal_opportunity_difference',
    'n_equal_opportunity_defined_target_labels',
    'mean_false_positive_rate_difference',
    'min_false_positive_rate_difference',
    'max_false_positive_rate_difference',
    'n_false_positive_rate_defined_target_labels',
    'mean_equalized_odds_difference',
    'min_equalized_odds_difference',
    'max_equalized_odds_difference',
    'n_equalized_odds_defined_target_labels',
    'mean_predictive_parity_difference',
    'min_predictive_parity_difference',
    'max_predictive_parity_difference',
    'n_predictive_parity_defined_target_labels',
    'mean_demographic_parity_ratio',
    'min_demographic_parity_ratio',
    'max_demographic_parity_ratio',
    'n_demographic_parity_ratio_defined_target_labels',
})


# Metric notation used in the calculations below:
# c is a target label, g is an audit group, K is the number of target labels,
# n_c is the true support of target label c, N is the number of evaluated rows,
# and N_g is the number of evaluated rows in an audit group g.
# TP, FP, FN, and TN are one-vs.-rest counts for c (and within g for audit-group metrics).
# Precision=PPV=TP/(TP+FP), Recall=TPR=TP/(TP+FN),
# F1=2TP/(2TP+FP+FN), and Specificity=TNR=TN/(TN+FP).
# Whenever defined, FPR=FP/(FP+TN)=1-Specificity and
# FNR=FN/(FN+TP)=1-Recall. NPV=TN/(TN+FN).
# D_m is the set of target labels where metric m is defined; D_m+ also requires
# n_c>0. Macro(m) averages over D_m, while Weighted(m) uses support weights over
# D_m+. Coverage columns report |D_m|. An empty aggregate remains NaN.
# For single-label multiclass predictions, sum_c(TP_c)=T and
# sum_c(FP_c)=sum_c(FN_c)=E, where N=T+E. Therefore, micro precision,
# micro recall, micro F1, and accuracy all equal T/N. Also,
# WeightedRecall=sum_c(n_c*TP_c/n_c)/N=sum_c(TP_c)/N=accuracy, while
# BalancedAccuracy averages R_c over the supported target labels and equals
# MacroRecall. Each equality family is stored once.
# For confusion matrix C, s=sum_ij(C_ij), c0=trace(C), p_k=sum_i(C_ik),
# and t_k=sum_j(C_kj): MCC=(c0*s-sum_k(p_k*t_k)) /
# sqrt((s^2-sum_k(p_k^2))*(s^2-sum_k(t_k^2))). Cohen's kappa is
# (p_o-p_e)/(1-p_e), with p_o=accuracy and p_e=sum_k((t_k/s)*(p_k/s)).


def resolve_metric_column(metric: str) -> str:
    """Return the result-column name for a standard metric name or shared column."""

    return METRIC_COLUMN_ALIASES.get(metric, metric)


def _safe_rate(numerator: float, denominator: float) -> float:
    """Return a rate, or NaN when its required denominator is absent."""

    return numerator / denominator if denominator else np.nan


def _rate_range(values: pd.Series) -> float:
    """Return max minus min when at least two rates are defined."""

    defined = [value for value in values if not pd.isna(value)]
    return max(defined) - min(defined) if len(defined) >= 2 else np.nan


def _rate_ratio(values: pd.Series) -> float:
    """Return min divided by max when at least two rates and a positive max exist."""

    defined = [value for value in values if not pd.isna(value)]
    if len(defined) < 2 or max(defined) == 0:
        return np.nan
    return min(defined) / max(defined)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Average defined values using their target-label supports as weights."""

    defined = values.notna() & weights.gt(0)
    if not defined.any():
        return np.nan

    return float(np.average(values.loc[defined], weights=weights.loc[defined]))


def _validate_metric_inputs(predictions: pd.DataFrame, target_labels: list[str]) -> None:
    """Validate the assumptions required by every metric table."""

    if predictions.empty:
        raise ValueError('predictions must contain at least one row')
    if not target_labels:
        raise ValueError('target_labels must contain at least one target label')
    if len(target_labels) != len(set(target_labels)):
        raise ValueError('target_labels cannot contain duplicates')

    value_columns = CONDITION_COLUMNS + [
        'true_label',
        'predicted_label',
        'audit_group',
    ]
    missing_columns = sorted(set(value_columns) - set(predictions.columns))
    if missing_columns:
        raise ValueError(f'predictions is missing required columns: {missing_columns}')

    columns_with_missing_values = [column for column in value_columns if predictions[column].isna().any()]
    if columns_with_missing_values:
        raise ValueError(f'predictions contains missing values in: {columns_with_missing_values}')

    observed_target_labels = set(predictions['true_label']) | set(predictions['predicted_label'])
    unknown_target_labels = sorted(observed_target_labels - set(target_labels))
    if unknown_target_labels:
        raise ValueError('predictions contains labels not configured in target_labels: {unknown_target_labels}')


def _condition_metadata(condition_predictions: pd.DataFrame) -> dict[str, Any]:
    """Return metadata after confirming it is constant within a condition."""

    inconsistent_columns = [
        column for column in CONDITION_COLUMNS if condition_predictions[column].nunique(dropna=False) != 1
    ]
    if inconsistent_columns:
        condition = condition_predictions['condition'].iloc[0]
        raise ValueError(f'Condition {condition!r} has inconsistent metadata columns: {inconsistent_columns}')

    return {column: condition_predictions[column].iloc[0] for column in CONDITION_COLUMNS}


def _calculate_one_vs_rest_metrics(
        true_labels: pd.Series,
        predicted_labels: pd.Series,
        target_label: str,
) -> dict[str, int | float]:
    """Calculate counts and rates with one target label treated as positive."""

    # For each target label c, treat c as positive and every other label as
    # negative. Positive support=TP+FN, negative support=FP+TN, and predicted
    # positives=TP+FP. Selection rate=(TP+FP)/N; precision=PPV=TP/(TP+FP);
    # recall=TPR=TP/(TP+FN); F1=2TP/(2TP+FP+FN); specificity=TNR=TN/(TN+FP);
    # FPR=FP/(FP+TN); FNR=FN/(FN+TP); and NPV=TN/(TN+FN). _safe_rate returns
    # NaN for a zero denominator. Whenever defined, FPR=1-specificity and
    # FNR=1-recall.
    actual_positive = true_labels == target_label
    predicted_positive = predicted_labels == target_label

    true_positive = int((actual_positive & predicted_positive).sum())
    false_positive = int((~actual_positive & predicted_positive).sum())
    false_negative = int((actual_positive & ~predicted_positive).sum())
    true_negative = int((~actual_positive & ~predicted_positive).sum())

    positive_support = true_positive + false_negative
    negative_support = false_positive + true_negative
    predicted_positive_count = true_positive + false_positive

    return {
        'positive_support': positive_support,
        'negative_support': negative_support,
        'predicted_positive': predicted_positive_count,
        'tp': true_positive,
        'fp': false_positive,
        'fn': false_negative,
        'tn': true_negative,
        'selection_rate': predicted_positive_count / len(true_labels),
        'precision': _safe_rate(true_positive, predicted_positive_count),
        'recall': _safe_rate(true_positive, positive_support),
        'f1': _safe_rate(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        'specificity': _safe_rate(true_negative, negative_support),
        'false_positive_rate': _safe_rate(false_positive, negative_support),
        'false_negative_rate': _safe_rate(false_negative, positive_support),
        'negative_predictive_value': _safe_rate(
            true_negative, true_negative + false_negative
        ),
    }


def _calculate_target_label_metrics(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        target_labels: list[str],
) -> pd.DataFrame:
    """Calculate one-vs.-rest metrics for every target label."""

    true_labels = condition_predictions['true_label']
    predicted_labels = condition_predictions['predicted_label']

    target_label_rows = [
        {
            **condition_metadata,
            'target_label': target_label,
            **_calculate_one_vs_rest_metrics(true_labels, predicted_labels, target_label)
        }
        for target_label in target_labels
    ]

    return pd.DataFrame(target_label_rows)


def _calculate_confusion_matrix(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        target_labels: list[str],
) -> pd.DataFrame:
    """Calculate the multiclass confusion matrix."""

    confusion_counts = pd.crosstab(
        condition_predictions['true_label'],
        condition_predictions['predicted_label'],
    ).reindex(
        index=target_labels,
        columns=target_labels,
        fill_value=0,
    )

    confusion_rows = [
        {
            **condition_metadata,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'count': cast(int, confusion_counts.loc[true_label, predicted_label]),
        }
        for true_label in target_labels
        for predicted_label in target_labels
    ]

    return pd.DataFrame(confusion_rows)


def _calculate_audit_group_metrics(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        target_labels: list[str],
) -> pd.DataFrame:
    """Calculate one-vs.-rest metrics within every audit group."""

    audit_groups: list[tuple[str, pd.DataFrame, float]] = []
    for audit_group, audit_group_predictions in condition_predictions.groupby('audit_group', sort=False):
        audit_group_accuracy = cast(
            float,
            (audit_group_predictions['true_label'] == audit_group_predictions['predicted_label']).mean(),
        )

        audit_groups.append(
            (cast(str, audit_group), audit_group_predictions, audit_group_accuracy)
        )

    audit_group_rows: list[dict[str, Any]] = []

    # Audit-group accuracy=correct_g/N_g. For every target label c within audit
    # group g, use the same one-vs.-rest supports, counts, and rates as above,
    # but calculate them only from rows in g and use N_g for selection rate.
    # _safe_rate again returns NaN when a required denominator is zero.
    for target_label in target_labels:
        for audit_group, audit_group_predictions, audit_group_accuracy in audit_groups:
            audit_group_rows.append(
                {
                    **condition_metadata,
                    'target_label': target_label,
                    'audit_group': audit_group,
                    'audit_group_n': len(audit_group_predictions),
                    'audit_group_accuracy': audit_group_accuracy,
                    **_calculate_one_vs_rest_metrics(
                        audit_group_predictions['true_label'],
                        audit_group_predictions['predicted_label'],
                        target_label,
                    ),
                }
            )

    return pd.DataFrame(audit_group_rows)


def _calculate_target_label_fairness_metrics(
        audit_group_metrics: pd.DataFrame,
        condition_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Compare each target label's rates across audit groups."""

    fairness_rows: list[dict[str, Any]] = []
    grouped_audit_group_metrics = audit_group_metrics.groupby(
        'target_label', sort=False
    )

    for target_label, target_label_audit_group_metrics in grouped_audit_group_metrics:
        selection_rates = target_label_audit_group_metrics['selection_rate']
        precisions = target_label_audit_group_metrics['precision']
        recalls = target_label_audit_group_metrics['recall']
        false_positive_rates = target_label_audit_group_metrics['false_positive_rate']

        equal_opportunity_difference = _rate_range(recalls)
        false_positive_rate_difference = _rate_range(false_positive_rates)

        equalized_odds_difference = (
            max(equal_opportunity_difference, false_positive_rate_difference)
            if not pd.isna(equal_opportunity_difference) and not pd.isna(false_positive_rate_difference)
            else np.nan
        )

        # Across audit groups, demographic-parity difference=range(selection rate),
        # demographic-parity ratio=min(selection rate)/max(selection rate),
        # equal-opportunity difference=range(recall), false-positive-rate
        # difference=range(FPR), equalized-odds difference=max(equal-opportunity
        # difference, FPR difference), and predictive-parity difference=range(precision).
        # A range needs at least two defined audit-group rates; the ratio additionally
        # needs a positive maximum. The coverage columns expose how many rates remain.
        fairness_rows.append(
            {
                **condition_metadata,
                'target_label': target_label,
                'demographic_parity_difference': _rate_range(selection_rates),
                'demographic_parity_ratio': _rate_ratio(selection_rates),
                'equal_opportunity_difference': equal_opportunity_difference,
                'false_positive_rate_difference': false_positive_rate_difference,
                'equalized_odds_difference': equalized_odds_difference,
                'predictive_parity_difference': _rate_range(precisions),
                'n_audit_groups_compared': len(target_label_audit_group_metrics),
                'n_selection_rate_defined_audit_groups': int(selection_rates.notna().sum()),
                'n_recall_defined_audit_groups': int(recalls.notna().sum()),
                'n_false_positive_rate_defined_audit_groups': int(false_positive_rates.notna().sum()),
                'n_precision_defined_audit_groups': int(precisions.notna().sum()),
            }
        )

    return pd.DataFrame(fairness_rows)


def _calculate_overall_fairness_metrics(
        fairness_metrics: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, int]]:
    """Calculate overall fairness metrics and their target-label coverage."""

    # If D_m contains the target labels where fairness metric m_c is defined,
    # mean=sum(m_c)/|D_m| and coverage=|D_m|; min and max use the same set.
    # Difference minima are best and maxima are worst. For demographic-parity
    # ratio, where 1 is best, those interpretations are reversed.
    coverage_column_by_fairness_metric = {
        'demographic_parity_difference': 'n_demographic_parity_defined_target_labels',
        'demographic_parity_ratio': 'n_demographic_parity_ratio_defined_target_labels',
        'equal_opportunity_difference': 'n_equal_opportunity_defined_target_labels',
        'false_positive_rate_difference': 'n_false_positive_rate_defined_target_labels',
        'equalized_odds_difference': 'n_equalized_odds_defined_target_labels',
        'predictive_parity_difference': 'n_predictive_parity_defined_target_labels',
    }

    summary = {
        f'mean_{metric}': float(fairness_metrics[metric].mean())
        for metric in coverage_column_by_fairness_metric
    }
    summary.update({
        f'min_{metric}': float(fairness_metrics[metric].min())
        for metric in coverage_column_by_fairness_metric
    })
    summary.update({
        f'max_{metric}': float(fairness_metrics[metric].max())
        for metric in coverage_column_by_fairness_metric
    })

    coverage = {
        defined_count_column: int(fairness_metrics[metric].notna().sum())
        for metric, defined_count_column in coverage_column_by_fairness_metric.items()
    }

    return summary, coverage


def _calculate_overall_condition_metrics(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        target_label_metrics: pd.DataFrame,
        audit_group_metrics: pd.DataFrame,
        fairness_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate the overall classification-and-fairness result for one condition."""

    true_labels = condition_predictions['true_label']
    predicted_labels = condition_predictions['predicted_label']
    sample_count = len(condition_predictions)
    accuracy = cast(float, (true_labels == predicted_labels).mean())
    target_label_weights = target_label_metrics['positive_support'] / sample_count
    audit_group_accuracies = audit_group_metrics.drop_duplicates('audit_group')['audit_group_accuracy']
    overall_fairness_metrics, fairness_metric_coverage = (
        _calculate_overall_fairness_metrics(fairness_metrics)
    )

    # The two shared columns below store formula-identical metrics once. Macro
    # and weighted averages ignore an undefined target-label rate and weight only
    # the remaining defined rates; defined-target-label counts expose that coverage.
    # Accuracy=sum_c(TP_c)/N, worst-audit-group accuracy=min_g(accuracy_g), and the
    # audit-group accuracy difference=max_g(accuracy_g)-min_g(accuracy_g). MCC and
    # Cohen's kappa use scikit-learn's degenerate-case behavior.
    return {
        **condition_metadata,
        'sample_count': sample_count,
        'n_audit_groups': len(audit_group_accuracies),
        ACCURACY_METRIC_COLUMN: accuracy,
        'macro_precision': target_label_metrics['precision'].mean(),
        MACRO_RECALL_METRIC_COLUMN: target_label_metrics['recall'].mean(),
        'macro_f1': target_label_metrics['f1'].mean(),
        'weighted_precision': _weighted_average(target_label_metrics['precision'], target_label_weights),
        'weighted_f1': _weighted_average(target_label_metrics['f1'], target_label_weights),
        'matthews_correlation_coefficient': matthews_corrcoef(true_labels, predicted_labels),
        'cohen_kappa': cohen_kappa_score(true_labels, predicted_labels),
        'worst_audit_group_accuracy': min(audit_group_accuracies),
        'audit_group_accuracy_difference': _rate_range(audit_group_accuracies),
        **overall_fairness_metrics,
        'n_target_labels': len(target_label_metrics),
        'n_precision_defined_target_labels': int(target_label_metrics['precision'].notna().sum()),
        'n_recall_defined_target_labels': int(target_label_metrics['recall'].notna().sum()),
        'n_f1_defined_target_labels': int(target_label_metrics['f1'].notna().sum()),
        **fairness_metric_coverage,
    }


def calculate_condition_metrics(
        predictions: pd.DataFrame,
        target_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate every metric table for one prediction condition."""

    _validate_metric_inputs(predictions, target_labels)
    condition_metadata = _condition_metadata(predictions)

    target_label_metrics = _calculate_target_label_metrics(predictions, condition_metadata, target_labels)
    confusion_matrix = _calculate_confusion_matrix(predictions, condition_metadata, target_labels)
    audit_group_metrics = _calculate_audit_group_metrics(predictions, condition_metadata, target_labels)
    fairness_metrics = _calculate_target_label_fairness_metrics(audit_group_metrics, condition_metadata)

    result = pd.DataFrame([
        _calculate_overall_condition_metrics(
            predictions,
            condition_metadata,
            target_label_metrics,
            audit_group_metrics,
            fairness_metrics,
        )
    ])

    return result, target_label_metrics, confusion_matrix, audit_group_metrics, fairness_metrics


def rank_results(results: pd.DataFrame, metric: str, direction: str) -> pd.DataFrame:
    """Rank prompt configurations separately within each language model."""

    if direction not in {'maximize', 'minimize'}:
        raise ValueError("direction must be 'maximize' or 'minimize'")

    metric_column = resolve_metric_column(metric)
    if metric_column not in results.columns:
        available = ', '.join(results.select_dtypes(include='number').columns)
        raise ValueError(f'Unknown ranking metric {metric!r}. Numeric result columns: {available}')
    if not pd.api.types.is_numeric_dtype(results[metric_column]):
        raise ValueError(f'Ranking metric {metric!r} must be numeric')

    defined_by_language_model = results.groupby('language_model', sort=False)[metric_column].apply(
        lambda values: values.notna().any()
    )

    undefined_language_models = defined_by_language_model.index[~defined_by_language_model].tolist()
    if undefined_language_models:
        raise ValueError(
            f'Ranking metric {metric!r} is undefined for every condition of language models: {undefined_language_models}'
        )

    ranked = results.sort_values(['language_model', metric_column, 'condition'],
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
        target_labels: list[str],
        ranking_metric: str,
        ranking_direction: str,
        evaluation_split: str,
) -> None:
    """Save the best current-split prompt for every language model."""

    metric_column = resolve_metric_column(ranking_metric)
    sections: list[str] = []

    for best in selected.to_dict('records'):
        resolved_prompt = prompt_templates[best['prompt_name']].format(
            target=display_column_name(best['target']),
            audit_column=display_column_name(best['audit_column']),
            labels=', '.join(target_labels),
        )

        sections.append(
            f'Language model: {best['language_model']}\n'
            f'Selected on {evaluation_split} metric: {ranking_metric} '
            f'({ranking_direction})\n'
            f'{evaluation_split.title()} score: {best[metric_column]}\n'
            f'Retrieval method: {best['retrieval_method']}\n'
            f'Embedding model: {best['embedding_model']}\n'
            f'Examples: {best['example_count']}\n'
            f'Example order: {best['example_order']}\n'
            f'Prompt name: {best['prompt_name']}\n'
            f'Prediction method: {best['prediction_method']}\n\n'
            f'{resolved_prompt}'
        )

    path.write_text('\n\n---\n\n'.join(sections) + '\n', encoding='utf-8')
