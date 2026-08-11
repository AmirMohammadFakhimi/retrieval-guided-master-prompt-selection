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
    'max_demographic_parity_difference',
    'n_demographic_parity_defined_target_labels',
    'mean_equal_opportunity_difference',
    'max_equal_opportunity_difference',
    'n_equal_opportunity_defined_target_labels',
    'mean_false_positive_rate_difference',
    'max_false_positive_rate_difference',
    'n_false_positive_rate_defined_target_labels',
    'mean_equalized_odds_difference',
    'max_equalized_odds_difference',
    'n_equalized_odds_defined_target_labels',
    'mean_predictive_parity_difference',
    'max_predictive_parity_difference',
    'n_predictive_parity_defined_target_labels',
    'mean_demographic_parity_ratio',
    'min_demographic_parity_ratio',
    'n_demographic_parity_ratio_defined_target_labels',
})

FAIRNESS_DIFFERENCE_METRICS = {
    'demographic_parity_difference': 'n_demographic_parity_defined_target_labels',
    'equal_opportunity_difference': 'n_equal_opportunity_defined_target_labels',
    'false_positive_rate_difference': 'n_false_positive_rate_defined_target_labels',
    'equalized_odds_difference': 'n_equalized_odds_defined_target_labels',
    'predictive_parity_difference': 'n_predictive_parity_defined_target_labels',
}


# Metric notation used in the calculations below:
# c is a target label, g is an audit group, K is the number of target labels,
# n_c is the true support of target label c, N is the number of evaluated rows,
# and N_g is the number of evaluated rows in audit group g.
# TP, FP, FN, and TN are one-vs-rest counts for c (and within g for audit-group metrics).
# Precision=PPV=TP/(TP+FP), Recall=TPR=TP/(TP+FN),
# F1=2TP/(2TP+FP+FN), and Specificity=TNR=TN/(TN+FP).
# Whenever defined, FPR=FP/(FP+TN)=1-Specificity and
# FNR=FN/(FN+TP)=1-Recall. NPV=TN/(TN+FN).
# D_m is the set of target labels where metric m is defined. Macro(m) averages m_c
# over D_m; Weighted(m)=sum_{c in D_m}(n_c*m_c)/sum_{c in D_m}(n_c).
# When m is defined for every target label, D_m contains all K target labels.
# For single-label multiclass predictions, sum_c(TP_c)=T and
# sum_c(FP_c)=sum_c(FN_c)=E, where N=T+E. Therefore micro precision,
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


def _calculate_target_label_metrics(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        target_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate one-vs-rest target-label rates and the multiclass confusion matrix."""

    true_labels = condition_predictions['true_label']
    predicted_labels = condition_predictions['predicted_label']
    sample_count = len(condition_predictions)
    target_label_rows: list[dict[str, Any]] = []

    for target_label in target_labels:
        actual_positive = true_labels == target_label
        predicted_positive = predicted_labels == target_label

        true_positive = int((actual_positive & predicted_positive).sum())
        false_positive = int((~actual_positive & predicted_positive).sum())
        false_negative = int((actual_positive & ~predicted_positive).sum())
        true_negative = int((~actual_positive & ~predicted_positive).sum())

        positive_support = true_positive + false_negative
        negative_support = false_positive + true_negative
        predicted_positive_count = true_positive + false_positive

        target_label_rows.append(
            {
                **condition_metadata,
                'target_label': target_label,
                'positive_support': positive_support,
                'negative_support': negative_support,
                'predicted_positive': predicted_positive_count,
                'tp': true_positive,
                'fp': false_positive,
                'fn': false_negative,
                'tn': true_negative,
                'selection_rate': predicted_positive_count / sample_count,
                'precision': _safe_rate(true_positive, predicted_positive_count),
                'recall': _safe_rate(true_positive, positive_support),
                'f1': _safe_rate(
                    2 * true_positive,
                    2 * true_positive + false_positive + false_negative,
                ),
                'specificity': _safe_rate(true_negative, negative_support),
                # Whenever defined, FPR = 1 - specificity.
                'false_positive_rate': _safe_rate(false_positive, negative_support),
                # Whenever defined, FNR = 1 - recall.
                'false_negative_rate': _safe_rate(false_negative, positive_support),
                'negative_predictive_value': _safe_rate(
                    true_negative, true_negative + false_negative
                ),
            }
        )

    confusion_counts = pd.crosstab(true_labels, predicted_labels).reindex(
        index=target_labels,
        columns=target_labels,
        fill_value=0,
    )
    confusion_rows = [
        {
            **condition_metadata,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'count': int(confusion_counts.loc[true_label, predicted_label]),
        }
        for true_label in target_labels
        for predicted_label in target_labels
    ]

    return pd.DataFrame(target_label_rows), pd.DataFrame(confusion_rows)


def _calculate_audit_group_metrics(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        target_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """Calculate rates within audit groups and disparities for every target label."""

    audit_grouped_predictions = list(condition_predictions.groupby('audit_group', sort=False))
    audit_group_accuracy_values = [
        cast(
            float,
            (audit_group_predictions['true_label'] == audit_group_predictions['predicted_label']).mean(),
        )
        for _, audit_group_predictions in audit_grouped_predictions
    ]

    audit_group_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    for target_label in target_labels:
        target_label_audit_group_rows: list[dict[str, Any]] = []
        for (audit_group, audit_group_predictions), audit_group_accuracy in zip(
                audit_grouped_predictions, audit_group_accuracy_values, strict=True
        ):
            true_labels = audit_group_predictions['true_label']
            predicted_labels = audit_group_predictions['predicted_label']

            actual_positive = true_labels == target_label
            predicted_positive = predicted_labels == target_label

            true_positive = int((actual_positive & predicted_positive).sum())
            false_positive = int((~actual_positive & predicted_positive).sum())
            false_negative = int((actual_positive & ~predicted_positive).sum())
            true_negative = int((~actual_positive & ~predicted_positive).sum())

            audit_group_n = len(audit_group_predictions)
            positive_support = true_positive + false_negative
            negative_support = false_positive + true_negative
            predicted_positive_count = true_positive + false_positive

            audit_group_metric_row = {
                **condition_metadata,
                'target_label': target_label,
                'audit_group': audit_group,
                'audit_group_n': audit_group_n,
                'audit_group_accuracy': audit_group_accuracy,
                'positive_support': positive_support,
                'negative_support': negative_support,
                'predicted_positive': predicted_positive_count,
                'tp': true_positive,
                'fp': false_positive,
                'fn': false_negative,
                'tn': true_negative,
                'selection_rate': predicted_positive_count / audit_group_n,
                'precision': _safe_rate(true_positive, predicted_positive_count),
                'recall': _safe_rate(true_positive, positive_support),
                'f1': _safe_rate(
                    2 * true_positive,
                    2 * true_positive + false_positive + false_negative,
                ),
                'specificity': _safe_rate(true_negative, negative_support),
                # Whenever defined, FPR = 1 - specificity.
                'false_positive_rate': _safe_rate(false_positive, negative_support),
                # Whenever defined, FNR = 1 - recall.
                'false_negative_rate': _safe_rate(false_negative, positive_support),
                'negative_predictive_value': _safe_rate(
                    true_negative, true_negative + false_negative
                ),
            }
            target_label_audit_group_rows.append(audit_group_metric_row)
            audit_group_rows.append(audit_group_metric_row)

        target_label_audit_group_frame = pd.DataFrame(target_label_audit_group_rows)
        selection_rates = target_label_audit_group_frame['selection_rate'].tolist()
        recalls = target_label_audit_group_frame['recall'].tolist()
        false_positive_rates = target_label_audit_group_frame['false_positive_rate'].tolist()
        precisions = target_label_audit_group_frame['precision'].tolist()
        equal_opportunity_difference = _rate_range(recalls)
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
        # is max(PPV_g)-min(PPV_g). At least two defined audit-group rates are required.
        fairness_rows.append(
            {
                **condition_metadata,
                'target_label': target_label,
                'n_audit_groups_compared': len(audit_grouped_predictions),
                'n_selection_rate_defined_audit_groups': int(
                    target_label_audit_group_frame['selection_rate'].notna().sum()
                ),
                'n_recall_defined_audit_groups': int(
                    target_label_audit_group_frame['recall'].notna().sum()
                ),
                'n_false_positive_rate_defined_audit_groups': int(
                    target_label_audit_group_frame['false_positive_rate'].notna().sum()
                ),
                'n_precision_defined_audit_groups': int(
                    target_label_audit_group_frame['precision'].notna().sum()
                ),
                'demographic_parity_difference': _rate_range(selection_rates),
                'demographic_parity_ratio': _rate_ratio(selection_rates),
                'equal_opportunity_difference': equal_opportunity_difference,
                'false_positive_rate_difference': false_positive_rate_difference,
                'equalized_odds_difference': equalized_odds_difference,
                'predictive_parity_difference': _rate_range(
                    precisions
                ),
            }
        )

    return pd.DataFrame(audit_group_rows), pd.DataFrame(fairness_rows), audit_group_accuracy_values


def _summarize_fairness(fairness_frame: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate disparities while reporting how many target labels define them."""

    # If D_d contains the target labels where disparity d_c is defined, the reported
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
    summary['n_demographic_parity_ratio_defined_target_labels'] = int(
        ratio.notna().sum()
    )
    return summary


def _summarize_condition(
        condition_predictions: pd.DataFrame,
        condition_metadata: dict[str, Any],
        target_label_frame: pd.DataFrame,
        fairness_frame: pd.DataFrame,
        audit_group_accuracy_values: list[float],
) -> dict[str, Any]:
    """Build one overall classification-and-fairness result row."""

    true_labels = condition_predictions['true_label']
    predicted_labels = condition_predictions['predicted_label']
    sample_count = len(condition_predictions)
    accuracy = float((true_labels == predicted_labels).mean())
    target_label_weights = target_label_frame['positive_support'] / sample_count

    # The two shared columns below store formula-identical metrics once. Macro
    # and weighted averages ignore an undefined target-label rate and weight only
    # the remaining defined rates; defined-target-label counts expose that coverage.
    # Accuracy=sum_c(TP_c)/N, worst-audit-group accuracy=min_g(accuracy_g), and the
    # audit-group accuracy difference=max_g(accuracy_g)-min_g(accuracy_g).
    return {
        **condition_metadata,
        'sample_count': sample_count,
        'n_target_labels': len(target_label_frame),
        'n_audit_groups': len(audit_group_accuracy_values),
        ACCURACY_METRIC_COLUMN: accuracy,
        'macro_precision': target_label_frame['precision'].mean(),
        MACRO_RECALL_METRIC_COLUMN: target_label_frame['recall'].mean(),
        'macro_f1': target_label_frame['f1'].mean(),
        'weighted_precision': _weighted_average(
            target_label_frame['precision'], target_label_weights
        ),
        'weighted_f1': _weighted_average(target_label_frame['f1'], target_label_weights),
        'n_precision_defined_target_labels': int(target_label_frame['precision'].notna().sum()),
        'n_recall_defined_target_labels': int(target_label_frame['recall'].notna().sum()),
        'n_f1_defined_target_labels': int(target_label_frame['f1'].notna().sum()),
        'matthews_correlation_coefficient': matthews_corrcoef(
            true_labels, predicted_labels
        ),
        'cohen_kappa': cohen_kappa_score(true_labels, predicted_labels),
        'worst_audit_group_accuracy': min(audit_group_accuracy_values),
        'audit_group_accuracy_difference': _rate_range(audit_group_accuracy_values),
        **_summarize_fairness(fairness_frame),
    }


def calculate_condition_metrics(
        predictions: pd.DataFrame,
        target_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate every metric table for one prediction condition."""

    _validate_metric_inputs(predictions, target_labels)
    metadata = _condition_metadata(predictions)
    target_label_frame, confusion_frame = _calculate_target_label_metrics(
        predictions,
        metadata,
        target_labels,
    )
    audit_group_frame, fairness_frame, audit_group_accuracy_values = _calculate_audit_group_metrics(
        predictions,
        metadata,
        target_labels,
    )

    result = pd.DataFrame([
        _summarize_condition(
            predictions,
            metadata,
            target_label_frame,
            fairness_frame,
            audit_group_accuracy_values,
        )
    ])

    return (
        result,
        target_label_frame,
        confusion_frame,
        audit_group_frame,
        fairness_frame,
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
            f'Prompt name: {best['prompt_name']}\n\n'
            f'{resolved_prompt}'
        )
    path.write_text('\n\n---\n\n'.join(sections) + '\n', encoding='utf-8')
