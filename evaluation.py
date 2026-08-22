import json
from itertools import combinations
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

FAIRNESS_METRIC_COLUMNS = frozenset({
    'audit_group_accuracy_difference',
    'mean_demographic_parity_difference',
    'min_demographic_parity_difference',
    'max_demographic_parity_difference',
    'mean_equal_opportunity_difference',
    'min_equal_opportunity_difference',
    'max_equal_opportunity_difference',
    'mean_false_positive_rate_difference',
    'min_false_positive_rate_difference',
    'max_false_positive_rate_difference',
    'mean_equalized_odds_difference',
    'min_equalized_odds_difference',
    'max_equalized_odds_difference',
    'mean_predictive_parity_difference',
    'min_predictive_parity_difference',
    'max_predictive_parity_difference',
    'mean_demographic_parity_ratio',
    'min_demographic_parity_ratio',
    'max_demographic_parity_ratio',
})

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

EXPERIMENT_FACTOR_COLUMNS = (
    'language_model',
    'prompt_name',
    'retrieval_method',
    'embedding_model',
    'example_count',
    'example_order',
)
FACTOR_CONTRAST_CONTEXT_COLUMNS = (
    'evaluation_split',
    'target',
    'audit_column',
    'prediction_method',
)
FACTOR_CONTRAST_DETAIL_COLUMNS = (
    'contrast_type',
    'factor',
    'from_factor_value',
    'to_factor_value',
    'evaluation_split',
    'target',
    'audit_column',
    'prediction_method',
    'language_model',
    'fixed_context',
    'metric',
    'direction',
    'from_metric_value',
    'to_metric_value',
    'delta',
    'improvement',
    'outcome',
    'from_condition_count',
    'to_condition_count',
)
FACTOR_CONTRAST_SUMMARY_COLUMNS = (
    'contrast_type',
    'aggregation_scope',
    'scope_language_model',
    'factor',
    'from_factor_value',
    'to_factor_value',
    'metric',
    'direction',
    'n_total_pairs',
    'n_defined_pairs',
    'mean_from_metric_value',
    'mean_to_metric_value',
    'mean_delta',
    'std_delta',
    'n_improved',
    'n_tied',
    'n_worsened',
    'improvement_rate',
)


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


def _factor_levels(config: dict[str, Any]) -> dict[str, list[Any]]:
    """Return experiment-factor levels in their configured order."""

    retrieval = config['retrieval']
    return {
        'language_model': [entry['id'] for entry in config['inference']['language_models']],
        'prompt_name': list(config['prompt_templates']),
        'retrieval_method': list(retrieval['methods']),
        'embedding_model': [entry['id'] for entry in retrieval['embedding_models']],
        'example_count': [count for count in retrieval['example_counts'] if count > 0],
        'example_order': list(retrieval['example_orders']),
    }


def _factor_contrast_metric_columns(results: pd.DataFrame) -> list[str]:
    """Return ordered condition-level rates and scores eligible for contrasts."""

    excluded = {
        'example_count',
        'sample_count',
        'n_target_labels',
        'n_audit_groups',
    }
    return [
        column
        for column in results.columns
        if column in RESULT_NUMERIC_COLUMNS
        and column not in excluded
        and not column.startswith('n_')
    ]


def _metric_direction(metric: str) -> str:
    """Return the preferred direction for one condition-level metric."""

    return 'minimize' if metric.endswith('_difference') else 'maximize'


def _contrast_outcome(improvement: float) -> str:
    """Describe a direction-adjusted metric change."""

    if pd.isna(improvement):
        return 'undefined'
    if improvement > 0:
        return 'improved'
    if improvement < 0:
        return 'worsened'
    return 'tied'


def _fixed_context_text(row: pd.Series, columns: list[str]) -> str:
    """Serialize the held-constant columns for one matched comparison."""

    return json.dumps(
        {column: row[column] for column in columns},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _strict_factor_contrast_details(
        results: pd.DataFrame,
        factor_levels: dict[str, list[Any]],
        metric_columns: list[str],
) -> list[dict[str, Any]]:
    """Return metric deltas for pairs that vary exactly one factor."""

    detail_rows: list[dict[str, Any]] = []
    factor_columns = list(EXPERIMENT_FACTOR_COLUMNS)
    context_columns = list(FACTOR_CONTRAST_CONTEXT_COLUMNS)

    for factor in factor_columns:
        levels = factor_levels[factor]
        if len(levels) < 2:
            continue

        eligible = results.loc[results[factor].isin(levels)].copy()
        fixed_columns = context_columns + [
            column for column in factor_columns if column != factor
        ]

        for from_level, to_level in combinations(levels, 2):
            from_rows = eligible.loc[eligible[factor].eq(from_level)]
            to_rows = eligible.loc[eligible[factor].eq(to_level)]

            if from_rows.duplicated(fixed_columns).any() or to_rows.duplicated(fixed_columns).any():
                raise ValueError(
                    f'Factor contrast {factor!r} has duplicate held-constant contexts'
                )

            matched = from_rows.merge(
                to_rows,
                on=fixed_columns,
                how='inner',
                suffixes=('_from', '_to'),
                validate='one_to_one',
            )
            if len(matched) != len(from_rows) or len(matched) != len(to_rows):
                raise ValueError(
                    f'Factor contrast {factor!r} has incomplete matched contexts for '
                    f'{from_level!r} and {to_level!r}'
                )

            for _, matched_row in matched.iterrows():
                fixed_context = _fixed_context_text(matched_row, fixed_columns)
                language_model = (
                    'not_applicable'
                    if factor == 'language_model'
                    else matched_row['language_model']
                )

                for metric in metric_columns:
                    direction = _metric_direction(metric)
                    from_metric_value = matched_row[f'{metric}_from']
                    to_metric_value = matched_row[f'{metric}_to']
                    delta = (
                        to_metric_value - from_metric_value
                        if pd.notna(from_metric_value) and pd.notna(to_metric_value)
                        else np.nan
                    )
                    improvement = delta if direction == 'maximize' else -delta
                    detail_rows.append({
                        'contrast_type': 'single_factor',
                        'factor': factor,
                        'from_factor_value': from_level,
                        'to_factor_value': to_level,
                        'evaluation_split': matched_row['evaluation_split'],
                        'target': matched_row['target'],
                        'audit_column': matched_row['audit_column'],
                        'prediction_method': matched_row['prediction_method'],
                        'language_model': language_model,
                        'fixed_context': fixed_context,
                        'metric': metric,
                        'direction': direction,
                        'from_metric_value': from_metric_value,
                        'to_metric_value': to_metric_value,
                        'delta': delta,
                        'improvement': improvement,
                        'outcome': _contrast_outcome(improvement),
                        'from_condition_count': 1,
                        'to_condition_count': 1,
                    })

    return detail_rows


def _zero_shot_contrast_details(
        results: pd.DataFrame,
        positive_example_counts: list[int],
        metric_columns: list[str],
        retrieval_configuration_count: int,
) -> list[dict[str, Any]]:
    """Compare zero-shot with the mean configured retrieval condition at each count."""

    if not positive_example_counts or not results['example_count'].eq(0).any():
        return []

    grouping_columns = [
        *FACTOR_CONTRAST_CONTEXT_COLUMNS,
        'language_model',
        'prompt_name',
    ]
    zero_shot = results.loc[results['example_count'].eq(0)]
    if zero_shot.duplicated(grouping_columns).any():
        raise ValueError('Zero-shot factor contrasts require one row per language model and prompt')

    detail_rows: list[dict[str, Any]] = []
    for example_count in positive_example_counts:
        few_shot = results.loc[results['example_count'].eq(example_count)]
        observed_configuration_counts = few_shot.groupby(
            grouping_columns,
            dropna=False,
        ).size()
        if (
                observed_configuration_counts.empty
                or not observed_configuration_counts.eq(retrieval_configuration_count).all()
        ):
            raise ValueError(
                f'Zero-shot comparison for example_count={example_count} requires '
                f'{retrieval_configuration_count} retrieval conditions per language model and prompt'
            )

        averaged_few_shot = few_shot.groupby(
            grouping_columns,
            as_index=False,
            dropna=False,
            sort=False,
        )[metric_columns].agg(lambda values: values.mean(skipna=False))
        matched = zero_shot.merge(
            averaged_few_shot,
            on=grouping_columns,
            how='inner',
            suffixes=('_from', '_to'),
            validate='one_to_one',
        )
        if len(matched) != len(zero_shot) or len(matched) != len(averaged_few_shot):
            raise ValueError(
                f'Zero-shot comparison for example_count={example_count} has incomplete contexts'
            )

        for _, matched_row in matched.iterrows():
            fixed_context = _fixed_context_text(matched_row, grouping_columns)
            for metric in metric_columns:
                direction = _metric_direction(metric)
                from_metric_value = matched_row[f'{metric}_from']
                to_metric_value = matched_row[f'{metric}_to']
                delta = (
                    to_metric_value - from_metric_value
                    if pd.notna(from_metric_value) and pd.notna(to_metric_value)
                    else np.nan
                )
                improvement = delta if direction == 'maximize' else -delta
                detail_rows.append({
                    'contrast_type': 'zero_shot_to_few_shot',
                    'factor': 'example_count',
                    'from_factor_value': 0,
                    'to_factor_value': example_count,
                    'evaluation_split': matched_row['evaluation_split'],
                    'target': matched_row['target'],
                    'audit_column': matched_row['audit_column'],
                    'prediction_method': matched_row['prediction_method'],
                    'language_model': matched_row['language_model'],
                    'fixed_context': fixed_context,
                    'metric': metric,
                    'direction': direction,
                    'from_metric_value': from_metric_value,
                    'to_metric_value': to_metric_value,
                    'delta': delta,
                    'improvement': improvement,
                    'outcome': _contrast_outcome(improvement),
                    'from_condition_count': 1,
                    'to_condition_count': retrieval_configuration_count,
                })

    return detail_rows


def _summarize_factor_contrasts(details: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched metric deltas overall and within language model."""

    grouping_columns = [
        'contrast_type',
        'factor',
        'from_factor_value',
        'to_factor_value',
        'metric',
        'direction',
    ]

    def summarize(table: pd.DataFrame, scope: str, language_model: str) -> dict[str, Any]:
        defined = table.loc[table['delta'].notna()]
        n_defined = len(defined)
        return {
            'contrast_type': table['contrast_type'].iloc[0],
            'aggregation_scope': scope,
            'scope_language_model': language_model,
            'factor': table['factor'].iloc[0],
            'from_factor_value': table['from_factor_value'].iloc[0],
            'to_factor_value': table['to_factor_value'].iloc[0],
            'metric': table['metric'].iloc[0],
            'direction': table['direction'].iloc[0],
            'n_total_pairs': len(table),
            'n_defined_pairs': n_defined,
            'mean_from_metric_value': defined['from_metric_value'].mean(),
            'mean_to_metric_value': defined['to_metric_value'].mean(),
            'mean_delta': defined['delta'].mean(),
            'std_delta': defined['delta'].std(),
            'n_improved': int(defined['outcome'].eq('improved').sum()),
            'n_tied': int(defined['outcome'].eq('tied').sum()),
            'n_worsened': int(defined['outcome'].eq('worsened').sum()),
            'improvement_rate': (
                float(defined['outcome'].eq('improved').mean())
                if n_defined
                else np.nan
            ),
        }

    summary_rows: list[dict[str, Any]] = []
    for _, group in details.groupby(grouping_columns, dropna=False, sort=False):
        summary_rows.append(summarize(group, 'overall', 'not_applicable'))

        if group['factor'].iloc[0] == 'language_model':
            continue
        for language_model, language_model_group in group.groupby(
                'language_model',
                dropna=False,
                sort=False,
        ):
            summary_rows.append(summarize(
                language_model_group,
                'language_model',
                str(language_model),
            ))

    return pd.DataFrame(summary_rows, columns=FACTOR_CONTRAST_SUMMARY_COLUMNS)


def calculate_factor_contrasts(
        results: pd.DataFrame,
        config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return matched one-factor details and descriptive aggregate summaries."""

    required_columns = {
        *FACTOR_CONTRAST_CONTEXT_COLUMNS,
        *EXPERIMENT_FACTOR_COLUMNS,
        'condition',
    }
    missing_columns = sorted(required_columns - set(results.columns))
    if missing_columns:
        raise ValueError(f'results is missing factor-contrast columns: {missing_columns}')
    if results.empty:
        raise ValueError('results must contain at least one condition')

    condition_key = [
        *FACTOR_CONTRAST_CONTEXT_COLUMNS,
        *EXPERIMENT_FACTOR_COLUMNS,
    ]
    if results.duplicated(condition_key).any():
        raise ValueError('results contains duplicate experiment-factor combinations')

    metric_columns = _factor_contrast_metric_columns(results)
    if not metric_columns:
        raise ValueError('results contains no eligible factor-contrast metrics')

    levels = _factor_levels(config)
    detail_rows = _strict_factor_contrast_details(results, levels, metric_columns)
    retrieval_configuration_count = (
        len(levels['retrieval_method'])
        * len(levels['embedding_model'])
        * len(levels['example_order'])
    )
    detail_rows.extend(_zero_shot_contrast_details(
        results,
        levels['example_count'],
        metric_columns,
        retrieval_configuration_count,
    ))

    details = pd.DataFrame(detail_rows, columns=FACTOR_CONTRAST_DETAIL_COLUMNS)
    if details.empty:
        return details, pd.DataFrame(columns=FACTOR_CONTRAST_SUMMARY_COLUMNS)
    return details, _summarize_factor_contrasts(details)


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


def rank_results(
        results: pd.DataFrame,
        quality_metric: str,
        quality_direction: str,
        fairness_metric: str,
        fairness_direction: str,
) -> pd.DataFrame:
    """Dense-rank configured quality and fairness metrics within each model."""

    for name, direction in (
            ('quality_direction', quality_direction),
            ('fairness_direction', fairness_direction),
    ):
        if direction not in {'maximize', 'minimize'}:
            raise ValueError(f"{name} must be 'maximize' or 'minimize'")

    metric_settings = (
        ('quality_metric', quality_metric),
        ('fairness_metric', fairness_metric),
    )
    metric_columns: dict[str, str] = {}
    for setting, metric in metric_settings:
        metric_column = resolve_metric_column(metric)
        if metric_column not in results.columns:
            available = ', '.join(results.select_dtypes(include='number').columns)
            raise ValueError(
                f'Unknown {setting} {metric!r}. Numeric result columns: {available}'
            )
        if not pd.api.types.is_numeric_dtype(results[metric_column]):
            raise ValueError(f'{setting} {metric!r} must be numeric')
        metric_columns[setting] = metric_column

    quality_metric_column = metric_columns['quality_metric']
    fairness_metric_column = metric_columns['fairness_metric']
    defined_by_language_model = results.groupby(
        'language_model',
        sort=False,
    )[quality_metric_column].apply(
        lambda values: values.notna().any()
    )

    undefined_language_models = defined_by_language_model.index[~defined_by_language_model].tolist()
    if undefined_language_models:
        raise ValueError(
            f'Quality metric {quality_metric!r} is undefined for every condition '
            f'of language models: {undefined_language_models}'
        )

    ranked = results.sort_values(['language_model', quality_metric_column, 'condition'],
        ascending=[True, quality_direction == 'minimize', True],
        na_position='last',
        kind='stable',
    ).reset_index(drop=True)
    ranked.insert(
        0,
        'quality_rank',
        ranked.groupby('language_model', sort=False)[quality_metric_column]
        .rank(
            method='dense',
            ascending=quality_direction == 'minimize',
            na_option='keep',
        )
        .astype('Int64'),
    )
    ranked.insert(
        1,
        'is_quality_best',
        ranked['quality_rank'].eq(1).fillna(False).astype(bool),
    )
    ranked.insert(
        2,
        'fairness_rank',
        ranked.groupby('language_model', sort=False)[fairness_metric_column]
        .rank(
            method='dense',
            ascending=fairness_direction == 'minimize',
            na_option='keep',
        )
        .astype('Int64'),
    )
    ranked.insert(
        3,
        'is_fairness_best',
        ranked['fairness_rank'].eq(1).fillna(False).astype(bool),
    )

    return ranked


def write_selected_prompts(
        path: Path,
        results: pd.DataFrame,
        language_model_ids: list[str],
        prompt_templates: dict[str, Any],
        target_labels: list[str],
        selection_name: str,
        selection_metric: str,
        selection_direction: str,
        evaluation_split: str,
) -> None:
    """Save every tied selected prompt and identify models without a winner."""

    if selection_name not in {'quality', 'fairness'}:
        raise ValueError("selection_name must be 'quality' or 'fairness'")

    metric_column = resolve_metric_column(selection_metric)
    winner_column = f'is_{selection_name}_best'
    sections: list[str] = []

    for language_model_id in language_model_ids:
        model_selected = results.loc[
            results['language_model'].eq(language_model_id)
            & results[winner_column].astype(bool)
        ].sort_values('condition', kind='stable')
        if model_selected.empty:
            sections.append(
                f'Language model: {language_model_id}\n'
                f'No defined {selection_name} winner for '
                f'{selection_metric} ({selection_direction}).'
            )
            continue

        for selected in model_selected.to_dict('records'):
            resolved_prompt = prompt_templates[selected['prompt_name']].format(
                target=display_column_name(selected['target']),
                audit_column=display_column_name(selected['audit_column']),
                labels=', '.join(target_labels),
            )

            sections.append(
                f'Language model: {selected['language_model']}\n'
                f'Selected on {evaluation_split} {selection_name} metric: '
                f'{selection_metric} ({selection_direction})\n'
                f'{evaluation_split.title()} score: {selected[metric_column]}\n'
                f'Retrieval method: {selected['retrieval_method']}\n'
                f'Embedding model: {selected['embedding_model']}\n'
                f'Examples: {selected['example_count']}\n'
                f'Example order: {selected['example_order']}\n'
                f'Prompt name: {selected['prompt_name']}\n'
                f'Prediction method: {selected['prediction_method']}\n\n'
                f'{resolved_prompt}'
            )

    path.write_text('\n\n---\n\n'.join(sections) + '\n', encoding='utf-8')
