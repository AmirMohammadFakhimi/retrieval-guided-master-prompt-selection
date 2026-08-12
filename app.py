"""Teaching-oriented Gradio interface for the single-split experiment."""

from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

import evaluation
from pipeline import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / 'config.yaml'
TABLE_PREVIEW_ROWS = 500


def _preview(table: pd.DataFrame) -> pd.DataFrame:
    """Keep the UI responsive while preserving complete CSV artifacts."""

    return table.head(TABLE_PREVIEW_ROWS)


def run_from_yaml(config_text: str):
    """Run the edited YAML and return split-aware teaching views."""

    try:
        config = yaml.safe_load(config_text)
        if not isinstance(config, dict):
            raise ValueError('The configuration must be a YAML mapping')

        output = run_experiment(config, PROJECT_ROOT, print)
        evaluation_split = output['evaluation_split']
        results = output['results'].copy()
        selected = results.loc[results['is_best'].astype(bool)].copy()
        language_model_ids = [
            entry['id'] for entry in config['inference']['language_models']
        ]
        if (
                len(selected) != len(language_model_ids)
                or selected['language_model'].nunique() != len(language_model_ids)
                or set(selected['language_model']) != set(language_model_ids)
        ):
            raise ValueError(f'Expected exactly one best {evaluation_split} condition per language model')

        configured_metric = config['defaults']['ranking_metric']
        metric_column = evaluation.resolve_metric_column(configured_metric)
        selection_lines = []
        for language_model_id in language_model_ids:
            row = selected.loc[
                selected['language_model'].eq(language_model_id)
            ].iloc[0]
            selection_lines.append(f'- **{language_model_id}**: rank 1 with `{configured_metric}={row[metric_column]}`')

        prediction_columns = [
            'evaluation_split',
            'query_id',
            'true_label',
            'audit_group',
            'predicted_label',
            'condition',
            'retrieval_method',
            'embedding_model',
            'language_model',
            'example_count',
            'example_order',
            'prompt_name',
            'label_scores',
        ]
        selected_predictions = output['predictions'].loc[
            output['predictions']['condition'].isin(selected['condition']),
            prediction_columns,
        ]
        plot_gallery = [
            (str(path), name.replace('_', ' ').title())
            for name, path in output['plots'].items()
        ]
        status = '\n'.join([
            f'Finished the **{evaluation_split}** workflow.  ',
            f'Complete artifacts: `{output["run_dir"]}`  ',
            f'Ranking metric: `{configured_metric}` '
            f'(`{config["defaults"]["ranking_direction"]}`)  ',
            '',
            *selection_lines,
            '',
            f'Best-prompt report: `{output["best_prompts"]}`  ',
            f'UI tables show at most {TABLE_PREVIEW_ROWS} rows; CSV files are complete.',
        ])

        return (
            status,
            selected,
            _preview(results),
            _preview(output['target_label_metrics']),
            _preview(output['fairness_metrics']),
            _preview(output['audit_group_metrics']),
            _preview(output['confusion_matrix']),
            _preview(selected_predictions),
            _preview(output['source_dataset_counts']),
            _preview(output['run_dataset_counts']),
            plot_gallery,
        )
    except Exception as exc:
        return (f'Error: {type(exc).__name__}: {exc}',) + (None,) * 10


def build_app():
    """Construct a detailed interface that follows the actual pipeline."""

    with gr.Blocks(title='Retrieval-guided master-prompt selection') as app:
        gr.Markdown(
            """
# Retrieval-guided master-prompt selection

This interface runs one complete **validation or test** experiment from the
YAML below. The selected split is used for every condition, metric, ranking,
plot, and best-prompt report. Loading the other source split for descriptive
counts never sends it to an encoder or language model.
"""
        )

        with gr.Accordion('1. Experiment flow and split contract', open=True):
            gr.Markdown(
                """
1. Load and normalize profession-filtered `train`, `validation` (source `dev`),
   and `test` rows.
2. Cap the retrieval train pool, then balance only the configured evaluation
   split by profession and gender.
3. Embed the selected train and evaluation rows.
4. Load one language model once; for each of its conditions, predict and
   immediately calculate metrics.
5. Rank conditions inside that language model and mark exactly one `is_best`.
6. Release the model, continue to the next model, then write complete tables,
   plots, and best prompts.

There is no automatic validation-to-test transition. Switching to `test` is a
deliberate configuration change and runs the same complete condition grid.
"""
            )

        with gr.Accordion('2. Configuration values and runtime controls'):
            gr.Markdown(
                """
- `target`: `profession` or `gender`; this changes the held-out label and audit
  column, so use separate runs.
- `evaluation_split`: `validation` or `test`.
- `professions`: `all` or a unique list of at least two supported professions.
- `train_size`: `all` or a positive integer. This caps retrieval rows only.
- `evaluation_per_profession_gender`: a positive integer, or `max_balanced` to
  use the smallest available cell in the selected split. The flow prints the
  resolved integer. Evaluation rows equal **profession count × 2 genders × the
  resolved value**.
- Retrieval methods: `semantic`, `balanced_semantic`. Example order:
  `as_retrieved`, `reverse`, or `shuffle`.
- Embedding dtype: `float32`, `float16`, or `bfloat16`; language-model dtype may
  additionally be `auto`. Device: `auto`, `cuda`, `mps`, or `cpu`.
- Prompt templates support exactly `{target}`, `{audit_column}`, and `{labels}`.
- `ranking_metric` must name a numeric results column or a documented alias;
  choose `maximize` for quality/ratios and `minimize` for error/differences.

Condition count per language model is retrieval methods × embedding models ×
example counts × example orders × prompt templates. Total prediction calls
also multiply by selected evaluation rows. Model context and embedding sequence
limits fail explicitly rather than silently truncating inputs.
"""
            )

        with gr.Accordion('3. Methodology and metric formulas'):
            gr.Markdown(
                r"""
`semantic` uses the nearest biographies. `balanced_semantic` takes the nearest
currently feasible rows while balancing profession, gender, and their joint
cells. The language model receives the master prompt and allowed-label
instruction, retrieved user/assistant demonstrations, then the evaluation
biography. It chooses the allowed label with the largest mean conditional token
log-probability:

$$
score(c)=\frac{1}{|T_c|}\sum_j\log P(t_j\mid prompt,t_{1:j-1}),\qquad
\hat c=\arg\max_c score(c)
$$

### Metric reference

#### 1. Notation, counts, and supports

For one prediction condition, $N$ is `sample_count`, $K$ is `n_target_labels`,
and $G$ is `n_audit_groups`. Row $i$ has true target label $y_i$, predicted
target label $\hat y_i$, and audit group $a_i$. Target label $c$ is evaluated
one-vs-rest: $c$ is positive and every other target label is negative. Audit
group $g$ contains $N_g$ rows, stored as `audit_group_n`.

$$
\begin{aligned}
TP_c&=\sum_i\mathbf{1}[y_i=c\land\hat y_i=c], &
FP_c&=\sum_i\mathbf{1}[y_i\ne c\land\hat y_i=c],\\
FN_c&=\sum_i\mathbf{1}[y_i=c\land\hat y_i\ne c], &
TN_c&=\sum_i\mathbf{1}[y_i\ne c\land\hat y_i\ne c].
\end{aligned}
$$

These four counts are stored as `tp`, `fp`, `fn`, and `tn`. The remaining
detailed output columns are:

- `positive_support`: $n_c=TP_c+FN_c$;
- `negative_support`: $FP_c+TN_c=N-n_c$;
- `predicted_positive`: $TP_c+FP_c$;
- `selection_rate`: $SR_c=(TP_c+FP_c)/N$.

The audit-group table applies exactly the same definitions after restricting
the sums to $a_i=g$; its counts use the subscript $(c,g)$ and its selection-rate
denominator is $N_g$. The long-form confusion-matrix output stores `count` as

$$
C_{r,s}=\sum_i\mathbf{1}[y_i=r\land\hat y_i=s]
$$

for every configured true-label row $r$ and predicted-label column $s$.

#### 2. Target-label and audit-group rates

For each target label, and identically within each audit group:

- `precision`: $PPV_c=TP_c/(TP_c+FP_c)$;
- `recall`: $TPR_c=TP_c/(TP_c+FN_c)$;
- `f1`: $F1_c=2TP_c/(2TP_c+FP_c+FN_c)$;
- `specificity`: $TNR_c=TN_c/(TN_c+FP_c)$;
- `false_positive_rate`: $FPR_c=FP_c/(FP_c+TN_c)$;
- `false_negative_rate`: $FNR_c=FN_c/(FN_c+TP_c)$;
- `negative_predictive_value`: $NPV_c=TN_c/(TN_c+FN_c)$.

`audit_group_accuracy` is the multiclass accuracy repeated for each target-label
row of the same audit group:

$$
Accuracy_g=\frac{\sum_{i:a_i=g}\mathbf{1}[y_i=\hat y_i]}{N_g}.
$$

Whenever their denominators are nonzero, $FPR=1-Specificity$ and
$FNR=1-Recall$. Higher selection rate is not inherently better or worse;
precision, recall, F1, specificity, NPV, and audit-group accuracy are better
when higher, while FPR and FNR are better when lower.

The implementation returns `NaN` for a rate whose required denominator is
zero. Macro and weighted aggregates omit undefined rates; a coverage output
reports how many values remained. Selection rate and audit-group accuracy are
defined because validated conditions and observed audit groups are nonempty.

#### 3. Overall quality and agreement

For any target-label rate $m_c$, let $D_m$ contain the target labels where it is
defined, and let $D_m^+$ additionally require positive true support $n_c>0$:

$$
\begin{aligned}
Accuracy&=\frac{\sum_cTP_c}{N},\\
Macro(m)&=\frac{1}{|D_m|}\sum_{c\in D_m}m_c,\\
Weighted(m)&=\frac{\sum_{c\in D_m^+}n_cm_c}
{\sum_{c\in D_m^+}n_c}.
\end{aligned}
$$

These formulas map to `macro_precision`, `macro_recall / balanced_accuracy`,
`macro_f1`, `weighted_precision`, and `weighted_f1`. The defined-label coverage
outputs are
`n_precision_defined_target_labels` $=|D_{Precision}|$,
`n_recall_defined_target_labels` $=|D_{Recall}|$, and
`n_f1_defined_target_labels` $=|D_{F1}|$. An aggregate is `NaN` when its defined
set is empty; a weighted aggregate also needs at least one positive weight.

In this single-label multiclass task, let $T=\sum_cTP_c$ be the number of
correct rows and $E=\sum_cFP_c=\sum_cFN_c$ the number of errors. Since
$N=T+E$:

$$
MicroPrecision=MicroRecall=MicroF1=Accuracy=\frac{T}{N},
$$

$$
WeightedRecall=\frac{1}{N}\sum_{c:n_c>0}n_c\frac{TP_c}{n_c}
=Accuracy,\qquad
BalancedAccuracy=\frac{1}{|D_{Recall}|}\sum_{c\in D_{Recall}}Recall_c
=MacroRecall.
$$

The results store those equality families once under
`accuracy / micro_precision / micro_recall / micro_f1 / weighted_recall` and
`macro_recall / balanced_accuracy`. Each individual standard name is still a
valid `ranking_metric` alias.

For multiclass agreement, let $C$ be the confusion matrix,
$s=\sum_{r,k}C_{r,k}=N$, $q=\operatorname{trace}(C)$,
$p_k=\sum_rC_{r,k}$ be predicted totals, and $t_k=\sum_rC_{k,r}$ be true totals:

The `matthews_correlation_coefficient` output is

$$
MCC=\frac{qs-\sum_kp_kt_k}
{\sqrt{(s^2-\sum_kp_k^2)(s^2-\sum_kt_k^2)}}.
$$

With observed agreement $p_o=Accuracy$ and chance-expected agreement
$p_e=\sum_k(t_k/s)(p_k/s)$:

The `cohen_kappa` output is

$$
\kappa=\frac{p_o-p_e}{1-p_e}.
$$

Higher accuracy, MCC, and kappa are better; one means perfect agreement. The
implementation uses scikit-learn's degenerate-case conventions: MCC is `0.0`
when its denominator is zero, while kappa is `NaN` when $1-p_e=0$.

#### 4. Target-label fairness across audit groups

For target label $c$ and rate $m$, let $G_{m,c}$ contain the audit groups where
$m_{c,g}$ is defined. A range ignores undefined values and exists only when at
least two values remain:

$$
Range_g(m_{c,g})=\max_{g\in G_{m,c}}m_{c,g}
-\min_{g\in G_{m,c}}m_{c,g}.
$$

The `fairness_metrics` columns are:

- `demographic_parity_difference`: $DPDiff_c=Range_g(SR_{c,g})$;
- `demographic_parity_ratio`:
  $DPRatio_c=\min_{g\in G_{SR,c}}SR_{c,g}/\max_{g\in G_{SR,c}}SR_{c,g}$;
- `equal_opportunity_difference`: $EODiff_c=Range_g(TPR_{c,g})$;
- `false_positive_rate_difference`: $FPRDiff_c=Range_g(FPR_{c,g})$;
- `equalized_odds_difference`: $EOddsDiff_c=\max(EODiff_c,FPRDiff_c)$;
- `predictive_parity_difference`: $PPDiff_c=Range_g(PPV_{c,g})$.

A difference of zero and a demographic-parity ratio of one mean equality across
the compared audit groups. The ratio requires at least two defined selection
rates and a positive maximum; if every group has zero selection rate, it is
`NaN` rather than $0/0$. Equalized-odds difference is `NaN` unless both its TPR-
and FPR-range components are defined.

The per-target-label coverage outputs are:

- `n_audit_groups_compared` $=G$;
- `n_selection_rate_defined_audit_groups` $=|G_{SR,c}|$;
- `n_recall_defined_audit_groups` $=|G_{TPR,c}|$;
- `n_false_positive_rate_defined_audit_groups` $=|G_{FPR,c}|$;
- `n_precision_defined_audit_groups` $=|G_{PPV,c}|$.

#### 5. Condition-level fairness summaries and coverage

Audit-group accuracy is summarized as:

- `worst_audit_group_accuracy`: $\min_gAccuracy_g$;
- `audit_group_accuracy_difference`: $Range_g(Accuracy_g)$.

The accuracy difference requires at least two audit groups. Higher worst-group
accuracy and a lower accuracy difference are preferable.

For each target-label fairness value $d_c$, let $D_d$ contain the target labels
where it is defined. Every fairness family receives the same three summaries:

$$
Mean(d)=\frac{1}{|D_d|}\sum_{c\in D_d}d_c,\qquad
Min(d)=\min_{c\in D_d}d_c,\qquad
Max(d)=\max_{c\in D_d}d_c.
$$

The exact result-column families are:

| Per-target-label value $d$ | Condition result columns |
|---|---|
| `demographic_parity_difference` | `mean_demographic_parity_difference`, `min_demographic_parity_difference`, `max_demographic_parity_difference` |
| `demographic_parity_ratio` | `mean_demographic_parity_ratio`, `min_demographic_parity_ratio`, `max_demographic_parity_ratio` |
| `equal_opportunity_difference` | `mean_equal_opportunity_difference`, `min_equal_opportunity_difference`, `max_equal_opportunity_difference` |
| `false_positive_rate_difference` | `mean_false_positive_rate_difference`, `min_false_positive_rate_difference`, `max_false_positive_rate_difference` |
| `equalized_odds_difference` | `mean_equalized_odds_difference`, `min_equalized_odds_difference`, `max_equalized_odds_difference` |
| `predictive_parity_difference` | `mean_predictive_parity_difference`, `min_predictive_parity_difference`, `max_predictive_parity_difference` |

Each family omits undefined target-label values. If $D_d$ is empty, all three
summaries are `NaN`. Its exact coverage output is:

- `n_demographic_parity_defined_target_labels` $=|D_{DPDiff}|$;
- `n_demographic_parity_ratio_defined_target_labels` $=|D_{DPRatio}|$;
- `n_equal_opportunity_defined_target_labels` $=|D_{EODiff}|$;
- `n_false_positive_rate_defined_target_labels` $=|D_{FPRDiff}|$;
- `n_equalized_odds_defined_target_labels` $=|D_{EOddsDiff}|$;
- `n_predictive_parity_defined_target_labels` $=|D_{PPDiff}|$.

For difference metrics, zero is best, so their minimum is the best target-label
value and their maximum is the worst. For demographic-parity ratio, one is best,
so the minimum ratio is the worst target-label value and the maximum is the best.
Coverage plots show $K$ or $G$ first so every defined count has an explicit
denominator.

Quality and fairness must be interpreted together. With `target: profession`
and audit column `gender`, these are conventional protected-group fairness
diagnostics. With `target: gender` and audit column `profession`, the same
mathematics is better described as profession-conditioned performance or
stereotype/association diagnostics. Profession and gender remain separate
experiments with the same prompt candidates and separate winners.
""",
                latex_delimiters=[
                    {'left': '$$', 'right': '$$', 'display': True},
                    {'left': '$', 'right': '$', 'display': False},
                ],
            )

        config_text = gr.Code(
            value=CONFIG_PATH.read_text(encoding='utf-8'),
            language='yaml',
            lines=38,
            label='Experiment configuration',
        )
        run_button = gr.Button('Run configured split', variant='primary')
        status = gr.Markdown()

        with gr.Tabs():
            with gr.Tab('Best conditions'):
                best_conditions = gr.Dataframe(
                    label='One current-split winner per language model',
                    interactive=False,
                )
            with gr.Tab('All rankings'):
                all_rankings = gr.Dataframe(
                    label='All conditions ranked within language model',
                    interactive=False,
                )
            with gr.Tab('Detailed metrics'):
                target_label_metrics = gr.Dataframe(
                    label='Per-target-label metrics', interactive=False
                )
                fairness_metrics = gr.Dataframe(
                    label='Target-label disparities across audit groups', interactive=False
                )
                audit_group_metrics = gr.Dataframe(
                    label='Target-label rates and supports by audit group', interactive=False
                )
                confusion_matrix = gr.Dataframe(label='Confusion counts', interactive=False)
            with gr.Tab('Best-condition predictions'):
                predictions = gr.Dataframe(
                    label='Current-split labels and scores for winning conditions',
                    interactive=False,
                )
            with gr.Tab('Source composition'):
                source_counts = gr.Dataframe(
                    label='All filtered train, validation, and test source rows',
                    interactive=False,
                )
            with gr.Tab('Run composition'):
                run_counts = gr.Dataframe(
                    label='Capped train plus only the configured evaluation split',
                    interactive=False,
                )
            with gr.Tab('Plots'):
                plot_gallery = gr.Gallery(
                    label='Rankings and best-condition diagnostics',
                    type='filepath',
                    columns=2,
                    height=720,
                    object_fit='contain',
                    buttons=['download', 'download_all', 'fullscreen'],
                    interactive=False,
                )

        run_button.click(
            run_from_yaml,
            inputs=config_text,
            outputs=[
                status,
                best_conditions,
                all_rankings,
                target_label_metrics,
                fairness_metrics,
                audit_group_metrics,
                confusion_matrix,
                predictions,
                source_counts,
                run_counts,
                plot_gallery,
            ],
            concurrency_limit=1,
        )
    return app


if __name__ == '__main__':
    build_app().queue(default_concurrency_limit=1).launch()
