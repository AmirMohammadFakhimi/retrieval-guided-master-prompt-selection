"""Teaching-oriented Gradio interface for the single-split experiment."""

from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

import evaluation
from pipeline import discard_incomplete_run, prepare_embedding_cache, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'validation.yaml'
TABLE_PREVIEW_ROWS = 500
RESULT_TABLE_NAMES = (
    'predictions',
    'results',
    'target_label_metrics',
    'confusion_matrix',
    'audit_group_metrics',
    'fairness_metrics',
    'source_dataset_counts',
    'run_dataset_counts',
)


def _preview(table: pd.DataFrame) -> pd.DataFrame:
    """Keep the UI responsive while preserving complete CSV artifacts."""

    return table.head(TABLE_PREVIEW_ROWS)


def _load_yaml_mapping(text: str, description: str) -> dict:
    """Load one YAML mapping with a useful error for UI actions."""

    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError(f'{description} must be a YAML mapping')
    return config


def _completed_run_choices(config_text: str) -> list[tuple[str, str]]:
    """Return completed run directories under the edited output directory."""

    config = _load_yaml_mapping(config_text, 'The configuration')
    output_dir = (PROJECT_ROOT / config['defaults']['output_dir']).resolve()
    if not output_dir.exists():
        return []

    run_dirs = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and '-run-' in path.name
    ]
    run_dirs.sort(
        key=lambda path: path.name.rsplit('-run-', maxsplit=1)[1],
        reverse=True,
    )

    choices: list[tuple[str, str]] = []
    for run_dir in run_dirs:
        config_path = run_dir / 'config_used.yaml'
        if not config_path.is_file():
            continue

        try:
            archived_config = _load_yaml_mapping(
                config_path.read_text(encoding='utf-8'),
                f'{config_path}',
            )
        except Exception:
            continue
        defaults = archived_config['defaults']
        evaluation_split = defaults['evaluation_split']
        required_reports = (
            run_dir / f'{evaluation_split}_best_quality_prompts.txt',
            run_dir / f'{evaluation_split}_best_fairness_prompts.txt',
        )
        if not all(report.is_file() for report in required_reports):
            continue

        label = f'{run_dir.name} — target: {defaults["target"]}'
        choices.append((label, str(run_dir.resolve())))

    return choices


def refresh_completed_runs(config_text: str):
    """Refresh completed-run choices and select the newest run."""

    try:
        choices = _completed_run_choices(config_text)
        selected = choices[0][1] if choices else None
        return gr.update(choices=choices, value=selected)
    except Exception as exc:
        gr.Warning(f'Could not refresh completed runs: {type(exc).__name__}: {exc}')
        return gr.update(choices=[], value=None)


def _present_output(
        output: dict,
        config: dict,
        action: str,
) -> tuple:
    """Return the common result views for fresh and archived runs."""

    evaluation_split = output['evaluation_split']
    results = output['results'].copy()
    quality_selected = results.loc[results['is_quality_best'].astype(bool)].copy()
    fairness_selected = results.loc[results['is_fairness_best'].astype(bool)].copy()
    selected_conditions = results.loc[
        results['is_quality_best'].astype(bool)
        | results['is_fairness_best'].astype(bool)
    ].copy()
    language_model_ids = [
        entry['id'] for entry in config['inference']['language_models']
    ]
    if (
            set(quality_selected['language_model']) != set(language_model_ids)
            or not quality_selected['quality_rank'].eq(1).all()
    ):
        raise ValueError(
            f'Expected at least one quality-rank-1 {evaluation_split} condition '
            f'per language model'
        )
    if not fairness_selected['fairness_rank'].eq(1).all():
        raise ValueError('Every fairness-selected condition must have fairness_rank=1')

    defaults = config['defaults']
    quality_metric = defaults['quality_metric']
    quality_metric_column = evaluation.resolve_metric_column(quality_metric)
    fairness_metric = defaults['fairness_metric']
    fairness_metric_column = evaluation.resolve_metric_column(fairness_metric)
    quality_selection_lines = []
    fairness_selection_lines = []
    for language_model_id in language_model_ids:
        model_quality_selected = quality_selected.loc[
            quality_selected['language_model'].eq(language_model_id)
        ]
        quality_values = model_quality_selected[
            quality_metric_column
        ].drop_duplicates()
        if len(quality_values) != 1:
            raise ValueError(
                f'Quality-rank-1 conditions for {language_model_id!r} do not '
                f'share one exact {quality_metric!r} value'
            )
        quality_tie_count = len(model_quality_selected)
        quality_condition_word = (
            'condition' if quality_tie_count == 1 else 'conditions'
        )
        quality_selection_lines.append(
            f'- **{language_model_id}**: {quality_tie_count} '
            f'{quality_condition_word} tied at quality rank 1 with '
            f'`{quality_metric}={quality_values.iloc[0]}`'
        )

        model_fairness_selected = fairness_selected.loc[
            fairness_selected['language_model'].eq(language_model_id)
        ]
        if model_fairness_selected.empty:
            fairness_selection_lines.append(
                f'- **{language_model_id}**: no defined fairness winner'
            )
            continue
        fairness_values = model_fairness_selected[
            fairness_metric_column
        ].drop_duplicates()
        if len(fairness_values) != 1:
            raise ValueError(
                f'Fairness-rank-1 conditions for {language_model_id!r} do not '
                f'share one exact {fairness_metric!r} value'
            )
        fairness_tie_count = len(model_fairness_selected)
        fairness_condition_word = (
            'condition' if fairness_tie_count == 1 else 'conditions'
        )
        fairness_selection_lines.append(
            f'- **{language_model_id}**: {fairness_tie_count} '
            f'{fairness_condition_word} tied at fairness rank 1 with '
            f'`{fairness_metric}={fairness_values.iloc[0]}`'
        )

    prediction_columns = [
        'evaluation_split',
        'query_id',
        'true_label',
        'audit_group',
        'predicted_label',
        *[
            column
            for column in ('prediction_method', 'model_output', 'label_scores')
            if column in output['predictions'].columns
        ],
        'condition',
        'retrieval_method',
        'embedding_model',
        'language_model',
        'example_count',
        'example_order',
        'prompt_name',
    ]
    quality_selected_predictions = output['predictions'].loc[
        output['predictions']['condition'].isin(quality_selected['condition']),
        prediction_columns,
    ]
    fairness_selected_predictions = output['predictions'].loc[
        output['predictions']['condition'].isin(fairness_selected['condition']),
        prediction_columns,
    ]
    plot_gallery = [
        (str(path), name.replace('_', ' ').title())
        for name, path in output['plots'].items()
    ]
    status_lines = [
        f'{action} **{evaluation_split}** run.  ',
        f'Complete artifacts: `{output["run_dir"]}`  ',
    ]
    if 'resumed_language_models' in output:
        resumed_language_models = output['resumed_language_models']
        resumed = ', '.join(resumed_language_models) if resumed_language_models else 'none'
        status_lines.append(f'Resumed completed models: `{resumed}`  ')
    status_lines.extend([
        f'Quality metric: `{quality_metric}` (`{defaults["quality_direction"]}`)  ',
        f'Fairness metric: `{fairness_metric}` '
        f'(`{defaults["fairness_direction"]}`)  ',
        '',
        '**Quality winners**',
        *quality_selection_lines,
        '',
        '**Fairness winners**',
        *fairness_selection_lines,
        '',
        f'Best-quality-prompt report: `{output["best_quality_prompts"]}`  ',
        f'Best-fairness-prompt report: `{output["best_fairness_prompts"]}`  ',
        f'UI tables show at most {TABLE_PREVIEW_ROWS} rows; CSV files are complete.',
    ])

    return (
        '\n'.join(status_lines),
        selected_conditions,
        _preview(results),
        _preview(output['target_label_metrics']),
        _preview(output['fairness_metrics']),
        _preview(output['audit_group_metrics']),
        _preview(output['confusion_matrix']),
        _preview(quality_selected_predictions),
        _preview(fairness_selected_predictions),
        _preview(output['source_dataset_counts']),
        _preview(output['run_dataset_counts']),
        plot_gallery,
    )


def load_completed_run(config_text: str, run_directory: str | None):
    """Load one completed run from its archived artifacts only."""

    try:
        if not run_directory:
            raise ValueError('Select a completed run first')

        run_dir = Path(run_directory).resolve()
        completed_run_paths = {
            path for _, path in _completed_run_choices(config_text)
        }
        if str(run_dir) not in completed_run_paths:
            raise ValueError('The selected directory is not a completed run')

        config_path = run_dir / 'config_used.yaml'
        config = _load_yaml_mapping(
            config_path.read_text(encoding='utf-8'),
            f'{config_path}',
        )
        evaluation_split = config['defaults']['evaluation_split']
        output = {
            'evaluation_split': evaluation_split,
            'run_dir': run_dir,
            **{
                table_name: pd.read_csv(
                    run_dir / f'{evaluation_split}_{table_name}.csv'
                )
                for table_name in RESULT_TABLE_NAMES
            },
            'plots': {
                path.stem: path
                for path in sorted((run_dir / 'plots').glob('*.png'))
            },
            'best_quality_prompts': (
                run_dir / f'{evaluation_split}_best_quality_prompts.txt'
            ),
            'best_fairness_prompts': (
                run_dir / f'{evaluation_split}_best_fairness_prompts.txt'
            ),
        }
        return _present_output(output, config, 'Loaded existing')
    except Exception as exc:
        return (f'Error: {type(exc).__name__}: {exc}',) + (None,) * 11


def prepare_from_yaml(config_text: str) -> str:
    """Prepare complete manifested training-embedding tables from the edited YAML."""

    try:
        config = _load_yaml_mapping(config_text, 'The configuration')

        row_counts = prepare_embedding_cache(config, PROJECT_ROOT, print)
        prepared_lines = [
            f'- **{model_id}**: {row_count:,} training rows'
            for model_id, row_count in row_counts.items()
        ]
        return '\n'.join([
            'Complete training-embedding tables are ready.  ',
            *prepared_lines,
        ])
    except Exception as exc:
        return f'Error: {type(exc).__name__}: {exc}'


def discard_incomplete_run_from_yaml(config_text: str) -> str:
    """Discard the one incomplete run under the edited YAML's output directory."""

    try:
        config = _load_yaml_mapping(config_text, 'The configuration')

        discarded = discard_incomplete_run(config, PROJECT_ROOT)
        if discarded:
            return 'Discarded the incomplete run. The next experiment will start from the first model.'
        return 'There is no incomplete run to discard.'
    except Exception as exc:
        return f'Error: {type(exc).__name__}: {exc}'


def run_from_yaml(config_text: str):
    """Run the edited YAML and return split-aware teaching views."""

    try:
        config = _load_yaml_mapping(config_text, 'The configuration')

        output = run_experiment(config, PROJECT_ROOT, print)
        return _present_output(output, config, 'Finished')
    except Exception as exc:
        return (f'Error: {type(exc).__name__}: {exc}',) + (None,) * 11


def build_app():
    """Construct a detailed interface that follows the actual pipeline."""

    with gr.Blocks(title='Retrieval-guided master-prompt selection') as app:
        gr.Markdown(
            """
# Retrieval-guided master-prompt selection

This interface runs one complete **validation or test** experiment from the
YAML below. The selected split is used for every condition, metric, ranking,
plot, and quality/fairness prompt report. Loading the other source split for
descriptive counts never sends it to an encoder or language model.
"""
        )

        with gr.Accordion('1. Experiment flow and split contract', open=True):
            gr.Markdown(
                """
1. If any positive example count is configured, explicitly prepare manifested
   embedding tables for every canonical training row; all-zero runs skip this.
2. Load profession-filtered `train`, `validation` (source `dev`), and `test`
   rows, then cap the retrieval training pool.
3. If a model CSV is missing and positive counts need retrieval, reuse or embed
   the selected evaluation queries, scan each eligible table once, and compute
   exact exhaustive cosine rankings over every eligible training vector.
4. Derive the configured retrieval methods in memory from those complete
   rankings, then release the retrieval arrays before loading a language model.
5. Load each missing language model once, predict every condition, then
   atomically save its raw predictions and count tables under
   `<output_dir>/incomplete_run/`.
6. For each completed or resumed model, independently dense-rank the configured
   quality and fairness metrics. Mark every exact defined rank-1 tie with
   `is_quality_best` or `is_fairness_best`. Condition text only determines
   reproducible display order; it never breaks either tie.
7. Write complete tables, matched factor contrasts, parallel quality/fairness
   plot suites, and both selected-prompt reports, then delete `incomplete_run`.

The app performs inference and metric calculation in one action. The notebook
separates them so metric code and the quality/fairness metric and direction
settings can be recalculated from saved predictions without model inference,
including after a kernel restart.
There is no automatic validation-to-test transition. Switching to `test` is a
deliberate configuration change and runs the same complete condition grid.
Restarting reuses every available model CSV. Checkpoints are not compared with
the edited YAML, so discard the incomplete run before changing experiment settings.
"""
            )

        with gr.Accordion('2. Configuration values and runtime controls'):
            gr.Markdown(
                """
- `target`: `profession` or `gender`; this changes the held-out label and audit
  column, so use separate runs.
- `evaluation_split`: `validation` or `test`.
- `dataset.revision`: a non-empty pinned Hugging Face dataset revision.
- `professions`: `all` or a unique list of at least two supported professions.
- `train_size`: `all` or a positive integer. This caps retrieval rows only.
- `evaluation_per_profession_gender`: a positive integer, or `max_balanced` to
  use the smallest available cell in the selected split. Immediately before
  inference, the flow always prints the maximum balanced capacity and selected
  value. Evaluation rows equal **profession count × 2 genders × the selected
  value**.
- Retrieval methods: `semantic`, `balanced_semantic`. Example order:
  `most_similar_first`, `most_similar_last`, or `shuffle`.
- `example_counts`: unique non-negative integers. Zero creates one condition
  per language model and prompt with retrieval controls set to
  `not_applicable`; positive counts use the retrieval cross-product.
- `retrieval.lancedb_path` stores manifested training-embedding tables;
  `retrieval.runtime_cache_path` stores fingerprinted evaluation-query-vector
  NPZ files. Exact maximum-count retrieval selections are recomputed in memory
  for each required run.
- Embedding dtype: `float32`, `float16`, or `bfloat16`; language-model dtype may
  additionally be `auto`. Device: `auto`, `cuda`, `mps`, or `cpu`.
- `inference.prediction_method`: `generated_output` uses deterministic free
  generation and lowercase exact-label validation; `log_probability`
  compares the mean conditional token log-probabilities of the allowed labels.
- `inference.generation_batch_size`: positive run-wide performance setting used
  only by `generated_output`. Larger values reduce generation calls but use more
  accelerator memory; 1 disables batching.
- Each `prompt_templates` entry names one prompt file relative to the project
  root. Prompt files support exactly `{target}`, `{audit_column}`, and `{labels}`.
- `quality_metric` may name any numeric result column or documented alias;
  `quality_direction` is `maximize` or `minimize`.
- `fairness_metric` must name a condition-level fairness summary: audit-group
  accuracy difference; a mean/minimum/maximum demographic-parity ratio; or a
  mean/minimum/maximum demographic-parity, equal-opportunity,
  false-positive-rate, equalized-odds, or predictive-parity difference.
  `fairness_direction` is `maximize` or `minimize`.
- `quality_rank` and `fairness_rank` are independent exact dense ranks within
  each language model. Every defined rank-1 tie receives the corresponding
  `is_quality_best` or `is_fairness_best` flag. Undefined values receive no rank
  and cannot win.

For **P** prompt templates, **R** retrieval methods, **E** embedding models,
**O** example orders, and **K+** positive example counts, conditions per
language model equal **P × (zero configured + R × E × O × K+)**. Total
prediction rows also multiply by selected evaluation rows. Language-model
context overflows fail explicitly; embedding inputs exceeding their configured
sequence limit are reported and truncated by the encoder.
"""
            )

        with gr.Accordion('3. Methodology and metric formulas'):
            gr.Markdown(
                r"""
`semantic` takes the nearest biographies from an exact exhaustive cosine
ranking over every eligible training vector; no approximate search is used.
`balanced_semantic` takes the nearest currently feasible rows from that same
complete ranking while balancing profession, gender, and their joint cells.
After the requested example-count prefix is selected, the similarity orders
sort that fixed set by retrieval score for prompt presentation; this keeps
balanced prefixes independent from presentation order. The language model
receives the complete configured master prompt, zero or more retrieved
user/assistant demonstrations, then the evaluation biography. The code only
substitutes the prompt placeholders; it does not append prompt wording. At zero
examples, the message list contains only the system instruction and evaluation
biography. The checked-in templates list the allowed labels and request exactly
one value.

With `prediction_method: generated_output`, generation is greedy
(`do_sample=False`), uses no temperature, allows up to 32 new tokens, and is
accepted only when its trimmed and lowercased text exactly matches one allowed
label. With `prediction_method: log_probability`, the selected label has the
largest mean conditional token log-probability:

$$
score(c)=\frac{1}{|T_c|}\sum_j\log P(t_j\mid prompt,t_{1:j-1}),\qquad
\hat c=\arg\max_c score(c)
$$

### Factor contrasts

The two factor-contrast CSVs compare every pair of configured levels while
holding every other applicable factor fixed. They cover language model, prompt,
retrieval method, embedding model, positive example count, and example order.
The detail table contains each matched metric delta. The summary aggregates
those deltas overall and, except for language-model contrasts, within each
language model. Difference metrics are lower-is-better; other eligible rates,
agreement scores, and parity ratios are higher-is-better. Undefined pairs are
retained in total counts but omitted from defined-pair aggregates.

Detail rows store the contrast type, factor transition, fixed JSON context,
source/target metric values, raw and direction-adjusted deltas, outcome, and
condition counts. Summary rows store scope, total/defined pairs, mean source and
target values, delta mean/standard deviation, improved/tied/worsened counts, and
the improvement rate.

Zero-shot is separate from strict example-count contrasts because retrieval
controls become applicable when examples are introduced. For each positive
count, `zero_shot_to_few_shot` first averages the complete retrieval grid within
one language-model and prompt context and then compares that mean with zero-shot.
The reported changes are descriptive paired differences, not causal estimates
or significance tests, and no composite quality/fairness score is created.

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
valid `quality_metric` alias.

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

Within each language model, `fairness_rank` is the exact dense rank on the
configured `fairness_metric` and `fairness_direction`; every exact defined
rank-1 tie has `is_fairness_best=True`. Independently, `quality_rank` uses
`quality_metric` and `quality_direction`, and every exact defined rank-1 tie has
`is_quality_best=True`. Undefined values receive no rank and cannot win.
Deterministic condition ordering is only for presentation and never breaks a
tie. Prompt reports and detailed plots are generated independently for quality
and fairness winners. A model with no defined fairness winner is recorded in
the fairness report and has no fairness-selected detailed rows; if every model
lacks one, the fairness-selected detailed plot suite is omitted. Frozen test
conditions remain unchanged.

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
        with gr.Row():
            prepare_button = gr.Button('Prepare all training embeddings')
            run_button = gr.Button('Run configured split', variant='primary')
            discard_button = gr.Button('Discard incomplete run')
        status = gr.Markdown()

        with gr.Row():
            completed_runs = gr.Dropdown(
                choices=[],
                label='Completed runs',
                info='Newest first; refresh after completing a run in the notebook.',
                scale=3,
            )
            refresh_runs_button = gr.Button('Refresh completed runs')
            load_run_button = gr.Button('Load selected run')

        with gr.Tabs():
            with gr.Tab('Quality and fairness winners'):
                selected_conditions = gr.Dataframe(
                    label='Current-split quality and fairness rank-1 conditions',
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
            with gr.Tab('Best-quality-condition predictions'):
                quality_predictions = gr.Dataframe(
                    label='Current-split labels and scores for quality winners',
                    interactive=False,
                )
            with gr.Tab('Best-fairness-condition predictions'):
                fairness_predictions = gr.Dataframe(
                    label='Current-split labels and scores for fairness winners',
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
                    label='Parallel quality and fairness selection diagnostics',
                    type='filepath',
                    columns=2,
                    height=720,
                    object_fit='contain',
                    buttons=['download', 'download_all', 'fullscreen'],
                    interactive=False,
                )

        prepare_button.click(
            prepare_from_yaml,
            inputs=config_text,
            outputs=status,
            concurrency_limit=1,
            concurrency_id='experiment_files',
        )
        run_button.click(
            run_from_yaml,
            inputs=config_text,
            outputs=[
                status,
                selected_conditions,
                all_rankings,
                target_label_metrics,
                fairness_metrics,
                audit_group_metrics,
                confusion_matrix,
                quality_predictions,
                fairness_predictions,
                source_counts,
                run_counts,
                plot_gallery,
            ],
            concurrency_limit=1,
            concurrency_id='experiment_files',
        )
        discard_button.click(
            discard_incomplete_run_from_yaml,
            inputs=config_text,
            outputs=status,
            concurrency_limit=1,
            concurrency_id='experiment_files',
        )
        refresh_runs_button.click(
            refresh_completed_runs,
            inputs=config_text,
            outputs=completed_runs,
            concurrency_limit=1,
            concurrency_id='experiment_files',
        )
        load_run_button.click(
            load_completed_run,
            inputs=[config_text, completed_runs],
            outputs=[
                status,
                selected_conditions,
                all_rankings,
                target_label_metrics,
                fairness_metrics,
                audit_group_metrics,
                confusion_matrix,
                quality_predictions,
                fairness_predictions,
                source_counts,
                run_counts,
                plot_gallery,
            ],
            concurrency_limit=1,
            concurrency_id='experiment_files',
        )
        app.load(
            refresh_completed_runs,
            inputs=config_text,
            outputs=completed_runs,
            concurrency_limit=1,
            concurrency_id='experiment_files',
        )
    return app


if __name__ == '__main__':
    build_app().queue(default_concurrency_limit=1).launch()
