"""Teaching-oriented Gradio interface for the single-split experiment."""

from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

import evaluation
from pipeline import discard_incomplete_run, prepare_embedding_cache, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / 'config.yaml'
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
        best_prompt_paths = list(run_dir.glob('*_best_prompts.txt'))
        if not config_path.is_file() or not best_prompt_paths:
            continue

        label = run_dir.name
        try:
            archived_config = _load_yaml_mapping(
                config_path.read_text(encoding='utf-8'),
                f'{config_path}',
            )
            label = f'{run_dir.name} — target: {archived_config["defaults"]["target"]}'
        except Exception:
            pass
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
        selection_lines.append(
            f'- **{language_model_id}**: rank 1 with '
            f'`{configured_metric}={row[metric_column]}`'
        )

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
    status_lines = [
        f'{action} **{evaluation_split}** run.  ',
        f'Complete artifacts: `{output["run_dir"]}`  ',
    ]
    if 'resumed_language_models' in output:
        resumed_language_models = output['resumed_language_models']
        resumed = ', '.join(resumed_language_models) if resumed_language_models else 'none'
        status_lines.append(f'Resumed completed models: `{resumed}`  ')
    status_lines.extend([
        f'Ranking metric: `{configured_metric}` '
        f'(`{config["defaults"]["ranking_direction"]}`)  ',
        '',
        *selection_lines,
        '',
        f'Best-prompt report: `{output["best_prompts"]}`  ',
        f'UI tables show at most {TABLE_PREVIEW_ROWS} rows; CSV files are complete.',
    ])

    return (
        '\n'.join(status_lines),
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
            'best_prompts': run_dir / f'{evaluation_split}_best_prompts.txt',
        }
        return _present_output(output, config, 'Loaded existing')
    except Exception as exc:
        return (f'Error: {type(exc).__name__}: {exc}',) + (None,) * 10


def prepare_from_yaml(config_text: str) -> str:
    """Prepare complete training embeddings from the edited YAML."""

    try:
        config = _load_yaml_mapping(config_text, 'The configuration')

        row_counts = prepare_embedding_cache(config, PROJECT_ROOT, print)
        prepared_lines = [
            f'- **{model_id}**: {row_count:,} training rows'
            for model_id, row_count in row_counts.items()
        ]
        return '\n'.join([
            'Complete training-embedding caches are ready.  ',
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
1. Explicitly prepare reusable embeddings for every canonical training row.
2. Load profession-filtered `train`, `validation` (source `dev`), and `test`
   rows, then cap the retrieval training pool.
3. Filter the complete vector table to that pool and balance only the evaluation
   split by profession and gender.
4. If a model CSV is missing, embed only the selected evaluation rows.
5. Load each missing language model once, predict every condition, then
   atomically save its raw predictions under `<output_dir>/incomplete_run/`.
6. For each completed or resumed model, calculate and rank every condition and
   mark exactly one `is_best`.
7. Write complete tables, plots, and best prompts, then delete `incomplete_run`.

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
also multiply by selected evaluation rows. Language-model context overflows
fail explicitly; embedding inputs exceeding their configured sequence limit
are reported and truncated by the encoder.
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
