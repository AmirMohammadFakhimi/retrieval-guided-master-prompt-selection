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


def _preview(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the UI responsive while preserving complete CSV artifacts."""

    return frame.head(TABLE_PREVIEW_ROWS)


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
            raise ValueError(
                f'Expected exactly one best {evaluation_split} condition per '
                'language model'
            )

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
score(c)=\frac{1}{|T_c|}\sum_j\log P(t_j\mid prompt,t_{<j}),\qquad
\hat c=\arg\max_c score(c)
$$

For target label $c$, precision $=TP/(TP+FP)$, recall $=TP/(TP+FN)$, and
$F1=2TP/(2TP+FP+FN)$. Macro metrics average defined target-label rates;
$Precision=PPV$, $Recall=TPR$, and $Specificity=TNR$. Whenever the corresponding
denominator is nonzero, $FPR=1-Specificity$ and $FNR=1-Recall$. Weighted metrics
use true target-label support. In single-label multiclass classification,
accuracy = micro precision = micro recall = micro F1 = weighted recall, and
balanced accuracy = macro recall.

For audit group $g$, selection rate $SR_{c,g}=(TP+FP)/N_g$. Demographic-parity
difference is $\max_g SR_{c,g}-\min_g SR_{c,g}$. Equal-opportunity difference
is the range of audit-group TPR; equalized-odds difference is the larger of the
TPR and FPR ranges. Predictive-parity difference is the range of audit-group
precision. A missing denominator produces `NaN`, and disparity needs two
defined audit groups.
Quality and fairness tables should be interpreted together.
"""
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
