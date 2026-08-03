"""Small Gradio UI for editing config.yaml and running the same pipeline."""

from pathlib import Path

import gradio as gr
import yaml

from pipeline import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def run_from_yaml(config_text: str):
    """Run one experiment from the YAML shown in the editor."""

    try:
        config = yaml.safe_load(config_text)
        if not isinstance(config, dict):
            raise ValueError("The configuration must be a YAML mapping")
        output = run_experiment(config, PROJECT_ROOT, print)
        best_prompts_path = output["best_prompts"]
        validation_results_frame = output["validation_results"]
        selected = validation_results_frame.loc[
            validation_results_frame["selected_for_test"].astype(bool)
        ].copy()
        final = output["results"].copy()
        ranking_metric = config["defaults"]["ranking_metric"]
        configured_models = [entry["id"] for entry in config["model"]["language_models"]]
        if (
            len(selected) != len(configured_models)
            or selected["model"].nunique() != len(configured_models)
            or set(selected["model"]) != set(configured_models)
        ):
            raise ValueError("Expected exactly one validation-selected condition per model")
        if (
            len(final) != len(configured_models)
            or final["model"].nunique() != len(configured_models)
            or set(final["model"]) != set(configured_models)
        ):
            raise ValueError("Expected exactly one independent final-test result per model")

        prediction_columns = [
            "evaluation_split",
            "query_id",
            "true_label",
            "audit_group",
            "predicted_label",
            "condition",
            "retrieval",
            "embedding_model",
            "model",
            "k",
            "example_order",
            "prompt_name",
            "label_scores",
        ]
        predictions = output["predictions"]
        selected_conditions = selected["condition"].tolist()
        selected_test_predictions = predictions.loc[
            predictions["evaluation_split"].eq("test")
            & predictions["condition"].isin(selected_conditions),
            prediction_columns,
        ]
        selection_lines = []
        for model_id in configured_models:
            selected_row = selected.loc[selected["model"].eq(model_id)].iloc[0]
            final_row = final.loc[final["model"].eq(model_id)].iloc[0]
            selection_lines.append(
                f"- **{model_id}**: "
                f"`{selected_row['condition']}` — validation "
                f"`{ranking_metric}={selected_row[ranking_metric]}`; final test "
                f"`{ranking_metric}={final_row[ranking_metric]}`"
            )
        status = "\n".join(
            [
                f"Finished. Results: `{output['run_dir']}`  ",
                f"One validation winner per model on `{ranking_metric}`:",
                "",
                *selection_lines,
                "",
                f"Selected prompts file: `{best_prompts_path}`",
            ]
        )
        return (
            status,
            validation_results_frame,
            final,
            output["class_metrics"],
            output["fairness_metrics"],
            output["group_metrics"],
            output["confusion_matrix"],
            output["dataset_counts"],
            selected_test_predictions,
            str(output["plot"]),
        )
    except Exception as exc:
        return (f"Error: {type(exc).__name__}: {exc}",) + (None,) * 9


def build_app():
    """Construct the UI; the YAML editor exposes every project setting."""

    with gr.Blocks(title="Prompt search and bias evaluation") as app:
        gr.Markdown(
            r"""
# Prompt search and bias evaluation

The YAML below is the single source of settings.

- `target: profession` supplies **hard_text + gender** and predicts profession.
- `target: gender` supplies **hard_text + profession** and predicts gender.
- `dataset.professions` accepts a list or `all`. With profession as the target,
  it defines allowed labels; with gender as the target, it defines audit groups.
- `dataset.train_size` accepts a positive integer or `all` for every matching
  cached training row. It limits the retrieval pool, not the downloaded cache.
- `dataset.shuffle_seed` reproducibly shuffles the official `train`, `dev`, and
  `test` splits before row selection. Keep it fixed across experiment seeds.
- Every language model × retrieval method × embedding model × `k` × example
  order × master prompt is a validation condition on `dev`. Validation selects
  one winner **within each language model**; those winners alone run on final
  `test` rows.
- `validation_per_profession_gender` is the main search-runtime multiplier.
  `test_per_profession_gender` controls the independent final evaluation.
  The thesis defaults are 5 validation and 10 test rows per cell.
- Larger `k` gives the model more examples but makes prompts longer.
- `semantic` takes the exact nearest biographies by cosine similarity.
- `balanced_semantic` repeatedly takes the nearest currently feasible
  biography while keeping profession, gender, and joint-cell counts as even
  as `k` permits. It expands the LanceDB search automatically when necessary.
- Every `embedding_models` entry creates separate `semantic` and
  `balanced_semantic` conditions.
- Every `model.language_models` entry creates a separate language-model
  condition. The pipeline loads and releases Hugging Face models sequentially.
- The master prompt and output constraints form the system message. Retrieved
  examples become user/assistant demonstrations, and the evaluation biography
  is the final user message rendered by each model's native chat template.
- Qwen uses a 1024-token cap, while BGE retains its architectural 512-token
  maximum. Both receive their own trained query prefix; cached training
  biographies remain raw `hard_text`.
- Each embedding-model ID owns one readable reusable LanceDB table. Delete
  `data/lancedb/` before changing data or embedding settings.
- The model scores every allowed label and chooses the largest mean conditional
  token log-probability. These scores rank labels; they are not calibrated
  probabilities.
- The selected-test-predictions tab shows allowed-label log-scores for every
  model's validation-selected condition.
- `ranking_metric: macro_f1` with `maximize` is the quality-oriented default.
  To rank by a disparity such as `max_equalized_odds_difference`, use
  `ranking_direction: minimize`.
- `device: mps` requires Apple GPU support and fails instead of falling back to
  CPU for these large models.

Key formulas:

$$
Accuracy=\frac{\#correct}{N},\qquad
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}
$$

$$
DPDiff_c=\max_g P(\hat Y=c\mid A=g)-\min_g P(\hat Y=c\mid A=g)
$$

$$
EqualOpportunityDiff_c=\max_g TPR_{c,g}-\min_g TPR_{c,g}
$$

$$
score(c)=\frac{1}{|T_c|}\sum_j
\log P(t_j\mid prompt,t_{&lt;j}),\qquad
\hat c=\arg\max_{c\in labels}score(c)
$$

Lower disparity is not useful by itself if predictive quality is poor. The
README explains every setting, metric, denominator, and interpretation.
"""
        )
        config_text = gr.Code(
            value=CONFIG_PATH.read_text(encoding="utf-8"),
            language="yaml",
            lines=40,
            label="Experiment configuration",
        )
        run_button = gr.Button("Run experiment", variant="primary")
        status = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Prompt selection and final test"):
                validation_results = gr.Dataframe(
                    label="Validation conditions; selected_for_test marks one winner per model",
                    interactive=False,
                )
                results = gr.Dataframe(
                    label="Independent final-test result for each model's selected condition",
                    interactive=False,
                )
            with gr.Tab("Metric details"):
                class_metrics = gr.Dataframe(
                    label="Per-class metrics",
                    interactive=False,
                )
                fairness_metrics = gr.Dataframe(
                    label="Group disparities by target class",
                    interactive=False,
                )
                group_metrics = gr.Dataframe(
                    label="Group rates and supports",
                    interactive=False,
                )
                confusion_matrix = gr.Dataframe(
                    label="Confusion counts",
                    interactive=False,
                )
            with gr.Tab("Selected test predictions"):
                predictions = gr.Dataframe(
                    label="Final-test labels and scores for all selected model conditions",
                    interactive=False,
                )
            with gr.Tab("Data and plot"):
                dataset_counts = gr.Dataframe(
                    label="Dataset composition",
                    interactive=False,
                )
                plot = gr.Image(
                    label="Quality and group-disparity comparison",
                    type="filepath",
                )

        # One event keeps the UI easy to follow and calls the exact notebook pipeline.
        run_button.click(
            run_from_yaml,
            inputs=config_text,
            outputs=[
                status,
                validation_results,
                results,
                class_metrics,
                fairness_metrics,
                group_metrics,
                confusion_matrix,
                dataset_counts,
                predictions,
                plot,
            ],
            concurrency_limit=1,
        )
    return app


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch()
