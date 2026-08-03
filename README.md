# Retrieval-guided master-prompt selection

B.Sc. project for comparing master prompts. Retrieved examples are controlled
few-shot support; retrieval is not an independently optimized research target.

For each configured language model, the pipeline evaluates the same prompt
conditions on Bias-in-Bios `dev`, selects that model's winner, and evaluates the
winner once on `test`. Profession and gender are separate runs with the same
prompt candidates and separate winners.

## Inference

The pipeline uses Hugging Face Transformers directly and loads models one at a
time. Qwen thinking is disabled by the native chat-template argument used for
all configured models. GPT-OSS is intentionally excluded: its mandatory
reasoning/Harmony protocol does not match direct final-label likelihood scoring
without adding a model-specific reasoning setting.

The checked-in device is `mps`, so a missing Apple-GPU runtime raises instead
of silently running these large models on CPU.

For every allowed label, the pipeline computes its mean conditional token
log-probability and chooses the largest score. These are relative label scores,
not calibrated probabilities.

Ollama and LM Studio are good local generation servers, but their normal APIs
do not provide arbitrary teacher-forced candidate likelihoods. Using structured
output through either server would be a different method: constrained
generation instead of the current label scorer.

## Structure

- `pipeline.py`: configuration validation and experiment orchestration;
- `dataset.py`: Bias-in-Bios loading, cleaning, and target settings;
- `modeling.py`: retrieval, prompt construction, and model scoring;
- `evaluation.py`: metrics, ranking, plots, and selected-prompt reporting.

## Run

```bash
conda activate prompt-selection
python -m pip install -r requirements.txt
python app.py
```

You can also run
[`Fairness_Aware_ICL_Complete_Pipeline.ipynb`](Fairness_Aware_ICL_Complete_Pipeline.ipynb).
For the study, run once with `defaults.target: profession` and once with
`defaults.target: gender`, changing no other experimental controls.

Important outputs in each timestamped result folder:

- `validation_results.csv`: all conditions and one selected row per model;
- `results.csv`: one independent final-test row per model;
- `predictions.csv`: prompts, retrieved-example IDs, label scores, and labels;
- `best_prompts.txt`: the selected prompt for each model;
- class, confusion, group, fairness, data-composition, and plot outputs.

The checked-in configuration has two retrieval methods, two embedding models,
two `k` values, one example order, two master prompts, and four language models:
64 validation conditions per target.
