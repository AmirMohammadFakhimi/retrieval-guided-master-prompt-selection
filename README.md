# Retrieval-guided master-prompt selection

B.Sc. project for comparing master prompts. Retrieved examples are controlled
few-shot support; retrieval is not an independently optimized research target.

For each configured language model, the pipeline evaluates and ranks the same
prompt conditions on exactly one configured Bias-in-Bios split. Use
`defaults.evaluation_split: validation` for `dev`, and switch it deliberately
to `test` only when you want a test run. Profession and gender remain separate
runs with the same prompt candidates and separate winners.

`configs/validation.yaml` contains the complete validation candidate grid.
Each file under `configs/test/` freezes one downstream language model and its
own validation-selected prompt, retrieval settings, and example count. This
keeps final test evaluation from recombining the three models' different
winners into a new condition grid.

## Inference

The pipeline uses Hugging Face Transformers directly and loads language models
one at a time. Qwen thinking is disabled by the native chat-template argument
used for all configured language models.

The checked-in configuration requests `mps`. Before running the project, set
`inference.device` to `auto` or to a backend available on the current machine:
`cuda`, `mps`, or `cpu`. An explicitly requested unavailable accelerator raises
an error instead of silently falling back to CPU.

`inference.prediction_method` selects exactly one prediction method for a run:

- `generated_output` renders the chat for an assistant response and generates
  up to 32 new tokens with deterministic greedy decoding (`do_sample=False`).
  Each configured prompt file contains the complete master-prompt wording,
  including the allowed labels and exact-output instruction. Configuration
  loading reads that text and the prompt builder only substitutes its
  placeholders. After surrounding whitespace is removed and the response is
  converted to lowercase, any output that is not exactly one allowed label
  stops the run with the raw response; there is no extraction or retry.
- `log_probability` computes each allowed label's mean conditional token
  log-probability and chooses the largest score. These are relative label scores,
  not calibrated probabilities.

The checked-in configuration uses `generated_output`. It does not sample and
does not use temperature. These two methods are different experimental methods,
so switch them only between deliberate runs rather than treating them as prompt
conditions.

`inference.generation_batch_size` is a positive run-wide performance setting
used only by `generated_output`. It batches rows within each condition; larger
values reduce generation calls but require more accelerator memory. The
checked-in value is 1, which disables batching.

## Structure

- `configs/validation.yaml`: complete candidate grid for model-specific winner
  selection on validation;
- `configs/test/`: one frozen, single-condition test configuration per language
  model;
- `configuration.py`: YAML loading and static configuration validation;
- `prompts/`: the frozen candidate-generation instruction, generated candidate
  pairs, and prompt-generation protocol;
- `pipeline.py`: experiment orchestration and artifact writing;
- `dataset.py`: source loading/normalization, current-run row selection, and
  dataset-composition counting;
- `embeddings.py`: embedding encoders, manifested LanceDB training tables, and
  evaluation-query-vector caching;
- `retrieval.py`: exact semantic retrieval, balanced selection, and example
  presentation ordering;
- `modeling.py`: device and model lifecycle, prompt construction, and language
  model prediction;
- `evaluation.py`: metrics, ranking, and selected-prompt reporting;
- `plotting.py`: current-split condition comparisons and metric diagnostics;
- `app.py`: teaching-oriented YAML editor with separate ranking, metric,
  prediction, source-composition, run-composition, and plot views;
- `Fairness_Aware_ICL_Complete_Pipeline.ipynb`: clean top-to-bottom experiment
  walkthrough using the same public interfaces as the application.

## Dataset flow

`load_source_rows(config, root)` loads every normalized source row once. When
the JSONL file is absent, the source dataset is downloaded using the pinned
`dataset.hub_id` and `dataset.revision`.
`select_profession_splits(config, rows)` then returns the profession-filtered
`train`, `validation` (source `dev`), and `test` mappings. This is a
data-loading boundary, not permission to evaluate every split.

`prepare_embedding_cache(config, root)` is the explicit one-time preparation
step. It embeds every canonical source-training row for each configured,
revision-pinned embedding model. Each model owns one manifested LanceDB table;
an unmanifested, missing, or incompatible table is rebuilt rather than reused.
New tables are compacted once after creation. Reused tables are not compacted
again. Training rows are stably sorted from longest to shortest before they are
divided into bounded input chunks. LanceDB stores that length order while each
row's `train_order` retains its canonical source position. Its tables are independent
of `dataset.professions` and `dataset.train_size`. Inputs exceeding an embedding
model's configured sequence limit are reported by row ID and token count, then
truncated by that encoder.

`select_run_data(config, split_rows)` applies `dataset.train_size`, verifies
that every retrieval example count fits the selected train pool, and returns
the train rows, evaluation rows, selected rows per profession-gender cell, and
maximum balanced rows per cell. An explicit positive integer uses that many
rows; `max_balanced` resolves to the smallest available cell in only
`defaults.evaluation_split`. Every run reports both the maximum and selected
values before inference.

For each embedding model, every selected evaluation query is compared with
every eligible training vector using true cosine similarity in fixed blocks of
64. A stable descending ranking preserves LanceDB scan position for exact-score
ties; no approximate search, quantization, or reduced-dimensional index is
used. `semantic` takes the nearest rows from that complete ranking.
`balanced_semantic` greedily selects the nearest currently feasible rows from
the same ranking while balancing profession, gender, and their joint cells; its
selection sequence intentionally preserves balanced prefixes for smaller
example counts. Only after the requested prefix is sliced,
`most_similar_first` or `most_similar_last` stably sorts that fixed set by
`retrieval_score` for prompt presentation. `shuffle` instead applies the
configured deterministic example-order seed. An example count of zero creates
one zero-shot condition per language model and master prompt. It supplies no
demonstrations and records retrieval method, embedding model, and example order
as `not_applicable`, because those controls cannot affect zero-shot prediction.

`calculate_dataset_counts(config, split_rows)` is the single composition-count
implementation. It creates two different views during a run:

- `source_dataset_counts`: every filtered source row in train, validation, and
  test, for descriptive inspection only;
- `run_dataset_counts`: the capped train pool plus only the configured
  evaluation rows, which is authoritative for rows used by the experiment.

When any positive example count is configured and a language-model checkpoint
is missing, an experiment opens the complete tables read-only, filters
professions first, then applies numeric `train_size` as the first matching rows
in the normalized dataset's fixed shuffled order. Before loading a language
model, it reuses or creates fingerprinted evaluation-query-vector NPZ files
under `retrieval.runtime_cache_path`, scans each eligible table once, and
computes the configured exact maximum-count retrieval selections in memory.
Malformed or stale query-vector files are ignored and regenerated atomically;
the large scan, vector, score, and ranking arrays are released before
language-model loading. Changing professions, target, prompts, or the train cap
does not rebuild training embeddings. Inputs that affect query vectors
invalidate the corresponding NPZ file; retrieval-pool and retrieval-method
changes are applied when selections are recomputed. A missing or incompatible
training table stops the run with an instruction to prepare it explicitly. An
all-zero run, or a fully resumed run with no missing model CSV, skips table
opening, evaluation-query embedding, and retrieval entirely.

## Setup and run

From the repository root, use any Python environment in which the project
requirements can be installed. No particular environment manager or
environment name is required.

```bash
python -m pip install -r requirements.txt
python app.py
```

The app and notebook default to `configs/validation.yaml`. To run one frozen
test condition in the notebook, change its visible `CONFIG_PATH` value to the
corresponding file under `configs/test/` before loading the configuration.

Before a run containing positive example counts, select **Prepare all training
embeddings**. When a language model CSV is missing, **Run configured split**
reuses or creates the query-vector NPZ files and computes every exact retrieval
selection in memory before loading the language model. An all-zero run can skip
preparation. The app calls inference and analysis in sequence. The notebook
exposes them as separate
`run_inference(...)` and `calculate_metrics(...)` sections.

`run_inference(...)` prepares any required query-vector cache, computes exact
retrieval selections in memory, performs the expensive model work, and returns
an `inference_run` containing the raw prediction table and its analysis context.
It also saves the combined
prediction CSV and the two count CSVs under `<output_dir>/incomplete_run/`.
`calculate_metrics(...)` consumes the object without loading a language model,
embedding a query, retrieving examples, or generating predictions. After a
kernel restart, run the notebook setup and configuration cells, set
`saved_predictions_path` to the previous run's `<split>_predictions.csv`, and
jump to the metric section. `load_inference_run(...)` reads that CSV and the two
count CSVs beside it.

After editing metric code or changing only `quality_metric`,
`quality_direction`, `fairness_metric`, or `fairness_direction`, rerun just the
metric section. Each successful metric recalculation creates a new timestamped
artifact directory.

During inference, each completed language model's raw predictions are saved
atomically under `<output_dir>/incomplete_run/`. On restart, an existing model
CSV is reused and a missing one is computed. This deliberately does not compare
the current YAML with the cached run, so discard the incomplete run before
changing experiment settings. Likewise, only load a prediction CSV created by
the same inference configuration; metric code and the configured quality and
fairness selection settings may differ. Use **Discard incomplete run** in the app or
`discard_incomplete_run(config, PROJECT_ROOT)` in the notebook. After every
final artifact is written successfully,
`incomplete_run/` is deleted automatically; the completed run retains
predictions, counts, and configuration
so metric-only recalculation remains available after a kernel restart.

Completed runs can be displayed without rerunning the experiment. In the app,
select a directory under **Completed runs** and choose **Load selected run**.
After finishing a run in the notebook, choose **Refresh completed runs** first;
the newest run is selected automatically, and older completed runs remain in
the dropdown. Loading reads that run's saved configuration, CSVs, plots, and
quality/fairness prompt reports only—it does not load models or recompute
results.

You can also run
[`Fairness_Aware_ICL_Complete_Pipeline.ipynb`](Fairness_Aware_ICL_Complete_Pipeline.ipynb).
It starts with the split contract and compact preflight information, previews
the exact language-model messages, separates inference from metric calculation,
then displays only the artifact directory, every tied winner for each language
model, and one editable factor-contrast summary slice. Complete CSVs and plots
are saved without being rendered in the notebook.

To benchmark the step-7 embedding batch size separately, run:

```bash
python benchmark_embedding_batch_sizes.py
```

The benchmark reads embedding settings from `configs/validation.yaml`. It tests
batch sizes 1 through 256 by powers of two. Qwen uses two 2,048-row
steps and BGE uses five. It times only embedding, prints the aggregate speed
for each batch size, and stores every step in
`results/embedding_batch_size_benchmark_measurements.csv`. The corresponding
aggregate summary is stored in
`results/embedding_batch_size_benchmark_summary.csv`.
Training texts are globally sorted from longest to shortest before the timed
chunks are selected, so the benchmark deliberately measures the longest rows.
Each batch size receives one untimed internal batch as a warm-up before its
measurements begin. Any selected input that the encoder will truncate is
reported before timing begins.

For the study, run once with `defaults.target: profession` and once with
`defaults.target: gender`, changing no other experimental controls.

Keep `defaults.evaluation_split: validation` while comparing prompts. The
single `dataset.evaluation_per_profession_gender` setting controls the selected
split: use 5 rows per profession-gender cell for validation and 10 for test, or
use `max_balanced` to select the largest equal cell size available. The flow
always prints both the maximum balanced capacity and selected value immediately
before inference begins. The total selected evaluation size is
`number of professions × 2 × selected rows per cell`.
The unselected split is loaded only for source composition; it is never passed
to embedding, prediction, metric calculation, ranking, or plotting.

Important outputs in each timestamped result folder:

- `<split>_results.csv`: every condition with independent `quality_rank` and
  `fairness_rank` values plus every exact tied `is_quality_best` and
  `is_fairness_best` winner;
- `<split>_factor_contrast_details.csv`: every matched one-factor metric delta
  and zero-shot-to-few-shot composite delta;
- `<split>_factor_contrast_summary.csv`: overall and per-language-model
  descriptive aggregates of those deltas;
- `<split>_predictions.csv`: prompts, retrieved-example metadata, seeds,
  prediction method, generated model output or label scores, and labels;
- `<split>_best_quality_prompts.txt`: every tied quality-selected prompt for
  each language model;
- `<split>_best_fairness_prompts.txt`: every tied fairness-selected prompt for
  each language model, plus an explicit note for any model without a defined
  fairness winner;
- `<split>_source_dataset_counts.csv`: all filtered train, validation, and test
  source composition;
- `<split>_run_dataset_counts.csv`: capped train plus the selected evaluation
  split composition;
- `plots/`: focused current-split plots for every numeric metric,
  count, support, coverage value, and confusion matrix;
- `<split>_target_label_metrics.csv`, `<split>_audit_group_metrics.csv`,
  `<split>_fairness_metrics.csv`, and `<split>_confusion_matrix.csv`.

The eight `<split>_quality_ranked_<family>.png` summary plots and eight
`<split>_fairness_ranked_<family>.png` summary plots compare every configured
condition in their corresponding within-language-model rank order and highlight
the matching winner flag. Nine `<split>_best_quality_*` and nine
`<split>_best_fairness_*` detailed target-label, audit-group, fairness, coverage,
and confusion diagnostics include every tied winner for their respective
criterion. Each confusion matrix is a separate condition panel rather than an
aggregate. The CSV metric tables retain every condition. When every fairness
value is undefined, the fairness-ranked summaries and prompt report remain,
while the nine fairness-selected detailed plots are omitted.

Quality and fairness ranks are independent exact dense ranks within each
language model and follow their configured directions. Equal values receive
ranks such as `1, 1, 2`; every defined rank-1 row receives its corresponding
winner flag. Undefined values are sorted last, receive no rank, and cannot win.
Lexicographic condition ordering makes tables deterministic but is never a
quality or fairness tie-breaker. An entirely undefined quality metric is an
error; an entirely undefined fairness metric produces no fairness winner for
that model.

## Factor contrasts

Factor contrasts answer how a metric changes when one configured factor changes
and every other applicable factor is held fixed. They cover language model,
prompt name, retrieval method, embedding model, positive example count, and
example order. Every pair of configured levels is compared in configuration
order. The detail artifact records the fixed context and raw delta
`to_metric_value - from_metric_value`; `improvement` reverses that sign for
lower-is-better difference metrics so a positive value always means improvement.

The summary reports total and defined pair counts, mean source and target values,
mean and standard deviation of raw deltas, improved/tied/worsened counts, and
the improved fraction. It contains an overall scope and, for factors other than
language model, a per-language-model scope. Undefined metric pairs remain
undefined and are excluded only from defined-pair aggregates.

The detail columns are:

- `contrast_type`, `factor`, `from_factor_value`, and `to_factor_value` identify
  the comparison;
- the split, target, audit column, prediction method, language model, and
  JSON `fixed_context` record its analysis context;
- `metric`, `direction`, `from_metric_value`, `to_metric_value`, and `delta`
  store the measured change;
- `improvement` is the direction-adjusted delta and `outcome` is
  `improved`, `tied`, `worsened`, or `undefined`;
- `from_condition_count` and `to_condition_count` expose whether a side is one
  condition or the averaged few-shot retrieval grid.

The summary retains the comparison identifiers and adds
`aggregation_scope`, `scope_language_model`, `n_total_pairs`,
`n_defined_pairs`, `mean_from_metric_value`, `mean_to_metric_value`,
`mean_delta`, `std_delta`, the three outcome counts, and `improvement_rate`.

Zero-shot is not part of the strict example-count contrast because retrieval
method, embedding model, and example order become applicable when examples are
introduced. Its separate `zero_shot_to_few_shot` contrast first averages every
configured retrieval condition at a positive count within one language-model
and prompt context, then compares that mean with the corresponding zero-shot
condition. These are descriptive paired differences, not causal estimates or
significance tests, and performance and fairness metrics remain separate rather
than being combined into one score.

## Metric formulas

### 1. Notation, counts, and supports

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

### 2. Target-label and audit-group rates

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

### 3. Overall quality and agreement

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

### 4. Target-label fairness across audit groups

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

### 5. Condition-level fairness summaries and coverage

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

| Per-target-label value $d$       | Condition result columns                                                                                          |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `demographic_parity_difference`  | `mean_demographic_parity_difference`, `min_demographic_parity_difference`, `max_demographic_parity_difference`    |
| `demographic_parity_ratio`       | `mean_demographic_parity_ratio`, `min_demographic_parity_ratio`, `max_demographic_parity_ratio`                   |
| `equal_opportunity_difference`   | `mean_equal_opportunity_difference`, `min_equal_opportunity_difference`, `max_equal_opportunity_difference`       |
| `false_positive_rate_difference` | `mean_false_positive_rate_difference`, `min_false_positive_rate_difference`, `max_false_positive_rate_difference` |
| `equalized_odds_difference`      | `mean_equalized_odds_difference`, `min_equalized_odds_difference`, `max_equalized_odds_difference`                |
| `predictive_parity_difference`   | `mean_predictive_parity_difference`, `min_predictive_parity_difference`, `max_predictive_parity_difference`       |

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

`fairness_metric` accepts only the condition-level fairness summaries above:
audit-group accuracy difference; the mean, minimum, and maximum
demographic-parity ratios; and the mean, minimum, and maximum difference
summaries for demographic parity, equal opportunity, false-positive rate,
equalized odds, and predictive parity. `fairness_direction` controls which end
receives rank 1. Within each language model, every exact defined rank-1 tie has
`is_fairness_best=True`; independently, `quality_metric` and
`quality_direction` determine `quality_rank` and `is_quality_best`.
Quality and fairness prompt reports, ranked summaries, and detailed plots use
their corresponding winners independently. Frozen test conditions remain
unchanged.

Quality and fairness must be interpreted together. With `target: profession`
and audit column `gender`, these are conventional protected-group fairness
diagnostics. With `target: gender` and audit column `profession`, the same
mathematics is better described as profession-conditioned performance or
stereotype/association diagnostics. Profession and gender remain separate
experiments with the same prompt candidates and separate winners.

For $P$ master prompts, $R$ retrieval methods, $E$ embedding models, $O$
example orders, and $K_+$ positive example counts, conditions per language
model equal $P\left(\mathbf{1}[0\text{ configured}]+R E O K_+\right)$. The
validation configuration therefore has 60 conditions per language model and
180 total. Its four professions, two genders, and ten evaluation rows per cell
produce 80 evaluation rows and 14,400 row-condition predictions.
