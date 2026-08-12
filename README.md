# Retrieval-guided master-prompt selection

B.Sc. project for comparing master prompts. Retrieved examples are controlled
few-shot support; retrieval is not an independently optimized research target.

For each configured language model, the pipeline evaluates and ranks the same
prompt conditions on exactly one configured Bias-in-Bios split. Use
`defaults.evaluation_split: validation` for `dev`, and switch it deliberately
to `test` only when you want a test run. Profession and gender remain separate
runs with the same prompt candidates and separate winners.

## Inference

The pipeline uses Hugging Face Transformers directly and loads language models
one at a time. Qwen thinking is disabled by the native chat-template argument
used for all configured language models. GPT-OSS is intentionally excluded:
its mandatory reasoning/Harmony protocol does not match direct final-label
likelihood scoring without adding a language-model-specific reasoning setting.

The checked-in device is `mps`, so a missing Apple-GPU runtime raises instead
of silently running these large language models on CPU.

For every allowed label, the pipeline computes its mean conditional token
log-probability and chooses the largest score. These are relative label scores,
not calibrated probabilities.

Ollama and LM Studio are good local generation servers, but their normal APIs
do not provide arbitrary teacher-forced candidate likelihoods. Using structured
output through either server would be a different method: constrained
generation instead of the current label scorer.

## Structure

- `configuration.py`: YAML loading and static configuration validation;
- `pipeline.py`: experiment orchestration and artifact writing;
- `dataset.py`: source loading/normalization, current-run row selection, and
  dataset-composition counting;
- `modeling.py`: retrieval, prompt construction, and language model scoring;
- `evaluation.py`: metrics, ranking, and selected-prompt reporting;
- `plotting.py`: current-split condition comparisons and metric diagnostics;
- `app.py`: teaching-oriented YAML editor with separate ranking, metric,
  prediction, source-composition, run-composition, and plot views;
- `Fairness_Aware_ICL_Complete_Pipeline.ipynb`: clean top-to-bottom experiment
  walkthrough using the same public interfaces as the application.

## Dataset flow

`load_data(config, root)` loads and normalizes all profession-filtered source
rows once and returns `train`, `validation` (source `dev`), and `test` mappings.
This is a data-loading boundary, not permission to evaluate every split.

`select_run_data(config, split_rows)` applies `dataset.train_size`, verifies
that every retrieval example count fits the selected train pool, and returns
the train rows, evaluation rows, and resolved rows per profession-gender cell.
An explicit positive integer uses that many rows; `max_balanced` resolves to
the smallest available cell in only `defaults.evaluation_split`.

`calculate_dataset_counts(config, split_rows)` is the single composition-count
implementation. It creates two different views during a run:

- `source_dataset_counts`: every filtered source row in train, validation, and
  test, for descriptive inspection only;
- `run_dataset_counts`: the capped train pool plus only the configured
  evaluation rows, which is authoritative for rows used by the experiment.

## Run

```bash
conda activate prompt-selection
python -m pip install -r requirements.txt
python app.py
```

You can also run
[`Fairness_Aware_ICL_Complete_Pipeline.ipynb`](Fairness_Aware_ICL_Complete_Pipeline.ipynb).
It starts with the split contract and configuration, displays both composition
tables, previews the language-model message format, runs the pipeline, and then
walks through rankings, winners, detailed metrics, plots, prompts, and bounded
prediction previews.

For the study, run once with `defaults.target: profession` and once with
`defaults.target: gender`, changing no other experimental controls.

Keep `defaults.evaluation_split: validation` while comparing prompts. The
single `dataset.evaluation_per_profession_gender` setting controls the selected
split: use 5 rows per profession-gender cell for validation and 10 for test, or
use `max_balanced` to select the largest equal cell size available. The flow
prints the resolved integer before embedding. The total is
`number of professions × 2 × resolved rows per cell`.
The unselected split is loaded only for source composition; it is never passed
to embedding, prediction, metric calculation, ranking, or plotting.

Important outputs in each timestamped result folder:

- `<split>_results.csv`: every ranked condition and one `is_best` row per language model;
- `<split>_predictions.csv`: prompts, retrieved-example metadata, seeds, label scores, and labels;
- `<split>_best_prompts.txt`: the selected prompt for each language model;
- `<split>_source_dataset_counts.csv`: all filtered train, validation, and test
  source composition;
- `<split>_run_dataset_counts.csv`: capped train plus the selected evaluation
  split composition;
- `plots/`: focused current-split plots for every numeric metric,
  count, support, coverage value, and confusion matrix;
- `<split>_target_label_metrics.csv`, `<split>_audit_group_metrics.csv`,
  `<split>_fairness_metrics.csv`, and `<split>_confusion_matrix.csv`.

Summary plots compare every configured condition in within-language-model rank
order and highlight `is_best`. Detailed target-label, audit-group, fairness,
coverage, and confusion diagnostics focus on the current split's winner for
each model; the CSV metric tables retain every condition.

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

Quality and fairness must be interpreted together. With `target: profession`
and audit column `gender`, these are conventional protected-group fairness
diagnostics. With `target: gender` and audit column `profession`, the same
mathematics is better described as profession-conditioned performance or
stereotype/association diagnostics. Profession and gender remain separate
experiments with the same prompt candidates and separate winners.

The checked-in configuration has two retrieval methods, two embedding models,
two example counts, one example order, two master prompts, and four language models:
64 conditions per target and configured evaluation split.
