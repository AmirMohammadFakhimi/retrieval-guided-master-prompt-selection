# Retrieval-guided master-prompt selection

B.Sc. project for comparing master prompts. Retrieved examples are controlled
few-shot support; retrieval is not an independently optimized research target.

For each configured language model, the pipeline evaluates the same prompt
conditions on Bias-in-Bios `dev`, selects that language model's winner, and
evaluates the winner once on `test`. Profession and gender are separate runs
with the same prompt candidates and separate winners.

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

- `pipeline.py`: configuration validation and experiment orchestration;
- `dataset.py`: Bias-in-Bios loading, cleaning, and target settings;
- `modeling.py`: retrieval, prompt construction, and language model scoring;
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

- `validation_results.csv`: all conditions and one selected row per language model;
- `results.csv`: one independent final-test row per language model;
- `predictions.csv`: prompts, retrieved-example metadata, seeds, label scores, and labels;
- `best_prompts.txt`: the selected prompt for each language model;
- class, confusion, group, fairness, data-composition, and plot outputs.

## Metric formulas

Notation: $c$ is a target class, $g$ is an audit group, $K$ is the number
of classes, $N$ is the total number of evaluated rows, $N_g$ is the size of
group $g$, and $n_c$ is the true support of class $c$. $D_m$ is the set
of classes where metric $m$ is defined. $TP$, $FP$, $FN$, and $TN$
are one-vs-rest confusion counts. A zero denominator produces `NaN`; a group
disparity requires at least two defined group rates.

For each class:

$$
Precision_c=PPV_c=\frac{TP_c}{TP_c+FP_c},\quad
Recall_c=TPR_c=\frac{TP_c}{TP_c+FN_c},\quad
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}
$$

$$
Specificity_c=TNR_c=\frac{TN_c}{TN_c+FP_c},\quad
FPR_c=\frac{FP_c}{FP_c+TN_c},\quad
FNR_c=\frac{FN_c}{FN_c+TP_c},\quad
NPV_c=\frac{TN_c}{TN_c+FN_c}
$$

For any class metric $m_c$, undefined values are omitted from its aggregate:

$$
Accuracy=\frac{\sum_cTP_c}{N},\quad
Macro(m)=\frac{1}{|D_m|}\sum_{c\in D_m}m_c,\quad
Weighted(m)=\frac{\sum_{c\in D_m}n_cm_c}{\sum_{c\in D_m}n_c}
$$

When $m$ is defined for every class, $|D_m|=K$.

In this single-label multiclass task, let $T=\sum_cTP_c$ be the number of
correct rows and $E=\sum_cFP_c=\sum_cFN_c$ the number of errors. Since
$N=T+E$:

$$
MicroPrecision=\frac{T}{T+E}=MicroRecall=MicroF1=Accuracy
$$

$$
WeightedRecall=\frac{1}{N}\sum_{c:n_c>0}n_c\frac{TP_c}{n_c}
=\frac{\sum_cTP_c}{N}=Accuracy,\qquad
BalancedAccuracy=\frac{1}{|D_R|}\sum_{c\in D_R}Recall_c=MacroRecall
$$

$D_R$ is the set of classes whose recall is defined. The results store these
two equality families once, under the columns
`accuracy / micro_precision / micro_recall / micro_f1 / weighted_recall` and
`macro_recall / balanced_accuracy`. Any individual standard name remains
valid as `ranking_metric`.

For multiclass MCC, $C$ is the confusion matrix,
$s=\sum_{ij}C_{ij}$, $q=\operatorname{trace}(C)$,
$p_k=\sum_iC_{ik}$ is the predicted total for class $k$, and
$t_k=\sum_jC_{kj}$ is its true total:

$$
MCC=\frac{qs-\sum_kp_kt_k}
{\sqrt{(s^2-\sum_kp_k^2)(s^2-\sum_kt_k^2)}}
$$

For Cohen's kappa, $p_o=Accuracy$ is observed agreement and
$p_e=\sum_k(t_k/s)(p_k/s)$ is chance-expected agreement:

$$
\kappa=\frac{p_o-p_e}{1-p_e}
$$

Within group $g$:

$$
SR_{c,g}=\frac{TP_{c,g}+FP_{c,g}}{N_g},\quad
TPR_{c,g}=\frac{TP_{c,g}}{TP_{c,g}+FN_{c,g}},\quad
FPR_{c,g}=\frac{FP_{c,g}}{FP_{c,g}+TN_{c,g}}
$$

$$
PPV_{c,g}=\frac{TP_{c,g}}{TP_{c,g}+FP_{c,g}},\qquad
Accuracy_g=\frac{\text{correct rows in }g}{N_g}
$$

Let $Range_g(x)=\max_gx_g-\min_gx_g$. The classwise fairness metrics are:

$$
DPDiff_c=Range_g(SR_{c,g}),\quad
DPRatio_c=\frac{\min_gSR_{c,g}}{\max_gSR_{c,g}},\quad
EqualOpportunityDiff_c=Range_g(TPR_{c,g})
$$

$$
FPRDiff_c=Range_g(FPR_{c,g}),\quad
EqualizedOddsDiff_c=\max(EqualOpportunityDiff_c,FPRDiff_c),\quad
PredictiveParityDiff_c=Range_g(PPV_{c,g})
$$

Finally, $WorstGroupAccuracy=\min_gAccuracy_g$ and
$GroupAccuracyDiff=Range_g(Accuracy_g)$. For classwise disparity $d_c$, let
$D_d$ be the classes where it is defined:

$$
MeanDisparity(d)=\frac{1}{|D_d|}\sum_{c\in D_d}d_c,\quad
WorstDifference(d)=\max_{c\in D_d}d_c,\quad
WorstDPRatio=\min_{c\in D_d}DPRatio_c
$$

The corresponding defined-class count is $|D_d|$.

The checked-in configuration has two retrieval methods, two embedding models,
two example counts, one example order, two master prompts, and four language models:
64 validation conditions per target.
