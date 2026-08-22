# Prompt candidates

`candidate_generation_prompt.txt` is the single frozen instruction given unchanged to every
closed-source generator. It is not a downstream master-prompt candidate and must not be listed
in `configs/validation.yaml`.

Each generator directory under `candidates/` owns one matched pair:

- `neutral.txt`
- `fairness_aware.txt`

Copy only the text inside the generator's two requested code blocks into those files. Do not add
the section headings or Markdown fences, and do not rewrite one candidate after generation. Add a
pair to `configs/validation.yaml` only after both files are non-empty and pass configuration
validation.

The current generator set is:

| Directory                                   | Generator selection                       |
|---------------------------------------------|-------------------------------------------|
| `openai_gpt_5_6_pro`                        | GPT-5.6 Pro                               |
| `openai_gpt_5_6_sol_high`                   | GPT-5.6 Sol with high reasoning           |
| `openai_gpt_5_6_thinking`                   | Best available OpenAI free-plan selection |
| `anthropic_claude_sonnet_5_max`             | Best available Claude free-plan selection |
| `google_gemini_3_6_flash_extended_thinking` | Best available Gemini free-plan selection |
| `xai_grok_4_5_fast`                         | Best available Grok free-plan selection   |

The free-plan descriptions record the selection protocol supplied on 2026-08-21; product
availability can change. Before final generation, record the exact model label shown by each
interface here if it differs from the directory name.

Use the same fresh-chat procedure for every generator: submit the frozen generation prompt once,
save the first valid pair verbatim, and record any mechanically necessary retry. Freeze the full
candidate set before evaluating it on the validation split. Compare neutral versus fairness-aware
within the same generator pair; let all candidates compete as master prompts, with a separate
winner selected for each downstream language model.

The `test/` pair contains only the original quick-run prompts. It is retained as test material and
is not part of the frozen validation candidate set.

Source configuration entries reference prompt files:

```yaml
prompt_templates:
  openai_gpt_5_6_sol_high_neutral:
    file: prompts/candidates/openai_gpt_5_6_sol_high/neutral.txt
  openai_gpt_5_6_sol_high_fairness_aware:
    file: prompts/candidates/openai_gpt_5_6_sol_high/fairness_aware.txt
```

Paths are resolved relative to the project root. The in-memory configuration and each run's
`config_used.yaml` contain the resolved text, so run artifacts remain self-contained if a source
file changes later.
