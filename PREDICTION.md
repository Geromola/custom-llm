# What I expect before training

Written and committed **before** the first real training run. Nothing here was edited
afterwards; the README compares these numbers with what actually happened.

## My three choices

| Choice | Value | Why |
|---|---|---|
| Corpus | Exp 1: classroom sentences only. Exp 2: classroom + 1,421 generated teaching passages in `corpus/` | The corpus is the only thing that differs between the two runs, so any change in the 48 eval scores has exactly one candidate cause. Generating the text, rather than importing outside documents, keeps the permissions trivial and lets me target specific eval skills instead of flooding the 509-type vocabulary with unrelated words. |
| Training steps | 3,000 in both runs | The assignment's suggested budget, and the reference run's validation loss is already flat well before it. Keeping it identical across runs matters more than tuning it: if I trained Exp 2 longer I could not tell whether a better score came from the new data or the extra steps. A 10-step run first, only to check the pipeline. |
| Learning rate | 0.001 in both runs | The notebook's default, with 100-step warmup and cosine decay to 10%. A much larger rate (say 0.05) would take steps too big for the loss surface — AdamW's updates would overshoot the minimum and the loss would oscillate or go non-finite, which the notebook explicitly guards against. A much smaller rate (say 1e-6) would move the weights so little that 3,000 steps would end near the random initialisation: the samples would stay as word salad and the training loss would barely leave ln(vocabulary size). |

## The design in one line

Four result sets — starter untrained/trained and expanded untrained/trained — let me
separate two different things that both look like "the model got better": **vocabulary
coverage** (a case stops being unscorable) and **learned patterns** (a scorable case
becomes correct). The expanded-untrained run has the new vocabulary but random weights,
so it is the control for coverage alone.

The extension corpus teaches 4 of the 8 extension categories — grammar, opposites, everyday knowledge,
categories/analogies — and deliberately contains **nothing** for negation, reference,
sequence and spatial relations. Those 12 cases are my control group: if their scores move,
something other than my teaching material is responsible.

## Numbers I expect

| Quantity | Prediction |
|---|---|
| Parameters | ~112k (fixed by the architecture, changes only through vocabulary size) |
| Exp 1 vocabulary | ~136 types, unknown-token rate ~0% on both splits |
| Exp 2 vocabulary | ~330 types (136 + the ~200 my files add), still far below the 509 cap, unknown rate ~0% |
| Exp 1 coverage | 24/48 scorable (the 24 starter cases); all 24 extension cases out-of-vocabulary |
| Exp 2 coverage | 36/48 scorable — the 24 starter cases plus my 12 taught cases; the 12 control cases stay out-of-vocabulary |
| Exp 1 untrained | 8–11 / 48 |
| Exp 1 trained | 18–22 / 48; starter_patterns 14–16/16, starter_transfer 3–5/8 |
| Exp 2 untrained | 10–13 / 48 — the taught cases become scorable but the weights are random, so ~25% of 12 by luck |
| Exp 2 trained | 26–31 / 48; of the 12 taught cases I expect 7–9 correct |
| Control cases | 0/12 in all four result sets |
| Loss | training panel ~4.9 → below 1.0 in both runs; validation close to training, because the 90/10 split shares sentence templates. Exp 2's losses will be **higher** than Exp 1's and are not comparable: different vocabulary, different corpus. |
| Runtime | under a minute per run on this M5 CPU |

## What I expect to fail

- **Grammar.** `yesterday she` → `walked` is the case I am least confident about. Every
  teaching line that contained the literal string "yesterday she" had to be withheld as
  eval material, so the model has to combine "yesterday … walked" from one set of
  sentences with "she walks / she walked" from another. If it fails, that is informative,
  not embarrassing.
- **Starter transfer.** The 8 new-wording cases were only 3–4/8 in the reference run. I
  expect my extension corpus to make them slightly *worse*, because roughly a quarter of
  the training batches now contain sentences that are not classroom templates at all.
- **Free continuations.** Even where the multiple-choice answer is right, I expect the
  free text to be a plausible-looking sentence in the corpus's style rather than a real
  answer to the prompt. Picking the right word out of four is a much easier task than
  producing one.
- **The chat interface.** Any word outside ~330 types comes back as `<UNK>`, prompts over
  48 tokens are truncated, and every prompt starts with no memory of the last one.

## What would make me wrong

If the 12 control cases move away from 0, my teaching material leaked into categories I
claimed not to teach. If Exp 2's taught cases score at chance (~3/12) after training, the
vocabulary arrived but the patterns did not — more data of the same kind would not be the
fix; a different sentence structure would be.
