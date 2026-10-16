**Metric-market fit — CHAIR, POPE and REFIND/CSR where each applies.**

| model          | CHAIR_i (clean)   | Resolution Threshold (σ)   | POPE F1 (adversarial)   | POPE yes-ratio   | Sycophancy slope   |   CSR (injected falsehood) |   Knowledge Inertia rate |
|:---------------|:------------------|:---------------------------|:------------------------|:-----------------|:-------------------|---------------------------:|-------------------------:|
| sim-frontier-a |                   | >max                       |                         |                  |                    |                     nan    |                 0.195125 |
| sim-open-13b   |                   | >max                       |                         |                  |                    |                       1.22 |                 0.231125 |
| sim-open-7b    |                   | >max                       |                         |                  |                    |                       1.1  |                 0.25275  |

_Resolution Threshold is the first σ where CHAIR exceeds clean by >10 pts; '>max' means the model never crossed it within the tested range. A positive sycophancy slope means the yes-ratio rises with distractor count. CSR is measured on the injected falsehood; high CSR alongside a high Knowledge Inertia rate is direct evidence the poisoned context is driving the tokens._
