**The Conflict Matrix — modality preference under cross-modal conflict.**

| model          | track   |   n |   accuracy |   trust_index |   knowledge_inertia |   sensory_capture |   reasoning_collapse |   csr_mean |   counterfactual_symmetry | verdict     |
|:---------------|:--------|----:|-----------:|--------------:|--------------------:|------------------:|---------------------:|-----------:|--------------------------:|:------------|
| sim-frontier-a | B       | 175 |   0.125714 |      0.741905 |            0.874286 |                 0 |            0         |     nan    |                  0.815476 | Text-Biased |
| sim-open-13b   | A       | 175 |   0.411429 |      0.609013 |            0.56     |                 0 |            0.0285714 |       1.22 |                  0.973772 | Text-Biased |
| sim-open-7b    | A       | 175 |   0.422857 |      0.548918 |            0.48     |                 0 |            0.0971429 |       1.1  |                  0.968368 | Balanced    |

_Trust Index > 0.5 indicates text bias. Counterfactual symmetry is computed only over genuinely twinned pairs; 1.0 means the bias flips symmetrically. CSR is the REFIND context-sensitivity ratio on the injected falsehood._
