**Retrieval-noise decay rates, $acc(d)=a\,e^{-\lambda d}+c$.**

| model          |   decay_lambda |   decay_a |   decay_floor |   decay_r2 |   decay_half_life |   decay_n_points | decay_converged   |
|:---------------|---------------:|----------:|--------------:|-----------:|------------------:|-----------------:|:------------------|
| sim-open-7b    |        0.17102 |    0.5221 |        0.2075 |     0.9889 |             4.053 |                6 | True              |
| sim-open-13b   |        0.08585 |    0.6989 |        0.0894 |     0.9959 |             8.074 |                6 | True              |
| sim-frontier-a |        0.06991 |    0.9548 |        0.0273 |     0.998  |             9.915 |                6 | True              |

_Higher $\lambda$ = faster collapse under hard-negative retrieval noise. Fits are weighted by per-dose sample count._
