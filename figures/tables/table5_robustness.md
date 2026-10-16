**Robustness Score $\mathcal{R} = (Acc_{adverse}/Acc_{clean})\times(1-ECE)$.**

| model          | track   |   clean_accuracy |   combined |   conflict |   noise |   pooled |   visual |
|:---------------|:--------|-----------------:|-----------:|-----------:|--------:|---------:|---------:|
| sim-frontier-a | B       |         0.981313 |     0.5436 |     0.0505 |  0.5849 |   0.503  |   0.4519 |
| sim-open-13b   | A       |         0.787374 |     0.4276 |     0.2256 |  0.4907 |   0.423  |   0.3772 |
| sim-open-7b    | A       |         0.728788 |     0.3796 |     0.2366 |  0.4187 |   0.3697 |   0.3133 |

_Uncertainty Error is operationalised as Expected Calibration Error. On Track A, confidence is the length-normalised sequence probability; on Track B it is self-consistency across N=10 samples. The two tracks are not pooled._
