**Model zoo accuracy by stressor axis.**

| Model          | Track   |    Clean |   Combined |   Conflict |    Noise |   Visual |    N |
|:---------------|:--------|---------:|-----------:|-----------:|---------:|---------:|-----:|
| sim-frontier-a | B       | 0.981313 |   0.617757 |   0.125714 | 0.638235 | 0.564269 | 8000 |
| sim-open-13b   | A       | 0.787374 |   0.437878 |   0.411429 | 0.502941 | 0.40566  | 8000 |
| sim-open-7b    | A       | 0.728788 |   0.35947  |   0.422857 | 0.426471 | 0.308373 | 8000 |

_Track A = white-box (constrained scoring); Track B = black-box (sampling, N=10 at T=0.7)._
