index build (incl. cached embeddings): 3.6s
query encoding (constant cost, excluded below): 5.3 ms

### scale x1: 640 patents, 32,298 passages, mode=dense
| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |
|---|---:|---:|---:|---:|:---:|
| none | 640 | 0.78 | 0.76 | 1.0x | yes |
| cpc=B60B (wheels, ~45%) | 298 | 0.81 | 0.82 | 1.0x | yes |
| cpc=B60D (couplings, ~3%) | 20 | 0.09 | 1.00 | 11.1x | yes |
| kw=tire | 315 | 0.87 | 1.53 | 1.8x | yes |
| kw=wheel AND hub | 100 | 0.77 | 2.54 | 3.3x | yes |
| cpc=B60C + kw=tread | 150 | 0.79 | 1.08 | 1.4x | yes |
| title~'wheel' | 189 | 0.77 | 0.77 | 1.0x | yes |

### scale x10: 6,400 patents, 322,980 passages, mode=dense
| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |
|---|---:|---:|---:|---:|:---:|
| none | 6,400 | 7.37 | 7.33 | 1.0x | yes |
| cpc=B60B (wheels, ~45%) | 2,980 | 7.75 | 8.01 | 1.0x | yes |
| cpc=B60D (couplings, ~3%) | 200 | 1.13 | 9.09 | 8.0x | yes |
| kw=tire | 3,150 | 7.10 | 8.61 | 1.2x | yes |
| kw=wheel AND hub | 1,000 | 7.25 | 11.61 | 1.6x | yes |
| cpc=B60C + kw=tread | 1,500 | 9.40 | 10.02 | 1.1x | yes |
| title~'wheel' | 1,890 | 8.31 | 7.53 | 0.9x | yes |

### scale x30: 19,200 patents, 968,940 passages, mode=dense
| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |
|---|---:|---:|---:|---:|:---:|
| none | 19,200 | 23.36 | 22.19 | 1.0x | yes |
| cpc=B60B (wheels, ~45%) | 8,940 | 22.65 | 22.39 | 1.0x | yes |
| cpc=B60D (couplings, ~3%) | 600 | 5.08 | 25.24 | 5.0x | yes |
| kw=tire | 9,450 | 21.87 | 25.33 | 1.2x | yes |
| kw=wheel AND hub | 3,000 | 21.39 | 29.86 | 1.4x | yes |
| cpc=B60C + kw=tread | 4,500 | 22.73 | 23.13 | 1.0x | yes |
| title~'wheel' | 5,670 | 22.46 | 22.24 | 1.0x | yes |

