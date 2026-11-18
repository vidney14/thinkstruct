load+clean 25,182 patents: 33.8s; index build: passages 0.2s, metadata_indexes 0.8s, bm25 4.5s, embeddings 1894.4s
query encoding (constant cost, excluded below): 4.8 ms

### scale x1: 25,182 patents, 747,834 passages, mode=dense
| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |
|---|---:|---:|---:|---:|:---:|
| none | 25,182 | 18.72 | 18.06 | 1.0x | yes |
| cpc=G06F (computing, ~5%) | 1,324 | 7.88 | 19.26 | 2.4x | yes |
| cpc=H01M (batteries, ~1%) | 258 | 0.87 | 22.04 | 25.4x | yes |
| cpc=B60 (vehicles, ~2%) | 439 | 3.42 | 22.25 | 6.5x | yes |
| cpc=H (electricity section) | 4,112 | 18.89 | 20.20 | 1.1x | yes |
| kw=battery | 449 | 3.17 | 45.57 | 14.4x | yes |
| kw=wireless AND network | 241 | 1.32 | 72.94 | 55.3x | yes |
| cpc=G06F + kw=memory | 220 | 1.12 | 23.08 | 20.6x | yes |
| title~'vehicle' | 660 | 3.62 | 19.04 | 5.3x | yes |
| year 2015-2017 | 3,594 | 23.40 | 18.81 | 0.8x | yes |

