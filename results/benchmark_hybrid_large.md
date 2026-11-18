load+clean 25,182 patents: 33.3s; index build: passages 0.2s, metadata_indexes 0.8s, bm25 4.5s, embeddings 0.4s
query encoding (constant cost, excluded below): 5.5 ms

### scale x1: 25,182 patents, 747,834 passages, mode=dense
| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |
|---|---:|---:|---:|---:|:---:|
| none | 25,182 | 18.01 | 18.00 | 1.0x | yes |
| cpc=G06F (computing, ~5%) | 1,324 | 7.57 | 19.21 | 2.5x | yes |
| cpc=H01M (batteries, ~1%) | 258 | 0.83 | 20.85 | 25.1x | yes |
| cpc=B60 (vehicles, ~2%) | 439 | 2.29 | 19.96 | 8.7x | yes |
| cpc=H (electricity section) | 4,112 | 17.22 | 18.80 | 1.1x | yes |
| kw=battery | 449 | 2.41 | 38.09 | 15.8x | yes |
| kw=wireless AND network | 241 | 0.95 | 65.46 | 68.9x | yes |
| cpc=G06F + kw=memory | 220 | 1.02 | 23.23 | 22.7x | yes |
| title~'vehicle' | 660 | 3.82 | 19.04 | 5.0x | yes |
| year 2015-2017 | 3,594 | 16.65 | 18.16 | 1.1x | yes |

