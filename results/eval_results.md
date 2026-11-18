Test set: 357 claims from 128 held-out patents, searched against all 640 patents (claims removed from index).

| model | R@1 | R@10 | MRR@10 | MRR@10 indep. | MRR@10 dep. |
|---|---:|---:|---:|---:|---:|
| BM25 (reference) | 0.759 | 0.874 | 0.802 | 0.901 | 0.748 |
| all-MiniLM-L6-v2 (base) | 0.658 | 0.852 | 0.724 | 0.817 | 0.673 |
| fine-tuned (1 ep, 2292 triplets) | 0.810 | 0.910 | 0.844 | 0.947 | 0.787 |

| model | cos(claim, pos) | cos(claim, hard neg) | margin | triplet acc |
|---|---:|---:|---:|---:|
| base | 0.676 | 0.649 | 0.027 | 0.594 |
| fine-tuned | 0.660 | 0.415 | 0.244 | 0.874 |

Per-query: 96 queries ranked their patent higher after tuning, 34 lower. Training took 126s.
