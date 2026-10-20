# Small model grid search (dev)

| C | class_weight | Entity P | Entity R | Entity F1 | Token F1 | Char-multi F1 | Predicted | Runtime (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | none | 0.1872 | 0.2918 | 0.2281 | 0.5111 | 0.5178 | 31651 | 658.8 |
| 0.1 | none | 0.1900 | 0.2832 | 0.2274 | 0.5063 | 0.5130 | 30248 | 290.2 |
| 0.01 | none | 0.1710 | 0.2288 | 0.1957 | 0.4680 | 0.4749 | 27165 | 84.9 |
| 0.1 | balanced | 0.1206 | 0.3616 | 0.1809 | 0.4847 | 0.4945 | 60880 | 162.0 |
| 1.0 | balanced | 0.1161 | 0.3513 | 0.1745 | 0.4838 | 0.4933 | 61408 | 349.6 |
| 0.01 | balanced | 0.1157 | 0.3525 | 0.1742 | 0.4697 | 0.4800 | 61858 | 83.6 |
| 10.0 | balanced | 0.1157 | 0.3501 | 0.1739 | 0.4826 | 0.4919 | 61437 | 6176.0 |
| 10.0 | none | skipped | skipped | skipped | skipped | skipped | skipped | not run (see README tuning note) |
