# Baseline experiment summary

Token metrics use the documented longest-non-overlapping character projection; char-multi metrics keep all entities per character (nested included); entity metrics use strict span matching.

| Method | Entity P | Entity R | Entity F1 | Token P | Token R | Token F1 | Token accuracy | Char-multi F1 | Char-multi acc | Predicted entities | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dictionary_exact | 0.1520 | 0.5415 | 0.2374 | 0.5058 | 0.4898 | 0.4977 | 0.7281 | 0.4815 | 0.6754 | 72331 | - |
| dictionary_longest | 0.2875 | 0.4916 | 0.3628 | 0.5058 | 0.4898 | 0.4977 | 0.7281 | 0.5116 | 0.7260 | 34713 | - |
| dictionary_longest_min_count_1_min_length_1 | 0.2875 | 0.4916 | 0.3628 | 0.5058 | 0.4898 | 0.4977 | 0.7281 | 0.5116 | 0.7260 | 34713 | 26.84 |
| dictionary_longest_min_count_1_min_length_2 | 0.3496 | 0.4796 | 0.4044 | 0.5350 | 0.4800 | 0.5060 | 0.7389 | 0.5188 | 0.7369 | 27851 | 43.50 |
| dictionary_longest_min_count_1_min_length_3 | 0.4610 | 0.3126 | 0.3725 | 0.6082 | 0.3675 | 0.4581 | 0.7392 | 0.4647 | 0.7373 | 13764 | - |
| dictionary_longest_min_count_2 | 0.3067 | 0.4433 | 0.3625 | 0.5329 | 0.4208 | 0.4703 | 0.7355 | 0.4873 | 0.7342 | 29338 | - |
| dictionary_longest_min_count_3 | 0.3162 | 0.4135 | 0.3584 | 0.5424 | 0.3829 | 0.4489 | 0.7345 | 0.4664 | 0.7334 | 26546 | - |
| dictionary_longest_min_count_5 | 0.3217 | 0.3678 | 0.3432 | 0.5339 | 0.3241 | 0.4033 | 0.7234 | 0.4218 | 0.7225 | 23205 | - |
| pure_handwritten_rules | 0.0908 | 0.0312 | 0.0465 | 0.2803 | 0.1797 | 0.2190 | 0.5972 | 0.2196 | 0.5967 | 6984 | 0.84 |
| character_linear_svm_longest_projection | 0.1872 | 0.2918 | 0.2281 | 0.6021 | 0.4440 | 0.5111 | 0.7661 | 0.5178 | 0.7629 | 31651 | 250.42 |
