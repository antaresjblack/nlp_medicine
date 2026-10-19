# Baseline experiment summary

Token metrics use the documented longest-non-overlapping character projection; entity metrics use strict span matching.

| Method | Entity P | Entity R | Entity F1 | Token F1 | Token accuracy | Predicted entities | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dictionary_exact | 0.1520 | 0.5415 | 0.2374 | 0.4977 | 0.7281 | 72331 | - |
| dictionary_longest | 0.2875 | 0.4916 | 0.3628 | 0.4977 | 0.7281 | 34713 | - |
| dictionary_longest_min_count_1_min_length_2 | 0.3496 | 0.4796 | 0.4044 | 0.5060 | 0.7389 | 27851 | - |
| dictionary_longest_min_count_1_min_length_3 | 0.4610 | 0.3126 | 0.3725 | 0.4581 | 0.7392 | 13764 | - |
| dictionary_longest_min_count_2 | 0.3067 | 0.4433 | 0.3625 | 0.4703 | 0.7355 | 29338 | - |
| dictionary_longest_min_count_3 | 0.3162 | 0.4135 | 0.3584 | 0.4489 | 0.7345 | 26546 | - |
| dictionary_longest_min_count_5 | 0.3217 | 0.3678 | 0.3432 | 0.4033 | 0.7234 | 23205 | - |
| small_model | 0.1161 | 0.3513 | 0.1745 | 0.4838 | 0.6976 | 61408 | - |
