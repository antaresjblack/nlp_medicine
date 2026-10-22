# Small model tune results

- Created: 2026-10-22T14:30:40+08:00
- Selection metric: `strict_entity_micro_f1`
- Selection split: `eval_tune`

| Rank | Candidate | C | class_weight | Dict span | Entity P | Entity R | Entity F1 | Token F1 | Char-multi F1 | Runtime (s) |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `span_c0.1_none` | 0.1 | none | true | 0.4276 | 0.4275 | 0.4276 | 0.5072 | 0.5187 | 48.9 |
| 2 | `span_c0.1_balanced` | 0.1 | balanced | true | 0.3858 | 0.4778 | 0.4269 | 0.5184 | 0.5318 | 55.7 |
| 3 | `span_c1.0_none` | 1.0 | none | true | 0.4168 | 0.4210 | 0.4189 | 0.5068 | 0.5181 | 73.9 |
| 4 | `span_c1.0_balanced` | 1.0 | balanced | true | 0.3776 | 0.4693 | 0.4185 | 0.5172 | 0.5305 | 144.3 |
| 5 | `no_span_c1.0_none` | 1.0 | none | false | 0.1842 | 0.2912 | 0.2257 | 0.5117 | 0.5175 | 172.7 |
| 6 | `no_span_c0.1_none` | 0.1 | none | false | 0.1875 | 0.2829 | 0.2255 | 0.5073 | 0.5134 | 108.5 |
| 7 | `no_span_c0.1_balanced` | 0.1 | balanced | false | 0.1179 | 0.3541 | 0.1769 | 0.4852 | 0.4938 | 71.6 |
| 8 | `no_span_c1.0_balanced` | 1.0 | balanced | false | 0.1134 | 0.3437 | 0.1705 | 0.4842 | 0.4925 | 244.0 |

Selected: `span_c0.1_none`
