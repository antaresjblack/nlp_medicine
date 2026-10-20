# Small model grid search (dev, with dictionary span features)

| C | class_weight | Entity P | Entity R | Entity F1 | Token F1 | Char-multi F1 | Predicted | Runtime (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | balanced | 0.3896 | 0.4799 | 0.4301 | 0.5241 | 0.5373 | 25004 | 84.0 |
| 0.1 | none | 0.4290 | 0.4277 | 0.4283 | 0.5119 | 0.5232 | 20240 | 66.8 |
| 1.0 | balanced | 0.3816 | 0.4712 | 0.4217 | 0.5226 | 0.5357 | 25064 | 236.6 |
| 1.0 | none | 0.4187 | 0.4214 | 0.4201 | 0.5109 | 0.5220 | 20432 | 110.8 |
