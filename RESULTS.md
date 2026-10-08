# Results and provenance

Copyright 2026 Parth Maniar. Apache-2.0.
[GitHub](https://github.com/officialpm) | [Portfolio](https://www.parthmaniar.tech/)

Measured V3 run 356231139 completed October 7, 2026: 37.7 seconds total, 15.10 seconds analysis, CPU only. It used 3,000,888 training rows, 28,512 test rows, 1,782 store-family series, and five validation windows of 28,512 rows each. No rows or zero-sales series were dropped.

| Window | 56-day seasonal log mean | 56-day promotion ridge | Difference |
|---|---:|---:|---:|
| May 28 - June 12, 2017 | 0.510024 | 0.470289 | -0.039734 |
| June 13 - June 28, 2017 | 0.442093 | 0.408833 | -0.033260 |
| June 29 - July 14, 2017 | 0.405931 | 0.396280 | -0.009652 |
| July 15 - July 30, 2017 | 0.416972 | 0.412081 | -0.004892 |
| July 31 - August 15, 2017 | 0.528593 | 0.489325 | -0.039268 |
| Mean RMSLE | 0.460723 | 0.435361 | -0.025362 |

Other five-window means: mean28 0.490619, weekday56 0.478226, weekday_log112 0.527057, promo_ridge112 0.463440. The longer promotion model wins the latest window at 0.471482. There is no formal uncertainty estimate or untouched final holdout here. Weekday regularization also differs slightly from the unpenalized seasonal control, so this is not a pure causal promotion ablation.

The exact selected CSV was submitted once October 7. Public RMSLE was 0.40508 vs the prior recorded 0.41602, a reduction of 0.01094. Public and development metrics are separate. This ongoing benchmark has no private leaderboard; no final ranking is claimed.

Original artifact: 724,703 bytes, recorded SHA256 `76006152c8f23b10acb1d092101bfaa5a7e75ef4103c417b6875227c3882f53e`. This is the run's recorded hash, not an independent rehash in this repository. Prediction CSVs and raw data are intentionally not published.
