# GPU Closure v10: tail-risk and key-control audit

All gains are macro-F1 percentage points. The dataset is the independent generalization unit; origin and seed are nested repeated measurements. Bootstrap intervals resample origins and carry all nested seeds together.

## Tail-risk table

| Group | Cells | Coverage | Served mean | Deployed min | Deployed q05 | Lower CVaR20 | Mild negative | Harmful < -0.5 | Missed > +0.5 | Worst regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2_unified_adapter_baselines::jigsaw_civil_comments_temporal_external | 60 | 38.3% | 0.103 | -0.558 | -0.258 | -0.250 | 6 | 1 | 0 | 0.558 |
| S2_unified_adapter_baselines::spatiotemporal_news_location_external | 60 | 55.0% | 0.725 | -0.318 | -0.104 | -0.011 | 4 | 0 | 1 | 0.530 |
| S2_unified_adapter_baselines::swiss_judgment_de_temporal_external | 60 | 11.7% | 0.426 | 0.000 | 0.124 | 0.206 | 0 | 0 | 0 | 0.000 |
| S3_cross_backbone_controls::jigsaw_civil_comments_temporal_external | 36 | 63.9% | 0.323 | -0.473 | -0.310 | -0.236 | 6 | 0 | 0 | 0.473 |
| S3_cross_backbone_controls::spatiotemporal_news_location_external | 36 | 44.4% | 1.012 | -0.185 | -0.046 | 0.061 | 1 | 0 | 0 | 0.185 |
| S3_cross_backbone_controls::swiss_judgment_de_temporal_external | 36 | 2.8% | 0.020 | 0.719 | 0.719 | 0.719 | 0 | 0 | 0 | 0.000 |
| S4_budget_and_geometry_sensitivity::jigsaw_civil_comments_temporal_external | 120 | 25.8% | 0.006 | -0.459 | -0.331 | -0.243 | 15 | 0 | 0 | 0.459 |
| S4_budget_and_geometry_sensitivity::spatiotemporal_news_location_external | 120 | 81.7% | 0.881 | -0.580 | -0.045 | -0.003 | 5 | 1 | 0 | 0.580 |
| S5_multieurlex_fourth_untouched_external::multieurlex_en_level1_temporal_external | 72 | 100.0% | 3.486 | 1.589 | 1.658 | 1.710 | 0 | 0 | 0 | 0.000 |

## Frozen-base calibration controls

These controls were specified after MultiEURLEX deployment outcomes were opened. They diagnose attribution; they are not a second prospective confirmation.

| Control | Coverage | Served mean | Deployed min | Harmful |
|---|---:|---:|---:|---:|
| global_threshold | 100.0% | 2.805 | 1.676 | 0 |
| per_label_thresholds | 100.0% | 4.050 | 1.799 | 0 |

## Interpretation boundary

- A zero harmful count at the preregistered -0.5 pp threshold does not imply that every deployment is non-negative; inspect the mild-negative column and the deployed minimum.
- Cell-level bootstrap intervals describe repeated origins and seeds. They do not turn one dataset into multiple independent external replications.
- S1--S4 and the frozen-base controls are post-outcome diagnostics. S5 remains the only newly locked prospective dataset in v10.
