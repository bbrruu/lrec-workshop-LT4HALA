# 0331 Parameter Test Summary

## Baselines
- Global baseline: K=20, merge_ratio=0.95, split_ratio=0.1
- Local baseline: K=20, merge_ratio=0.95, split_ratio=0.3

## Best Configs
- Global best: run_id=global_k25_m0p95_s0p10, silhouette=0.053998, final_k=16.00, cfg=(K=25, merge=0.95, split=0.1)
- Local best (weighted by dynasty n): run_id=local_k20_m0p90_s0p30, silhouette=0.050103, weighted_final_k=5.95, cfg=(K=20, merge=0.9, split=0.3)

## Baseline vs Best
- Global silhouette delta: +0.004802
- Local silhouette delta: +0.005305

## Artifacts
- global_results.csv
- local_results.csv
- global_silhouette.png
- global_k_distribution.png
- local_silhouette.png
- local_k_distribution.png

## Top 10 Global
| run_id | group | group_value | k | merge_ratio | split_ratio | silhouette | final_k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global_k25_m0p95_s0p10 | K | 25 | 25 | 0.95 | 0.1 | 0.053998 | 16.0000 |
| global_k15_m0p95_s0p10 | K | 15 | 15 | 0.95 | 0.1 | 0.051201 | 17.0000 |
| global_k20_m0p90_s0p10 | merge_ratio | 0.9 | 20 | 0.9 | 0.1 | 0.050589 | 6.0000 |
| global_k20_m0p95_s0p30 | split_ratio | 0.3 | 20 | 0.95 | 0.3 | 0.049304 | 12.0000 |
| global_k20_m0p95_s0p50 | split_ratio | 0.5 | 20 | 0.95 | 0.5 | 0.049304 | 12.0000 |
| global_k20_m0p95_s0p10 | K | 20 | 20 | 0.95 | 0.1 | 0.049196 | 17.0000 |
| global_k20_m0p95_s0p10 | merge_ratio | 0.95 | 20 | 0.95 | 0.1 | 0.049196 | 17.0000 |
| global_k20_m0p95_s0p10 | split_ratio | 0.1 | 20 | 0.95 | 0.1 | 0.049196 | 17.0000 |
| global_k20_m0p97_s0p10 | merge_ratio | 0.97 | 20 | 0.97 | 0.1 | 0.042886 | 20.0000 |
| global_k30_m0p95_s0p10 | K | 30 | 30 | 0.95 | 0.1 | 0.039968 | 22.0000 |

## Top 10 Local
| run_id | group | group_value | k | merge_ratio | split_ratio | silhouette | final_k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| local_k20_m0p90_s0p30 | merge_ratio | 0.9 | 20 | 0.9 | 0.3 | 0.050103 | 5.9509 |
| local_k10_m0p95_s0p30 | K | 10 | 10 | 0.95 | 0.3 | 0.047940 | 9.2957 |
| local_k20_m0p92_s0p30 | merge_ratio | 0.92 | 20 | 0.92 | 0.3 | 0.047920 | 7.5484 |
| local_k20_m0p95_s0p50 | split_ratio | 0.5 | 20 | 0.95 | 0.5 | 0.044802 | 13.2233 |
| local_k20_m0p95_s0p30 | K | 20 | 20 | 0.95 | 0.3 | 0.044798 | 13.2261 |
| local_k20_m0p95_s0p30 | merge_ratio | 0.95 | 20 | 0.95 | 0.3 | 0.044798 | 13.2261 |
| local_k20_m0p95_s0p30 | split_ratio | 0.3 | 20 | 0.95 | 0.3 | 0.044798 | 13.2261 |
| local_k30_m0p95_s0p30 | K | 30 | 30 | 0.95 | 0.3 | 0.044705 | 15.8702 |
| local_k15_m0p95_s0p30 | K | 15 | 15 | 0.95 | 0.3 | 0.044614 | 10.8325 |
| local_k20_m0p97_s0p30 | merge_ratio | 0.97 | 20 | 0.97 | 0.3 | 0.043156 | 18.8948 |
