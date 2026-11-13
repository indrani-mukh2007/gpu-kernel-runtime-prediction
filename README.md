# GPU Kernel Runtime Prediction & SJF Scheduling

Operation-class-specific gradient-boosted trees for predicting GPU kernel execution time from operator configurations across NVIDIA T4 and P100 GPUs.

## Overview

- **Data**: 2,000 configurations across 10 operations (5 compute-bound, 5 memory-bound) profiled on T4 and P100, holding out whole operations (`sdpa_attention` and `rmsnorm`, `N = 400`) for test evaluation.
- **Models**: Separate compute-bound and memory-bound GBDT regressors compared against a single pooled global model and an analytical roofline baseline.
- **Ablation**: Drop-column retraining across 6 feature groups (`FLOPs`, `Memory Footprint`, `Roofline Estimate`, `Arithmetic Intensity`, `Input/Output Shapes`, `Data Type`).
- **Transfer**: Warm-start boosting continuation from T4 to P100 across calibration fractions (`1%` to `100%`).
- **Scheduling**: Non-preemptive discrete-event Shortest-Job-First (SJF) queue simulator (`300` queues, `40` jobs/queue).

## Summary Results

| Model (NVIDIA T4 Held-Out Ops) | Overall MAPE | Compute-Bound (`sdpa_attention`) | Memory-Bound (`rmsnorm`) |
|---|---|---|---|
| Class-Specific GBDT | 8.94% | 8.71% | 9.16% |
| Global GBDT | 14.05% | 12.80% | 15.30% |
| Analytical Roofline | 26.95% | 20.86% | 33.04% |

- **Cross-GPU Transfer (`T4 -> P100`)**: Zero-shot gives `31.14%` MAPE; warm-start fine-tuning on `10%` P100 data (`160` configs) reaches `10.86%` MAPE.
- **SJF Scheduling**: Class-specific SJF reduces mean JCT by `11.67%` (`~12%`) vs. roofline-guided SJF (global model: `5.95%`) and captures `60.88%` (`~60%`) of the `19.68%` (`~20%`) Oracle SJF gain over FIFO.

## Running

```bash
pip install -r requirements.txt
python run_all.py
```
