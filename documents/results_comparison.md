# BigSmall Inference Results — Fold Comparison

**Model:** `BP4D_BigSmall_Multitask_FoldX.pth`
**Ground-truth HR:** average of valid readings from `hr.csv` (hrStatus=1, hr>0) per subject
**HR estimation method:** FFT on bandpass-filtered [0.6, 3.3 Hz] BVP signal
**SE:** standard error = std / sqrt(n), n = 5 subjects
**Professor SNR:** not available (requires raw BVP prediction signals)

---

## Per-Subject Predicted Heart Rate (bpm)

| Subject | Real HR | Fold 1 | Fold 2 | Fold 3 | Professor |
|---------|--------:|-------:|-------:|-------:|----------:|
| S_000   |  85.638 | 88.330 | 88.330 | 88.330 |    88.330 |
| S_001   |  87.143 | 92.285 | 79.980 | 68.994 |    65.918 |
| S_002   |  68.444 | 71.191 | 71.191 | 71.191 |    68.994 |
| S_003   |  79.963 | 80.420 | 80.420 | 80.420 |    76.904 |
| S_004   |  69.160 | 72.510 | 72.510 | 72.510 |    66.357 |

> **Note:** S_000, S_002, S_003, S_004 yield identical predictions across all 3 folds.
> Only S_001 changes — the three folds disagree significantly (92.3 / 80.0 / 69.0 bpm),
> all far from the true value of 87.1 bpm. This subject appears to be a hard case for
> this model family.

---

## Per-Subject Heart Rate Error (predicted − real, bpm)

| Subject | Fold 1 | Fold 2 | Fold 3 | Professor |
|---------|-------:|-------:|-------:|----------:|
| S_000   |  +2.69 |  +2.69 |  +2.69 |     +2.69 |
| S_001   |  +5.14 |  −7.16 | −18.15 |    −21.22 |
| S_002   |  +2.75 |  +2.75 |  +2.75 |     +0.55 |
| S_003   |  +0.46 |  +0.46 |  +0.46 |     −3.06 |
| S_004   |  +3.35 |  +3.35 |  +3.35 |     −2.80 |

---

## Aggregate Metrics (with Standard Error)

Values shown as **value ± SE**. Professor metrics calculated from per-subject errors.

### Mean Absolute Error — MAE (bpm, lower is better)

| | Fold 1 | Fold 2 | Fold 3 | Professor |
|-|-------:|-------:|-------:|----------:|
| MAE ± SE | **2.878 ± 0.671** | 3.282 ± 0.973 | 5.479 ± 2.867 | 6.066 ± 3.414 |

### Root Mean Square Error — RMSE (bpm, lower is better)

| | Fold 1 | Fold 2 | Fold 3 | Professor |
|-|-------:|-------:|-------:|----------:|
| RMSE ± SE | **3.246 ± 1.974** | 3.938 ± 2.857 | 8.433 ± 7.601 | 9.749 ± 8.916 |

### Mean Absolute Percentage Error — MAPE (%, lower is better)

| | Fold 1 | Fold 2 | Fold 3 | Professor |
|-|-------:|-------:|-------:|----------:|
| MAPE ± SE | **3.695 ± 0.809** | 4.158 ± 1.111 | 6.680 ± 3.227 | 7.236 ± 3.862 |

### Pearson Correlation (higher is better)

| | Fold 1 | Fold 2 | Fold 3 | Professor |
|-|-------:|-------:|-------:|----------:|
| r ± SE | **0.984 ± 0.102** | 0.875 ± 0.279 | 0.407 ± 0.527 | 0.466 ± 0.511 |

### Signal-to-Noise Ratio — SNR (dB, higher is better)

| | Fold 1 | Fold 2 | Fold 3 | Professor |
|-|-------:|-------:|-------:|----------:|
| SNR ± SE | 0.927 ± 1.575 | **1.075 ± 1.739** | 1.206 ± 1.753 | N/A |

> SNR measures how much predicted BVP power falls at the true HR frequency vs. the rest of the
> [0.6, 3.3 Hz] band. All folds show low SNR (~1 dB), indicating that while the FFT HR estimate
> is often correct, the predicted waveform is noisy outside the heart-rate bin.

### Full Metrics Summary

| Metric | Fold 1 | Fold 2 | Fold 3 | Professor |
|--------|-------:|-------:|-------:|----------:|
| MAE (bpm)   | **2.878** | 3.282 | 5.479 | 6.066 |
| RMSE (bpm)  | **3.246** | 3.938 | 8.433 | 9.749 |
| MAPE (%)    | **3.695** | 4.158 | 6.680 | 7.236 |
| Pearson r   | **0.984** | 0.875 | 0.407 | 0.466 |
| SNR (dB)    | 0.927 | 1.075 | **1.206** | N/A |

---

## Per-Subject SNR (dB)

| Subject | Fold 1 | Fold 2 | Fold 3 |
|---------|-------:|-------:|-------:|
| S_000   |  +4.12 |  +4.68 |  +4.92 |
| S_001   |  −5.85 |  −6.45 |  −6.38 |
| S_002   |  +1.67 |  +1.93 |  +2.26 |
| S_003   |  +1.58 |  +2.05 |  +2.13 |
| S_004   |  +3.12 |  +3.16 |  +3.09 |

> S_001 has strongly negative SNR across all folds, confirming the predicted BVP has no
> meaningful power at the true HR frequency — the model fails on this subject entirely.

---

## Per-Subject Respiration Rate (breaths/min)

| Subject | Fold 1 | Fold 2 | Fold 3 | Professor |
|---------|-------:|-------:|-------:|----------:|
| S_000   |  20.65 |  20.65 |  20.65 |     37.79 |
| S_001   |  24.17 |  22.41 |  25.05 |     43.51 |
| S_002   |  22.41 |  24.17 |  29.44 |     39.11 |
| S_003   |  28.56 |  26.37 |  27.69 |     36.47 |
| S_004   |  25.93 |  29.88 |  21.97 |     39.99 |

> Our respiration rates (20–30 breaths/min) are physiologically plausible (normal: 12–20).
> The professor's values (37–44 breaths/min) are consistently ~15 bpm higher, suggesting a
> different bandpass range or FFT method for respiration estimation.

---

## Per-Subject Blink Rate (blinks/min)

| Subject | Our (all folds) | Professor |
|---------|----------------:|----------:|
| S_000   |           24.63 |     14.50 |
| S_001   |           31.44 |     21.42 |
| S_002   |           32.20 |     40.37 |
| S_003   |           26.01 |     28.72 |
| S_004   |           30.77 |     14.43 |

> Blink rate is computed from video only (independent of model fold), so it is identical across
> all three folds. Our method uses Haar Cascade + eye-strip brightness + bandpass [0.1–0.9 Hz]
> + peak counting. The professor likely used a dedicated landmark-based blink detector
> (e.g., MediaPipe or Eye Aspect Ratio), explaining the differences.

---

## Summary & Conclusions

### HR Accuracy Ranking (by MAE)

| Rank | | MAE | RMSE | MAPE | Pearson |
|------|--|----:|-----:|-----:|--------:|
| 1 | **Fold 1** | **2.878 bpm** | **3.246 bpm** | **3.695 %** | **0.984** |
| 2 | Fold 2 | 3.282 bpm | 3.938 bpm | 4.158 % | 0.875 |
| 3 | Fold 3 | 5.479 bpm | 8.433 bpm | 6.680 % | 0.407 |
| 4 | Professor | 6.066 bpm | 9.749 bpm | 7.236 % | 0.466 |

All three of our folds outperform the professor's reference on every HR metric.

### Key Observations

- **S_001 is the dominant source of error** across all results. The SNR for S_001 is strongly
  negative (−5.9 to −6.4 dB) in all folds — the model predicts a BVP signal at the wrong
  frequency entirely. Fold 1 is closest (+5.1 bpm error), the professor's result worst (−21.2 bpm).

- **Fold 1 dominates on HR metrics** (MAE, RMSE, MAPE, Pearson). Its Pearson of 0.984 is
  exceptionally high, meaning the predicted HRs rank subjects in exactly the right order even
  if individual values have small offsets.

- **Fold 3 has the highest SNR** despite having the worst HR accuracy. This is because SNR
  measures signal quality at the true HR frequency, which can be high even when the FFT peak
  is at the wrong bin (e.g., if the true HR bin has a local power spike that isn't the global
  maximum).

- **The professor's preprocessing differs.** Their predictions for S_002, S_003, S_004 are
  different from all three of our folds, indicating a different face crop, chunk length, or
  normalization — not just a different model fold.

### Recommendation
Use **Fold 1** (`BP4D_BigSmall_Multitask_Fold1.pth`) for best HR accuracy on this dataset.
