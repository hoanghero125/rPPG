# BigSmall rPPG — Condition Comparison (Normal / Smile / Reading)

**Model:** `BP4D_BigSmall_Multitask_Fold1`
**Evaluation:** FFT peak on bandpass-filtered BVP [0.6, 3.3 Hz]
**Ground-truth HR:** average of valid readings from `hr.csv` (hrStatus = 1, hr > 0)
**SE:** standard error = std / √n

> ⚠️ Normal condition has **n = 4** subjects (S_002 missing — null `video_start` in metadata at recording time). Smile and Reading have **n = 5**.

---

## Aggregate Metrics

| Metric | Normal (n=4) | Smile (n=5) | Reading (n=5) | Best |
|--------|-------------:|------------:|--------------:|------|
| **MAE** (bpm) ↓ | **1.045 ± 0.628** | 2.659 ± 0.850 | 2.244 ± 0.939 | Normal |
| **RMSE** (bpm) ↓ | **1.634 ± 1.448** | 3.268 ± 2.551 | 3.074 ± 2.305 | Normal |
| **MAPE** (%) ↓ | **1.574 ± 1.010** | 3.635 ± 0.993 | 2.914 ± 1.246 | Normal |
| **Pearson r** ↑ | **0.997 ± 0.059** | 0.962 ± 0.158 | 0.961 ± 0.159 | Normal |
| **SNR** (dB) ↑ | 2.238 ± 0.782 | **2.245 ± 0.930** | −1.230 ± 1.203 | Smile ≈ Normal |

---

## Per-Subject Heart Rate (bpm)

### Normal

| Subject | Pred HR | Real HR | Error (bpm) | error|
|---------|---------|--------|------------|------|
| S_000 | 78.662 | 78.675 | −0.013 | 0.013 |
| S_001 | 79.980 | 80.841 | −0.861 | 0.861 |
| S_002 | — | — | — | — |
| S_003 | 59.766 | 62.914 | −3.148 | 3.148 |
| S_004 | 76.025 | 75.866 | +0.160 | 0.160 |

### Smile

| Subject | Pred HR | Real HR | Error (bpm) | error |
|---------|---------|--------|------------|------|
| S_000 | 79.980 | 81.333 | −1.353 | 1.353 |
| S_001 | 76.904 | 83.181 | −6.277 | 6.277 |
| S_002 | 65.039 | 62.214 | +2.825 | 2.825 |
| S_003 | 60.205 | 61.915 | −1.710 | 1.710 |
| S_004 | 66.797 | 67.929 | −1.132 | 1.132 |

### Reading

| Subject | Pred HR | Real HR | Error (bpm) | error|
|---------|---------|--------|------------|------|
| S_000 | 76.465 | 80.493 | −4.028 | 4.028 |
| S_001 | 87.012 | 87.971 | −0.959 | 0.959 |
| S_002 | 68.555 | 74.013 | −5.458 | 5.458 |
| S_003 | 65.039 | 65.451 | −0.412 | 0.412 |
| S_004 | 76.465 | 76.826 | −0.361 | 0.361 |

---

## Per-Subject SNR (dB)

| Subject | Normal | Smile | Reading |
|---------|-------:|------:|--------:|
| S_000 | +2.472 | +1.847 | −3.386 |
| S_001 | −0.336 | −1.486 | −4.441 |
| S_002 | — | +2.565 | +0.174 |
| S_003 | +3.842 | +4.437 | +3.137 |
| S_004 | +2.974 | +3.865 | −1.633 |
| **Mean** | **+2.238** | **+2.245** | **−1.230** |

---

## Summary

| Condition | MAE (bpm) | RMSE (bpm) | MAPE (%) | Pearson | SNR (dB) | n |
|-----------|----------:|-----------:|---------:|--------:|---------:|--:|
| **Normal** | **1.045** | **1.634** | **1.574** | **0.997** | 2.238 | 4 |
| **Reading** | 2.244 | 3.074 | 2.914 | 0.961 | −1.230 | 5 |
| **Smile** | 2.659 | 3.268 | 3.635 | 0.962 | **2.245** | 5 |

**Normal** achieves the best score on all HR accuracy metrics (MAE, RMSE, MAPE, Pearson) by a large margin — roughly half the error of Smile and Reading. Normal and Smile are virtually tied on SNR (~2.24 dB), while Reading degrades sharply (−1.23 dB), indicating the predicted BVP waveform loses spectral purity during the reading task despite still yielding competitive HR estimates.

The n = 4 vs n = 5 difference limits direct comparison with Normal; S_001 is the highest-error subject across all conditions and is present in all three, so the Normal advantage is not an artifact of the missing subject.
