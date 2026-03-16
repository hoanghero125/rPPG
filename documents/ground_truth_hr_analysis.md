# Phân tích Ground Truth HR trong bài toán rPPG

## 1. Bối cảnh

Trong quá trình đánh giá các mô hình rPPG trên tập PPG_Dataset, chúng ta sử dụng **device HR** (nhịp tim trung bình từ smartwatch, đọc từ `hr.csv`) làm ground truth. Tuy nhiên, cách làm chuẩn trong nghiên cứu rPPG lại khác.

Tài liệu này phân tích sự khác biệt giữa hai cách tiếp cận và lý do nên chuyển sang phương pháp chuẩn.

---

## 2. Hai nguồn Ground Truth HR

### 2.1. Device HR (cách hiện tại của chúng ta)

**Nguồn**: File `hr.csv` từ smartwatch/cảm biến đeo tay.

**Cách tính**: Lấy trung bình tất cả các giá trị HR hợp lệ (`hrStatus == 1` và `hr > 0`) trong toàn bộ phiên thu.

```python
# Trong notebook hiện tại
def read_gt_hr(session_path):
    df = pd.read_csv(hr_path, usecols=[0,1,2,3,4], ...)
    valid = df[(df["hrStatus"] == 1) & (df["hr"] > 0)]["hr"]
    return float(valid.mean())  # <-- một giá trị scalar duy nhất cho cả video
```

**Đặc điểm**:
- Là **một giá trị trung bình** cho toàn bộ video (ví dụ: 83.54 bpm)
- Không phản ánh sự biến thiên HR trong video
- Phụ thuộc vào thuật toán nội bộ của smartwatch (black-box)
- Có thể bị lệch nếu smartwatch mất tín hiệu hoặc có nhiễu

### 2.2. Label HR / BVP-derived HR (cách chuẩn trong nghiên cứu)

**Nguồn**: Tín hiệu PPG dạng sóng (waveform) từ cảm biến tiếp xúc, được lưu trong `ppg.csv` (kênh green).

**Cách tính**: Áp dụng cùng một pipeline xử lý tín hiệu cho cả predicted BVP và ground truth BVP:

```
BVP signal (T mẫu)
  --> Detrend (loại bỏ drift, lambda=100)
  --> Bandpass filter (0.6-3.3 Hz = 36-198 bpm)
  --> FFT (tìm tần số chủ đạo)
  --> HR = f_peak * 60 (bpm)
```

```python
# Trong rPPG-Toolbox chuẩn (evaluation/post_process.py)
def calculate_metric_per_video(predictions, labels, fs, hr_method='FFT'):
    # Cả predictions và labels đều đi qua CÙNG pipeline
    predictions = detrend(predictions)
    labels = detrend(labels)

    predictions = bandpass_filter(predictions, 0.6, 3.3)
    labels = bandpass_filter(labels, 0.6, 3.3)

    hr_pred  = fft_hr(predictions, fs)   # HR từ model prediction
    hr_label = fft_hr(labels, fs)        # HR từ BVP ground truth
    # --> So sánh hr_pred vs hr_label
```

**Đặc điểm**:
- HR được tính **per-chunk** hoặc **per-window** (ví dụ: mỗi 6 giây)
- Phản ánh HR tại thời điểm cụ thể, không phải trung bình cả video
- Đánh giá công bằng: cùng pipeline cho cả prediction và ground truth
- Cho phép tính các metric dạng waveform (Pearson correlation, MACC)

---

## 3. Tại sao nghiên cứu rPPG dùng BVP-derived HR?

### 3.1. Tính đối xứng trong đánh giá

Nguyên tắc cốt lõi: **prediction và ground truth phải đi qua cùng một pipeline xử lý tín hiệu**.

```
Predicted BVP ──> [Detrend] ──> [Bandpass] ──> [FFT] ──> HR_pred
                                                              |
                                                         So sánh
                                                              |
Label BVP     ──> [Detrend] ──> [Bandpass] ──> [FFT] ──> HR_label
```

Nếu dùng device HR, ta phá vỡ tính đối xứng này — prediction được xử lý bằng FFT nhưng ground truth là một scalar từ thuật toán khác hoàn toàn.

### 3.2. Độ phân giải thời gian

| Phương pháp | Độ phân giải | Ví dụ (video 60s, 30fps) |
|-------------|-------------|--------------------------|
| Device HR | Toàn bộ video | 1 giá trị: 83.54 bpm |
| BVP-derived HR (chunk=180 frames) | Mỗi 6 giây | 10 giá trị HR, mỗi chunk một giá trị |
| BVP-derived HR (window=10s) | Mỗi 10 giây | 6 giá trị HR |

Device HR là trung bình cả phiên, nên nếu HR thay đổi trong video (ví dụ: từ 70 lên 90 bpm), giá trị trung bình 80 bpm sẽ **sai** cho mọi thời điểm cụ thể.

### 3.3. Cách các dataset chuẩn lưu ground truth

Tất cả các dataset lớn trong lĩnh vực rPPG đều lưu **tín hiệu BVP dạng sóng**, không phải scalar HR:

| Dataset | Ground Truth | Định dạng |
|---------|-------------|-----------|
| UBFC-rPPG | BVP waveform từ finger pulse oximeter | `.txt` (chuỗi giá trị) |
| PURE | BVP waveform + HR + SpO2 | `.json` |
| SCAMPS | BVP tổng hợp (perfect) | `.mat` |
| MMPD | BVP waveform từ cảm biến cổ tay | `.mat` |
| BP4D+ | BVP + respiration + action units | `.mat` + `.csv` |
| iBVP | BVP với quality labels | `.csv` |

Lý do: BVP waveform chứa nhiều thông tin hơn scalar HR — cho phép đánh giá cả hình dạng sóng, không chỉ tần số.

### 3.4. Trong rPPG-Toolbox chính thức

File `unsupervised_methods/unsupervised_predictor.py` cho thấy rõ:

```python
# Line 64-73: cả gt_hr và pre_hr đều được tính từ BVP waveform
gt_hr, pre_hr, SNR, macc = calculate_metric_per_video(
    BVP_window,      # predicted BVP (T,)
    label_window,    # ground truth BVP (T,)  <-- KHÔNG phải device HR
    diff_flag=False,
    fs=config.UNSUPERVISED.DATA.FS,
    hr_method='FFT'
)
```

Bland-Altman plots cũng ghi rõ label: `'GT PPG HR [bpm]'` — tức HR trích xuất từ PPG, không phải device HR.

---

## 4. Vấn đề cụ thể trong notebook hiện tại

### 4.1. `hr_label` bị bỏ phí

Trong hàm `process_bvp()`, ta đã tính cả `hr_label` (FFT từ BVP label) nhưng **không dùng**:

```python
def process_bvp(pred_chunks, label_chunks, fs=30, diff_flag=False):
    ...
    hr_pred  = fft_peak_hz(pred_processed,  fs, 0.6, 3.3) * 60.0
    hr_label = fft_peak_hz(label_processed, fs, 0.6, 3.3) * 60.0  # <-- tính rồi bỏ
    ...
    return hr_pred, hr_label, snr_db, pred_processed
```

Nhưng trong vòng lặp đánh giá:

```python
hr_pred, hr_label, _, pred_processed = process_bvp(...)  # hr_label bị bỏ qua
gt_hr = subjects_by_key[subj_key]["gt_hr"]               # dùng device HR thay thế
hr_err = hr_pred - gt_hr                                  # so sánh với device HR
```

### 4.2. SNR cũng dùng sai tham chiếu

```python
# Hiện tại: SNR tính dựa trên device HR
snr_db = calculate_snr(pred_processed, gt_hr, FS)

# Chuẩn: SNR nên tính dựa trên HR từ label BVP
snr_db = calculate_snr(pred_processed, hr_label, FS)
```

SNR đo năng lượng tín hiệu tại tần số HR và các harmonic. Nếu tần số tham chiếu (device HR) lệch so với tần số thực trong tín hiệu, kết quả SNR sẽ bị sai.

---

## 5. So sánh hai cách tính trên dữ liệu thực

Ví dụ từ model `UBFC-rPPG_RhythmFormer` trên PPG_Dataset:

| Subject | HR_pred (bpm) | Device HR (bpm) | Error vs Device |
|---------|--------------|----------------|-----------------|
| S_000 | 82.18 | 83.54 | -1.36 |
| S_001 | 80.42 | 80.40 | +0.02 |
| S_002 | 67.68 | 122.48 | **-54.80** |
| S_003 | 78.66 | 114.92 | **-36.26** |
| S_004 | 78.66 | 79.12 | -0.46 |

Với S_002 và S_003, error rất lớn. Nhưng câu hỏi quan trọng: **lỗi này do model predict sai, hay do device HR không phản ánh đúng HR trong video?**

Nếu dùng BVP-derived HR (`hr_label` từ FFT của PPG green), ta có thể kiểm chứng:
- Nếu `hr_label` cũng gần 122 bpm → model thực sự predict sai
- Nếu `hr_label` gần 67 bpm → device HR bị lệch, model predict đúng, nhưng metric bị phạt oan

Đây chính là lý do BVP-derived HR đáng tin cậy hơn device HR trong đánh giá.

---

## 6. Kết luận

| Tiêu chí | Device HR (`hr.csv`) | BVP-derived HR (FFT từ `ppg.csv`) |
|----------|---------------------|-----------------------------------|
| Chuẩn nghiên cứu | Không | **Có** |
| Độ phân giải thời gian | Toàn video (1 giá trị) | **Per-chunk / per-window** |
| Tính đối xứng đánh giá | Không (khác pipeline) | **Có (cùng pipeline)** |
| Phản ánh HR tại thời điểm video | Không (trung bình) | **Có** |
| Cho phép metric waveform (MACC, Pearson signal-level) | Không | **Có** |
| Dùng trong rPPG-Toolbox chính thức | Không | **Có** |

**Khuyến nghị**: Chuyển sang dùng `hr_label` (FFT từ PPG green BVP) làm `real_heartrate` trong tất cả các notebook, thay vì `gt_hr` từ `hr.csv`. Code đã có sẵn trong `process_bvp()` — chỉ cần dùng giá trị `hr_label` thay vì bỏ qua nó.
