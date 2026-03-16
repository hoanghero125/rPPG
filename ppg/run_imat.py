#!/usr/bin/env python
"""Run IMAT denoising pipeline for FactorizePhys inference.

IMAT core reimplemented inline to avoid NumPy 1.22 compatibility issues
with the original GalaxyPPG code (np.linalg.eig produces complex arrays
that trigger memory corruption). Uses np.linalg.eigh instead.
"""
import sys, os, json, glob, shutil
import numpy as np
import pandas as pd
import cv2
from scipy import signal
from scipy.signal import periodogram
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = "/home/naver/disk2/HoangDPB/rPPG-Toolbox"
sys.path.insert(0, REPO_ROOT)
from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys

# ---- Config ----
RAW_DATA_PATH     = os.path.join(REPO_ROOT, "raw_data/5_demo")
PREPROCESSED_PATH = os.path.join(REPO_ROOT, "preprocessed_data/ppg/5_demo/UBFC-rPPG_FactorizePhys")
OUTPUT_DIR        = os.path.join(REPO_ROOT, "results/ppg/5_demo/UBFC-rPPG_FactorizePhys")
VIDEO_FPS = 30; PPG_FS = 25; CHUNK_LENGTH = 160; IMG_H = IMG_W = 72
LABEL_TYPE = "Standardized"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_LABEL = "UBFC-rPPG_FactorizePhys"
MODEL_PATH  = os.path.join(REPO_ROOT, "final_model_release/UBFC-rPPG_FactorizePhys_FSAM_Res.pth")
os.makedirs(PREPROCESSED_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# IMAT Denoising — Reimplemented (Mashhadi et al. 2015)
# Core: SVD reference generation + LMS adaptive filter
# Uses np.linalg.eigh (safe) instead of np.linalg.eig (crashes on NumPy 1.22)
# =====================================================================

class IMATDenoiser:
    """IMAT motion artifact removal for wearable PPG signals."""

    def __init__(self, fs=25, target_length=200):
        self.fs = fs
        self.target_length = target_length
        self.window_count = 0
        self.Nprv = 0.0
        self.previous_ppg = None
        self.previous_acc_x = None
        self.previous_acc_y = None
        self.previous_acc_z = None

    def process_window(self, ppg, acc_x, acc_y, acc_z):
        """Process one 8-second window. Returns (denoised_signal, bpm)."""
        tl = self.target_length

        def pad_or_trim(sig):
            if len(sig) < tl:
                return np.pad(sig, (0, tl - len(sig)), mode='edge')
            return sig[:tl]

        self.window_count += 1
        ppg = np.array(pad_or_trim(ppg), dtype=np.float64)
        acc_x = np.array(pad_or_trim(acc_x), dtype=np.float64)
        acc_y = np.array(pad_or_trim(acc_y), dtype=np.float64)
        acc_z = np.array(pad_or_trim(acc_z), dtype=np.float64)

        if self.window_count == 1:
            self.Nprv = self._init_hr(ppg) * 4
            x3 = ppg.copy()
            self.previous_ppg = ppg.copy()
            self.previous_acc_x = acc_x.copy()
            self.previous_acc_y = acc_y.copy()
            self.previous_acc_z = acc_z.copy()
        else:
            # Current window MA cancellation
            x2 = self._ma_cancellation(ppg, acc_x, acc_y, acc_z, 30, 10, self.Nprv)

            # Extended window (previous + current) for better estimation
            ext_ppg = np.concatenate([self.previous_ppg, ppg])
            ext_ax = np.concatenate([self.previous_acc_x, acc_x])
            ext_ay = np.concatenate([self.previous_acc_y, acc_y])
            ext_az = np.concatenate([self.previous_acc_z, acc_z])
            z2 = self._ma_cancellation(ext_ppg, ext_ax, ext_ay, ext_az, 30, 10, self.Nprv)

            # Average current and extended results
            ml = min(len(x2), len(z2))
            x3 = (x2[:ml] + z2[:ml]) / 2

            self.previous_ppg = ppg.copy()
            self.previous_acc_x = acc_x.copy()
            self.previous_acc_y = acc_y.copy()
            self.previous_acc_z = acc_z.copy()

        # Simple FFT HR estimation to update Nprv
        bpm = self._estimate_hr(x3)
        self.Nprv = bpm * 4
        return x3, bpm

    def _init_hr(self, ppg):
        """Initialize HR from FFT in [1.2, 1.5] Hz range."""
        f = np.abs(np.fft.fft(ppg))
        freqs = np.fft.fftfreq(len(ppg), 1.0 / self.fs)
        mask = (freqs >= 1.2) & (freqs <= 1.5)
        if not mask.any():
            return 75.0  # default 75 BPM
        idx = np.argmax(f[mask])
        return float(freqs[mask][idx]) * 60

    def _estimate_hr(self, sig):
        """Simple FFT HR estimation in [0.75, 3.5] Hz."""
        sig = np.ascontiguousarray(sig, dtype=np.float64)
        N = max(2048, len(sig))
        f = np.abs(np.fft.fft(sig, N))
        freqs = np.fft.fftfreq(N, 1.0 / self.fs)
        mask = (freqs >= 0.75) & (freqs <= 3.5)
        if not mask.any():
            return 75.0
        return float(freqs[mask][np.argmax(f[mask])]) * 60

    def _ma_cancellation(self, ppg, acc_x, acc_y, acc_z, k, deg, Nprv):
        """Motion artifact cancellation: SVD reference generation + LMS filter."""
        tool = len(ppg)
        y1 = self._ref_ma_svd(acc_x, k, tool)
        y2 = self._ref_ma_svd(acc_y, k, tool)
        y3 = self._ref_ma_svd(acc_z, k, tool)

        s1 = np.ascontiguousarray(y1.reshape(k, tool).T)
        s2 = np.ascontiguousarray(y2.reshape(k, tool).T)
        s3 = np.ascontiguousarray(y3.reshape(k, tool).T)

        x2 = ppg - np.mean(ppg)
        en = np.sqrt(2 * np.mean(x2 ** 2)) * 2
        x2 = np.clip(x2, -en, en)

        x2 = self._adaptive_lms(s1, s2, s3, x2, k, deg, Nprv)
        return x2

    def _ref_ma_svd(self, sig, L, tool):
        """SVD-based reference motion artifact generation.

        Uses np.linalg.eigh (for symmetric matrices) instead of np.linalg.eig
        to avoid NumPy 1.22 complex array memory corruption.
        """
        N = tool
        if L > N // 2:
            L = N - L
        K = N - L + 1

        # Trajectory matrix
        X = np.zeros((L, K), dtype=np.float64)
        for i in range(K):
            X[:, i] = sig[i:i + L]

        # Covariance matrix (symmetric positive semi-definite)
        S = X @ X.T

        # eigh returns real eigenvalues in ascending order for symmetric matrices
        eigenvals, U = np.linalg.eigh(S)
        # Sort descending (eigh returns ascending)
        idx = np.argsort(-eigenvals)
        U = U[:, idx]

        V = X.T @ U

        # Diagonal averaging to reconstruct SSA components
        Lp, Kp = min(L, K), max(L, K)
        p = []
        for j in range(min(L, K)):
            rca = np.outer(U[:, j], V[:, j])
            y = np.zeros(N, dtype=np.float64)

            for kk in range(Lp - 1):
                s = 0.0
                for m in range(kk + 1):
                    if m < rca.shape[0] and kk - m < rca.shape[1]:
                        s += rca[m, kk - m]
                y[kk] = s / (kk + 1)

            for kk in range(Lp - 1, Kp):
                s = 0.0
                for m in range(Lp):
                    if m < rca.shape[0] and kk - m < rca.shape[1]:
                        s += rca[m, kk - m]
                y[kk] = s / Lp

            for kk in range(Kp, N):
                s = 0.0
                for m in range(kk - Kp + 2, min(N - Kp + 1, rca.shape[0])):
                    if m < rca.shape[0] and kk - m < rca.shape[1]:
                        s += rca[m, kk - m]
                y[kk] = s / max(1, N - kk)

            p.append(y)

        return np.concatenate(p)

    def _adaptive_lms(self, s1, s2, s3, x2, k, deg, Nprv):
        """LMS adaptive filter to remove motion artifacts from PPG."""
        ms = 0.005
        fft_length = 6000

        def get_freq_peak(col):
            f = np.abs(np.fft.fft(np.ascontiguousarray(col, dtype=np.float64), fft_length))
            return int(np.argmax(f[:800]))

        k = min(20, k)

        frqs1 = np.array([get_freq_peak(s1[:, i]) for i in range(min(k, s1.shape[1]))])
        frqs2 = np.array([get_freq_peak(s2[:, i]) for i in range(min(k, s2.shape[1]))])
        frqs3 = np.array([get_freq_peak(s3[:, i]) for i in range(min(k, s3.shape[1]))])

        def lms_filter(x_ref, d, flen, step):
            N = len(x_ref)
            w = np.zeros(flen, dtype=np.float64)
            y = np.zeros(N, dtype=np.float64)
            xr = np.ascontiguousarray(x_ref, dtype=np.float64)
            dd = np.ascontiguousarray(d, dtype=np.float64)
            for n in range(flen, N):
                xv = xr[n - flen:n][::-1].copy()
                y[n] = np.dot(w, xv)
                e = dd[n] - y[n]
                w = w + 2 * step * e * xv
            return y

        # Process each axis's components
        for refs, frqs in [(s1, frqs1), (s2, frqs2), (s3, frqs3)]:
            for i in range(min(k, refs.shape[1])):
                ref_sig = np.ascontiguousarray(refs[:, i], dtype=np.float64)
                if frqs[i] < 250 or frqs[i] > 800:
                    x2 = x2 - lms_filter(ref_sig, x2, deg, ms)
                else:
                    x2 = x2 - lms_filter(ref_sig, x2, deg, ms)

        return x2


# =====================================================================
# Signal processing helpers
# =====================================================================

def _bandpass(sig, fs, low=0.75, high=4.0, order=2):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, np.float64(sig))

def _minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-10)

def read_acc_aligned(session_path, ppg_timestamps_ms):
    acc_df = pd.read_csv(os.path.join(session_path, "acc.csv"))
    acc_ts = acc_df["timestamp"].values.astype(np.float64)
    return np.column_stack([
        np.interp(ppg_timestamps_ms, acc_ts, acc_df[c].values.astype(np.float64))
        for c in ("x", "y", "z")
    ])

def denoise_imat(ppg, acc, fs=25):
    """Denoise full PPG signal using IMAT (SVD + LMS adaptive filter).

    Processes in 8-second non-overlapping windows (200 samples at 25 Hz),
    matching the original GalaxyPPG pipeline. IMAT is stateful: uses
    previous window for extended analysis.
    """
    win_len = 200; N = len(ppg)
    ppg_f = _bandpass(ppg, fs, low=0.5, high=4.0, order=4)
    acc_f = np.column_stack([_bandpass(acc[:, i], fs, low=0.5, high=4.0, order=4) for i in range(3)])
    ppg_norm = _minmax(ppg_f)
    acc_norm = np.column_stack([_minmax(acc_f[:, i]) for i in range(3)])

    denoiser = IMATDenoiser(fs=fs, target_length=win_len)
    parts = []
    start = 0
    while start + win_len <= N:
        d, _ = denoiser.process_window(
            ppg_norm[start:start+win_len],
            acc_norm[start:start+win_len, 0],
            acc_norm[start:start+win_len, 1],
            acc_norm[start:start+win_len, 2],
        )
        parts.append(np.asarray(d, dtype=np.float64))
        start += win_len
    if start < N:
        rem = N - start
        wp = np.pad(ppg_norm[start:], (0, win_len - rem), mode='edge')
        wa = [np.pad(acc_norm[start:, i], (0, win_len - rem), mode='edge') for i in range(3)]
        d, _ = denoiser.process_window(wp, wa[0], wa[1], wa[2])
        parts.append(np.asarray(d, dtype=np.float64)[:rem])
    return np.concatenate(parts)

def read_ppg_denoised(session_path, num_frames, fps=30, ppg_fs=25):
    ppg_df = pd.read_csv(os.path.join(session_path, "ppg.csv"))
    ppg_ts = ppg_df["timestamp"].values.astype(np.float64)
    ppg_green = ppg_df["green"].values.astype(np.float64)
    acc_data = read_acc_aligned(session_path, ppg_ts)
    ppg_clean = denoise_imat(ppg_green, acc_data, fs=ppg_fs)
    meta_path = os.path.join(session_path, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    vs = meta["sync_markers"].get("video_start") or meta.get("start_timestamp")
    ppg_t = (ppg_ts - float(vs)) / 1000.0
    frame_t = np.arange(num_frames, dtype=np.float64) / fps
    ftc = np.clip(frame_t, ppg_t[0], ppg_t[-1])
    return np.interp(ftc, ppg_t, ppg_clean).astype(np.float32)


# =====================================================================
# Video / face / preprocessing
# =====================================================================

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)

def crop_face_resize(frames, out_h, out_w, large_box_coef=1.5):
    xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(xml_path)
    frame0 = frames[0]
    if frame0.dtype != np.uint8:
        frame0 = np.clip(frame0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    H, W = frames.shape[1], frames.shape[2]
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2])
        x = max(0, int(x - (large_box_coef-1.0)/2.0*fw))
        y = max(0, int(y - (large_box_coef-1.0)/2.0*fh))
        fw = min(int(fw*large_box_coef), W-x)
        fh = min(int(fh*large_box_coef), H-y)
    else:
        x, y, fw, fh = 0, 0, W, H
    C = frames.shape[3]
    resized = np.zeros((len(frames), out_h, out_w, C), dtype=np.float32)
    for i, frame in enumerate(frames):
        crop = frame[y:y+fh, x:x+fw]
        if crop.size == 0: crop = frame
        resized[i] = cv2.resize(crop.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    return resized

def standardized_label(label):
    label = label.astype(np.float64)
    m, s = np.mean(label), np.std(label)
    if s > 0: label = (label - m) / s
    else: label = np.zeros_like(label)
    return label.astype(np.float32)


# =====================================================================
# Post-processing / evaluation
# =====================================================================

def detrend(signal_in, lambda_val=100):
    T_len = len(signal_in); H_mat = np.eye(T_len); ones = np.ones(T_len)
    D_mat = np.diag(ones[:-2], -2) - 2*np.diag(ones[:-1], -1) + np.diag(ones)
    D_mat = D_mat[2:, :]
    inv = np.linalg.inv(H_mat + lambda_val**2 * D_mat.T @ D_mat)
    return (H_mat - inv) @ signal_in

def bandpass_filter(sig, fs, low, high, order=1):
    b, a = signal.butter(order, [low/fs*2, high/fs*2], btype="bandpass")
    return signal.filtfilt(b, a, sig.astype(np.float64))

def fft_peak_hz(sig, fs, low, high):
    N = 1
    while N < len(sig): N *= 2
    freqs, pxx = periodogram(sig, fs=fs, nfft=N, detrend=False)
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any(): return 0.0
    return float(freqs[mask][np.argmax(pxx[mask])])

def calculate_snr(pred_ppg, hr_label_bpm, fs, low_pass=0.6, high_pass=3.3):
    N = 1
    while N < len(pred_ppg): N *= 2
    freqs, pxx = periodogram(pred_ppg, fs=fs, nfft=N, detrend=False)
    f1 = hr_label_bpm/60.0; f2 = 2*f1; dev = 6.0/60.0
    sig_mask = ((freqs >= f1-dev)&(freqs <= f1+dev)) | ((freqs >= f2-dev)&(freqs <= f2+dev))
    noise_mask = (freqs >= low_pass)&(freqs <= high_pass)&~sig_mask
    sp = pxx[sig_mask].sum(); np_ = pxx[noise_mask].sum()
    if np_ == 0: return float("inf")
    return float(10.0*np.log10(sp/np_))

def _reform(chunk_dict):
    return np.concatenate([chunk_dict[k] for k in sorted(chunk_dict.keys())])

def process_bvp(pred_chunks, label_chunks, fs=30):
    pred = _reform(pred_chunks).astype(np.float64)
    label = _reform(label_chunks).astype(np.float64)
    pred = detrend(pred, 100); label = detrend(label, 100)
    pred = bandpass_filter(pred, fs, 0.6, 3.3)
    label = bandpass_filter(label, fs, 0.6, 3.3)
    hr_pred = fft_peak_hz(pred, fs, 0.6, 3.3)*60.0
    hr_label = fft_peak_hz(label, fs, 0.6, 3.3)*60.0
    snr_db = calculate_snr(pred, hr_label, fs)
    return hr_pred, hr_label, snr_db


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("Denoise method: imat\n")

    # Discover subjects
    all_dirs = sorted([d for d in glob.glob(os.path.join(RAW_DATA_PATH, "*"))
                       if os.path.isdir(d) and os.path.basename(d) != "videos"])
    subjects = []
    for subj_dir in all_dirs:
        subj_id = os.path.basename(subj_dir)
        subj_key = subj_id.replace("_", "")
        session_dirs = glob.glob(os.path.join(RAW_DATA_PATH, subj_id, "*"))
        if not session_dirs: continue
        session_path = session_dirs[0]
        session_name = os.path.basename(session_path)
        vf = glob.glob(os.path.join(RAW_DATA_PATH, "videos", f"{subj_id}_{session_name}.mp4"))
        if not vf: continue
        subjects.append({"subj_id": subj_id, "subj_key": subj_key,
                          "video_path": vf[0], "session_path": session_path})
        print(f"  {subj_id}")

    # Preprocess
    if os.path.exists(PREPROCESSED_PATH):
        shutil.rmtree(PREPROCESSED_PATH)
    os.makedirs(PREPROCESSED_PATH)

    all_input_files = []
    for subj in subjects:
        print(f"\n=== {subj['subj_key']} ===")
        frames = read_video_frames(subj["video_path"])
        T = frames.shape[0]
        print(f"  Video: {T} frames")

        ppg_signal = read_ppg_denoised(subj["session_path"], T, fps=VIDEO_FPS, ppg_fs=PPG_FS)
        print(f"  PPG (imat): min={ppg_signal.min():.4f}, max={ppg_signal.max():.4f}")

        frames_cropped = crop_face_resize(frames, IMG_H, IMG_W)
        label = standardized_label(ppg_signal)
        clip_num = T // CHUNK_LENGTH

        subj_out = os.path.join(PREPROCESSED_PATH, subj["subj_key"])
        os.makedirs(subj_out)

        for ci in range(clip_num):
            inp = frames_cropped[ci*CHUNK_LENGTH:(ci+1)*CHUNK_LENGTH]
            lab = label[ci*CHUNK_LENGTH:(ci+1)*CHUNK_LENGTH]
            ip = os.path.join(subj_out, f"{subj['subj_key']}_input{ci}.npy")
            lp = os.path.join(subj_out, f"{subj['subj_key']}_label{ci}.npy")
            np.save(ip, inp); np.save(lp, lab)
            all_input_files.append(ip)
        print(f"  {clip_num} clips")

    # Dataset
    class PPGDataset(Dataset):
        def __init__(self, input_files):
            self.inputs = sorted(input_files)
            self.labels = [f.replace("input", "label") for f in self.inputs]
        def __len__(self): return len(self.inputs)
        def __getitem__(self, index):
            data = np.float32(np.load(self.inputs[index]))
            label = np.float32(np.load(self.labels[index]))
            data = np.transpose(data, (3,0,1,2))
            fname = os.path.basename(self.inputs[index])
            si = fname.index("_"); subj = fname[:si]; cid = fname[si+6:].split(".")[0]
            return data, label, subj, cid

    dataset = PPGDataset(all_input_files)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f"\nDataset: {len(dataset)} clips, {len(loader)} batches")

    # Model
    MD_CONFIG = {"FRAME_NUM": CHUNK_LENGTH, "MD_FSAM": True, "MD_TYPE": "NMF",
                 "MD_TRANSFORM": "T_KAB", "MD_R": 1, "MD_S": 1, "MD_STEPS": 3,
                 "MD_INFERENCE": True, "MD_RESIDUAL": True}
    model = FactorizePhys(frames=CHUNK_LENGTH, md_config=MD_CONFIG, in_channels=3,
                          dropout=0.1, device=torch.device(DEVICE))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE); model.eval()

    # Inference
    preds_dict = {}; labels_dict = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            data_t, labels_t, batch_subjects, batch_chunk_ids = batch
            data_in = data_t.float().to(DEVICE)
            last_frame = data_in[:, :, -1:, :, :].clone()
            data_padded = torch.cat([data_in, last_frame], dim=2)
            out = model(data_padded)
            pred_np = out[0].cpu().numpy()
            label_np = labels_t.numpy()
            for i in range(data_t.shape[0]):
                subj = batch_subjects[i]; cid = int(batch_chunk_ids[i])
                if subj not in preds_dict:
                    preds_dict[subj] = {}; labels_dict[subj] = {}
                preds_dict[subj][cid] = pred_np[i]
                labels_dict[subj][cid] = label_np[i]

    # Evaluate
    FS = VIDEO_FPS
    hr_preds_all = []; hr_labels_all = []; snr_all = []
    per_subject = []

    print(f"\n{'Subject':<10} {'HR_pred':>10} {'HR_label':>10} {'HR_err':>8} {'SNR':>7}")
    print("-" * 50)

    for subj_key in sorted(preds_dict.keys()):
        hr_pred, hr_label, snr_db = process_bvp(preds_dict[subj_key], labels_dict[subj_key], fs=FS)
        hr_err = hr_pred - hr_label
        subj_id = subj_key[0] + "_" + subj_key[1:]
        per_subject.append({"name": subj_id, "predicted_heartrate": hr_pred,
                            "real_heartrate": hr_label, "heartrate_error": hr_err, "snr_db": snr_db})
        hr_preds_all.append(hr_pred); hr_labels_all.append(hr_label); snr_all.append(snr_db)
        print(f"{subj_id:<10} {hr_pred:>10.3f} {hr_label:>10.3f} {hr_err:>8.3f} {snr_db:>7.2f}")

    hr_preds_all = np.array(hr_preds_all); hr_labels_all = np.array(hr_labels_all)
    n = len(hr_preds_all)
    err = hr_preds_all - hr_labels_all; abs_e = np.abs(err)
    mae = float(np.mean(abs_e)); mae_se = float(np.std(abs_e)/np.sqrt(n))
    rmse = float(np.sqrt(np.mean(err**2))); rmse_se = float(np.sqrt(np.std(err**2)/np.sqrt(n)))
    mape = float(np.mean(abs_e/(np.abs(hr_labels_all)+1e-9))*100)
    mape_se = float(np.std(abs_e/(np.abs(hr_labels_all)+1e-9))/np.sqrt(n)*100)
    pearson_r = float(np.corrcoef(hr_preds_all, hr_labels_all)[0,1]) if n >= 2 else float("nan")
    pearson_se = float(np.sqrt(max(0,(1-pearson_r**2)/(n-2)))) if n >= 2 else float("nan")
    snr_all = np.array(snr_all)
    mean_snr = float(np.mean(snr_all)); mean_snr_se = float(np.std(snr_all)/np.sqrt(n))

    print(f"\nAggregate metrics (IMAT denoising):")
    print(f"  MAE     : {mae:.4f} +/- {mae_se:.4f} bpm")
    print(f"  RMSE    : {rmse:.4f} +/- {rmse_se:.4f} bpm")
    print(f"  MAPE    : {mape:.4f} +/- {mape_se:.4f} %")
    print(f"  Pearson : {pearson_r:.4f} +/- {pearson_se:.4f}")
    print(f"  SNR     : {mean_snr:.4f} +/- {mean_snr_se:.4f} dB")

    # Save
    model_output_dir = os.path.join(OUTPUT_DIR, MODEL_LABEL)
    os.makedirs(model_output_dir, exist_ok=True)
    metrics_dict = {
        "model": MODEL_LABEL, "n_subjects": n, "evaluation_method": "FFT",
        "ground_truth": "BVP-derived HR (FFT of PPG green label)",
        "label_type": LABEL_TYPE, "bvp_bandpass_hz": [0.6, 3.3],
        "aggregate_metrics": {
            "MAE": {"value": mae, "se": mae_se, "unit": "bpm"},
            "RMSE": {"value": rmse, "se": rmse_se, "unit": "bpm"},
            "MAPE": {"value": mape, "se": mape_se, "unit": "%"},
            "Pearson": {"value": pearson_r, "se": pearson_se, "unit": ""},
            "SNR": {"value": mean_snr, "se": mean_snr_se, "unit": "dB"},
        },
        "per_subject": per_subject,
    }
    with open(os.path.join(model_output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_dict, fh, indent=2)

    csv_rows = pd.DataFrame(per_subject)
    csv_path = os.path.join(model_output_dir, "ppg_results.csv")
    csv_rows.to_csv(csv_path, index=False)
    print(f"\nSaved to {model_output_dir}")
