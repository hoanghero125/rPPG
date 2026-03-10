"""
bigsmall_inference.py
=====================
Python script version of bigsmall_inference.ipynb.

Usage:
    python bigsmall_inference.py [--config <yaml>]

All paths and parameters are set in the CONFIG section below (or via a YAML
override — see --config flag).
"""

import os
import sys
import json
import glob
import pickle
import shutil
import argparse

import numpy as np
import pandas as pd
import cv2
from scipy import signal
from scipy.signal import periodogram
from tqdm import tqdm
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from neural_methods.model.BigSmall import BigSmall


# ===========================================================================
# CONFIG — edit these paths/params or override via --config <yaml>
# ===========================================================================
DEFAULT_CFG = {
    "RAW_DATA_PATH":     os.path.join(REPO_ROOT, "raw_data"),
    "PREPROCESSED_PATH": os.path.join(REPO_ROOT, "preprocessed_bigsmall_nb"),
    "MODEL_PATH":        os.path.join(REPO_ROOT, "final_model_release",
                                      "BP4D_BigSmall_Multitask_Fold1.pth"),
    "OUTPUT_DIR":        os.path.join(REPO_ROOT, "results_bigsmall"),
    "VIDEO_FPS":         30,
    "PPG_FS":            25,
    "CHUNK_LENGTH":      3,
    "BIG_H":             144,
    "BIG_W":             144,
    "SMALL_H":           9,
    "SMALL_W":           9,
    "BATCH_SIZE":        32,
    "NUM_WORKERS":       4,
    "DEVICE":            "cuda:0" if torch.cuda.is_available() else "cpu",
    # 49-channel label layout
    "LABEL_IDX_BVP":     0,
    "LABEL_IDX_HR":      1,
    "LABEL_IDX_RESP":    5,
    "NUM_LABEL_CH":      49,
}


DEFAULT_CONFIG_FILE = os.path.join(
    REPO_ROOT, "configs", "infer_configs", "bigsmall_inference.yaml"
)


def load_config(yaml_path=None):
    import yaml

    cfg = DEFAULT_CFG.copy()

    if yaml_path is None:
        yaml_path = DEFAULT_CONFIG_FILE

    with open(yaml_path) as f:
        overrides = yaml.safe_load(f)

    cfg.update(overrides)

    # Resolve relative paths against the repo root
    for key in ("RAW_DATA_PATH", "PREPROCESSED_PATH", "MODEL_PATH", "OUTPUT_DIR"):
        if not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(REPO_ROOT, cfg[key])

    return cfg


# ===========================================================================
# I/O helpers
# ===========================================================================

def read_video_frames(video_path):
    """Read all frames from a video file.

    Returns:
        np.ndarray: shape (T, H, W, 3), dtype uint8, RGB order.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Empty video: {video_path}")
    return np.stack(frames, axis=0)


def read_ppg_synced(session_path, num_frames, fps=30):
    """Read PPG green channel from ppg.csv and resample to video frame times.

    Returns:
        np.ndarray: shape (num_frames,), dtype float32.
    """
    meta_path = os.path.join(session_path, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    video_start_ms = meta["sync_markers"].get("video_start")
    if video_start_ms is None:
        video_start_ms = meta.get("start_timestamp")
    if video_start_ms is None:
        raise ValueError(f"No video_start or start_timestamp found in {meta_path}")
    video_start_ms = float(video_start_ms)

    ppg_df = pd.read_csv(os.path.join(session_path, "ppg.csv"))
    ppg_ts    = ppg_df["timestamp"].values.astype(np.float64)
    ppg_green = ppg_df["green"].values.astype(np.float64)

    ppg_t   = (ppg_ts - video_start_ms) / 1000.0
    frame_t = np.arange(num_frames, dtype=np.float64) / fps
    frame_t_clipped = np.clip(frame_t, ppg_t[0], ppg_t[-1])

    return np.interp(frame_t_clipped, ppg_t, ppg_green).astype(np.float32)


def read_gt_hr(session_path):
    """Return average HR from hr.csv (hrStatus==1, hr>0)."""
    hr_path = os.path.join(session_path, "hr.csv")
    df = pd.read_csv(
        hr_path,
        usecols=[0, 1, 2, 3, 4],
        names=["id", "dataReceived", "timestamp", "hr", "hrStatus"],
        header=0,
    )
    valid = df[(df["hrStatus"] == 1) & (df["hr"] > 0)]["hr"]
    return float(valid.mean()) if len(valid) else float("nan")


def discover_subjects(raw_data_path):
    """Scan raw_data_path for subject folders and locate video + session files.

    Does not require metadata.csv. Subject folders are discovered via glob
    (any directory that is not 'videos'). avg_hr is set to 0.0 as a placeholder
    since it is only used to fill an unused label channel and does not affect
    any reported metrics.

    Returns:
        list[dict]: one entry per subject with keys:
            subj_id, subj_key, video_path, session_path, avg_hr, gt_hr.
    """
    all_dirs = sorted([
        d for d in glob.glob(os.path.join(raw_data_path, "*"))
        if os.path.isdir(d) and os.path.basename(d) != "videos"
    ])
    print(f"Found {len(all_dirs)} subject folders\n")

    subjects = []
    for subj_dir in all_dirs:
        subj_id  = os.path.basename(subj_dir)
        subj_key = subj_id.replace("_", "")

        session_dirs = glob.glob(os.path.join(raw_data_path, subj_id, "*"))
        if not session_dirs:
            print(f"No session folder for {subj_id}, skipping.")
            continue
        session_path = session_dirs[0]

        session_name  = os.path.basename(session_path)
        video_pattern = os.path.join(raw_data_path, "videos",
                                     f"{subj_id}_{session_name}.mp4")
        video_files = glob.glob(video_pattern)
        if not video_files:
            print(f"No video found for {subj_id}, skipping.")
            continue
        video_path = video_files[0]

        gt_hr = read_gt_hr(session_path)

        subjects.append({
            "subj_id":      subj_id,
            "subj_key":     subj_key,
            "video_path":   video_path,
            "session_path": session_path,
            "avg_hr":       0.0,   # placeholder — not used in any reported metric
            "gt_hr":        gt_hr,
        })
        print(f"  {subj_id}  HR_device={gt_hr:.4f} bpm  "
              f"video={os.path.basename(video_path)}")

    print(f"\nTotal subjects: {len(subjects)}")
    return subjects


# ===========================================================================
# Normalization / preprocessing
# ===========================================================================

def diff_normalize_data(data):
    """DiffNormalized: (frame[t+1]-frame[t]) / (frame[t+1]+frame[t]+1e-7) / std."""
    data = data.astype(np.float32)
    n = data.shape[0]
    out = np.zeros_like(data)
    out[:n - 1] = (data[1:] - data[:-1]) / (data[1:] + data[:-1] + 1e-7)
    std = np.std(out)
    if std > 0:
        out /= std
    return out


def standardized_data(data):
    """Standardized: global z-score over all pixels and frames."""
    data = data.astype(np.float32)
    m, s = np.mean(data), np.std(data)
    data = (data - m) / s if s > 0 else np.zeros_like(data)
    return np.where(np.isnan(data), np.zeros_like(data), data)


def diff_normalize_label(label):
    """DiffNormalized label: finite difference / std, zero-padded."""
    diff = np.diff(label.astype(np.float64), axis=0)
    s = np.std(diff)
    if s > 0:
        diff /= s
    return np.append(diff, [0.0]).astype(np.float32)


def crop_face_resize(frames, out_h, out_w, large_box_coef=1.5):
    """Detect face on frame 0, expand bbox, resize all frames."""
    xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(xml_path)
    frame0 = frames[0]
    if frame0.dtype != np.uint8:
        frame0 = np.clip(frame0, 0, 255).astype(np.uint8)
    gray  = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    H, W  = frames.shape[1], frames.shape[2]

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2])
        x  = max(0, int(x  - (large_box_coef - 1.0) / 2.0 * fw))
        y  = max(0, int(y  - (large_box_coef - 1.0) / 2.0 * fh))
        fw = min(int(fw * large_box_coef), W - x)
        fh = min(int(fh * large_box_coef), H - y)
    else:
        x, y, fw, fh = 0, 0, W, H

    C = frames.shape[3]
    resized = np.zeros((len(frames), out_h, out_w, C), dtype=np.float32)
    for i, frame in enumerate(frames):
        crop = frame[y: y + fh, x: x + fw]
        if crop.size == 0:
            crop = frame
        resized[i] = cv2.resize(crop.astype(np.float32), (out_w, out_h),
                                interpolation=cv2.INTER_AREA)
    return resized


def resize_frames(frames, out_h, out_w):
    """Resize every frame to (out_h, out_w) using INTER_AREA."""
    T, _H, _W, C = frames.shape
    resized = np.zeros((T, out_h, out_w, C), dtype=np.float32)
    for i in range(T):
        resized[i] = cv2.resize(frames[i], (out_w, out_h),
                                interpolation=cv2.INTER_AREA)
    return resized


# ===========================================================================
# Preprocessing pipeline
# ===========================================================================

def preprocess_subjects(subjects, cfg):
    """Preprocess all subjects and save clips to PREPROCESSED_PATH.

    Returns:
        list[str]: paths to all saved input pickle files.
    """
    preprocessed_path = cfg["PREPROCESSED_PATH"]
    video_fps         = cfg["VIDEO_FPS"]
    chunk_length      = cfg["CHUNK_LENGTH"]
    big_h, big_w      = cfg["BIG_H"], cfg["BIG_W"]
    small_h, small_w  = cfg["SMALL_H"], cfg["SMALL_W"]
    num_label_ch      = cfg["NUM_LABEL_CH"]
    label_idx_bvp     = cfg["LABEL_IDX_BVP"]
    label_idx_hr      = cfg["LABEL_IDX_HR"]

    if os.path.exists(preprocessed_path):
        shutil.rmtree(preprocessed_path)
    os.makedirs(preprocessed_path)
    print(f"Cleared and recreated: {preprocessed_path}\n")

    all_input_files = []

    for subj in subjects:
        subj_key     = subj["subj_key"]
        video_path   = subj["video_path"]
        session_path = subj["session_path"]
        avg_hr       = subj["avg_hr"]

        print(f"=== Processing {subj_key} ===")

        frames = read_video_frames(video_path)
        T = frames.shape[0]
        print(f"  Video: {T} frames @ {video_fps} fps")

        ppg_signal = read_ppg_synced(session_path, T, fps=video_fps)
        print(f"  PPG green: min={ppg_signal.min():.0f}, max={ppg_signal.max():.0f}")

        labels = np.full((T, num_label_ch), -1.0, dtype=np.float32)
        labels[:, label_idx_bvp] = ppg_signal
        labels[:, label_idx_hr]  = avg_hr

        frames_big = crop_face_resize(frames, big_h, big_w)
        big_data   = standardized_data(frames_big)
        diff_big   = diff_normalize_data(frames_big)
        small_data = resize_frames(diff_big, small_h, small_w)

        labels[:, label_idx_bvp] = diff_normalize_label(ppg_signal)

        clip_num    = T // chunk_length
        big_clips   = np.array([big_data[i * chunk_length:(i + 1) * chunk_length]   for i in range(clip_num)])
        small_clips = np.array([small_data[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)])
        label_clips = np.array([labels[i * chunk_length:(i + 1) * chunk_length]     for i in range(clip_num)])

        subj_dir = os.path.join(preprocessed_path, subj_key)
        os.makedirs(subj_dir)

        for chunk_idx in range(clip_num):
            input_path = os.path.join(subj_dir, f"{subj_key}_input{chunk_idx}.pickle")
            label_path = os.path.join(subj_dir, f"{subj_key}_label{chunk_idx}.npy")

            frames_dict = {0: big_clips[chunk_idx], 1: small_clips[chunk_idx]}
            with open(input_path, "wb") as fh:
                pickle.dump(frames_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(label_path, label_clips[chunk_idx])
            all_input_files.append(input_path)

        print(f"  {clip_num} clips -> {subj_dir}\n")

    print(f"Total clips saved: {len(all_input_files)}")
    print("\nFolder structure:")
    for subj in subjects:
        d = os.path.join(preprocessed_path, subj["subj_key"])
        n = len(glob.glob(os.path.join(d, "*.pickle")))
        print(f"  {subj['subj_key']}/  ({n} clips)")

    return all_input_files


# ===========================================================================
# Dataset & DataLoader
# ===========================================================================

class BigSmallDataset(Dataset):
    def __init__(self, input_files):
        self.inputs = sorted(input_files)
        self.labels = [
            f.replace("input", "label").replace(".pickle", ".npy")
            for f in self.inputs
        ]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        with open(self.inputs[index], "rb") as fh:
            data = pickle.load(fh)

        # NDHWC -> NDCHW
        data[0] = np.float32(np.transpose(data[0], (0, 3, 1, 2)))
        data[1] = np.float32(np.transpose(data[1], (0, 3, 1, 2)))

        label = np.float32(np.load(self.labels[index]))  # (D, 49)

        fname      = os.path.basename(self.inputs[index])
        split_idx  = fname.index("_")
        subject_id = fname[:split_idx]                     # e.g. "S000"
        chunk_id   = fname[split_idx + 6:].split(".")[0]  # +6 skips "_input"

        return data, label, subject_id, chunk_id


def bigsmall_collate(batch):
    """Stack dict-based data into tensors."""
    data_dicts, labels, subjects, chunk_ids = zip(*batch)
    data_big   = torch.stack([torch.from_numpy(d[0]) for d in data_dicts], dim=0)
    data_small = torch.stack([torch.from_numpy(d[1]) for d in data_dicts], dim=0)
    labels_t   = torch.stack([torch.from_numpy(l)    for l in labels],     dim=0)
    return data_big, data_small, labels_t, list(subjects), list(chunk_ids)


def build_dataloader(input_files, cfg):
    dataset = BigSmallDataset(input_files)
    print(f"Dataset: {len(dataset)} clips")
    loader = DataLoader(
        dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        collate_fn=bigsmall_collate,
    )
    print(f"DataLoader ready: {len(loader)} batches")
    return loader


# ===========================================================================
# Model loading
# ===========================================================================

def load_model(cfg):
    model = BigSmall(n_segment=cfg["CHUNK_LENGTH"])
    state_dict = torch.load(cfg["MODEL_PATH"], map_location=cfg["DEVICE"])
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(cfg["DEVICE"])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded and set to eval mode. Parameters: {num_params:,}")
    return model


# ===========================================================================
# Inference
# ===========================================================================

def run_inference(model, loader, cfg):
    """Run BigSmall forward pass over all batches.

    Returns:
        bvp_preds_dict  (dict): subj -> {chunk_id: np.ndarray}
        bvp_labels_dict (dict): subj -> {chunk_id: np.ndarray}
        resp_preds_dict (dict): subj -> {chunk_id: np.ndarray}
    """
    device        = cfg["DEVICE"]
    chunk_length  = cfg["CHUNK_LENGTH"]
    num_label_ch  = cfg["NUM_LABEL_CH"]
    label_idx_bvp = cfg["LABEL_IDX_BVP"]

    bvp_preds_dict  = {}
    bvp_labels_dict = {}
    resp_preds_dict = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            data_big, data_small, labels_t, batch_subjects, batch_chunk_ids = batch

            N, D, C_b, H_b, W_b = data_big.shape
            _, _, C_s, H_s, W_s = data_small.shape

            big_flat   = data_big.view(N * D, C_b, H_b, W_b).to(device)
            small_flat = data_small.view(N * D, C_s, H_s, W_s).to(device)

            trim       = (N * D) // chunk_length * chunk_length
            big_flat   = big_flat[:trim]
            small_flat = small_flat[:trim]

            labels_flat = labels_t.view(N * D, num_label_ch)

            _au_out, bvp_out, resp_out = model((big_flat, small_flat))

            bvp_pred_np  = bvp_out.squeeze(-1).cpu().numpy()
            resp_pred_np = resp_out.squeeze(-1).cpu().numpy()
            bvp_label_np = labels_flat[:trim, label_idx_bvp].numpy()

            for i in range(N):
                if i * chunk_length >= trim:
                    break
                subj  = batch_subjects[i]
                cid   = int(batch_chunk_ids[i])
                start = i * chunk_length
                end   = start + chunk_length

                bvp_preds_dict.setdefault(subj, {})
                bvp_labels_dict.setdefault(subj, {})
                resp_preds_dict.setdefault(subj, {})

                bvp_preds_dict[subj][cid]  = bvp_pred_np[start:end]
                bvp_labels_dict[subj][cid] = bvp_label_np[start:end]
                resp_preds_dict[subj][cid] = resp_pred_np[start:end]

    print("\nInference complete.")
    print("Subjects:", sorted(bvp_preds_dict.keys()))
    return bvp_preds_dict, bvp_labels_dict, resp_preds_dict


# ===========================================================================
# Post-processing & metrics
# ===========================================================================

def detrend(signal_in, lambda_val=100):
    """Smoothness-priors detrending (Tarvainen et al.)."""
    T_len = len(signal_in)
    H_mat = np.eye(T_len)
    ones  = np.ones(T_len)
    D_mat = (np.diag(ones[:-2], -2)
             - 2 * np.diag(ones[:-1], -1)
             + np.diag(ones))
    D_mat = D_mat[2:, :]
    inv   = np.linalg.inv(H_mat + lambda_val ** 2 * D_mat.T @ D_mat)
    return (H_mat - inv) @ signal_in


def bandpass_filter(sig, fs, low, high, order=1):
    b, a = signal.butter(order, [low / fs * 2, high / fs * 2], btype="bandpass")
    return signal.filtfilt(b, a, sig.astype(np.float64))


def fft_peak_hz(sig, fs, low, high):
    N = 1
    while N < len(sig):
        N *= 2
    freqs, pxx = periodogram(sig, fs=fs, nfft=N, detrend=False)
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(freqs[mask][np.argmax(pxx[mask])])


def calculate_snr(pred_ppg, hr_label_bpm, fs, low_pass=0.6, high_pass=3.3):
    N = 1
    while N < len(pred_ppg):
        N *= 2
    freqs, pxx = periodogram(pred_ppg, fs=fs, nfft=N, detrend=False)
    f1, f2 = hr_label_bpm / 60.0, 2 * hr_label_bpm / 60.0
    dev    = 6.0 / 60.0
    sig_mask   = (((freqs >= f1 - dev) & (freqs <= f1 + dev))
                  | ((freqs >= f2 - dev) & (freqs <= f2 + dev)))
    noise_mask = (freqs >= low_pass) & (freqs <= high_pass) & ~sig_mask
    sig_power  = pxx[sig_mask].sum()
    noise_power = pxx[noise_mask].sum()
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(sig_power / noise_power))


def _reform_from_dict(chunk_dict):
    return np.concatenate([chunk_dict[k] for k in sorted(chunk_dict.keys())])


def process_bvp(pred_chunks, label_chunks, fs=30):
    pred  = _reform_from_dict(pred_chunks).astype(np.float64)
    label = _reform_from_dict(label_chunks).astype(np.float64)
    pred  = detrend(np.cumsum(pred),  100)
    label = detrend(np.cumsum(label), 100)
    pred_p  = bandpass_filter(pred,  fs, low=0.6, high=3.3)
    label_p = bandpass_filter(label, fs, low=0.6, high=3.3)
    hr_pred  = fft_peak_hz(pred_p,  fs, 0.6, 3.3) * 60.0
    hr_label = fft_peak_hz(label_p, fs, 0.6, 3.3) * 60.0
    snr_db   = calculate_snr(pred_p, hr_label, fs)
    return hr_pred, hr_label, snr_db, pred_p


def process_resp(pred_chunks, fs=30):
    pred = _reform_from_dict(pred_chunks).astype(np.float64)
    pred = detrend(np.cumsum(pred), 100)
    pred = bandpass_filter(pred, fs, low=0.13, high=0.5)
    return fft_peak_hz(pred, fs, 0.13, 0.5) * 60.0


# ===========================================================================
# Blink rate detection
# ===========================================================================

def detect_blink_rate(video_path, fps=30, large_box_coef=1.5):
    """Estimate blink rate (blinks/min) via eye-strip brightness analysis."""
    frames = read_video_frames(video_path)
    T = len(frames)

    xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(xml_path)
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray0, scaleFactor=1.3, minNeighbors=5)
    H, W  = frames[0].shape[:2]

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2])
        x  = max(0, int(x  - (large_box_coef - 1.0) / 2.0 * fw))
        y  = max(0, int(y  - (large_box_coef - 1.0) / 2.0 * fh))
        fw = min(int(fw * large_box_coef), W - x)
        fh = min(int(fh * large_box_coef), H - y)
    else:
        x, y, fw, fh = 0, 0, W, H

    eye_top    = y + int(fh * 0.20)
    eye_bottom = y + int(fh * 0.50)
    eye_left, eye_right = x, x + fw

    brightness = np.array([
        np.mean(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)[eye_top:eye_bottom, eye_left:eye_right])
        for f in frames
    ])

    brightness_inv = -brightness.astype(np.float64)
    b, a = signal.butter(2, [0.1 / fps * 2, 0.9 / fps * 2], btype="bandpass")
    filtered = signal.filtfilt(b, a, brightness_inv)

    min_dist = int(fps)
    peaks, _ = signal.find_peaks(filtered, distance=min_dist)

    duration_min = T / fps / 60.0
    return len(peaks) / duration_min


# ===========================================================================
# Results collection & export
# ===========================================================================

def collect_results(subjects, bvp_preds_dict, bvp_labels_dict,
                    resp_preds_dict, blink_rate_dict, fs):
    subjects_by_key = {s["subj_key"]: s for s in subjects}
    per_subject_results = []
    hr_preds_all = []
    gt_hrs_all   = []
    snr_all      = []
    rr_preds_all = []

    print(f"\n{'Subject':<10} {'HR_pred':>10} {'HR_gt':>10} "
          f"{'HR_err':>8} {'RR_pred':>8} {'SNR':>7} {'Blinks':>7}")
    print("-" * 68)

    for subj_key in sorted(bvp_preds_dict.keys()):
        hr_pred, hr_label, _, pred_processed = process_bvp(
            bvp_preds_dict[subj_key], bvp_labels_dict[subj_key], fs=fs
        )
        rr_pred    = process_resp(resp_preds_dict[subj_key], fs=fs)
        gt_hr      = subjects_by_key[subj_key]["gt_hr"]
        snr_db     = calculate_snr(pred_processed, gt_hr, fs)
        blink_rate = blink_rate_dict.get(subj_key, float("nan"))
        hr_err     = hr_pred - gt_hr

        subj_id = subj_key[0] + "_" + subj_key[1:]

        per_subject_results.append({
            "name":                subj_id,
            "predicted_heartrate": hr_pred,
            "real_heartrate":      gt_hr,
            "heartrate_error":     hr_err,
            "respiration_rate":    rr_pred,
            "blink_rate":          blink_rate,
            "snr_db":              snr_db,
        })

        hr_preds_all.append(hr_pred)
        gt_hrs_all.append(gt_hr)
        snr_all.append(snr_db)
        rr_preds_all.append(rr_pred)

        print(f"{subj_id:<10} {hr_pred:>10.3f} {gt_hr:>10.3f} "
              f"{hr_err:>8.3f} {rr_pred:>8.3f} {snr_db:>7.2f} {blink_rate:>7.2f}")

    return (per_subject_results,
            np.array(hr_preds_all),
            np.array(gt_hrs_all),
            np.array(snr_all))


def compute_aggregate_metrics(hr_preds_all, gt_hrs_all, snr_all):
    n     = len(hr_preds_all)
    err   = hr_preds_all - gt_hrs_all
    abs_e = np.abs(err)
    sq_e  = err ** 2
    rel_e = abs_e / (np.abs(gt_hrs_all) + 1e-9)

    mae      = float(np.mean(abs_e))
    mae_se   = float(np.std(abs_e) / np.sqrt(n))
    rmse     = float(np.sqrt(np.mean(sq_e)))
    rmse_se  = float(np.sqrt(np.std(sq_e) / np.sqrt(n)))
    mape     = float(np.mean(rel_e) * 100.0)
    mape_se  = float(np.std(rel_e) / np.sqrt(n) * 100.0)

    if n >= 2:
        pearson_r  = float(np.corrcoef(hr_preds_all, gt_hrs_all)[0, 1])
        pearson_se = float(np.sqrt(max(0.0, (1 - pearson_r ** 2) / (n - 2))))
    else:
        pearson_r, pearson_se = float("nan"), float("nan")

    mean_snr    = float(np.mean(snr_all))
    mean_snr_se = float(np.std(snr_all) / np.sqrt(n))

    print(f"\nMAE     : {mae:.4f} +/- {mae_se:.4f} bpm")
    print(f"RMSE    : {rmse:.4f} +/- {rmse_se:.4f} bpm")
    print(f"MAPE    : {mape:.4f} +/- {mape_se:.4f} %")
    print(f"Pearson : {pearson_r:.4f} +/- {pearson_se:.4f}")
    print(f"SNR     : {mean_snr:.4f} +/- {mean_snr_se:.4f} dB")

    return {
        "MAE":     {"value": mae,      "se": mae_se,      "unit": "bpm"},
        "RMSE":    {"value": rmse,     "se": rmse_se,     "unit": "bpm"},
        "MAPE":    {"value": mape,     "se": mape_se,     "unit": "%"},
        "Pearson": {"value": pearson_r, "se": pearson_se, "unit": ""},
        "SNR":     {"value": mean_snr, "se": mean_snr_se, "unit": "dB"},
    }


def export_results(per_subject_results, aggregate_metrics, n, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.basename(DEFAULT_CFG["MODEL_PATH"]).replace(".pth", "")
    metrics_dict = {
        "model":               model_name,
        "n_subjects":          n,
        "evaluation_method":   "FFT",
        "bvp_bandpass_hz":     [0.6, 3.3],
        "resp_bandpass_hz":    [0.13, 0.5],
        "aggregate_metrics":   aggregate_metrics,
        "per_subject": [
            {k: v for k, v in r.items() if k != "blink_rate"}
            for r in per_subject_results
        ],
    }

    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as fh:
        json.dump(metrics_dict, fh, indent=2)
    print(f"\nMetrics saved to: {json_path}")

    results_df = pd.DataFrame(per_subject_results, columns=[
        "name", "predicted_heartrate", "real_heartrate",
        "heartrate_error", "respiration_rate", "blink_rate",
    ])
    csv_path = os.path.join(output_dir, "ppg_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    print()
    print(results_df.to_string(index=False))


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="BigSmall inference script")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file "
                             "(default: configs/infer_configs/bigsmall_inference.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=== Configuration ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print(f"  Device: {cfg['DEVICE']}\n")

    os.makedirs(cfg["PREPROCESSED_PATH"], exist_ok=True)
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)

    # 1. Discover subjects
    print("=== Discovering subjects ===")
    subjects = discover_subjects(cfg["RAW_DATA_PATH"])

    # 2. Preprocess
    print("\n=== Preprocessing ===")
    all_input_files = preprocess_subjects(subjects, cfg)

    # 3. Dataset & DataLoader
    print("\n=== Building DataLoader ===")
    loader = build_dataloader(all_input_files, cfg)

    # 4. Load model
    print("\n=== Loading model ===")
    model = load_model(cfg)

    # 5. Inference
    print("\n=== Running inference ===")
    bvp_preds_dict, bvp_labels_dict, resp_preds_dict = run_inference(model, loader, cfg)

    # 6. Blink rate
    print("\n=== Detecting blink rates ===")
    blink_rate_dict = {}
    for subj in subjects:
        sk = subj["subj_key"]
        print(f"  {sk} ...", end=" ", flush=True)
        br = detect_blink_rate(subj["video_path"], fps=cfg["VIDEO_FPS"])
        blink_rate_dict[sk] = br
        print(f"{br:.2f} blinks/min")

    # 7. Collect per-subject results
    print("\n=== Per-subject results ===")
    per_subject_results, hr_preds_all, gt_hrs_all, snr_all = collect_results(
        subjects, bvp_preds_dict, bvp_labels_dict,
        resp_preds_dict, blink_rate_dict, fs=cfg["VIDEO_FPS"]
    )

    # 8. Aggregate metrics
    print("\n=== Aggregate metrics ===")
    aggregate_metrics = compute_aggregate_metrics(hr_preds_all, gt_hrs_all, snr_all)

    # 9. Export
    print("\n=== Exporting results ===")
    export_results(per_subject_results, aggregate_metrics,
                   len(hr_preds_all), cfg["OUTPUT_DIR"])


if __name__ == "__main__":
    main()
