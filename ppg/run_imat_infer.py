#!/usr/bin/env python
"""Step 2: Run FactorizePhys inference on IMAT-denoised preprocessed data.

Run after run_imat_preprocess.py which saves .npy files.
Separated to avoid memory corruption from IMAT's heavy array operations.
"""
import sys, os, json, glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import periodogram
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = "/home/naver/disk2/HoangDPB/rPPG-Toolbox"
sys.path.insert(0, REPO_ROOT)
from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys

PREPROCESSED_PATH = os.path.join(REPO_ROOT, "preprocessed_data/ppg/5_demo/UBFC-rPPG_FactorizePhys")
OUTPUT_DIR        = os.path.join(REPO_ROOT, "results/ppg/5_demo/UBFC-rPPG_FactorizePhys")
VIDEO_FPS = 30; CHUNK_LENGTH = 160; LABEL_TYPE = "Standardized"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_LABEL = "UBFC-rPPG_FactorizePhys"
MODEL_PATH  = os.path.join(REPO_ROOT, "final_model_release/UBFC-rPPG_FactorizePhys_FSAM_Res.pth")

# Post-processing
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

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Load preprocessed data
    all_input_files = sorted(glob.glob(os.path.join(PREPROCESSED_PATH, "*", "*_input*.npy")))
    print(f"Found {len(all_input_files)} preprocessed clips")
    assert len(all_input_files) > 0, f"No clips found in {PREPROCESSED_PATH}"

    dataset = PPGDataset(all_input_files)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f"DataLoader: {len(loader)} batches")

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
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

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
