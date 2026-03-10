# Model Groups Reference

Groups are organized so that all models within a group share **identical preprocessing**. One preprocess run per notebook, then loop over models.

---

## BigSmall (standalone notebook)
- **Notebook**: `bigsmall_inference.ipynb`
- **Model**: BigSmall (multi-task: BVP + respiration + AU)
- **Preprocessing**: Dual-resolution (Big: 144x144 Standardized, Small: 9x9 DiffNormalized), chunk=3, label=DiffNormalized (49ch)
- **Data format**: Pickle dict {0: big, 1: small}, NDCHW

| Weights |
|---------|
| `BP4D_BigSmall_Multitask_Fold1.pth` |
| `BP4D_BigSmall_Multitask_Fold2.pth` |
| `BP4D_BigSmall_Multitask_Fold3.pth` |

---

## Group A — DeepPhys + TS-CAN
- **Notebook**: `groupA_inference.ipynb`
- **Preprocessing**: DiffNormalized + Standardized (6ch), 72x72, chunk=180, label=DiffNormalized
- **Data format**: NDCHW `(N, D, 6, 72, 72)`
- **Post-processing**: cumsum → detrend → bandpass

| Name | Model Class | Weights |
|------|-------------|---------|
| PURE_DeepPhys | DeepPhys | `PURE_DeepPhys.pth` |
| PURE_TSCAN | Tscan | `PURE_TSCAN.pth` |
| SCAMPS_DeepPhys | DeepPhys | `SCAMPS_DeepPhys.pth` |
| SCAMPS_TSCAN | Tscan | `SCAMPS_TSCAN.pth` |
| UBFC-rPPG_DeepPhys | DeepPhys | `UBFC-rPPG_DeepPhys.pth` |
| UBFC-rPPG_TSCAN | Tscan | `UBFC-rPPG_TSCAN.pth` |
| BP4D_PseudoLabel_DeepPhys | DeepPhys | `BP4D_PseudoLabel_DeepPhys.pth` |
| BP4D_PseudoLabel_TSCAN | Tscan | `BP4D_PseudoLabel_TSCAN.pth` |
| MA-UBFC_deepphys | DeepPhys | `MA-UBFC_deepphys.pth` |
| MA-UBFC_tscan | Tscan | `MA-UBFC_tscan.pth` |

**Inference notes:**
- **DeepPhys**: `DeepPhys(img_size=72)`. Input `(N*D, 6, H, W)`, output `(N*D, 1)`. Frame-independent.
- **TS-CAN**: `TSCAN(frame_depth=10, img_size=72)`. Needs `base_len = frame_depth` alignment. Output `(N*D, 1)`.

---

## Group B — EfficientPhys
- **Notebook**: `groupB_inference.ipynb`
- **Preprocessing**: Standardized (3ch), 72x72, chunk=180, label=DiffNormalized
- **Data format**: NDCHW `(N, D, 3, 72, 72)`
- **Post-processing**: cumsum → detrend → bandpass

| Name | Weights |
|------|---------|
| PURE_EfficientPhys | `PURE_EfficientPhys.pth` |
| SCAMPS_EfficientPhys | `SCAMPS_EfficientPhys.pth` |
| UBFC-rPPG_EfficientPhys | `UBFC-rPPG_EfficientPhys.pth` |
| iBVP_EfficientPhys | `iBVP_EfficientPhys.pth` |
| BP4D_PseudoLabel_EfficientPhys | `BP4D_PseudoLabel_EfficientPhys.pth` |
| MA-UBFC_efficientphys | `MA-UBFC_efficientphys.pth` |

**Inference notes:**
- `EfficientPhys(frame_depth=10, img_size=72)`. Flatten to `(N*D, 3, H, W)`, align to `base_len=frame_depth`, **append one extra frame**, then forward → `(N*D, 1)`. Model does `torch.diff` internally.

---

## Group C — PhysNet
- **Notebook**: `groupC_inference.ipynb`
- **Preprocessing**: DiffNormalized (3ch), 72x72, chunk=128, label=DiffNormalized
- **Data format**: NCDHW `(N, 3, 128, 72, 72)`
- **Post-processing**: cumsum → detrend → bandpass

| Name | Weights |
|------|---------|
| PURE_PhysNet | `PURE_PhysNet_DiffNormalized.pth` |
| SCAMPS_PhysNet | `SCAMPS_PhysNet_DiffNormalized.pth` |
| UBFC-rPPG_PhysNet | `UBFC-rPPG_PhysNet_DiffNormalized.pth` |
| BP4D_PseudoLabel_PhysNet | `BP4D_PseudoLabel_PhysNet_DiffNormalized.pth` |
| MA-UBFC_physnet | `MA-UBFC_physnet.pth` |

**Inference notes:**
- `PhysNet_padding_Encoder_Decoder_MAX(frames=128)`. Output tuple `(rPPG, _, _, _)` — use `[0]`, shape `(N, T)`.

---

## Group D — PhysFormer
- **Notebook**: `groupD_inference.ipynb`
- **Preprocessing**: DiffNormalized (3ch), 128x128, chunk=160, label=DiffNormalized
- **Data format**: NCDHW `(N, 3, 160, 128, 128)`
- **Post-processing**: cumsum → detrend → bandpass

| Name | Weights |
|------|---------|
| PURE_PhysFormer | `PURE_PhysFormer_DiffNormalized.pth` |
| SCAMPS_PhysFormer | `SCAMPS_PhysFormer_DiffNormalized.pth` |
| UBFC-rPPG_PhysFormer | `UBFC-rPPG_PhysFormer_DiffNormalized.pth` |

**Inference notes:**
- `ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.2, theta=0.7)`. **Requires `gra_sharp=2.0` in forward**. Output tuple `(rPPG, _, _, _)` — use `[0]`.

---

## Group E — PhysMamba
- **Notebook**: `groupE_inference.ipynb`
- **Preprocessing**: DiffNormalized (3ch), 128x128, chunk=128, label=DiffNormalized
- **Data format**: NCDHW `(N, 3, 128, 128, 128)`
- **Post-processing**: cumsum → detrend → bandpass

| Name | Weights |
|------|---------|
| PURE_PhysMamba | `PURE_PhysMamba_DiffNormalized.pth` |
| UBFC-rPPG_PhysMamba | `UBFC-rPPG_PhysMamba_DiffNormalized.pth` |

**Inference notes:**
- `PhysMamba(frames=128)`. Output `rPPG` shape `(N, T)` directly (no tuple).

---

## Group F — iBVPNet + FactorizePhys
- **Notebook**: `groupF_inference.ipynb`
- **Preprocessing**: Raw (3ch, no normalization), 72x72, chunk=160, label=Standardized
- **Data format**: NCDHW `(N, 3, 160, 72, 72)`
- **Post-processing**: detrend → bandpass (no cumsum)
- **Both models need frame padding** (append 1 extra frame in temporal dim)

| Name | Model Class | Weights |
|------|-------------|---------|
| PURE_iBVPNet | iBVPNet | `PURE_iBVPNet.pth` |
| PURE_FactorizePhys | FactorizePhys | `PURE_FactorizePhys_FSAM_Res.pth` |
| iBVP_FactorizePhys | FactorizePhys | `iBVP_FactorizePhys_FSAM_Res.pth` |
| SCAMPS_FactorizePhys | FactorizePhys | `SCAMPS_FactorizePhys_FSAM_Res.pth` |
| UBFC-rPPG_FactorizePhys | FactorizePhys | `UBFC-rPPG_FactorizePhys_FSAM_Res.pth` |

**Inference notes:**
- **iBVPNet**: `iBVPNet(frames=160, in_channels=3)`. Pad → `(N, 3, T+1, H, W)`. Output `(N, T)`.
- **FactorizePhys**: `FactorizePhys(frames=160, md_config=..., in_channels=3)`. Same padding. Output tuple — use `[0]`. Load with `strict=False`.
  - md_config: `{FRAME_NUM: 160, MD_FSAM: True, MD_TYPE: "NMF", MD_R: 1, MD_S: 1, MD_STEPS: 3, MD_RESIDUAL: True, MD_INFERENCE: True, MD_TRANSFORM: "T_KAB"}`

---

## Group G — RhythmFormer
- **Notebook**: `groupG_inference.ipynb`
- **Preprocessing**: Standardized (3ch), 128x128, chunk=160, label=Standardized
- **Data format**: NDCHW `(N, D, 3, 128, 128)`
- **Post-processing**: detrend → bandpass (no cumsum)

| Name | Weights |
|------|---------|
| PURE_RhythmFormer | `PURE_RhythmFormer.pth` |
| UBFC-rPPG_RhythmFormer | `UBFC-rPPG_RhythmFormer.pth` |

**Inference notes:**
- `RhythmFormer()`. Input `(N, D, 3, H, W)`. Output `(N, D)`. **Normalize output per-sample**: `(pred - mean) / std`.

---

## Quick reference: preprocessing configs

| Group | Data Type | Channels | Resolution | Chunk | Label | Format |
|-------|-----------|----------|------------|-------|-------|--------|
| BigSmall | Std + DiffNorm (dual-res) | special | 144+9 | 3 | DiffNorm 49ch | NDCHW |
| A | DiffNorm + Std | 6 | 72x72 | 180 | DiffNorm | NDCHW |
| B | Standardized | 3 | 72x72 | 180 | DiffNorm | NDCHW |
| C | DiffNormalized | 3 | 72x72 | 128 | DiffNorm | NCDHW |
| D | DiffNormalized | 3 | 128x128 | 160 | DiffNorm | NCDHW |
| E | DiffNormalized | 3 | 128x128 | 128 | DiffNorm | NCDHW |
| F | Raw | 3 | 72x72 | 160 | Standardized | NCDHW |
| G | Standardized | 3 | 128x128 | 160 | Standardized | NDCHW |

## Weight files

All in `final_model_release/`. Naming: `{TrainDataset}_{ModelName}[_Variant].pth`.

```
final_model_release/
├── BP4D_BigSmall_Multitask_Fold{1,2,3}.pth
├── BP4D_PseudoLabel_{DeepPhys,TSCAN,EfficientPhys,PhysNet_DiffNormalized}.pth
├── MA-UBFC_{deepphys,tscan,efficientphys,physnet}.pth
├── PURE_{DeepPhys,TSCAN,EfficientPhys,RhythmFormer}.pth
├── PURE_{PhysNet,PhysFormer,PhysMamba}_DiffNormalized.pth
├── PURE_{iBVPNet,FactorizePhys_FSAM_Res}.pth
├── SCAMPS_{DeepPhys,TSCAN,EfficientPhys}.pth
├── SCAMPS_{PhysNet,PhysFormer}_DiffNormalized.pth
├── SCAMPS_FactorizePhys_FSAM_Res.pth
├── UBFC-rPPG_{DeepPhys,TSCAN,EfficientPhys,RhythmFormer}.pth
├── UBFC-rPPG_{PhysNet,PhysFormer,PhysMamba}_DiffNormalized.pth
├── UBFC-rPPG_FactorizePhys_FSAM_Res.pth
├── iBVP_{EfficientPhys,FactorizePhys_FSAM_Res}.pth
```
