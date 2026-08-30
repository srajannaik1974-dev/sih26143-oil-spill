# M1 U-Net Checkpoint Compatibility Test

**Test Date**: 2026-08-29  
**Auditor**: Member 6 (Backend / Integration Layer)  
**Target Module**: Member 1 Satellite Oil Spill Detection (`ml/training/unet.py` & `ml/training/inference.py`)

---

## 1. Checkpoint

- **Path**: `C:\Users\prapt\OneDrive\Desktop\best_unet.pth`
- **File Size**: 31,417,344 bytes (~31.4 MB total tensor payload size)
- **Format**: PyTorch Checkpoint Archive (Extracted directory containing `data.pkl` and `data/` tensor files)
- **Saved Hyperparameters**:
  ```python
  {
      'data_dir': '/content/drive/MyDrive/SIH26143/prototype',
      'ckpt_dir': '/content/drive/MyDrive/SIH26143/checkpoints',
      'epochs': 10,
      'batch_size': 1,
      'lr': 0.0001,
      'image_size': 256,
      'base_features': 32,
      'patience': 5,
      'seed': 42
  }
  ```

---

## 2. PyTorch Loading

- **Result**: **PASS** (`torch.load()` executed cleanly with `weights_only=False`)
- **Errors**: None. All metadata and weights loaded into PyTorch `UNet` instance without error.

---

## 3. Existing M1 Architecture

- **Class**: `UNet` (`ml/training/unet.py`)
- **Input Channels**: 1 (Sentinel-1 SAR VV channel)
- **Output Channels**: 1 (Binary segmentation raw logit)
- **Base Features**: 32
- **Parameter Count**: **7,849,025** trainable parameters

---

## 4. State Dict Compatibility

- **Matching Keys**: **91 / 91 keys (100% exact match)**
- **Missing Keys**: **0**
- **Unexpected Keys**: **0**
- **Shape Mismatches**: **0**
- **`load_state_dict(strict=True)`**: **PASSED**

---

## 5. Inference Test

- **Input Used**: None (*Sample 1-channel Sentinel-1 SAR `.tif` GeoTIFF image files are currently uncommitted in the repository*).
- **Inference Result**: **BLOCKED** by missing `.tif` sample image asset in repository.
- **Output Shape**: N/A
- **Spill Detection Result**: N/A
- **Errors**: None

---

## 6. get_spill_location()

- **Tested / Not Tested**: **Not Tested**
- **Actual Result**: Blocked by missing GeoTIFF sample image asset (`rasterio` requires a valid `.tif` file with CRS geospatial metadata).

---

## 7. FINAL VERDICT

**PASS — checkpoint loads but inference test unavailable**

- The trained U-Net checkpoint `C:\Users\prapt\OneDrive\Desktop\best_unet.pth` is **100% architecturally compatible** with the repository's existing `UNet(in_channels=1, out_channels=1, base_features=32)` model class. All 91 state_dict tensor names, shapes, and parameter counts align perfectly.

---

## 8. Recommended Next Step

1. Package/compress `C:\Users\prapt\OneDrive\Desktop\best_unet.pth` into a standard `.pth` single zip file and place it at `ml/training/checkpoints/best_unet.pth`.
2. Add a sample Sentinel-1 SAR GeoTIFF (`.tif`) image to `ml/dataset/` to enable automated end-to-end inference verification.
