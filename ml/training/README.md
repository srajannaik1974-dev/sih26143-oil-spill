# Training Module — SIH 2026 PS 26143

## Files

| File | Purpose |
|---|---|
| [`dataset.py`](dataset.py) | PyTorch `Dataset` — loads 2-ch SAR TIFFs, normalises, resizes |
| [`unet.py`](unet.py) | U-Net architecture — 2 input ch, 1 output ch, raw logits |
| [`train.py`](train.py) | Training loop — BCE+Dice loss, early stopping, checkpoint |
| [`evaluate.py`](evaluate.py) | Test-set evaluation — Dice/IoU table + 5-panel visualisation |
| [`inference.py`](inference.py) | Single-image prediction — binary mask + probability map |
| `checkpoints/best_unet.pth` | Best saved model (created after training) |

---

## Quick Start (Google Colab)

```python
# 1. Install dependencies
!pip install -q torch torchvision rasterio tqdm matplotlib

# 2. Clone the repo (or mount Drive)
!git clone https://github.com/your-org/sih26143-oil-spill.git
import sys; sys.path.insert(0, "/content/sih26143-oil-spill")

# 3. Train
!python ml/training/train.py \
    --data-dir /content/sih26143/prototype \
    --epochs   50 \
    --batch-size 4

# 4. Evaluate on test set
!python ml/training/evaluate.py \
    --data-dir /content/sih26143/prototype \
    --ckpt     ml/training/checkpoints/best_unet.pth

# 5. Inference on a new image
from ml.training.inference import OilSpillPredictor
predictor = OilSpillPredictor("ml/training/checkpoints/best_unet.pth")
mask, prob = predictor.predict("/path/to/new_image.tif")
predictor.visualise("/path/to/new_image.tif")
```

---

## Dataset Format Expected

```
prototype/
├── train/
│   ├── images/   # (2, 2048, 2048) float32 TIFF
│   └── masks/    # (1, 2048, 2048) uint8  TIFF  values: {0, 1}
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Filenames must match between `images/` and `masks/` (same numeric stem, e.g. `0001.tif`).

---

## Architecture

```
Input (B, 2, 512, 512)
    │
    ├─ Encoder Block 1  →  (B,  64, 512, 512) skip
    ├─ Encoder Block 2  →  (B, 128, 256, 256) skip
    ├─ Encoder Block 3  →  (B, 256, 128, 128) skip
    ├─ Encoder Block 4  →  (B, 512,  64,  64) skip
    │
    ├─ Bottleneck       →  (B, 1024, 32,  32)
    │
    ├─ Decoder Block 4  ←  skip4
    ├─ Decoder Block 3  ←  skip3
    ├─ Decoder Block 2  ←  skip2
    ├─ Decoder Block 1  ←  skip1
    │
    └─ Output Conv 1×1 →  (B, 1, 512, 512)  [RAW LOGITS]
```

**No sigmoid in the forward pass** — apply externally for evaluation/inference.

---

## Loss Function

```
Total Loss = 0.5 × BCEWithLogitsLoss  +  0.5 × Soft Dice Loss
```

- **BCEWithLogitsLoss**: pixel-level binary cross-entropy (numerically stable — sigmoid applied internally).
- **Soft Dice Loss**: overlap-based loss — critical for class-imbalanced oil-spill segmentation.

---

## Metrics

| Metric | Formula | When used |
|---|---|---|
| Dice Score | 2·\|P∩T\| / (\|P\|+\|T\|) | Training, Validation, Test |
| IoU | \|P∩T\| / \|P∪T\| | Validation, Test |

---

## Training Hyperparameters (defaults)

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Standard, adaptive LR |
| Learning rate | 1e-4 | Safe starting LR for U-Net |
| Weight decay | 1e-5 | L2 regularisation |
| Batch size | 4 | Conservative — fits most GPU VRAM |
| Image size | 512×512 | Standard for U-Net segmentation |
| LR scheduler | ReduceLROnPlateau (×0.5) | Adapts when val Dice plateaus |
| Early stopping | 10 epochs | Prevents overfitting on small set |
| Seed | 42 | Full reproducibility |

---

## Notes

- SAR images have **2 channels** (VV + VH polarisation) — **not RGB**.
- Normalisation uses **2nd–98th percentile clipping per channel per image** — handles the wide Sigma0 dB range robustly.
- The model is trained at **512×512** resolution — inference automatically resizes input and upsizes output back to original dimensions.
- Masks are **binary** (0 = background, 1 = oil spill).
- The best checkpoint is saved based on **validation Dice**, not training loss.
