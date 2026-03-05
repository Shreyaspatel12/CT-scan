# CT Scan Classification Pipeline

CT scan classification using CTViT (Vision Transformer for CT) on the LIDC-IDRI dataset.
Covers the full pipeline from raw DICOM loading to fine-tuning with PEFT methods.

---

## Project Structure

```
CT-scan/
├── models/
│   ├── ctvit.py              ← CTViT transformer architecture (predefined, do not modify)
│   └── util.py               ← Attention, dropout, window partition utilities (predefined, do not modify)
├── utils.py                  ← Shared data loading, preprocessing, datasets
├── ct_pipeline.ipynb         ← Stage 1: Original baseline pipeline
├── pretrain.py               ← Stage 2: Standalone pretraining script
├── finetune.py               ← Stage 2: Standalone fine-tuning script
├── huggingface1.ipynb        ← Stage 3: HuggingFace implementation with PEFT
└── tcia-diagnosis-data-2012-04-20.xls  ← TCIA diagnosis labels
```

---

## Dataset

**LIDC-IDRI** (Lung Image Database Consortium)
- 1018 CT scans total (unlabeled — used for pretraining)
- 157 labeled patients from TCIA diagnosis Excel file (used for fine-tuning)
- Download from: https://www.cancerimagingarchive.net/collection/lidc-idri/

**Diagnosis classes:**
| Label | Class |
|-------|-------|
| 0 | Unknown |
| 1 | Benign |
| 2 | Malignant Primary |
| 3 | Malignant Metastatic |

**Folder structure expected:**
```
Data/
└── LIDC-IDRI-0001/
    └── study/
        └── series/
            ├── slice001.dcm
            ├── slice002.dcm
            └── ...
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Shreyaspatel12/CT-scan.git
cd CT-scan

# Create and activate virtual environment
python -m venv menv
source menv/bin/activate          # Mac/Linux
menv\Scripts\activate             # Windows

# Install dependencies
pip install pydicom opencv-python-headless torch torchvision
pip install transformers peft accelerate scikit-learn pandas openpyxl
```

---

## Stage 1 — Baseline Pipeline (`ct_pipeline.ipynb`)

**What it is:**
The original working pipeline built from scratch. Everything in one notebook — data loading, preprocessing, tokenization, pretraining, fine-tuning, and evaluation.

**What it does:**
1. Loads DICOM CT scans from nested LIDC-IDRI folder structure
2. Converts to Hounsfield Units, normalizes, picks 3 middle slices, resizes to 224×224
3. Tokenizes images into 196 patches using `PatchEmbed` (Conv2D → 196 × 1024 tokens)
4. Pretrains CTViT using Masked Patch Modeling (75% mask ratio, MSE loss) on 10 samples
5. Fine-tunes on 157 labeled patients with weighted CrossEntropy loss
6. Evaluates with Macro F1 score

**How to run:**
```bash
jupyter notebook
# Open ct_pipeline.ipynb and run all cells
```

**Results:**
- Macro F1: **0.2936**
- Best class: Malignant-Metastatic (F1 = 0.57)
- Worst class: Benign (F1 = 0.00)

**Limitations of this stage:**
- Everything hardcoded in one file
- Manual training loop
- Random (non-stratified) train/val split
- No learning rate scheduling
- Single GPU only

---

## Stage 2 — Refactored Pipeline (`pretrain.py` + `finetune.py`)

**What changed:**
Code was restructured into separate files with shared utilities extracted to `utils.py`.

**`utils.py` — shared across all files:**
- `get_series_paths()` — finds all DICOM folders
- `load_ct_series()` — loads and sorts DICOM slices, converts HU
- `normalize_volume()`, `get_three_slices()`, `resize_slices()` — preprocessing
- `LIDCDataset` — unlabeled dataset for pretraining
- `LIDCLabeledDataset` — labeled dataset for fine-tuning (cross-references Excel file)
- `PatchEmbed` — tokenization (Conv2D → 196 × 1024)
- `load_labels()` — reads TCIA Excel file

**`pretrain.py` — self-supervised pretraining:**
- Masked Patch Modeling — randomly zeros 75% of the 196 tokens
- MSE reconstruction loss on masked positions only
- Saves weights → `pretrained_weights.pth`

```bash
python pretrain.py
```

**`finetune.py` — supervised fine-tuning:**
- Loads pretrained weights from `pretrained_weights.pth`
- Stratified 80/20 train/val split (guarantees all 4 classes in both sets)
- Weighted CrossEntropy loss (handles class imbalance)
- Cosine LR scheduler
- Different learning rates: backbone `1e-5`, classifier head `1e-4`
- Saves weights → `finetuned_weights.pth`

```bash
python finetune.py
```

**Improvements over Stage 1:**
| Stage 1 | Stage 2 |
|---------|---------|
| All code in one notebook | Separated into focused files |
| Random train/val split | Stratified split |
| Fixed learning rate | Cosine annealing scheduler |
| No config block | CONFIG block at top of each file |
| Single GPU | Ready for multi-GPU |

---

## Stage 3 — HuggingFace Pipeline (`huggingface1.ipynb`)

**What it is:**
Full reimplementation using HuggingFace libraries for both pretraining and fine-tuning with PEFT methods.

**How to run:**
```bash
jupyter notebook
# Open huggingface1.ipynb and run cells in order 1 → 15
```

### Part A — Pretraining with HuggingFace Trainer

**Key design decision — masking inside `forward()`:**

Instead of a manual for loop, masking is implemented inside `CTViTForPretraining.forward()`. This allows HuggingFace Trainer to handle the entire pretraining loop automatically.

```
Old (Stage 1 & 2):    manual loop → mask tokens → CTViT → loss
New (Stage 3):        Trainer → CTViTForPretraining.forward() → mask inside → loss
```

**Why this works:**
Trainer's internal loop just calls `loss = model(**batch).loss`. Since masking and loss computation both happen inside `forward()`, Trainer works without any changes.

**`CTViTForPretraining`** (extends `PreTrainedModel`):
- `mask_tokens()` lives inside the model
- Fresh random mask generated every forward call
- Returns `PretrainOutput(loss=mse_loss)`

**`PretrainConfig`** (extends `PretrainedConfig`):
- Stores `embed_dim`, `depth`, `num_heads`, `mask_ratio`, `patch_size`

### Part B — Fine-tuning with PEFT

**`CTViTClassifier`** (extends `PreTrainedModel` — not `nn.Module`):

This is the critical design decision. Extending `PreTrainedModel` instead of `nn.Module` gives the model a proper `config` object which all HuggingFace PEFT methods require.

**`CTViTConfig`** (extends `PretrainedConfig`):
- Stores architecture settings
- Includes `vocab_size=1` (dummy value required by some PEFT methods)

### PEFT Methods Compared

All 3 methods are run automatically in sequence and a final comparison table is printed.

| Method | What it trains | Parameters | Best for |
|--------|---------------|------------|----------|
| **LoRA** | Small matrices added to q, k, v attention layers | ~0.77% | General purpose, most popular |
| **AdaLoRA** | Like LoRA but auto-allocates rank to important layers | ~0.77% | When unsure which layers matter |
| **IA3** | Tiny scaling vectors on activations | ~0.07% | Minimal resources, large datasets |

**PEFT results on 157 patients:**
| Method | Macro F1 | Trainable Params |
|--------|----------|-----------------|
| Full fine-tuning (Stage 2) | 0.2936 | 100% |
| LoRA | ~0.28 | 0.77% |
| AdaLoRA | ~0.28 | 0.77% |
| IA3 | ~0.22 | 0.07% |

---

## Alternatives Considered but Not Included

### Libraries evaluated for pretraining efficiency

| Library | Decision | Reason |
|---------|----------|--------|
| **HuggingFace Accelerate** | ✅ Used | Simple multi-GPU, minimal code changes, wraps existing loop |
| **DeepSpeed** | ❌ Not used | Overkill for current dataset size, complex config, best for 10B+ parameter models |
| **PyTorch FSDP** | ❌ Not used | Requires multiple GPUs, unnecessary for single machine |
| **NVIDIA Apex** | ❌ Not used | Replaced by native PyTorch AMP, no longer recommended |

### PEFT methods evaluated

| Method | Decision | Reason |
|--------|----------|--------|
| **LoRA** | ✅ Used | Works perfectly with custom vision models |
| **AdaLoRA** | ✅ Used | Works perfectly, adaptive rank allocation |
| **IA3** | ✅ Used | Works perfectly, very lightweight |
| **Prefix Tuning** | ❌ Removed | Built for language models only — requires `vocab_size`, `get_text_config()`, internal text token embeddings that don't exist in vision models |

### Pretraining masking approaches evaluated

| Approach | Decision | Reason |
|----------|----------|--------|
| **Mask in training loop** | Stage 1 & 2 | Simple, works, but requires manual loop — can't use Trainer |
| **Mask in Dataset/DataCollator** | ❌ Not used | Fixed mask per epoch — model sees same masked patches every epoch, worse learning |
| **Mask inside model.forward()** | ✅ Stage 3 | Fresh random mask every call, fully compatible with Trainer, cleanest approach |

---

## Architecture Summary

```
Input CT Scan (224×224×3)
        ↓
PatchEmbed (Conv2D) → 196 tokens × 1024 dims
        ↓
CTViT (24 transformer blocks, 16 heads, embed_dim=1024)
  - BlockDrop: LayerNorm → AttentionDrop → residual → MlpDrop
  - AttentionDrop: multi-head self-attention with optional windowing
  - Window attention disabled (window_block_indexes=[])
        ↓
Mean Pool → 1024-dim vector
        ↓
Linear Classifier → 4 classes
```

---

## Requirements

```
torch
torchvision
pydicom
opencv-python-headless
transformers
peft
accelerate
scikit-learn
pandas
openpyxl
fairscale          ← required by models/ctvit.py
```
