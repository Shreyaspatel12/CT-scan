"""
finetune.py — Supervised Fine-tuning on CT Diagnosis Labels
Run with: python finetune.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

from utils import LIDCLabeledDataset, PatchEmbed, load_labels
from models.ctvit import CTViT


# =============================================================================
# CONFIG — change these as needed
# =============================================================================

ROOT_DIR       = "/Volumes/Expansion/Data/manifest-1600709154662/LIDC-IDRI/"
LABELS_PATH    = "tcia-diagnosis-data-2012-04-20.xls"
PRETRAIN_CKPT  = "pretrained_weights.pth"
SAVE_PATH      = "finetuned_weights.pth"

BATCH_SIZE     = 4
NUM_EPOCHS     = 5
LR_BACKBONE    = 1e-5    # lower lr for pretrained backbone
LR_HEAD        = 1e-4    # higher lr for new classification head
NUM_CLASSES    = 4

# Class counts from the dataset (used for weighted loss)
CLASS_COUNTS   = [27, 36, 43, 51]   # Unknown, Benign, Malignant-Primary, Malignant-Metastatic


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Labels ---
    df = load_labels(LABELS_PATH)
    print(f"Total labeled patients: {len(df)}")
    print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")

    # --- Dataset ---
    labeled_dataset = LIDCLabeledDataset(ROOT_DIR, df)

    # Stratified split — ensures all classes appear in both train and val
    indices = list(range(len(labeled_dataset)))
    labels  = [labeled_dataset.samples[i][1] for i in indices]

    train_idx, val_idx = train_test_split(
        indices, test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(labeled_dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(labeled_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}\n")

    # --- Model ---
    patch_embed = PatchEmbed(in_channels=3, embed_dim=1024, patch_size=16).to(device)

    model = CTViT(
        embed_dim=1024,
        depth=24,
        num_heads=16,
    ).to(device)

    # Load pretrained weights
    checkpoint = torch.load(PRETRAIN_CKPT, map_location=device)
    model.load_state_dict(checkpoint["model"])
    patch_embed.load_state_dict(checkpoint["patch_embed"])
    print(f"Pretrained weights loaded from {PRETRAIN_CKPT}")

    # Classification head
    classifier = nn.Linear(1024, NUM_CLASSES).to(device)

    # --- Loss ---
    class_counts  = torch.tensor(CLASS_COUNTS, dtype=torch.float32)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.Adam([
        {"params": model.parameters(),       "lr": LR_BACKBONE},
        {"params": patch_embed.parameters(), "lr": LR_BACKBONE},
        {"params": classifier.parameters(),  "lr": LR_HEAD},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- Fine-tuning Loop ---
    print("Starting fine-tuning...")
    model.train()
    classifier.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            tokens = patch_embed(images)       # (B, 196, 1024)
            output = model(
                tokens,
                window_size=(7, 7),
                window_block_indexes=[],
                spatial_size=(14, 14),
                cls_embed=False,
            )                                  # (B, 196, 1024)

            pooled = output.mean(dim=1)        # (B, 1024)
            logits = classifier(pooled)        # (B, 4)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Loss: {total_loss/len(train_loader):.4f}")

    # --- Evaluation ---
    print("\nEvaluating...")
    model.eval()
    classifier.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            tokens = patch_embed(images)
            output = model(
                tokens,
                window_size=(7, 7),
                window_block_indexes=[],
                spatial_size=(14, 14),
                cls_embed=False,
            )
            pooled = output.mean(dim=1)
            logits = classifier(pooled)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\nMacro F1 Score: {f1:.4f}\n")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Unknown", "Benign", "Malignant-Primary", "Malignant-Metastatic"],
        zero_division=0
    ))

    # --- Save ---
    print(f"Saving model to {SAVE_PATH}...")
    torch.save({
        "model":       model.state_dict(),
        "patch_embed": patch_embed.state_dict(),
        "classifier":  classifier.state_dict(),
    }, SAVE_PATH)
    print(f"Saved → {SAVE_PATH}")

    # ============================================================
    # ✅ FINE-TUNING COMPLETE
    # This script covers:
    #   - Label loading from TCIA Excel file          ✅
    #   - Stratified train/val split (80/20)          ✅
    #   - Pretrained weights loaded                   ✅
    #   - Classification head (1024 → 4 classes)      ✅
    #   - Weighted CrossEntropy Loss                  ✅
    #   - Cosine LR Scheduler                         ✅
    #   - Macro F1 Evaluation                         ✅
    #   - Saved weights → finetuned_weights.pth       ✅
    #
    # Next step: huggingface.ipynb (PEFT methods)
    # ============================================================


if __name__ == "__main__":
    main()
