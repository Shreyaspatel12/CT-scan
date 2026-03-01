import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import LIDCDataset, PatchEmbed
from models.ctvit import CTViT

# CONFIG — change these as needed
ROOT_DIR     = "Data/"
NUM_PRETRAIN = 10
BATCH_SIZE   = 2
NUM_EPOCHS   = 5
LEARNING_RATE= 1e-4
MASK_RATIO   = 0.75        # fraction of patches to mask
SAVE_PATH    = "pretrained_weights.pth"

# MASKING
def mask_tokens(tokens, mask_ratio=0.75):
    B, N, C = tokens.shape
    num_masked = int(N * mask_ratio)

    noise = torch.rand(B, N, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask_ids = ids_shuffle[:, :num_masked]

    mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
    mask.scatter_(1, mask_ids, True)

    masked_tokens = tokens.clone()
    masked_tokens[mask] = 0.0

    return masked_tokens, mask


# MAIN
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = LIDCDataset(ROOT_DIR)
    pretrain_subset = torch.utils.data.Subset(dataset, range(NUM_PRETRAIN))
    pretrain_loader = DataLoader(pretrain_subset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Pretraining on {len(pretrain_subset)} samples")

    # Model
    patch_embed = PatchEmbed(in_channels=3, embed_dim=1024, patch_size=16).to(device)

    model = CTViT(
        embed_dim=1024,
        depth=24,
        num_heads=16,
    ).to(device)

    # Reconstruction head
    patch_pixels = 16 * 16 * 3
    reconstruction_head = nn.Linear(1024, patch_pixels).to(device)

    # Optimizer
    optimizer = optim.Adam(
        list(model.parameters()) +
        list(patch_embed.parameters()) +
        list(reconstruction_head.parameters()),
        lr=LEARNING_RATE
    )

    # Pretraining Loop
    model.train()
    reconstruction_head.train()

    print("\nStarting pretraining...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        for batch in pretrain_loader:
            batch = batch.to(device)

            tokens = patch_embed(batch)
            masked_tokens, mask = mask_tokens(tokens, MASK_RATIO)

            output = model(
                masked_tokens,
                window_size=(7, 7),
                window_block_indexes=[],
                spatial_size=(14, 14),
                cls_embed=False,
            )

            pred   = reconstruction_head(output)
            target = reconstruction_head(tokens.detach())

            # MSE loss on masked positions only
            loss = ((pred - target) ** 2)[mask].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Loss: {total_loss/len(pretrain_loader):.4f}")

    # Save the weights
    print(f"\nPretraining done! Saving weights to {SAVE_PATH}...")
    torch.save({
        "model":       model.state_dict(),
        "patch_embed": patch_embed.state_dict(),
    }, SAVE_PATH)
    print(f"Saved → {SAVE_PATH}")

if __name__ == "__main__":
    main()
