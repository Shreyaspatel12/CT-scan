import os
import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


# DATA LOADING
def get_series_paths(root_dir):
    series_paths = []
    for root, dirs, files in os.walk(root_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            series_paths.append(root)
    return series_paths


def load_ct_series(series_path):
    slices = []
    for file in os.listdir(series_path):
        if file.lower().endswith(".dcm"):
            try:
                dicom = pydicom.dcmread(
                    os.path.join(series_path, file),
                    force=True
                )
                if hasattr(dicom, "pixel_array"):
                    slices.append(dicom)
            except Exception:
                pass

    if len(slices) == 0:
        raise ValueError(f"No valid DICOM slices found in {series_path}")

    slices.sort(key=lambda x: int(x.InstanceNumber))

    images = []
    for s in slices:
        image = s.pixel_array.astype(np.float32)
        if hasattr(s, "RescaleSlope") and hasattr(s, "RescaleIntercept"):
            image = image * s.RescaleSlope + s.RescaleIntercept
        if image.shape != (512, 512):
            image = cv2.resize(image, (512, 512))
        images.append(image)

    volume = np.stack(images, axis=0)  # (num_slices, H, W)
    return volume


# PREPROCESSING
def normalize_volume(volume):
    return (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)


def get_three_slices(volume):
    mid = volume.shape[0] // 2
    if volume.shape[0] < 3:
        return np.stack([volume[0]] * 3, axis=0)
    return volume[mid - 1: mid + 2]


def resize_slices(slices, size=224):
    return np.stack([cv2.resize(slices[i], (size, size)) for i in range(3)], axis=0)


def preprocess(series_path):
    volume = load_ct_series(series_path)
    volume = normalize_volume(volume)
    slices = get_three_slices(volume)
    slices = resize_slices(slices, size=224)
    return torch.tensor(slices, dtype=torch.float32)  # (3, 224, 224)


# DATASETS
class LIDCDataset(Dataset):
    def __init__(self, root_dir):
        self.series_paths = get_series_paths(root_dir)
        print(f"Unlabeled dataset: {len(self.series_paths)} CT series")

    def __len__(self):
        return len(self.series_paths)

    def __getitem__(self, idx):
        return preprocess(self.series_paths[idx])


class LIDCLabeledDataset(Dataset):
    def __init__(self, root_dir, df):
        self.samples = []

        for _, row in df.iterrows():
            patient_id = str(row["patient_id"]).strip()
            label = int(row["diagnosis"])

            patient_dir = os.path.join(root_dir, patient_id)
            if not os.path.exists(patient_dir):
                continue

            series = get_series_paths(patient_dir)
            if len(series) == 0:
                continue

            self.samples.append((series[0], label))

        print(f"Labeled dataset: {len(self.samples)} samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        series_path, label = self.samples[idx]
        image = preprocess(series_path)
        return image, torch.tensor(label, dtype=torch.long)


# PATCH EMBEDDING (Tokenization)
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Splits image into 16x16 patches and projects to embed_dim.
    Input:  (B, 3, 224, 224)
    Output: (B, 196, 1024)  — flat sequence for CTViT
    """
    def __init__(self, in_channels=3, embed_dim=1024, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)        # (B, 1024, 14, 14)
        x = x.flatten(2)        # (B, 1024, 196)
        x = x.transpose(1, 2)  # (B, 196, 1024)
        return x


# LABEL LOADING
import pandas as pd

def load_labels(xls_path):
    df = pd.read_excel(xls_path)
    df.columns = [col.strip().replace("\n", " ") for col in df.columns]
    df = df.iloc[:, [0, 1]].copy()
    df.columns = ["patient_id", "diagnosis"]
    df = df.dropna(subset=["diagnosis"])
    df["diagnosis"] = df["diagnosis"].astype(int)
    return df
