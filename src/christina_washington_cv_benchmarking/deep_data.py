"""Reproducible data loading for Assignment 3 deep benchmarks."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FixedFashionDataset(Dataset):
    """Dataset backed by the saved 900-image Fashion-MNIST subset."""

    def __init__(self, images, labels, indices, transform=None):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        idx = int(self.indices[position])

        image = Image.fromarray(
            self.images[idx].astype(np.uint8),
            mode="L"
        )

        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_deep_transform():
    """Return standardized preprocessing for pretrained CNNs."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_fixed_dataloaders(
    split_file="data/fashion_mnist_fixed_split.npz",
    batch_size=32,
    num_workers=0,
):
    """Build train/validation/test loaders from the saved fixed split."""

    split_file = Path(split_file)

    if not split_file.exists():
        raise FileNotFoundError(
            f"Fixed split file not found: {split_file}"
        )

    saved = np.load(split_file)

    required = {
        "images",
        "labels",
        "train_idx",
        "val_idx",
        "test_idx",
    }

    missing = required.difference(saved.files)

    if missing:
        raise ValueError(
            f"Fixed split is missing arrays: {sorted(missing)}"
        )

    images = saved["images"]
    labels = saved["labels"]

    transform = get_deep_transform()

    train_dataset = FixedFashionDataset(
        images,
        labels,
        saved["train_idx"],
        transform,
    )

    val_dataset = FixedFashionDataset(
        images,
        labels,
        saved["val_idx"],
        transform,
    )

    test_dataset = FixedFashionDataset(
        images,
        labels,
        saved["test_idx"],
        transform,
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
