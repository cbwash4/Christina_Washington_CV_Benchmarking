#!/usr/bin/env python3
"""Command-line runner for Assignment 3 deep architecture benchmarking."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from christina_washington_cv_benchmarking.deep_data import (
    build_fixed_dataloaders,
)
from christina_washington_cv_benchmarking.deep_runner import (
    run_experiment,
)


TORCHVISION_MODELS = [
    "resnet18",
    "resnet50",
    "densenet121",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "alexnet",
    "vgg16",
    "googlenet",
    "convnext_tiny",
]

YOLO_MODEL = "yolo11n_cls"

SUPPORTED_MODELS = TORCHVISION_MODELS + [YOLO_MODEL, "all"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Assignment 3 deep architecture benchmarks "
            "using the saved fixed Fashion-MNIST split."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Architecture to benchmark, or 'all'.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20).",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader batch size (default: 32).",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device (default: auto).",
    )

    parser.add_argument(
        "--split-file",
        default=str(
            ROOT / "data" / "fashion_mnist_fixed_split.npz"
        ),
        help="Path to the saved fixed split NPZ file.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate arguments, dataset, split, DataLoaders, "
            "and model selection without training."
        ),
    )

    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no CUDA device is available."
        )

    return requested


def main():
    args = parse_args()

    device = resolve_device(args.device)

    train_loader, val_loader, test_loader = (
        build_fixed_dataloaders(
            split_file=args.split_file,
            batch_size=args.batch_size,
        )
    )

    models = (
        TORCHVISION_MODELS + [YOLO_MODEL]
        if args.model == "all"
        else [args.model]
    )

    print("=" * 70)
    print("ASSIGNMENT 3 DEEP BENCHMARK")
    print("=" * 70)

    print(f"Model selection : {args.model}")
    print(f"Models to run   : {', '.join(models)}")
    print(f"Device          : {device}")
    print(f"Epochs          : {args.epochs}")
    print(f"Learning rate   : {args.learning_rate}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Train images    : {len(train_loader.dataset)}")
    print(f"Validation      : {len(val_loader.dataset)}")
    print(f"Test images     : {len(test_loader.dataset)}")

    if args.dry_run:
        images, labels = next(iter(train_loader))

        print(f"Input shape     : {tuple(images.shape)}")
        print(f"Label shape     : {tuple(labels.shape)}")

        print("\nDRY RUN PASSED")
        print("No models were trained.")
        return

    for model_name in models:
        print("\n" + "=" * 70)
        print(f"RUNNING: {model_name}")
        print("=" * 70)

        if model_name == YOLO_MODEL:
            from ultralytics import YOLO

            yolo_data = ROOT / "data" / "yolo_fashion"

            if not yolo_data.exists():
                raise FileNotFoundError(
                    f"YOLO dataset not found: {yolo_data}"
                )

            yolo_device = 0 if device == "cuda" else "cpu"

            model = YOLO("yolo11n-cls.pt")

            model.train(
                data=str(yolo_data),
                epochs=args.epochs,
                batch=args.batch_size,
                imgsz=224,
                device=yolo_device,
                optimizer="auto",
                lr0=0.01,
                seed=42,
                pretrained=True,
                workers=2,
                project=str(ROOT / "yolo_runs"),
                name="yolo11n_cls",
            )

        else:
            run_experiment(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                num_classes=3,
                device=device,
            )


if __name__ == "__main__":
    main()
