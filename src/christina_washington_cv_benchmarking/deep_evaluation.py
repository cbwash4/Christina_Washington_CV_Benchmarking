
import os
import time
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
    model,
    test_loader,
    model_name,
    checkpoint_path=None,
    device="cuda"
):
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    # Parameter count
    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    # Checkpoint size
    checkpoint_size_mb = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_size_mb = (
            os.path.getsize(checkpoint_path)
            / (1024 ** 2)
        )

    # Reset GPU memory statistics
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warm-up GPU
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            for _ in range(3):
                outputs = model(inputs)

            break

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    total_images = 0

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            predictions = outputs.argmax(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            total_images += labels.size(0)

    if device == "cuda":
        torch.cuda.synchronize()

    total_inference_time = (
        time.perf_counter() - start_time
    )

    inference_ms_per_image = (
        total_inference_time
        / total_images
        * 1000
    )

    fps = (
        total_images / total_inference_time
    )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    per_class_report = classification_report(
        all_labels,
        all_predictions,
        output_dict=True,
        zero_division=0
    )

    peak_gpu_memory_mb = None

    if device == "cuda":
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "inference_ms_per_image": inference_ms_per_image,
        "fps": fps,
        "parameter_count": parameter_count,
        "checkpoint_size_mb": checkpoint_size_mb,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "confusion_matrix": cm,
        "per_class_report": per_class_report,
        "predictions": all_predictions,
        "labels": all_labels
    }
