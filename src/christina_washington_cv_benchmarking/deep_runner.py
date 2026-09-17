
import json
import gc
from pathlib import Path

import pandas as pd
import torch

from .deep_models import get_model
from .deep_train import train_model
from .deep_evaluation import evaluate_model
from .deep_plots import save_training_plots


def run_experiment(
    model_name,
    train_loader,
    val_loader,
    test_loader,
    epochs=20,
    learning_rate=0.001,
    num_classes=3,
    device="cuda"
):

    print("=" * 60)
    print(f"STARTING: {model_name}")
    print("=" * 60)

    # Create model
    model = get_model(
        model_name,
        num_classes=num_classes,
        pretrained=True
    )

    # Train model
    training = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device
    )

    # Evaluate BEST validation checkpoint
    evaluation = evaluate_model(
        model=training["model"],
        test_loader=test_loader,
        model_name=model_name,
        checkpoint_path=training["checkpoint_path"],
        device=device
    )

    # -----------------------------
    # Save training history
    # -----------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    history_df = pd.DataFrame(
        training["history"]
    )

    history_path = (
        results_dir /
        f"{model_name}_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    # Save loss and accuracy curves
    save_training_plots(
        training["history"],
        model_name
    )

    # -----------------------------
    # Save summary metrics
    # -----------------------------
    summary = {
        "Model": model_name,
        "Accuracy": evaluation["accuracy"],
        "Macro Precision": evaluation["macro_precision"],
        "Macro Recall": evaluation["macro_recall"],
        "Macro F1": evaluation["macro_f1"],
        "Weighted Precision": evaluation["weighted_precision"],
        "Weighted Recall": evaluation["weighted_recall"],
        "Weighted F1": evaluation["weighted_f1"],
        "Best Val Accuracy": training["best_val_accuracy"],
        "Best Epoch": training["best_epoch"],
        "Train Time (s)": training["total_training_time"],
        "Infer (ms/img)": evaluation["inference_ms_per_image"],
        "FPS": evaluation["fps"],
        "Inference Benchmark Images": evaluation["inference_benchmark_images"],
        "Parameters": evaluation["parameter_count"],
        "Trainable Parameters": evaluation["trainable_parameter_count"],
        "Checkpoint Size (MB)": evaluation["checkpoint_size_mb"],
        "Peak GPU Memory (MB)": evaluation["peak_gpu_memory_mb"]
    }

    summary_path = (
        results_dir /
        f"{model_name}_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # -----------------------------
    # Save confusion matrix
    # -----------------------------
    cm_dir = Path("confusion_matrices")
    cm_dir.mkdir(exist_ok=True)

    cm_df = pd.DataFrame(
        evaluation["confusion_matrix"]
    )

    cm_df.to_csv(
        cm_dir / f"{model_name}_confusion_matrix.csv",
        index=False
    )

    # -----------------------------
    # Save per-class metrics
    # -----------------------------
    per_class_df = pd.DataFrame(
        evaluation["per_class_report"]
    ).transpose()

    per_class_df.to_csv(
        results_dir /
        f"{model_name}_per_class_metrics.csv"
    )

    print("\n" + "=" * 60)
    print(f"COMPLETED: {model_name}")
    print(
        f"Test Accuracy: "
        f"{evaluation['accuracy']:.4f}"
    )
    print(
        f"Macro F1: "
        f"{evaluation['macro_f1']:.4f}"
    )
    print(
        f"Best Validation Epoch: "
        f"{training['best_epoch']}"
    )
    print("=" * 60)

    # Clean GPU memory
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary
