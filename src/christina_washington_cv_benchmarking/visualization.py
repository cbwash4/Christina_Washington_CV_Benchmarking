
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def create_output_directories(output_dir="benchmark_results"):
    output_path = Path(output_dir)

    (output_path / "confusion_matrices").mkdir(
        parents=True,
        exist_ok=True
    )

    (output_path / "classification_reports").mkdir(
        parents=True,
        exist_ok=True
    )

    return output_path


def safe_filename(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def save_class_distribution(
    full_distribution,
    output_path
):
    classes = list(full_distribution.keys())
    counts = list(full_distribution.values())

    plt.figure(figsize=(8, 5))
    plt.bar(classes, counts)

    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        output_path / "class_distribution.png",
        dpi=150
    )

    plt.close()


def save_confusion_matrix(
    matrix,
    class_names,
    model_name,
    output_path
):
    plt.figure(figsize=(7, 6))

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size
        else 0
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                horizontalalignment="center",
                color=(
                    "white"
                    if matrix[i, j] > threshold
                    else "black"
                )
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.tight_layout()

    filename = (
        safe_filename(model_name)
        + ".png"
    )

    plt.savefig(
        output_path
        / "confusion_matrices"
        / filename,
        dpi=150
    )

    plt.close()


def save_model_comparison(
    summary,
    output_path
):
    successful = summary.dropna(
        subset=["Macro F1"]
    ).copy()

    if successful.empty:
        return

    models = successful["Model"].tolist()

    metrics = [
        ("Accuracy", "Accuracy"),
        ("Macro F1", "Macro F1"),
        ("Train (s)", "Training Time (seconds)"),
        (
            "Infer (ms/img)",
            "Inference Time (ms/image)"
        )
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9)
    )

    axes = axes.flatten()

    for ax, (column, title) in zip(
        axes,
        metrics
    ):
        ax.bar(
            models,
            successful[column]
        )

        ax.set_title(title)
        ax.tick_params(
            axis="x",
            rotation=45
        )

    plt.tight_layout()

    plt.savefig(
        output_path / "model_comparison.png",
        dpi=150
    )

    plt.close()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(v)
            for v in value
        ]

    return value


def save_benchmark_outputs(
    summary,
    model_results,
    class_names,
    full_distribution,
    run_configuration,
    benchmark_metrics,
    output_dir="benchmark_results"
):
    output_path = create_output_directories(
        output_dir
    )

    summary.to_csv(
        output_path / "benchmark_summary.csv",
        index=False
    )

    with open(
        output_path / "benchmark_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            _json_safe(benchmark_metrics),
            file,
            indent=4
        )

    with open(
        output_path / "run_configuration.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            _json_safe(run_configuration),
            file,
            indent=4
        )

    save_class_distribution(
        full_distribution,
        output_path
    )

    for model_name, result in model_results.items():

        if result.get("status") == "failed":
            continue

        save_confusion_matrix(
            result["confusion_matrix"],
            class_names,
            model_name,
            output_path
        )

        report_filename = (
            safe_filename(model_name)
            + ".csv"
        )

        result[
            "classification_report"
        ].to_csv(
            output_path
            / "classification_reports"
            / report_filename
        )

    save_model_comparison(
        summary,
        output_path
    )

    return output_path
