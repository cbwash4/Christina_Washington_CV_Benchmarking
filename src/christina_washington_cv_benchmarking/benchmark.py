
import time
import numpy as np

from .data_loader import load_dataset
from .preprocessing import preprocess_dataset
from .classical_models import get_classical_models
from .neural_models import train_simple_cnn
from .evaluation import (
    train_and_evaluate_classical_model,
    evaluate_trained_cnn,
    create_summary_dataframe
)
from .visualization import save_benchmark_outputs


PACKAGE_NAME = "Christina_Washington_CV_Benchmarking"
PACKAGE_VERSION = "1.0.0"


def benchmark_image_classification(
    dataset,
    dataset_type: str,
    target_labels,
    color_mode: str
) -> dict:
    """Load an image dataset, train all required classifiers,
    and return a complete benchmark comparison.
    """

    valid_dataset_types = {
        "folder",
        "csv",
        "json",
        "array"
    }

    valid_color_modes = {
        "grayscale",
        "rgb"
    }

    dataset_type = str(dataset_type).lower()
    color_mode = str(color_mode).lower()

    if dataset_type not in valid_dataset_types:
        raise ValueError(
            "dataset_type must be one of: "
            "folder, csv, json, array."
        )

    if color_mode not in valid_color_modes:
        raise ValueError(
            "color_mode must be 'grayscale' or 'rgb'."
        )

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    loaded = load_dataset(
        dataset=dataset,
        dataset_type=dataset_type,
        target_labels=target_labels
    )

    warnings = []

    if loaded["number_skipped"] > 0:
        warnings.append(
            f"{loaded['number_skipped']} invalid or unreadable "
            "image(s) were skipped."
        )

        warnings.extend(
            [
                f"Skipped: {path}"
                for path in loaded["skipped_files"]
            ]
        )

    for class_name, count in loaded["class_counts"].items():
        if count < 5:
            warnings.append(
                f"Class '{class_name}' contains only {count} "
                "valid images. At least 5 images per class "
                "are recommended."
            )

    # ---------------------------------------------------------
    # PREPROCESS + CREATE ONE SHARED SPLIT
    # ---------------------------------------------------------

    processed = preprocess_dataset(
        data=loaded["data"],
        labels=loaded["labels"],
        color_mode=color_mode
    )

    X_flat_train = processed["X_flat_train"]
    X_flat_test = processed["X_flat_test"]

    X_tensor_train = processed["X_tensor_train"]
    X_tensor_test = processed["X_tensor_test"]

    y_train = processed["y_train"]
    y_test = processed["y_test"]

    class_names = processed["class_names"]

    # ---------------------------------------------------------
    # TRAIN + EVALUATE FIVE CLASSICAL MODELS
    # ---------------------------------------------------------

    model_results = {}
    trained_models = {}

    classical_models = get_classical_models()

    for model_name, model in classical_models.items():

        try:
            result = train_and_evaluate_classical_model(
                model_name=model_name,
                model=model,
                X_train=X_flat_train,
                y_train=y_train,
                X_test=X_flat_test,
                y_test=y_test,
                class_names=class_names
            )

            result["status"] = "success"

            model_results[model_name] = result
            trained_models[model_name] = model

        except Exception as exc:

            model_results[model_name] = {
                "status": "failed",
                "error": str(exc)
            }

            warnings.append(
                f"{model_name} failed: {exc}"
            )

    # ---------------------------------------------------------
    # TRAIN + EVALUATE CNN
    # ---------------------------------------------------------

    try:
        cnn_start = time.perf_counter()

        cnn_model, cnn_history = train_simple_cnn(
            X_train=X_tensor_train,
            y_train=y_train,
            number_of_classes=len(class_names)
        )

        cnn_training_time = (
            time.perf_counter() - cnn_start
        )

        cnn_result = evaluate_trained_cnn(
            model=cnn_model,
            X_test=X_tensor_test,
            y_test=y_test,
            class_names=class_names,
            training_time=cnn_training_time
        )

        cnn_result["status"] = "success"

        model_results["Simple CNN"] = cnn_result
        trained_models["Simple CNN"] = cnn_model

    except Exception as exc:

        model_results["Simple CNN"] = {
            "status": "failed",
            "error": str(exc)
        }

        warnings.append(
            f"Simple CNN failed: {exc}"
        )

    # ---------------------------------------------------------
    # SUMMARY + RANKING
    # ---------------------------------------------------------

    summary = create_summary_dataframe(
        model_results
    )

    successful_summary = summary.dropna(
        subset=["Macro F1"]
    )

    if successful_summary.empty:
        best_model = None
    else:
        best_model = successful_summary.iloc[0][
            "Model"
        ]

    # ---------------------------------------------------------
    # DATASET INFORMATION
    # ---------------------------------------------------------

    dataset_information = {
        "dataset_type": dataset_type,
        "color_mode": color_mode,
        "image_size": [64, 64],
        "number_of_valid_images": loaded[
            "number_of_samples"
        ],
        "number_of_classes": loaded[
            "number_of_classes"
        ],
        "class_names": class_names,
        "class_counts": loaded[
            "class_counts"
        ],
        "number_skipped": loaded[
            "number_skipped"
        ],
        "skipped_files": loaded[
            "skipped_files"
        ],
        "label_mapping": processed[
            "label_mapping"
        ]
    }

    # ---------------------------------------------------------
    # SPLIT INFORMATION
    # ---------------------------------------------------------

    split_information = {
        "random_seed": 42,
        "test_size": 0.20,
        "stratified": True,
        "same_split_for_all_models": True,
        "number_training_images": int(
            len(processed["train_indices"])
        ),
        "number_testing_images": int(
            len(processed["test_indices"])
        ),
        "train_indices": processed[
            "train_indices"
        ].tolist(),
        "test_indices": processed[
            "test_indices"
        ].tolist(),
        "full_class_distribution": processed[
            "full_distribution"
        ],
        "training_class_distribution": processed[
            "training_distribution"
        ],
        "testing_class_distribution": processed[
            "testing_distribution"
        ]
    }

    # ---------------------------------------------------------
    # CONFUSION MATRICES + REPORTS
    # ---------------------------------------------------------

    confusion_matrices = {}

    classification_reports = {}

    for model_name, result in model_results.items():

        if result.get("status") == "success":

            confusion_matrices[model_name] = (
                result["confusion_matrix"]
            )

            classification_reports[model_name] = (
                result["classification_report"]
            )

    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------

    run_configuration = {
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "random_seed": 42,
        "image_size": [64, 64],
        "color_mode": color_mode,
        "dataset_type": dataset_type,
        "test_size": 0.20,
        "split_strategy": "stratified",
        "scaling": (
            "Training-data-only scaling for Logistic "
            "Regression, SVM, and Neural Network."
        ),
        "cnn_epochs_max": 20,
        "cnn_batch_size": 32,
        "cnn_early_stopping_patience": 3,
        "cnn_pretrained": False,
        "augmentation_primary_benchmark": False
    }

    benchmark_metrics = {}

    for model_name, result in model_results.items():

        if result.get("status") == "success":

            benchmark_metrics[model_name] = {
                "accuracy": result["accuracy"],
                "macro_precision": result[
                    "macro_precision"
                ],
                "macro_recall": result[
                    "macro_recall"
                ],
                "macro_f1": result[
                    "macro_f1"
                ],
                "weighted_f1": result[
                    "weighted_f1"
                ],
                "training_time_seconds": result[
                    "training_time_seconds"
                ],
                "inference_time_ms_per_image": result[
                    "inference_time_ms_per_image"
                ]
            }

        else:

            benchmark_metrics[model_name] = {
                "status": "failed",
                "error": result.get(
                    "error",
                    "Unknown error"
                )
            }

    # ---------------------------------------------------------
    # SAVE REQUIRED FILES
    # ---------------------------------------------------------

    output_path = save_benchmark_outputs(
        summary=summary,
        model_results=model_results,
        class_names=class_names,
        full_distribution=processed[
            "full_distribution"
        ],
        run_configuration=run_configuration,
        benchmark_metrics=benchmark_metrics,
        output_dir="benchmark_results"
    )

    # ---------------------------------------------------------
    # REQUIRED RETURN DICTIONARY
    # ---------------------------------------------------------

    return {
        "package_information": {
            "name": PACKAGE_NAME,
            "version": PACKAGE_VERSION
        },

        "summary": summary,

        "best_model": best_model,

        "dataset_information": dataset_information,

        "split_information": split_information,

        "model_results": model_results,

        "confusion_matrices": confusion_matrices,

        "classification_reports": classification_reports,

        "warnings": warnings,

        "output_directory": str(output_path)
    }
