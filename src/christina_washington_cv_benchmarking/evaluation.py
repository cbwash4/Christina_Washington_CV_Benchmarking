
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from .neural_models import predict_simple_cnn


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )
    }


def evaluate_predictions(
    model_name,
    y_true,
    y_pred,
    class_names,
    training_time,
    inference_time
):
    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    number_test_images = len(y_true)

    inference_ms_per_image = (
        inference_time / number_test_images
    ) * 1000.0

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_dataframe = pd.DataFrame(
        report_dict
    ).transpose()

    return {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "training_time_seconds": float(training_time),
        "inference_time_ms_per_image": float(
            inference_ms_per_image
        ),
        "predictions": np.asarray(y_pred),
        "confusion_matrix": cm,
        "classification_report": report_dataframe
    }


def train_and_evaluate_classical_model(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    class_names
):
    start_train = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter() - start_train
    )

    start_inference = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    inference_time = (
        time.perf_counter() - start_inference
    )

    return evaluate_predictions(
        model_name=model_name,
        y_true=y_test,
        y_pred=predictions,
        class_names=class_names,
        training_time=training_time,
        inference_time=inference_time
    )


def evaluate_trained_cnn(
    model,
    X_test,
    y_test,
    class_names,
    training_time
):
    start_inference = time.perf_counter()

    predictions = predict_simple_cnn(
        model,
        X_test
    )

    inference_time = (
        time.perf_counter() - start_inference
    )

    return evaluate_predictions(
        model_name="Simple CNN",
        y_true=y_test,
        y_pred=predictions,
        class_names=class_names,
        training_time=training_time,
        inference_time=inference_time
    )


def create_summary_dataframe(model_results):
    rows = []

    for model_name, result in model_results.items():

        if result.get("status") == "failed":
            rows.append({
                "Model": model_name,
                "Accuracy": np.nan,
                "Macro Precision": np.nan,
                "Macro Recall": np.nan,
                "Macro F1": np.nan,
                "Weighted F1": np.nan,
                "Train (s)": np.nan,
                "Infer (ms/img)": np.nan
            })

        else:
            rows.append({
                "Model": model_name,
                "Accuracy": result["accuracy"],
                "Macro Precision": result["macro_precision"],
                "Macro Recall": result["macro_recall"],
                "Macro F1": result["macro_f1"],
                "Weighted F1": result["weighted_f1"],
                "Train (s)": result["training_time_seconds"],
                "Infer (ms/img)": result[
                    "inference_time_ms_per_image"
                ]
            })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=[
            "Macro F1",
            "Infer (ms/img)"
        ],
        ascending=[
            False,
            True
        ],
        na_position="last"
    ).reset_index(drop=True)

    return summary
