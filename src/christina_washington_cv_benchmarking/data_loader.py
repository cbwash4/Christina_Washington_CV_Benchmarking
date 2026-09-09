
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _validate_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_folder_dataset(dataset, target_labels):
    root = Path(dataset)

    if not root.exists() or not root.is_dir():
        raise ValueError(f"Folder dataset does not exist: {dataset}")

    if not isinstance(target_labels, (list, tuple)):
        raise ValueError("For folder datasets, target_labels must be a list of class-folder names.")

    image_paths = []
    labels = []
    skipped = []

    for class_name in target_labels:
        class_dir = root / class_name

        if not class_dir.exists():
            raise ValueError(f"Class folder not found: {class_dir}")

        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            if _validate_image(path):
                image_paths.append(str(path))
                labels.append(class_name)
            else:
                skipped.append(str(path))

    return image_paths, np.asarray(labels), skipped


def load_csv_dataset(dataset, target_labels):
    csv_path = Path(dataset)

    if not csv_path.exists():
        raise ValueError(f"CSV file does not exist: {dataset}")

    df = pd.read_csv(csv_path)

    if "image_path" not in df.columns:
        raise ValueError("CSV must contain an image_path column.")

    if not isinstance(target_labels, str) or target_labels not in df.columns:
        raise ValueError("target_labels must identify a valid CSV label column.")

    image_paths = []
    labels = []
    skipped = []

    for _, row in df.iterrows():
        path = Path(str(row["image_path"]))

        if not path.is_absolute():
            path = csv_path.parent / path

        if (
            path.exists()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and _validate_image(path)
        ):
            image_paths.append(str(path))
            labels.append(row[target_labels])
        else:
            skipped.append(str(path))

    return image_paths, np.asarray(labels), skipped


def load_json_dataset(dataset, target_labels):
    json_path = Path(dataset)

    if not json_path.exists():
        raise ValueError(f"JSON/JSONL file does not exist: {dataset}")

    if not isinstance(target_labels, str):
        raise ValueError("For JSON datasets, target_labels must be the label-field name.")

    if json_path.suffix.lower() == ".jsonl":
        records = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("JSON dataset must contain a list of records.")

    image_paths = []
    labels = []
    skipped = []

    for record in records:
        if "image_path" not in record or target_labels not in record:
            raise ValueError(
                f"Each JSON record must contain image_path and {target_labels}."
            )

        path = Path(str(record["image_path"]))

        if not path.is_absolute():
            path = json_path.parent / path

        if (
            path.exists()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and _validate_image(path)
        ):
            image_paths.append(str(path))
            labels.append(record[target_labels])
        else:
            skipped.append(str(path))

    return image_paths, np.asarray(labels), skipped


def load_array_dataset(dataset, target_labels):
    if isinstance(dataset, pd.DataFrame):
        if not isinstance(target_labels, str):
            raise ValueError(
                "For DataFrame input, target_labels must be the target-column name."
            )

        if target_labels not in dataset.columns:
            raise ValueError(f"Target column not found: {target_labels}")

        feature_columns = [c for c in dataset.columns if c != target_labels]

        if "image" in dataset.columns:
            images = np.stack(dataset["image"].to_numpy())
        elif "image_path" in dataset.columns:
            image_paths = dataset["image_path"].tolist()
            labels = dataset[target_labels].to_numpy()
            skipped = []

            valid_paths = []
            valid_labels = []

            for path, label in zip(image_paths, labels):
                path = Path(str(path))

                if (
                    path.exists()
                    and path.suffix.lower() in SUPPORTED_EXTENSIONS
                    and _validate_image(path)
                ):
                    valid_paths.append(str(path))
                    valid_labels.append(label)
                else:
                    skipped.append(str(path))

            return valid_paths, np.asarray(valid_labels), skipped
        else:
            raise ValueError(
                "DataFrame must contain an image column or image_path column."
            )

        labels = dataset[target_labels].to_numpy()

    else:
        images = np.asarray(dataset)
        labels = np.asarray(target_labels)

    if images.ndim not in (3, 4):
        raise ValueError(
            "Array images must have shape (N,H,W), (N,H,W,1), or (N,H,W,3)."
        )

    if images.ndim == 4 and images.shape[-1] not in (1, 3):
        raise ValueError("Array image channel dimension must be 1 or 3.")

    if len(images) != len(labels):
        raise ValueError("Number of images must equal number of labels.")

    return images, labels, []


def load_dataset(dataset, dataset_type, target_labels):
    dataset_type = str(dataset_type).lower()

    if dataset_type == "folder":
        result = load_folder_dataset(dataset, target_labels)

    elif dataset_type == "csv":
        result = load_csv_dataset(dataset, target_labels)

    elif dataset_type == "json":
        result = load_json_dataset(dataset, target_labels)

    elif dataset_type == "array":
        result = load_array_dataset(dataset, target_labels)

    else:
        raise ValueError(
            "dataset_type must be one of: folder, csv, json, array."
        )

    data, labels, skipped = result

    if len(data) != len(labels):
        raise ValueError("Number of valid images does not equal number of labels.")

    unique, counts = np.unique(labels, return_counts=True)

    if len(unique) < 2:
        raise ValueError("At least two target classes are required.")

    class_counts = {
        str(label): int(count)
        for label, count in zip(unique, counts)
    }

    return {
        "data": data,
        "labels": labels,
        "skipped_files": skipped,
        "number_skipped": len(skipped),
        "class_counts": class_counts,
        "number_of_samples": len(labels),
        "number_of_classes": len(unique),
    }
