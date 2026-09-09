
import json
import numpy as np
import pandas as pd
from PIL import Image

from christina_washington_cv_benchmarking.data_loader import (
    load_dataset
)


def create_test_images(tmp_path):
    paths = []

    for i in range(6):
        array = np.full(
            (20, 20, 3),
            i * 30,
            dtype=np.uint8
        )

        path = tmp_path / f"image_{i}.png"

        Image.fromarray(array).save(path)

        paths.append(path)

    return paths


def test_array_loader():
    X = np.zeros(
        (10, 20, 20, 3),
        dtype=np.uint8
    )

    y = np.array(
        ["a"] * 5 +
        ["b"] * 5
    )

    result = load_dataset(
        X,
        "array",
        y
    )

    assert result["number_of_samples"] == 10
    assert result["number_of_classes"] == 2


def test_csv_loader(tmp_path):
    paths = create_test_images(tmp_path)

    df = pd.DataFrame({
        "image_path": [
            str(p) for p in paths
        ],
        "label": [
            "a", "a", "a",
            "b", "b", "b"
        ]
    })

    csv_path = tmp_path / "manifest.csv"
    df.to_csv(csv_path, index=False)

    result = load_dataset(
        csv_path,
        "csv",
        "label"
    )

    assert result["number_of_samples"] == 6
    assert result["number_of_classes"] == 2


def test_json_loader(tmp_path):
    paths = create_test_images(tmp_path)

    records = []

    labels = [
        "a", "a", "a",
        "b", "b", "b"
    ]

    for path, label in zip(
        paths,
        labels
    ):
        records.append({
            "image_path": str(path),
            "label": label
        })

    json_path = tmp_path / "manifest.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(records, file)

    result = load_dataset(
        json_path,
        "json",
        "label"
    )

    assert result["number_of_samples"] == 6
    assert result["number_of_classes"] == 2


def test_folder_loader(tmp_path):
    for class_name in ["a", "b"]:

        class_dir = (
            tmp_path / class_name
        )

        class_dir.mkdir()

        for i in range(3):

            array = np.full(
                (20, 20, 3),
                i * 40,
                dtype=np.uint8
            )

            Image.fromarray(
                array
            ).save(
                class_dir / f"{i}.png"
            )

    result = load_dataset(
        tmp_path,
        "folder",
        ["a", "b"]
    )

    assert result["number_of_samples"] == 6
    assert result["number_of_classes"] == 2


def test_corrupt_image_skipped(tmp_path):
    for class_name in ["a", "b"]:

        class_dir = (
            tmp_path / class_name
        )

        class_dir.mkdir()

        for i in range(3):
            array = np.zeros(
                (20, 20, 3),
                dtype=np.uint8
            )

            Image.fromarray(
                array
            ).save(
                class_dir / f"{i}.png"
            )

    corrupt = (
        tmp_path / "a" / "corrupt.png"
    )

    corrupt.write_text(
        "not a real image"
    )

    result = load_dataset(
        tmp_path,
        "folder",
        ["a", "b"]
    )

    assert result["number_skipped"] == 1
    assert str(corrupt) in result["skipped_files"]
