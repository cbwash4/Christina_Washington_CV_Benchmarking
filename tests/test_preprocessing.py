
import numpy as np

from christina_washington_cv_benchmarking.preprocessing import (
    standardize_images,
    encode_labels,
    create_feature_representations
)


def test_grayscale_shape():
    X = np.random.randint(
        0, 256, (10, 40, 50),
        dtype=np.uint8
    )

    result = standardize_images(
        X,
        "grayscale"
    )

    assert result.shape == (10, 64, 64, 1)
    assert result.min() >= 0
    assert result.max() <= 1


def test_rgb_shape():
    X = np.random.randint(
        0, 256, (10, 40, 50, 3),
        dtype=np.uint8
    )

    result = standardize_images(
        X,
        "rgb"
    )

    assert result.shape == (10, 64, 64, 3)


def test_label_encoding():
    labels = np.array([
        "cat", "dog", "horse",
        "cat", "dog", "horse"
    ])

    y, encoder, names, mapping = (
        encode_labels(labels)
    )

    assert len(np.unique(y)) == 3
    assert len(names) == 3
    assert len(mapping) == 3


def test_flatten_grayscale():
    X = np.zeros(
        (5, 64, 64, 1),
        dtype=np.float32
    )

    flat, tensor = (
        create_feature_representations(X)
    )

    assert flat.shape == (5, 4096)
    assert tensor.shape == (5, 64, 64, 1)


def test_flatten_rgb():
    X = np.zeros(
        (5, 64, 64, 3),
        dtype=np.float32
    )

    flat, tensor = (
        create_feature_representations(X)
    )

    assert flat.shape == (5, 12288)
    assert tensor.shape == (5, 64, 64, 3)
