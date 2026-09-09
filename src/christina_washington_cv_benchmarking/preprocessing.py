
import numpy as np
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (64, 64)
RANDOM_SEED = 42
TEST_SIZE = 0.20


def _standardize_pil_image(image, color_mode):
    if color_mode == "grayscale":
        image = image.convert("L")
    elif color_mode == "rgb":
        image = image.convert("RGB")
    else:
        raise ValueError("color_mode must be 'grayscale' or 'rgb'.")

    image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)

    if color_mode == "grayscale":
        background = Image.new("L", IMAGE_SIZE, 0)
    else:
        background = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))

    x = (IMAGE_SIZE[0] - image.width) // 2
    y = (IMAGE_SIZE[1] - image.height) // 2

    background.paste(image, (x, y))

    array = np.asarray(background, dtype=np.float32) / 255.0

    if color_mode == "grayscale":
        array = np.expand_dims(array, axis=-1)

    return array


def standardize_images(data, color_mode):
    processed = []

    if isinstance(data, np.ndarray):
        iterable = data
    else:
        iterable = list(data)

    for item in iterable:

        if isinstance(item, (str, bytes)):
            with Image.open(item) as img:
                processed.append(
                    _standardize_pil_image(img.copy(), color_mode)
                )

        else:
            array = np.asarray(item)

            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]

            if array.dtype != np.uint8:
                if array.max() <= 1.0:
                    array = (array * 255).clip(0, 255).astype(np.uint8)
                else:
                    array = array.clip(0, 255).astype(np.uint8)

            image = Image.fromarray(array)

            processed.append(
                _standardize_pil_image(image, color_mode)
            )

    X = np.stack(processed).astype(np.float32)

    expected_channels = 1 if color_mode == "grayscale" else 3

    expected_shape = (
        len(X),
        IMAGE_SIZE[1],
        IMAGE_SIZE[0],
        expected_channels
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected standardized image shape: {X.shape}. "
            f"Expected {expected_shape}."
        )

    return X


def encode_labels(labels):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    class_names = [str(name) for name in encoder.classes_]

    mapping = {
        str(class_name): int(index)
        for index, class_name in enumerate(encoder.classes_)
    }

    return y, encoder, class_names, mapping


def create_feature_representations(X):
    X_tensor = np.asarray(X, dtype=np.float32)
    X_flat = X_tensor.reshape(X_tensor.shape[0], -1)

    return X_flat, X_tensor


def create_stratified_split(y):
    y = np.asarray(y)

    indices = np.arange(len(y))

    unique, counts = np.unique(y, return_counts=True)

    if len(unique) < 2:
        raise ValueError("At least two classes are required.")

    if np.any(counts < 2):
        raise ValueError(
            "Each class must contain at least two samples "
            "for a stratified split."
        )

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_SEED
        )
    except ValueError as exc:
        raise ValueError(
            f"Unable to create stratified train-test split: {exc}"
        ) from exc

    return train_idx, test_idx


def apply_split(X_flat, X_tensor, y, train_idx, test_idx):
    return {
        "X_flat_train": X_flat[train_idx],
        "X_flat_test": X_flat[test_idx],
        "X_tensor_train": X_tensor[train_idx],
        "X_tensor_test": X_tensor[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "train_indices": train_idx,
        "test_indices": test_idx
    }


def get_class_distribution(y, class_names):
    unique, counts = np.unique(y, return_counts=True)

    return {
        class_names[int(label)]: int(count)
        for label, count in zip(unique, counts)
    }


def preprocess_dataset(data, labels, color_mode):
    X = standardize_images(data, color_mode)

    y, encoder, class_names, label_mapping = encode_labels(labels)

    X_flat, X_tensor = create_feature_representations(X)

    train_idx, test_idx = create_stratified_split(y)

    split = apply_split(
        X_flat,
        X_tensor,
        y,
        train_idx,
        test_idx
    )

    split["class_names"] = class_names
    split["label_mapping"] = label_mapping
    split["label_encoder"] = encoder

    split["full_distribution"] = get_class_distribution(
        y,
        class_names
    )

    split["training_distribution"] = get_class_distribution(
        split["y_train"],
        class_names
    )

    split["testing_distribution"] = get_class_distribution(
        split["y_test"],
        class_names
    )

    return split
