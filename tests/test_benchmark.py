
import numpy as np

from christina_washington_cv_benchmarking import (
    benchmark_image_classification
)


def test_package_import():
    from christina_washington_cv_benchmarking import (
        benchmark_image_classification
    )

    assert callable(
        benchmark_image_classification
    )


def test_complete_benchmark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(42)

    class_a = rng.integers(
        0, 80,
        size=(15, 20, 20),
        dtype=np.uint8
    )

    class_b = rng.integers(
        175, 256,
        size=(15, 20, 20),
        dtype=np.uint8
    )

    X = np.concatenate([
        class_a,
        class_b
    ])

    y = np.array(
        ["a"] * 15 +
        ["b"] * 15
    )

    results = benchmark_image_classification(
        dataset=X,
        dataset_type="array",
        target_labels=y,
        color_mode="grayscale"
    )

    assert len(
        results["summary"]
    ) == 6

    assert len(
        results["split_information"][
            "train_indices"
        ]
    ) > 0

    assert len(
        results["split_information"][
            "test_indices"
        ]
    ) > 0

    train_indices = set(
        results["split_information"][
            "train_indices"
        ]
    )

    test_indices = set(
        results["split_information"][
            "test_indices"
        ]
    )

    assert train_indices.isdisjoint(
        test_indices
    )

    assert (
        tmp_path
        / "benchmark_results"
        / "benchmark_summary.csv"
    ).exists()

    assert (
        tmp_path
        / "benchmark_results"
        / "benchmark_metrics.json"
    ).exists()

    assert (
        tmp_path
        / "benchmark_results"
        / "run_configuration.json"
    ).exists()

    assert (
        tmp_path
        / "benchmark_results"
        / "class_distribution.png"
    ).exists()

    assert (
        tmp_path
        / "benchmark_results"
        / "model_comparison.png"
    ).exists()


def test_version_metadata():
    import christina_washington_cv_benchmarking as package

    assert package.__version__ == "1.0.0"
