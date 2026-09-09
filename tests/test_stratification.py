
import numpy as np

from christina_washington_cv_benchmarking.preprocessing import (
    create_stratified_split
)


def test_no_train_test_overlap():
    y = np.array(
        [0] * 20 +
        [1] * 20 +
        [2] * 20
    )

    train_idx, test_idx = (
        create_stratified_split(y)
    )

    overlap = set(train_idx).intersection(
        set(test_idx)
    )

    assert len(overlap) == 0


def test_all_samples_used():
    y = np.array(
        [0] * 20 +
        [1] * 20 +
        [2] * 20
    )

    train_idx, test_idx = (
        create_stratified_split(y)
    )

    assert (
        len(train_idx) +
        len(test_idx)
        == len(y)
    )


def test_split_is_reproducible():
    y = np.array(
        [0] * 20 +
        [1] * 20 +
        [2] * 20
    )

    train1, test1 = (
        create_stratified_split(y)
    )

    train2, test2 = (
        create_stratified_split(y)
    )

    assert np.array_equal(
        train1,
        train2
    )

    assert np.array_equal(
        test1,
        test2
    )


def test_stratification_preserved():
    y = np.array(
        [0] * 20 +
        [1] * 20 +
        [2] * 20
    )

    train_idx, test_idx = (
        create_stratified_split(y)
    )

    assert set(
        np.unique(y[train_idx])
    ) == {0, 1, 2}

    assert set(
        np.unique(y[test_idx])
    ) == {0, 1, 2}
