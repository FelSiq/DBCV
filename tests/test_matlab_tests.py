import os

import pytest
import pandas as pd
import numpy as np

import dbcv


@pytest.mark.parametrize(
    "dataset_uri,expected_output",
    (
        ("dataset_1.txt", 0.857574140050071),
        ("dataset_2.txt", 0.810334358909311),
        ("dataset_3.txt", 0.631879697008377),
        ("dataset_4.txt", 0.868775827633395),
        ("dataset_100.txt", 0.750000000000000),
        ("dataset_101.txt", 0.694201697828989),
    ),
)
def test_compare_to_original(dataset_uri: str, expected_output: float):
    base_uri = os.path.dirname(__file__)
    dataset_uri = os.path.abspath(os.path.join(base_uri, "assets", dataset_uri))

    df = pd.read_csv(dataset_uri, sep=" ", header=None, index_col=None)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    assert X.shape[1] == 2
    assert y.size == X.shape[0]

    output = dbcv.dbcv(
        X=X,
        y=y,
        use_original_mst_implementation=True,
        metric="sqeuclidean",
        noise_id=0,
    )

    assert np.isclose(expected_output, output, rtol=0.001, atol=1e-5)
