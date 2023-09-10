import functools
import itertools
import timeit

import numpy as np
import tqdm

import dbcv


def compute(k: int = 100) -> None:
    rng = np.random.RandomState(1892)
    combs = itertools.product([100, 200, 1000], [10, 30, 50], [2, 4, 8, 16])
    for n_instances, n_dims, n_clusters in tqdm.tqdm(combs):
        for _ in range(k):
            X = rng.randn(n_instances, n_dims)
            y = rng.randint(0, n_clusters, size=n_instances)
            score_a = dbcv.dbcv(X, y, n_processes=1)
            score_b = dbcv.dbcv(X, y, n_processes=4)
            score_c = dbcv.dbcv(X, y, n_processes=8)
            assert np.isclose(score_a, score_b)
            assert np.isclose(score_a, score_c)


if __name__ == "__main__":
    compute()
