import functools
import itertools
import timeit

import numpy as np
import tqdm

import dbcv


def compute(n_instances: int, n_dims: int, n_clusters: int) -> None:
    X = np.random.randn(n_instances, n_dims)
    y = np.random.randint(0, n_clusters, size=n_instances)
    score = dbcv.dbcv(X, y)
    assert -1.0 <= score <= 1.0, score


def test():
    combs = itertools.product([100, 1000, 10000], [10, 100, 1000], [2, 4, 8, 16])
    results = []
    for args in tqdm.tqdm(combs, total=3 * 3 * 4):
        fn = functools.partial(compute, *args)
        min_time = min(timeit.repeat(fn, repeat=3, number=20))
        results.append((args, min_time))

    for item in results:
        print(item)


if __name__ == "__main__":
    test()
