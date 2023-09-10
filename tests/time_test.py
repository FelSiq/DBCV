import functools
import itertools
import timeit

import numpy as np
import tqdm

import dbcv


def compute(n_instances: int, n_dims: int, n_clusters: int) -> None:
    rng = np.random.RandomState(1892)
    X = rng.randn(n_instances, n_dims)
    y = rng.randint(0, n_clusters, size=n_instances)
    score = dbcv.dbcv(X, y, n_processes=8)
    assert -1.0 <= score <= 1.0, score


def test():
    combs = itertools.product([100, 1000, 5000], [10, 100, 150], [2, 4, 8, 16])
    results = []
    pbar = tqdm.tqdm(combs, total=3 * 3 * 4)

    for args in pbar:
        fn = functools.partial(compute, *args)
        pbar.set_description(str(args))
        min_time = min(timeit.repeat(fn, repeat=2, number=10))
        results.append((args, min_time))

    for item in results:
        print(item)


if __name__ == "__main__":
    test()
