import timeit
import functools

import numpy as np
import dbcv


def run(enable_dynamic_precision: bool, bits_of_precision: int | None = None):
    rng = np.random.RandomState(182)

    X = rng.randn(1000, 786)
    y = rng.randint(0, 3, size=X.shape[0])

    X += y.reshape(-1, 1) / float(y.max())

    dbcv.dbcv(
        X,
        y,
        enable_dynamic_precision=enable_dynamic_precision,
        bits_of_precision=bits_of_precision,
    )


def test():
    kwargs = {"repeat": 3, "number": 100}
    print(timeit.repeat(functools.partial(run, enable_dynamic_precision=False), **kwargs))
    print(timeit.repeat(functools.partial(run, enable_dynamic_precision=True, bits_of_precision=64), **kwargs))
    print(timeit.repeat(functools.partial(run, enable_dynamic_precision=True, bits_of_precision=128), **kwargs))
    print(timeit.repeat(functools.partial(run, enable_dynamic_precision=True, bits_of_precision=256), **kwargs))


if __name__ == "__main__":
    test()
