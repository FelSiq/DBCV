import timeit
import functools

import numpy as np
import dbcv


def run(enable_dynamic_precision: bool, bits_of_precision: int = 256):
    rng = np.random.RandomState(182)

    X = rng.randn(1000, 786)
    y = rng.randint(0, 3, size=X.shape[0])
    y[rng.random() <= 0.025] = -1

    X[y == 1] += np.maximum(np.random.randn(1, 786), 0.25)
    X[y == 2] += np.clip(np.random.randn(1, 786), -1.0, -0.25)
    X += 0.25 * rng.randn(1000, 786)

    dbcv.dbcv(
        X,
        y,
        enable_dynamic_precision=enable_dynamic_precision,
        bits_of_precision=bits_of_precision,
    )


def test():
    kwargs = {"repeat": 5, "number": 20}

    ts = timeit.repeat(functools.partial(run, enable_dynamic_precision=False), **kwargs)
    print(f"Base: {np.mean(ts):.4f} \\pm {np.std(ts):.4f}")

    for bop in [64, 128, 256, 512, 1024]:
        ts = timeit.repeat(functools.partial(run, enable_dynamic_precision=True, bits_of_precision=bop), **kwargs)
        print(f"{bop}: {np.mean(ts):.4f} \\pm {np.std(ts):.4f}")


if __name__ == "__main__":
    test()
