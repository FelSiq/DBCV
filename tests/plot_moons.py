import os

import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np

import dbcv


def test():
    fig, axes = plt.subplots(2, 5, figsize=(12, 7), sharex=True, sharey=True, layout="tight")
    colors = ["black", "red"]

    for i, noise in enumerate(np.linspace(0, 0.50, 10)):
        X, y = sklearn.datasets.make_moons(n_samples=300, noise=noise, random_state=1782)
        score = dbcv.dbcv(X, y)
        ax = axes[i // 5][i % 5]
        ax.scatter(*X.T, c=[colors[yi] for yi in y])
        ax.set_title(f"$\sigma={noise:.3f}$\ndbcv={score:.3f}")

    output_dir = os.path.abspath("./assets/")
    output_uri = os.path.join(output_dir, "example_moons_with_noise.pdf")

    fig.savefig(output_uri, format="pdf", bbox_inches=0)
    fig.savefig(output_uri.replace(".pdf", ".png"), format="png", bbox_inches=0)

    plt.show()


if __name__ == "__main__":
    test()
