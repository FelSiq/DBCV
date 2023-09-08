import os

import matplotlib.pyplot as plt
import sklearn.neighbors
import sklearn.datasets
import numpy as np

import dbcv


def test():
    fig, axes = plt.subplots(2, 5, figsize=(12, 7), sharex=True, sharey=True, layout="tight")
    colors = ["black", "red", "blue"]
    rng = np.random.RandomState(182)

    for i, noise in enumerate(np.linspace(0, 0.50, 10)):
        X, y = sklearn.datasets.make_moons(n_samples=300, noise=noise, random_state=1782)

        X_noise = rng.uniform(*(1.50 * np.quantile(X, (0, 1))), size=(30, 2))
        y_noise = np.asarray([-1] * len(X_noise), dtype=int)

        dists, _ = sklearn.neighbors.NearestNeighbors(n_neighbors=1).fit(X).kneighbors(X_noise)
        dists = dists.squeeze()
        is_farther_away = dists > 0.5
        X_noise = X_noise[is_farther_away, :]
        y_noise = y_noise[is_farther_away]

        score = dbcv.dbcv(np.vstack((X, X_noise)), np.hstack((y, y_noise)))
        ax = axes[i // 5][i % 5]
        ax.scatter(*X.T, c=[colors[yi] for yi in y])
        ax.scatter(*X_noise.T, c=[colors[yi] for yi in y_noise], marker=".")
        ax.set_title(f"$\sigma={noise:.3f}$\ndbcv={score:.3f}")

    output_dir = os.path.abspath("./assets/")
    output_uri = os.path.join(output_dir, "example_moons_with_noise.pdf")

    fig.savefig(output_uri, format="pdf", bbox_inches=0)
    fig.savefig(output_uri.replace(".pdf", ".png"), format="png", bbox_inches=0)

    plt.show()


if __name__ == "__main__":
    test()
