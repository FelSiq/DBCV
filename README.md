# Fast Density-Based Clustering Validation (DBCV) 

![Moon with noise example.](./assets/example_moons_with_noise.png)

---

## Install
```bash
python -m pip install "git+https://github.com/FelSiq/DBCV"
```

---

## Usage

### Basic usage
```python
import dbcv
import sklearn.datasets

X, y = sklearn.datasets.make_moons(n_samples=300, noise=0.05, random_state=1782)

score = dbcv.dbcv(X, y)
print(score)
# 0.978017974682571
```

### Example with noise "cluster"
```python
import sklearn.datasets
import numpy as np
import dbcv

X, y = sklearn.datasets.make_moons(n_samples=500, noise=0.10, random_state=1782)

noise_id = -1
# NOTE: dbcv.dbcv(..., noise_id=-1) by default; you don't have to set this up.

rng = np.random.RandomState(1082)
X_noise = rng.uniform(*np.quantile(X, (0, 1), axis=0), size=(100, 2))
y_noise = 100 * [noise_id]

score = dbcv.dbcv(np.vstack((X, X_noise)), np.hstack((y, y_noise)), noise_id=noise_id)
print(score)
# 0.8072520501068048
```

### Multiprocessing
You can use the `dbcv.dbcv(..., n_processes=n)` argument to specify the number of parallel processes during computations. The default value of `n_processes` is set to `"auto"`. If `n_processes="auto"`, the number of parallel processes will be set to 1 for datasets with 200 or fewer instances, and 4 for datasets with more than 200 instances.

```python
import dbcv
import sklearn.datasets

X, y = sklearn.datasets.make_moons(n_samples=300, noise=0.05, random_state=1782)

score = dbcv.dbcv(X, y, n_processes=2)
print(score)
# 0.978017974682571
```

---

## Reference
"Density-Based Clustering Validation". Davoud Moulavi, Pablo A. Jaskowiak, Ricardo J. G. B. Campello, Arthur Zimek, Jörg Sander.
