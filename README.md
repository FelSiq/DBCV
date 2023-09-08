![Moon with noise example.](./assets/example_moons_with_noise.png)

---

# Install
```bash
python -m pip install "git+https://github.com/FelSiq/DBCV"
```

---

# Usage
```python
import dbcv
import sklearn.datasets

X, y = sklearn.datasets.make_moons(n_samples=300, noise=0.05, random_state=1782)

score = dbcv.dbcv(X, y)

print(score)
# 0.978017974682571
```

---

# Reference
"Density-Based Clustering Validation". Davoud Moulavi, Pablo A. Jaskowiak, Ricardo J. G. B. Campello, Arthur Zimek, Jörg Sander.
