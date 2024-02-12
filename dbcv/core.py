import multiprocessing
import typing as t
import itertools
import functools

import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import scipy.spatial.distance
import scipy.sparse.csgraph
import scipy.stats
import mpmath


_MP = mpmath.mp.clone()


def compute_pair_to_pair_dists(X: npt.NDArray[np.float64], metric: str) -> npt.NDArray[np.float64]:
    dists = scipy.spatial.distance.cdist(X, X, metric=metric)
    np.maximum(dists, 1e-12, out=dists)
    np.fill_diagonal(dists, val=np.inf)
    return dists


def get_subarray(
    arr: npt.NDArray[np.float64],
    /,
    inds_a: t.Optional[npt.NDArray[np.int32]] = None,
    inds_b: t.Optional[npt.NDArray[np.int32]] = None,
) -> npt.NDArray[np.float64]:
    if inds_a is None:
        return arr
    if inds_b is None:
        inds_b = inds_a
    inds_a_mesh, inds_b_mesh = np.meshgrid(inds_a, inds_b)
    return arr[inds_a_mesh, inds_b_mesh]


def get_internal_objects(mutual_reach_dists: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mst = scipy.sparse.csgraph.minimum_spanning_tree(mutual_reach_dists)
    mst = mst.toarray()

    is_mst_edges = mst > 0.0

    internal_node_inds = (is_mst_edges + is_mst_edges.T).sum(axis=0) > 1
    internal_node_inds = np.flatnonzero(internal_node_inds)

    internal_edge_weights = get_subarray(mst, inds_a=internal_node_inds)

    return internal_node_inds, internal_edge_weights


def compute_cluster_core_distance(
    dists: npt.NDArray[np.float64], d: int, enable_dynamic_precision: bool
) -> npt.NDArray[np.float64]:
    n, m = dists.shape

    if n == m and n > 800:
        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=801, metric="precomputed")
        dists, _ = nn.fit(np.nan_to_num(dists, posinf=0.0)).kneighbors(return_distance=True)
        n = dists.shape[1]

    orig_dists_dtype = dists.dtype

    if enable_dynamic_precision:
        dists = np.asarray(_MP.matrix(dists), dtype=object).reshape(*dists.shape)

    core_dists = np.power(dists, -d).sum(axis=-1, keepdims=True) / (n - 1 + 1e-12)

    if not enable_dynamic_precision:
        np.clip(core_dists, a_min=1e-12, a_max=1e12, out=core_dists)

    np.power(core_dists, -1.0 / d, out=core_dists)

    if enable_dynamic_precision:
        core_dists = np.asfarray(core_dists, dtype=orig_dists_dtype)

    return core_dists


def compute_mutual_reach_dists(
    dists: npt.NDArray[np.float64],
    d: float,
    enable_dynamic_precision: bool,
    is_symmetric: bool,
    cls_inds_a: t.Optional[npt.NDArray[np.int32]] = None,
    cls_inds_b: t.Optional[npt.NDArray[np.int32]] = None,
) -> npt.NDArray[np.float64]:
    cls_dists = get_subarray(dists, inds_a=cls_inds_a, inds_b=cls_inds_b)

    if is_symmetric:
        core_dists_a = core_dists_b = compute_cluster_core_distance(
            d=d, dists=cls_dists, enable_dynamic_precision=enable_dynamic_precision
        )

    else:
        core_dists_a = compute_cluster_core_distance(d=d, dists=cls_dists, enable_dynamic_precision=enable_dynamic_precision)
        core_dists_b = compute_cluster_core_distance(
            d=d, dists=cls_dists.T, enable_dynamic_precision=enable_dynamic_precision
        ).T

    mutual_reach_dists = cls_dists.copy()
    np.maximum(mutual_reach_dists, core_dists_a, out=mutual_reach_dists)
    np.maximum(mutual_reach_dists, core_dists_b, out=mutual_reach_dists)

    return mutual_reach_dists


def fn_density_sparseness(
    cls_inds: npt.NDArray[np.int32],
    dists: npt.NDArray[np.float64],
    d: int,
    enable_dynamic_precision: bool,
) -> tuple[float, npt.NDArray[np.int32]]:
    if cls_inds.size <= 3:
        return (0.0, np.empty(0, dtype=int))

    mutual_reach_dists = compute_mutual_reach_dists(
        dists=dists, d=d, is_symmetric=True, enable_dynamic_precision=enable_dynamic_precision
    )
    internal_node_inds, internal_edge_weights = get_internal_objects(mutual_reach_dists)

    dsc = float(internal_edge_weights.max())
    internal_node_inds = cls_inds[internal_node_inds]

    return (dsc, internal_node_inds)


def fn_density_separation(
    cls_i: int, cls_j: int, dists: npt.NDArray[np.float64], d: int, enable_dynamic_precision: bool
) -> tuple[int, int, float]:
    mutual_reach_dists = compute_mutual_reach_dists(
        dists=dists, d=d, is_symmetric=False, enable_dynamic_precision=enable_dynamic_precision
    )
    dspc_ij = float(mutual_reach_dists.min()) if mutual_reach_dists.size else np.inf
    return (cls_i, cls_j, dspc_ij)


def _check_duplicated_samples(X: npt.NDArray[np.float64], threshold: float = 1e-9):
    if X.shape[0] <= 1:
        return

    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    nn.fit(X)
    dists, _ = nn.kneighbors(return_distance=True)

    if np.any(dists < threshold):
        raise ValueError("Duplicated samples have been found in X.")


def dbcv(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int32],
    metric: str = "sqeuclidean",
    noise_id: int = -1,
    check_duplicates: bool = True,
    n_processes: int = 4,
    enable_dynamic_precision: bool = False,
    bits_of_precision: int = 512,
) -> float:
    """Compute DBCV metric.

    Density-Based Clustering Validation (DBCV) is an intrinsic (= unsupervised/unlabeled)
    relative metric. See [1] for the original reference.

    Parameters
    ----------
    X : npt.NDArray[np.float64] of shape (N, D)
        Sample embeddings.

    y : npt.NDArray[np.int32] of shape (N,)
        Cluster IDs assigned for each sample in X.

    metric : str, default="sqeuclidean"
        Metric function to compute dissimilarity between observations.
        This argument is passed to `scipy.spatial.distance.cdist`.

    noise_id : int, default=-1
        Noise "cluster" ID.

    check_duplicates : bool, default=True
        If True, check for duplicated samples.
        Instances with Euclidean distance to their nearest neighbor below 1e-9 are considered
        duplicates.

    n_processes : int or "auto", default="auto"
        Maximum number of parallel processes for processing clusters and cluster pairs.
        If `n_processes="auto"`, the number of parallel processes will be set to 1 for
        datasets with 200 or fewer instances, and 4 for datasets with more than 200 instances.

    enable_dynamic_precision : bool, default=False
        If True, activate dynamic quantity of bits of precision for floating point
        during density calculation, as defined by `bits_of_precision` argument below.
        This argument enables proper density calculation for very high dimensional data,
        although it is much slower than the standard calculations.

    bits_of_precision : int, default=512
        Bits of precision for density calculation. High values are necessary for high
        dimensions to avoid underflow/overflow.

    Returns
    -------
    DBCV : float
        DBCV metric estimation.

    Source
    ------
    .. [1] "Density-Based Clustering Validation". Davoud Moulavi, Pablo A. Jaskowiak,
           Ricardo J. G. B. Campello, Arthur Zimek, Jörg Sander.
           https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf
    """
    X = np.asfarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = np.asarray(y, dtype=int)

    n, d = X.shape  # NOTE: 'n' must be calculated before removing noise.

    if n != y.size:
        raise ValueError(f"Mismatch in {X.shape[0]=} and {y.size=} dimensions.")

    non_noise_inds = y != noise_id
    X = X[non_noise_inds, :]
    y = y[non_noise_inds]

    if y.size == 0:
        return 0.0

    y = scipy.stats.rankdata(y, method="dense") - 1
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)

    if check_duplicates:
        _check_duplicated_samples(X)

    dists = compute_pair_to_pair_dists(X=X, metric=metric)

    # DSC: 'Density Sparseness of a Cluster'
    dscs = np.empty(cluster_ids.size, dtype=float)

    # DSPC: 'Density Separation of a Pair of Clusters'
    min_dspcs = np.full(cluster_ids.size, fill_value=np.inf)

    # Internal objects = Internal nodes = nodes such that degree(node) > 1 in MST.
    internal_objects_per_cls: dict[int, npt.NDArray[np.int32]] = {}

    cls_inds = [np.flatnonzero(y == cls_id) for cls_id in cluster_ids]

    if n_processes == "auto":
        n_processes = 4 if y.size > 200 else 1

    with _MP.workprec(bits_of_precision), multiprocessing.Pool(processes=min(n_processes, cluster_ids.size)) as ppool:
        fn_density_sparseness_ = functools.partial(
            fn_density_sparseness,
            d=d,
            enable_dynamic_precision=enable_dynamic_precision,
        )

        args = [(cls_ind, get_subarray(dists, inds_a=cls_ind)) for cls_ind in cls_inds]

        for cls_id, (dsc, internal_node_inds) in enumerate(ppool.starmap(fn_density_sparseness_, args)):
            internal_objects_per_cls[cls_id] = internal_node_inds
            dscs[cls_id] = dsc

    n_cls_pairs = (cluster_ids.size * (cluster_ids.size - 1)) // 2

    if n_cls_pairs > 0:
        with _MP.workprec(bits_of_precision), multiprocessing.Pool(processes=min(n_processes, n_cls_pairs)) as ppool:
            fn_density_separation_ = functools.partial(
                fn_density_separation,
                d=d,
                enable_dynamic_precision=enable_dynamic_precision,
            )

            args = [
                (cls_i, cls_j, get_subarray(dists, internal_objects_per_cls[cls_i], internal_objects_per_cls[cls_j]))
                for cls_i, cls_j in itertools.combinations(cluster_ids, 2)
            ]

            for cls_i, cls_j, dspc_ij in ppool.starmap(fn_density_separation_, args):
                min_dspcs[cls_i] = min(min_dspcs[cls_i], dspc_ij)
                min_dspcs[cls_j] = min(min_dspcs[cls_j], dspc_ij)

    np.nan_to_num(min_dspcs, copy=False, posinf=1e12)
    vcs = (min_dspcs - dscs) / (1e-12 + np.maximum(min_dspcs, dscs))
    np.nan_to_num(vcs, copy=False, nan=0.0)
    dbcv = float(np.sum(vcs * cluster_sizes)) / n

    return dbcv
