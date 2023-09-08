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
    if inds_a is None: return arr
    if inds_b is None: inds_b = inds_a
    return arr[*np.meshgrid(inds_a, inds_b)]


def get_internal_objects(mutual_reach_dists: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mst = scipy.sparse.csgraph.minimum_spanning_tree(mutual_reach_dists)
    mst = mst.toarray()

    is_mst_edges = mst > 0.0

    internal_node_inds = (is_mst_edges + is_mst_edges.T).sum(axis=0) > 1
    internal_node_inds = np.flatnonzero(internal_node_inds)

    internal_edge_weights = get_subarray(mst, inds_a=internal_node_inds)

    return internal_node_inds, internal_edge_weights


def compute_cluster_core_distance(dists: npt.NDArray[np.float64], d: int) -> npt.NDArray[np.float64]:
    n, m = dists.shape

    if n == m and n > 800:
        nn = sklearn.neighbors.NearestNeighbors(n_neighbors=801, metric="precomputed")
        dists, _ = nn.fit(np.nan_to_num(dists, posinf=0.0)).kneighbors(return_distance=True)
        dists = dists[:, 1:]
        n = dists.shape[1]

    core_dists = np.power(dists, -d).sum(axis=-1, keepdims=True) / (n - 1 + 1e-12)
    np.maximum(core_dists, 1e-12, out=core_dists)
    np.power(core_dists, -1.0 / d, out=core_dists)
    return core_dists


def compute_mutual_reach_dists(
    dists: npt.NDArray[np.float64],
    d: float,
    cls_inds_a: t.Optional[npt.NDArray[np.int32]] = None,
    cls_inds_b: t.Optional[npt.NDArray[np.int32]] = None,
    is_symmetric: bool = False,
) -> npt.NDArray[np.float64]:
    cls_dists = get_subarray(dists, inds_a=cls_inds_a, inds_b=cls_inds_b)

    if is_symmetric:
        core_dists_a = core_dists_b = compute_cluster_core_distance(d=d, dists=cls_dists)

    else:
        core_dists_a = compute_cluster_core_distance(d=d, dists=cls_dists)
        core_dists_b = compute_cluster_core_distance(d=d, dists=cls_dists.T).T

    mutual_reach_dists = cls_dists.copy()
    np.maximum(mutual_reach_dists, core_dists_a, out=mutual_reach_dists)
    np.maximum(mutual_reach_dists, core_dists_b, out=mutual_reach_dists)

    return mutual_reach_dists


def fn_density_sparseness(
    cls_inds: npt.NDArray[np.int32], dists: npt.NDArray[np.float64], d: int
) -> tuple[float, npt.NDArray[np.int32]]:
    if cls_inds.size <= 3:
        return (0.0, np.empty(0, dtype=int))

    mutual_reach_dists = compute_mutual_reach_dists(dists=dists, d=d, is_symmetric=True)
    internal_node_inds, internal_edge_weights = get_internal_objects(mutual_reach_dists)

    dsc = float(internal_edge_weights.max())
    internal_node_inds = cls_inds[internal_node_inds]

    return (dsc, internal_node_inds)


def fn_density_separation(cls_i: int, cls_j: int, dists: npt.NDArray[np.float64], d: int) -> tuple[int, int, float]:
    mutual_reach_dists = compute_mutual_reach_dists(dists=dists, d=d, is_symmetric=False)
    dspc_ij = float(mutual_reach_dists.min()) if mutual_reach_dists.size else np.inf
    return (cls_i, cls_j, dspc_ij)


def dbcv(
    X: npt.NDArray[np.float64], y: npt.NDArray[np.int32], metric: str = "sqeuclidean", noise_id: int = -1, n_processes: int = 4
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

    n_processes : int or "auto", default="auto"
        Maximum number of parallel processes for processing clusters and cluster pairs.
        If `n_processes="auto"`, the number of parallel processes will be set to 1 for
        datasets with 200 or fewer instances, and 4 for datasets with more than 200 instances.

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
    X = np.atleast_2d(X)

    y = np.asarray(y, dtype=int)

    n, d = X.shape  # NOTE: 'n' must be calculated before removing noise.

    if n != y.size:
        raise ValueError(f"Mismatch in {X.shape[0]=} and {y.size=} dimensions.")

    non_noise_inds = y != noise_id
    X = X[non_noise_inds, :]
    y = y[non_noise_inds]

    y = scipy.stats.rankdata(y, method="dense") - 1
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)

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

    with multiprocessing.Pool(processes=min(n_processes, cluster_ids.size)) as ppool:
        fn_density_sparseness_ = functools.partial(fn_density_sparseness, d=d)
        args = [(cls_ind, get_subarray(dists, inds_a=cls_ind)) for cls_ind in cls_inds]
        for cls_id, (dsc, internal_node_inds) in enumerate(ppool.starmap(fn_density_sparseness_, args)):
            internal_objects_per_cls[cls_id] = internal_node_inds
            dscs[cls_id] = dsc

    n_cls_pairs = (cluster_ids.size * (cluster_ids.size - 1)) // 2

    with multiprocessing.Pool(processes=min(n_processes, n_cls_pairs)) as ppool:
        fn_density_separation_ = functools.partial(fn_density_separation, d=d)

        args = [
            (cls_i, cls_j, get_subarray(dists, internal_objects_per_cls[cls_i], internal_objects_per_cls[cls_j]))
            for cls_i, cls_j in itertools.combinations(cluster_ids, 2)
        ]

        for cls_i, cls_j, dspc_ij in ppool.starmap(fn_density_separation_, args):
            min_dspcs[cls_i] = min(min_dspcs[cls_i], dspc_ij)
            min_dspcs[cls_j] = min(min_dspcs[cls_j], dspc_ij)

    vcs = (min_dspcs - dscs) / (1e-12 + np.maximum(min_dspcs, dscs))
    np.nan_to_num(vcs, copy=False, nan=0.0)
    dbcv = float(np.sum(vcs * cluster_sizes)) / n

    return dbcv
