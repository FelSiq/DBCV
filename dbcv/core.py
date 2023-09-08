import typing as t
import itertools

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


def get_internal_objects(mutual_reach_dists: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mst = scipy.sparse.csgraph.minimum_spanning_tree(mutual_reach_dists)
    mst = mst.toarray()

    is_mst_edges = mst > 0.0

    internal_node_inds = (is_mst_edges + is_mst_edges.T).sum(axis=0) > 1
    internal_node_inds = np.flatnonzero(internal_node_inds)

    internal_edge_weights = mst[*np.meshgrid(internal_node_inds, internal_node_inds)]

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
    cls_inds_a: npt.NDArray[np.int32],
    cls_inds_b: t.Optional[npt.NDArray[np.int32]] = None,
) -> npt.NDArray[np.float64]:
    if cls_inds_b is None:
        cls_dists = dists[*np.meshgrid(cls_inds_a, cls_inds_a)]
        core_dists_a = core_dists_b = compute_cluster_core_distance(d=d, dists=cls_dists)

    else:
        cls_dists = dists[*np.meshgrid(cls_inds_a, cls_inds_b)]
        core_dists_a = compute_cluster_core_distance(d=d, dists=cls_dists)
        core_dists_b = compute_cluster_core_distance(d=d, dists=cls_dists.T).T

    mutual_reach_dists = cls_dists.copy()
    np.maximum(mutual_reach_dists, core_dists_a, out=mutual_reach_dists)
    np.maximum(mutual_reach_dists, core_dists_b, out=mutual_reach_dists)

    return mutual_reach_dists


def dbcv(X: npt.NDArray[np.float64], y: npt.NDArray[np.int32], metric: str = "sqeuclidean", noise_id: int = -1) -> float:
    """Compute DBCV metric.

    DBCV is an intrinsic (= unsupervised/unlabeled) relative metric.

    Parameters
    ----------
    X : npt.NDArray[np.float64] of shape (N, D)
        Data embeddings

    Y : npt.NDArray[np.int32] of shape (N,)
        Cluster assignments.

    Returns
    -------
    DBCV : float
        DBCV metric estimation.

    Source
    ------
    ..[1] "Density-Based Clustering Validation". Davoud Moulavi, Pablo A. Jaskowiak,
          Ricardo J. G. B. Campello, Arthur Zimek, Jörg Sander.
          https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf
    """
    X = np.asfarray(X)
    X = np.atleast_2d(X)

    y = np.asarray(y, dtype=int)

    n, d = X.shape  # NOTE: 'n' must be calculated before removing noise.

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

    for cls_id in cluster_ids:
        cls_inds = np.flatnonzero(y == cls_id)

        if cls_inds.size <= 3:
            internal_objects_per_cls[cls_id] = np.empty(0, dtype=int)
            dscs[cls_id] = 0.0
            continue

        mutual_reach_dists = compute_mutual_reach_dists(dists=dists, d=d, cls_inds_a=cls_inds)
        internal_node_inds, internal_edge_weights = get_internal_objects(mutual_reach_dists)

        internal_objects_per_cls[cls_id] = cls_inds[internal_node_inds]
        dscs[cls_id] = float(internal_edge_weights.max())

    for cls_i, cls_j in itertools.combinations(cluster_ids, 2):
        mutual_reach_dists = compute_mutual_reach_dists(
            dists=dists,
            d=d,
            cls_inds_a=internal_objects_per_cls[cls_i],
            cls_inds_b=internal_objects_per_cls[cls_j],
        )

        if mutual_reach_dists.size == 0:
            continue

        dspc_ij = float(mutual_reach_dists.min())
        min_dspcs[cls_i] = min(min_dspcs[cls_i], dspc_ij)
        min_dspcs[cls_j] = min(min_dspcs[cls_j], dspc_ij)

    vcs = (min_dspcs - dscs) / (1e-12 + np.maximum(min_dspcs, dscs))
    np.nan_to_num(vcs, copy=False, nan=0.0)
    dbcv = float(np.sum(vcs * cluster_sizes)) / n

    if np.isnan(dbcv):
        print(min_dspcs, dscs, vcs, sep="\n", end="\n\n")

    return dbcv
