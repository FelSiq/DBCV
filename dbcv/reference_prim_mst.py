"""Python translation of the original implementation of Prim's MST in MATLAB.

Reference source: https://github.com/pajaskowiak/dbcv/blob/main/src/MST_Edges.m
"""

import numpy as np
import numpy.typing as npt


def prim_mst(
    graph: npt.NDArray[np.float32], ind_root: int = 0
) -> npt.NDArray[np.float32]:
    n = len(graph)
    intree = np.full(n, fill_value=False)
    d = np.full(n, fill_value=np.inf)

    d[ind_root] = 0
    v = ind_root
    counter = 0

    G = {
        "MST_edges": {
            "node_inds": np.zeros((n - 1, 2), dtype=int),
            "weights": np.zeros(n - 1, dtype=float),
        },
        "MST_degrees": np.zeros(n, dtype=int),
        "MST_parent": np.arange(n),
    }

    while counter < n - 1:
        intree[v] = True
        dist = np.inf

        for w in np.arange(n):
            if w != v and not intree[w]:
                weight = graph[v, w]

                if d[w] > weight:
                    d[w] = weight
                    G["MST_parent"][w] = v

                if dist > d[w]:
                    dist = d[w]
                    next_v = w

        counter += 1
        G["MST_edges"]["node_inds"][counter - 1, :] = (G["MST_parent"][next_v], next_v)
        G["MST_edges"]["weights"][counter - 1] = graph[G["MST_parent"][next_v], next_v]
        G["MST_degrees"][G["MST_parent"][next_v]] += 1
        G["MST_degrees"][next_v] += 1
        v = next_v

    (inds_a, inds_b) = G["MST_edges"]["node_inds"].T
    weights = G["MST_edges"]["weights"]

    mst = np.zeros_like(graph)
    mst[inds_a, inds_b] = weights
    mst[inds_b, inds_a] = weights

    return mst
