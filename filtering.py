# Copyright 2025 Kevin Zambello
#
# This program is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

"""Module providing utilities for filtering and classification of singularities."""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def get_optimal_eps(data):
    """
    Find optimal eps for DBSCAN().
    """
    neighbors = NearestNeighbors(n_neighbors=4).fit(data)
    neigh_dist, neigh_ind = neighbors.kneighbors(data)

    y = np.sort(neigh_dist, axis=0)[:, 1]

    x = np.array([float(el) for el in np.arange(1, y.size + 1)])

    my_dist = np.abs(
        (y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0]
    ) / np.sqrt((y[-1] - y[0]) ** 2 + (x[-1] - x[0]) ** 2)

    return y[np.argmax(my_dist)]


def get_cluster(my_blob, which_cluster="random"):
    """
    Find a cluster.
    """
    X = np.array([[np.real(z), np.imag(z)] for z in my_blob])

    clustering = DBSCAN(min_samples=4, eps=get_optimal_eps(X))

    clustering.fit(X)

    good_labels = clustering.labels_[clustering.labels_ >= 0]
    if good_labels.size == 0:
        # DBSCAN has failed
        return np.array([])

    # labels corresponding to the (up to) 3 largest clusters
    unique_labels, counts = np.unique(good_labels, return_counts=True)
    unique_labels = unique_labels[np.argsort(-counts)[0 : np.min([counts.size, 3])]]

    ul_min = unique_labels[0]
    for ul in unique_labels:
        old_idx = clustering.labels_ == ul_min
        new_idx = clustering.labels_ == ul

        if which_cluster == "closest_to_zero":
            old_dist = np.abs(np.mean(X[old_idx, 0] + X[old_idx, 1] * 1.0j))
            new_dist = np.abs(np.mean(X[new_idx, 0] + X[new_idx, 1] * 1.0j))
        elif which_cluster == "closest_to_real":
            old_dist = np.abs(np.mean(X[old_idx, 1] * 1.0j))
            new_dist = np.abs(np.mean(X[new_idx, 1] * 1.0j))
        elif which_cluster == "closest_to_imaginary":
            old_dist = np.abs(np.mean(X[old_idx, 0]))
            new_dist = np.abs(np.mean(X[new_idx, 0]))
        elif which_cluster != "random":
            print("\nERROR: unrecognized which_cluster option.")
            return np.array([])

        if new_dist < old_dist:
            ul_min = ul

    ul_rnd = np.random.choice(unique_labels)

    # print(f"ul_min = {ul_min} ul_rnd = {ul_rnd} unique_labels = {unique_labels}")

    if which_cluster == "random":
        return my_blob[clustering.labels_ == ul_rnd]

    return my_blob[clustering.labels_ == ul_min]


def apply_filter(
    sing_in, boundaries=(-np.inf, np.inf, -np.inf, np.inf), which_cluster=None
):
    """
    Filter singularities within given boundaries optionally extracting a cluster.
    """

    if sing_in.size == 0:
        return sing_in

    sing_out = np.array([])

    idx_a = np.where(sing_in.imag > boundaries[0])[0]
    idx_b = np.where(sing_in.imag < boundaries[1])[0]
    idx_c = np.where(sing_in.real > boundaries[2])[0]
    idx_d = np.where(sing_in.real < boundaries[3])[0]
    idx = np.unique(np.concatenate((idx_a, idx_b, idx_c, idx_d)))
    idx = list(set(idx_a).intersection(idx_b).intersection(idx_c).intersection(idx_d))

    if len(idx) > 0:
        sing_out = np.append(sing_out, sing_in[idx])
        if which_cluster is not None and sing_out.size > 0:
            sing_out = get_cluster(sing_out, which_cluster=which_cluster)

    return sing_out
