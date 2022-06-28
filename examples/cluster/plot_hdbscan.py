# -*- coding: utf-8 -*-
"""
====================================
Demo of HDBSCAN clustering algorithm
====================================

In this demo we will take a look at :class:`sklearn.cluster.HDBSCAN` from the
perspective of generalizing the :class:`sklearn.cluster.DBSCAN` algorithm.
We'll compare both algorithms on specific datasets. Finally we'll evaluate
HDBSCAN's sensitivity to certain hyperparameters. We first define a couple
utility functions for convenience.
"""
# %%
import numpy as np

from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # The probability of a point belonging to its labeled cluster determines
    # the size of its marker
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if k == -1 else "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=4 if k == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)


# %%
# Generate sample data
# --------------------
# One of the greatest advantages of HDBSCAN over DBSCAN is its out-of-the-box
# robustness. It's especially remarkable on heterogenous mixtures of data.
# Like DBSCAN, it can model arbitrary shapes and distributions, however unlike
# DBSCAN it does not require specification of an arbitray (and indeed tricky)
# `eps` hyperparameter. For example, below we generate a dataset composed of
# a mixture of three diagonal Gaussians.
fig, axis = plt.subplots(1, 1, figsize=(12, 5))
centers = [[1, 1], [-1, -1], [1.5, -1.5]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=[0.4, 0.1, 0.75], random_state=0
)
plot(X, labels=labels_true, ground_truth=True, ax=axis)
# %%
# Scale Invariance
# -----------------
# It's worth remembering that, while DBSCAN provides a default value for `eps`
# parameter, it is entirely meaningless and must be tuned for your specific
# dataset. As a simple demonstration, consider what happens when we find an
# epsilon value that works for one dataset, and try to apply it to a
# similar but rescaled versions of the dataset. Below are plots of the original
# dataset, and versions rescaled by 0.5 and 3 respectively.
fig, axes = plt.subplots(3, 1, figsize=(12, 16))
parameters = {"eps": 0.3}
dbs = DBSCAN(**parameters).fit(X)
plot(X, dbs.labels_, parameters=parameters, ax=axes[0])
dbs.fit(0.5 * X)
plot(0.5 * X, dbs.labels_, parameters=parameters, ax=axes[1])
dbs.fit(3 * X)
plot(3 * X, dbs.labels_, parameters=parameters, ax=axes[2])

# %%
# Indeed, in order to maintain the same results we would have to scale `eps` by
# the same factor.
fig, axis = plt.subplots(1, 1, figsize=(12, 5))
dbs = DBSCAN(eps=0.9).fit(3 * X)
plot(3 * X, dbs.labels_, parameters={"eps": 0.9}, ax=axis)

# %%
# While standardizing data (e.g. using
# :class:`sklearn.preprocessing.StandardScaler`) helps mitigate this problem,
# great care must be taken to select the appropriate value for `eps`. HDBSCAN
# is much more robust in this sense. HDBSCAN can be seen as clustering over
# all possible values of `eps` and extracting the best clusters from all
# possible clusters (see :ref:`HDBSCAN`). One immediate advantage is that
# HDBSCAN is scale-invariant.
fig, axes = plt.subplots(3, 1, figsize=(12, 16))
hdb = HDBSCAN().fit(X)
plot(X, hdb.labels_, hdb.probabilities_, ax=axes[0])
hdb.fit(0.5 * X)
plot(0.5 * X, hdb.labels_, hdb.probabilities_, ax=axes[1])
hdb.fit(3 * X)
plot(3 * X, hdb.labels_, hdb.probabilities_, ax=axes[2])

# %%
# Multi-Scale Clustering
# ----------------------
# HDBSCAN is much more than scale invariant though -- it is capable of
# multi-scale clustering, which accounts for clusters with varying density.
# Traditional DBSCAN assumes that any potential clusters are homogenous in
# density. HDBSCAN is free from such constraints. To demonstrate this we
# consider the following dataset
fig, axis = plt.subplots(1, 1, figsize=(12, 5))
centers = [[-0.85, -0.85], [-0.85, 0.85], [3, 3], [3, -3]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=[0.2, 0.35, 1.35, 1.35], random_state=0
)
plot(X, labels=labels_true, ground_truth=True, ax=axis)

# %%
# This dataset is more difficult for DBSCAN due to the varying densities and
# spatial separation. If `eps` is too large then we risk falsely clustering the
# two dense clusters as one since their mutual reachability will extend across
# clusters. If `eps` is too small, then we risk fragmenting the sparser
# clusters into many false clusters. Not to mention this requires manually
# tuning choices of `eps` until we find a tradeoff that we are comfortable
# with. Let's see how DBSCAN tackles this.
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
params = {"eps": 0.7}
dbs = DBSCAN(**params).fit(X)
plot(X, dbs.labels_, parameters=params, ax=axes[0])
params = {"eps": 0.3}
dbs = DBSCAN(**params).fit(X)
plot(X, dbs.labels_, parameters=params, ax=axes[1])

# %%
# To properly cluster the two dense clusters, we would need a smaller value of
# epsilon, however at `eps=0.3` we are already fragmenting the sparse clusters,
# which would only become more severe as we decrease epsilon. Indeed it seems
# that DBSCAN is incapable of simultaneously separating the two dense clusters
# while preventing the sparse clusters from fragmenting. Let's compare with
# HDBSCAN.
fig, axis = plt.subplots(1, 1, figsize=(12, 5))
hdb = HDBSCAN().fit(X)
plot(X, hdb.labels_, hdb.probabilities_, ax=axis)

# %%
# HDBSCAN is able to pick up and preserve the multi-scale structure of the
# dataset, all the while requiring no parameter tuning. Of course in practice
# on any sufficiently interesting dataset, there will be some tuning required,
# but this demonstrates the fact that HDBSCAN can yield an entire class of
# solutions that are inaccessible to DBSCAN without nearly as much manual
# intervention and tuning.

# %%
# Hyperparameter Robustness
# -------------------------
# Ultimately tuning will be an important step in any real world application, so
# let's take a look at some of the most important hyperparameters for HDBSCAN.
# While HDBSCAN is free from the `eps` parameter of DBSCAN, it does still have
# some hyperparemeters like `min_cluster_size` and `min_samples` which tune its
# sense of density. We will however see that HDBSCAN is relatively robust to
# these parameters, and these parameters hold clear semantic meaning which help
# in tuning them.
#
# `min_cluster_size`
# ^^^^^^^^^^^^^^^^^^
# This hyperparameter is the minimum number of samples in a group for that
# group to be considered a cluster; groupings smaller than this size will be
# left as noise. The default value is 5. This parameter is generally tuned to
# larger values as needed. Smaller values will likely to lead to results with
# fewer points labeled as noise, however values too small will lead to false
# sub-clusters being picked up and preferred. Larger values tend to be more
# robust w.r.t noisy datasets, e.g. high-variance clusters with significant
# overlap.

PARAM = ({"min_cluster_size": 5}, {"min_cluster_size": 3}, {"min_cluster_size": 25})
fig, axes = plt.subplots(3, 1, figsize=(12, 16))
for i, param in enumerate(PARAM):
    hdb = HDBSCAN(**param).fit(X)
    labels = hdb.labels_

    plot(X, labels, hdb.probabilities_, param, ax=axes[i])

# %%
# `min_samples`
# ^^^^^^^^^^^^^
# This hyperparameter is the number of samples in a neighborhood for a point to
# be considered as a core point. This includes the point itself. defaults to
# the `min_cluster_size`. Similarly to `min_cluster_size`, larger values
# increase the model's robustness to noise, but risks ignoring or discarding
# potentially valid but small clusters. Best tuned after finding a good value
# for `min_cluster_size`.

PARAM = (
    {"min_cluster_size": 20, "min_samples": 5},
    {"min_cluster_size": 20, "min_samples": 3},
    {"min_cluster_size": 20, "min_samples": 25},
)
fig, axes = plt.subplots(3, 1, figsize=(12, 16))
for i, param in enumerate(PARAM):
    hdb = HDBSCAN(**param).fit(X)
    labels = hdb.labels_

    plot(X, labels, hdb.probabilities_, param, ax=axes[i])