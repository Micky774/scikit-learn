import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest

from sklearn.manifold import _mds as mds


def test_smacof():
    # test metric smacof using the data of "Modern Multidimensional Scaling",
    # Borg & Groenen, p 154
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    Z = np.array([[-0.266, -0.539], [0.451, 0.252], [0.016, -0.238], [-0.200, 0.524]])
    X, _ = mds.smacof(sim, init=Z, n_components=2, max_iter=1, n_init=1)
    X_true = np.array(
        [[-1.415, -2.471], [1.633, 1.107], [0.249, -0.067], [-0.468, 1.431]]
    )
    assert_array_almost_equal(X, X_true, decimal=3)


def test_smacof_error():
    # Not symmetric similarity matrix:
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    with pytest.raises(ValueError):
        mds.smacof(sim)

    # Not squared similarity matrix:
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [4, 2, 1, 0]])

    with pytest.raises(ValueError):
        mds.smacof(sim)

    # init not None and not correct format:
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    Z = np.array([[-0.266, -0.539], [0.016, -0.238], [-0.200, 0.524]])
    with pytest.raises(ValueError):
        mds.smacof(sim, init=Z, n_init=1)


def test_eigh_error():
    # Non symmetric (dis)similarity matrix:
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    with pytest.raises(ValueError, match="Array must be symmetric"):
        mds.eigh_scaler(sim)

    # Non squared (dis)similarity matrix:
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [4, 2, 1, 0]])

    with pytest.raises(ValueError, match="array must be 2-dimensional and square"):
        mds.eigh_scaler(sim)

    # Non Euclidean (dis)similarity matrix:
    sim = np.array([[0, 12, 3, 4], [12, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    with pytest.raises(ValueError, match="Dissimilarity matrix must be euclidean"):
        mds.eigh_scaler(sim)


def test_MDS():
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    mds_clf = mds.MDS(metric=False, n_jobs=3, dissimilarity="precomputed")
    mds_clf.fit(sim)


def test_MDS_eigh():
    # Test eigh using example data from "An Introduction to MDS"
    # Florian Wickelmaier, p 11. Validated against R implementation
    # (cmdscale) as well.
    sim = np.array(
        [[0, 93, 82, 133], [93, 0, 52, 60], [82, 52, 0, 111], [133, 60, 111, 0]]
    )

    mds_clf = mds.MDS(metric=True, solver="eigh", dissimilarity="precomputed")
    mds_clf.fit(sim)

    X_true_1 = np.array(
        [
            [-62.831, -32.97448],
            [18.403, 12.02697],
            [-24.960, 39.71091],
            [69.388, -18.76340],
        ]
    )
    X_true_2 = np.copy(X_true_1)
    X_true_2[:, 0] *= -1

    # Signs of columns are dependent on signs of computed eigenvectors
    # which are arbitrary and meaningless
    match = False
    for X_possible in (X_true_1, -X_true_1, X_true_2, -X_true_2):
        match = match or np.allclose(mds_clf.embedding_, X_possible)
    assert match


def test_nonmetric_mds_eigh_error():
    sim = np.ones((2, 2))
    mds_clf = mds.MDS(metric=False, solver="eigh")
    msg = "Using the eigh solver requires metric=True"
    with pytest.raises(ValueError, match=msg):
        mds_clf.fit(sim)
    with pytest.raises(ValueError, match=msg):
        mds_clf.fit_transform(sim)


@pytest.mark.parametrize("k", [0.5, 1.5, 2])
def test_normed_stress(k):
    """Test that non-metric MDS normalized stress is scale-invariant."""
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])

    X1, stress1 = mds.smacof(
        sim, metric=False, normalized_stress=True, max_iter=5, random_state=0
    )
    X2, stress2 = mds.smacof(
        k * sim, metric=False, normalized_stress=True, max_iter=5, random_state=0
    )

    assert_allclose(stress1, stress2, rtol=1e-5)
    assert_allclose(X1, X2, rtol=1e-5)


def test_normalize_metric_warning():
    """
    Test that a UserWarning is emitted when using normalized stress with
    metric-MDS.
    """
    msg = "Normalized stress is not supported"
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    with pytest.raises(ValueError, match=msg):
        mds.smacof(sim, metric=True, normalized_stress=True)
