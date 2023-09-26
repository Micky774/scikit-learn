# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes
# License: 3-clause BSD

# Code to implement a Dual Tree Boruvka Minimimum Spanning Tree computation
# The algorithm is largely tree independent, but some fine details still
# depend on the particular choice of tree.
#
# The core idea of the algorithm is to do repeated sweeps through the dataset,
# adding edges to the tree with each sweep until a full tree is formed.
# To do this, start with each node (or point) existing in it's own component.
# On each sweep find all the edges of minimum weight (in this instance
# of minimal mutual reachability distance) that join separate components.
# Add all these edges to the list of edges in the spanning tree, and then
# combine together all the components joined by edges. Begin the next sweep ...
#
# Eventually we end up with only one component, and all edges in we added
# form the minimum spanning tree. The key insight is that each sweep is
# essentially akin to a nearest neighbor search (with the caveat about being
# in separate components), and so can be performed very efficiently using
# a space tree such as a kdtree or ball tree. By using a dual tree formalism
# with a query tree and reference tree we can prune when all points im the
# query node are in the same component, as are all the points of the reference
# node. This allows for rapid pruning in the dual tree traversal in later
# stages. Importantly, we can construct the full tree in O(log N) sweeps
# and since each sweep has complexity equal to that of an all points
# nearest neighbor query within the tree structure we are using we end
# up with sub-quadratic complexity at worst.
#
# This code is based on the papers:
#
# Fast Euclidean Minimum Spanning Tree: Algorithm, analysis, and applications
# William B. March, Parikshit Ram, Alexander Gray
# Conference: Proceedings of the 16th ACM SIGKDD International Conference on
#  Knowledge Discovery and Data Mining
# 2010
#
# Tree-Independent Dual-Tree Algorithms
# Ryan R. Curtin, William B. March, Parikshit Ram, David V. Anderson,
# Alexander G. Gray, Charles L. Isbell Jr
# 2013, arXiv 1304.4327
#
# As per the sklearn BallTree and KDTree implementations we make use of
# the rdist for KDTree, which is a faster-to-compute notion of distance
# (for example in the euclidean case it is the distance squared).
#
# To combine together components in between sweeps we make use of
# a union find data structure. This is a separate implementation
# from that used in the labelling of the single linkage tree as
# we can perform more specific optimizations here for what
# is a simpler version of the structure.

import numpy as np

cimport numpy as cnp
from libc.float cimport DBL_MAX
from libc.math cimport fabs, pow

from sklearn.neighbors import BallTree, KDTree

from ...metrics._dist_metrics cimport DistanceMetric
from ...utils._typedefs cimport intp_t, float64_t, int64_t, uint8_t, int8_t

from joblib import Parallel, delayed


cdef float64_t INF = np.inf


# Define the NodeData struct used in sklearn trees for faster
# access to the node data internals in Cython.
cdef struct NodeData_t:
    intp_t idx_start
    intp_t idx_end
    intp_t is_leaf
    float64_t radius


# Define a function giving the minimum distance between two
# nodes of a ball tree
cdef inline float64_t balltree_min_dist_dual(
    float64_t radius1,
    float64_t radius2,
    intp_t node1,
    intp_t node2,
    float64_t[:, ::1] centroid_dist
) nogil except -1:

    cdef float64_t dist_pt = centroid_dist[node1, node2]
    return max(0, (dist_pt - radius1 - radius2))


# Define a function giving the minimum distance between two
# nodes of a kd-tree
cdef inline float64_t kdtree_min_dist_dual(
    DistanceMetric metric,
    intp_t node1,
    intp_t node2,
    float64_t[:, :, ::1] node_bounds,
    intp_t num_features
) except -1:

    cdef float64_t d, d1, d2, rdist = 0.0
    cdef float64_t zero = 0.0
    cdef intp_t j

    if metric.p == INF:
        for j in range(num_features):
            d1 = node_bounds[0, node1, j] - node_bounds[1, node2, j]
            d2 = node_bounds[0, node2, j] - node_bounds[1, node1, j]
            d = (d1 + fabs(d1)) + (d2 + fabs(d2))

            rdist = max(rdist, 0.5 * d)
    else:
        # here we'll use the fact that x + abs(x) = 2 * max(x, 0)
        for j in range(num_features):
            d1 = node_bounds[0, node1, j] - node_bounds[1, node2, j]
            d2 = node_bounds[0, node2, j] - node_bounds[1, node1, j]
            d = (d1 + fabs(d1)) + (d2 + fabs(d2))

            rdist += pow(0.5 * d, metric.p)

    return metric._rdist_to_dist(rdist)


# As above, but this time we use the rdist as per the kdtree
# implementation. This allows us to release the GIL over
# larger sections of code
cdef inline float64_t kdtree_min_rdist_dual(
    DistanceMetric metric,
    intp_t node1,
    intp_t node2,
    float64_t[:, :, ::1] node_bounds,
    intp_t num_features
) nogil except -1:

    cdef float64_t d, d1, d2, rdist = 0.0
    cdef float64_t zero = 0.0
    cdef intp_t j

    if metric.p == INF:
        for j in range(num_features):
            d1 = node_bounds[0, node1, j] - node_bounds[1, node2, j]
            d2 = node_bounds[0, node2, j] - node_bounds[1, node1, j]
            d = (d1 + fabs(d1)) + (d2 + fabs(d2))

            rdist = max(rdist, 0.5 * d)
    else:
        # here we'll use the fact that x + abs(x) = 2 * max(x, 0)
        for j in range(num_features):
            d1 = node_bounds[0, node1, j] - node_bounds[1, node2, j]
            d2 = node_bounds[0, node2, j] - node_bounds[1, node1, j]
            d = (d1 + fabs(d1)) + (d2 + fabs(d2))

            rdist += pow(0.5 * d, metric.p)

    return rdist


cdef class BoruvkaUnionFind(object):
    """Efficient union find implementation.

    Parameters
    ----------

    size : int
        The total size of the set of objects to
        track via the union find structure.

    Attributes
    ----------

    is_component : array of bool; shape (size, 1)
        Array specifying whether each element of the
        set is the root node, or identifier for
        a component.
    """

    cdef intp_t[::1] _parent
    cdef uint8_t[::1] _rank
    cdef cnp.ndarray is_component

    def __init__(self, size):
        self._parent = np.arange(size, dtype=np.intp)
        self._rank = np.zeros(size, dtype=np.uint8)
        self.is_component = np.ones(size, dtype=bool)

    cdef int union_(self, intp_t x, intp_t y) except -1:
        """Union together elements x and y"""
        cdef intp_t x_root = self.find(x)
        cdef intp_t y_root = self.find(y)

        if x_root == y_root:
            return 0

        if self._rank[x_root] < self._rank[y_root]:
            self._parent[x_root] = y_root
            self.is_component[x_root] = False
        elif self._rank[x_root] > self._rank[y_root]:
            self._parent[y_root] = x_root
            self.is_component[y_root] = False
        else:
            self._rank[x_root] += 1
            self._parent[y_root] = x_root
            self.is_component[y_root] = False

        return 0

    cdef intp_t find(self, intp_t x) except -1:
        """Find the root or identifier for the component that x is in"""
        cdef intp_t x_parent
        cdef intp_t x_grandparent

        x_parent = self._parent[x]
        while True:
            if x_parent == x:
                return x
            x_grandparent = self._parent[x_parent]
            self._parent[x] = x_grandparent
            x = x_parent
            x_parent = x_grandparent

    cdef cnp.ndarray[intp_t, ndim=1] components(self):
        """Return an array of all component roots/identifiers"""
        return self.is_component.nonzero()[0]


def _core_dist_query(tree, data, min_samples):
    return tree.query(data, k=min_samples, dualtree=True, breadth_first=True)

cdef class BoruvkaAlgorithm:
    """A Dual Tree Boruvka Algorithm implemented for the sklearn
    KDTree space tree implementation.

    Parameters
    ----------

    tree : KDTree
        The kd-tree to run Dual Tree Boruvka over.

    min_samples : int, optional (default= 5)
        The min_samples parameter of HDBSCAN used to
        determine core distances.

    metric : string, optional (default='euclidean')
        The metric used to compute distances for the tree

    leaf_size : int, optional (default=20)
        The Boruvka algorithm benefits from a smaller leaf size than
        standard kd-tree nearest neighbor searches. The tree passed in
        is used for a kNN search for core distance. A second tree is
        constructed with a smaller leaf size for Boruvka; this is that
        leaf size.

    alpha : float, optional (default=1.0)
        The alpha distance scaling parameter as per Robust Single Linkage.

    approx_min_span_tree : bool, optional (default=False)
        Take shortcuts and only approximate the min spanning tree.
        This is considerably faster but does not return a true
        minimal spanning tree.

    n_jobs : int, optional (default=4)
        The number of parallel jobs used to compute core distances.

    **kwargs :
        Keyword args passed to the metric.
    """

    cdef object tree
    cdef object core_dist_tree
    cdef DistanceMetric dist
    cdef cnp.ndarray _data
    cdef readonly const float64_t[:, ::1] raw_data
    cdef float64_t[:, :, ::1] node_bounds
    cdef float64_t alpha
    cdef int8_t approx_min_span_tree
    cdef intp_t n_jobs
    cdef intp_t min_samples
    cdef intp_t num_points
    cdef intp_t num_nodes
    cdef intp_t num_features
    cdef bint is_KDTree

    cdef public float64_t[::1] core_distance
    cdef public float64_t[::1] bounds
    cdef public intp_t[::1] component_of_point
    cdef public intp_t[::1] component_of_node
    cdef public intp_t[::1] candidate_neighbor
    cdef public intp_t[::1] candidate_point
    cdef public float64_t[::1] candidate_distance
    cdef public float64_t[:, ::1] centroid_distances
    cdef public intp_t[::1] idx_array
    cdef public NodeData_t[::1] node_data
    cdef BoruvkaUnionFind component_union_find
    cdef cnp.ndarray edges
    cdef intp_t num_edges

    cdef intp_t *component_of_point_ptr
    cdef intp_t *component_of_node_ptr
    cdef float64_t *candidate_distance_ptr
    cdef intp_t *candidate_neighbor_ptr
    cdef intp_t *candidate_point_ptr
    cdef float64_t *core_distance_ptr
    cdef float64_t *bounds_ptr

    cdef cnp.ndarray components
    cdef cnp.ndarray core_distance_arr
    cdef cnp.ndarray bounds_arr
    cdef cnp.ndarray _centroid_distances_arr
    cdef cnp.ndarray component_of_point_arr
    cdef cnp.ndarray component_of_node_arr
    cdef cnp.ndarray candidate_point_arr
    cdef cnp.ndarray candidate_neighbor_arr
    cdef cnp.ndarray candidate_distance_arr

    def __init__(self, tree, min_samples=5, metric='euclidean', leaf_size=20,
                 alpha=1.0, approx_min_span_tree=False, n_jobs=4, **kwargs):

        self.core_dist_tree = tree
        self.tree = tree
        self.is_KDTree = isinstance(tree, KDTree)
        self._data = np.array(self.tree.data)
        self.raw_data = self.tree.data
        self.node_bounds = self.tree.node_bounds
        self.alpha = alpha
        self.approx_min_span_tree = approx_min_span_tree
        self.n_jobs = n_jobs

        self.num_points = self.tree.data.shape[0]
        self.num_features = self.tree.data.shape[1]
        self.num_nodes = self.tree.node_data.shape[0]

        self.dist = DistanceMetric.get_metric(metric, **kwargs)

        self.components = np.arange(self.num_points)
        self.bounds_arr = np.empty(self.num_nodes, np.double)
        self.component_of_point_arr = np.empty(self.num_points, dtype=np.intp)
        self.component_of_node_arr = np.empty(self.num_nodes, dtype=np.intp)
        self.candidate_neighbor_arr = np.empty(self.num_points, dtype=np.intp)
        self.candidate_point_arr = np.empty(self.num_points, dtype=np.intp)
        self.candidate_distance_arr = np.empty(self.num_points,
                                               dtype=np.double)
        self.component_union_find = BoruvkaUnionFind(self.num_points)

        self.edges = np.empty((self.num_points - 1, 3))
        self.num_edges = 0

        self.idx_array = self.tree.idx_array
        self.node_data = self.tree.node_data

        self.bounds = (<float64_t[:self.num_nodes:1]> (<float64_t *>
                                                         self.bounds_arr.data))
        self.component_of_point = (<intp_t[:self.num_points:1]> (
            <intp_t *> self.component_of_point_arr.data))
        self.component_of_node = (<intp_t[:self.num_nodes:1]> (
            <intp_t *> self.component_of_node_arr.data))
        self.candidate_neighbor = (<intp_t[:self.num_points:1]> (
            <intp_t *> self.candidate_neighbor_arr.data))
        self.candidate_point = (<intp_t[:self.num_points:1]> (
            <intp_t *> self.candidate_point_arr.data))
        self.candidate_distance = (<float64_t[:self.num_points:1]> (
            <float64_t *> self.candidate_distance_arr.data))

        if not self.is_KDTree:
            # Compute centroids for BallTree
            self._centroid_distances_arr = self.dist.pairwise(self.tree.node_bounds[0])
            self.centroid_distances = (
                <float64_t [:self.num_nodes,
                            :self.num_nodes:1]> (
                                <float64_t *>
                                self._centroid_distances_arr.data))

        self._initialize_components()
        self._compute_bounds()

        # Set up fast pointer access to arrays
        self.component_of_point_ptr = <intp_t *> &self.component_of_point[0]
        self.component_of_node_ptr = <intp_t *> &self.component_of_node[0]
        self.candidate_distance_ptr = <float64_t *> &self.candidate_distance[0]
        self.candidate_neighbor_ptr = <intp_t *> &self.candidate_neighbor[0]
        self.candidate_point_ptr = <intp_t *> &self.candidate_point[0]
        self.core_distance_ptr = <float64_t *> &self.core_distance[0]
        self.bounds_ptr = <float64_t *> &self.bounds[0]

    cdef _compute_bounds(self):
        """Initialize core distances"""

        cdef intp_t n
        cdef intp_t i
        cdef intp_t m

        cdef cnp.ndarray[float64_t, ndim=2] knn_dist
        cdef cnp.ndarray[intp_t, ndim=2] knn_indices

        # A shortcut: if we have a lot of points then we can split the points
        # into four piles and query them in parallel. On multicore systems
        # (most systems) this amounts to a 2x-3x wall clock improvement.
        if self.tree.data.shape[0] > 16384 and self.n_jobs > 1:
            split_cnt = self.num_points // self.n_jobs
            datasets = []
            for i in range(self.n_jobs):
                if i == self.n_jobs - 1:
                    datasets.append(np.asarray(self.tree.data[i*split_cnt:]))
                else:
                    datasets.append(np.asarray(self.tree.data[i*split_cnt:(i+1)*split_cnt]))

            knn_data = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
                delayed(_core_dist_query)
                (self.core_dist_tree, points,
                 self.min_samples)
                for points in datasets)
            knn_dist = np.vstack([x[0] for x in knn_data])
            knn_indices = np.vstack([x[1] for x in knn_data])
        else:
            knn_dist, knn_indices = self.core_dist_tree.query(
                self.tree.data,
                k=self.min_samples,
                dualtree=True,
                breadth_first=True)

        self.core_distance_arr = knn_dist[:, self.min_samples - 1].copy()
        self.core_distance = (<float64_t[:self.num_points:1]> (
            <float64_t *> self.core_distance_arr.data))


        if self.is_KDTree:
            # Since we do everything in terms of rdist to free up the GIL
            # we need to convert all the core distances beforehand
            # to make comparison feasible.
            for n in range(self.num_points):
                self.core_distance[n] = self.dist._dist_to_rdist(
                    self.core_distance[n])

        # Since we already computed NN distances for the min_samples closest
        # points we can use this to do the first round of boruvka -- we won't
        # get every point due to core_distance/mutual reachability distance
        # issues, but we'll get quite a few, and they are the hard ones to
        # get, so fill in any we can and then run update components.
        for n in range(self.num_points):
            for i in range(0, self.min_samples):
                m = knn_indices[n, i]
                if n == m:
                    continue
                if self.core_distance[m] <= self.core_distance[n]:
                    self.candidate_point[n] = n
                    self.candidate_neighbor[n] = m
                    self.candidate_distance[n] = self.core_distance[n]
                    break

        self.update_components()

        for n in range(self.num_nodes):
            self.bounds_arr[n] = <float64_t> DBL_MAX

    cdef _initialize_components(self):
        """Initialize components of the min spanning tree (eventually there
        is only one component; initially each point is its own component)"""

        cdef intp_t n

        for n in range(self.num_points):
            self.component_of_point[n] = n
            self.candidate_neighbor[n] = -1
            self.candidate_point[n] = -1
            self.candidate_distance[n] = DBL_MAX

        for n in range(self.num_nodes):
            self.component_of_node[n] = -(n+1)

    cdef int update_components(self) except -1:
        """Having found the nearest neighbor not in the same component for
        each current component (via tree traversal), run through adding
        edges to the min spanning tree and recomputing components via
        union find."""

        cdef intp_t source
        cdef intp_t sink
        cdef intp_t c
        cdef intp_t component
        cdef intp_t n
        cdef intp_t i
        cdef intp_t p
        cdef intp_t current_component
        cdef intp_t current_source_component
        cdef intp_t current_sink_component
        cdef intp_t child1
        cdef intp_t child2

        cdef NodeData_t node_info

        # For each component there should be a:
        #   - candidate point (a point in the component)
        #   - candiate neighbor (the point to join with)
        #   - candidate_distance (the distance from point to neighbor)
        #
        # We will go through and and an edge to the edge list
        # for each of these, and the union the two points
        # together in the union find structure

        for c in range(self.components.shape[0]):
            component = self.components[c]
            source = self.candidate_point[component]
            sink = self.candidate_neighbor[component]
            if source == -1 or sink == -1:
                continue
                # raise ValueError('Source or sink of edge is not defined!')
            current_source_component = self.component_union_find.find(source)
            current_sink_component = self.component_union_find.find(sink)
            if current_source_component == current_sink_component:
                # We've already joined these, so ignore this edge
                self.candidate_point[component] = -1
                self.candidate_neighbor[component] = -1
                self.candidate_distance[component] = DBL_MAX
                continue
            self.edges[self.num_edges, 0] = source
            self.edges[self.num_edges, 1] = sink
            if self.is_KDTree:
                self.edges[self.num_edges, 2] = self.dist._rdist_to_dist(
                    self.candidate_distance[component])
            else:
                self.edges[self.num_edges, 2] = self.candidate_distance[component]
            self.num_edges += 1

            self.component_union_find.union_(source, sink)

            # Reset everything,and check if we're done
            self.candidate_distance[component] = DBL_MAX
            if self.num_edges == self.num_points - 1:
                self.components = self.component_union_find.components()
                return self.components.shape[0]

        # After having joined everything in the union find data
        # structure we need to go through and determine the components
        # of each point for easy lookup.
        #
        # Have done that we then go through and set the component
        # of each node, as this provides fast pruning in later
        # tree traversals.
        for n in range(self.tree.data.shape[0]):
            self.component_of_point[n] = self.component_union_find.find(n)

        for n in range(self.tree.node_data.shape[0] - 1, -1, -1):
            node_info = self.node_data[n]
            # Case 1:
            #    If the node is a leaf we need to check that every point
            #    in the node is of the same component
            if node_info.is_leaf:
                current_component = self.component_of_point[
                    self.idx_array[node_info.idx_start]]
                for i in range(node_info.idx_start + 1, node_info.idx_end):
                    p = self.idx_array[i]
                    if self.component_of_point[p] != current_component:
                        break
                else:
                    self.component_of_node[n] = current_component
            # Case 2:
            #    If the node is not a leaf we only need to check
            #    that both child nodes are in the same component
            else:
                child1 = 2 * n + 1
                child2 = 2 * n + 2
                if (self.component_of_node[child1] ==
                        self.component_of_node[child2]):
                    self.component_of_node[n] = self.component_of_node[child1]

        # Since we're working with mutual reachability distance we often have
        # ties or near ties; because of that we can benefit by not resetting
        # the bounds unless we get stuck (don't join any components). Thus
        # we check for that, and only reset bounds in the case where we have
        # the same number of components as we did going in. This doesn't
        # produce a true min spanning tree, but only and approximation
        # Thus only do this if the caller is willing to accept such
        if self.approx_min_span_tree:
            last_num_components = self.components.shape[0]
            self.components = self.component_union_find.components()

            if self.components.shape[0] == last_num_components:
                # Reset bounds
                for n in range(self.num_nodes):
                    self.bounds_arr[n] = <float64_t> DBL_MAX
        else:
            self.components = self.component_union_find.components()

            for n in range(self.num_nodes):
                self.bounds_arr[n] = <float64_t> DBL_MAX

        return self.components.shape[0]

    cdef int dual_tree_traversal(self, intp_t node1,
                                 intp_t node2) nogil except -1:
        """Perform a dual tree traversal, pruning wherever possible, to find
        the nearest neighbor not in the same component for each component.
        This is akin to a standard dual tree NN search, but we also prune
        whenever all points in query and reference nodes are in the same
        component."""

        cdef intp_t[::1] point_indices1, point_indices2

        cdef intp_t i
        cdef intp_t j

        cdef intp_t p
        cdef intp_t q

        cdef intp_t parent
        cdef intp_t child1
        cdef intp_t child2

        cdef double node_dist

        cdef NodeData_t node1_info = self.node_data[node1]
        cdef NodeData_t node2_info = self.node_data[node2]
        cdef NodeData_t parent_info
        cdef NodeData_t left_info
        cdef NodeData_t right_info

        cdef intp_t component1
        cdef intp_t component2

        cdef float64_t d

        cdef float64_t mr_dist
        cdef float64_t _radius

        cdef float64_t new_bound
        cdef float64_t new_upper_bound
        cdef float64_t new_lower_bound
        cdef float64_t bound_max
        cdef float64_t bound_min

        cdef intp_t left
        cdef intp_t right
        cdef float64_t left_dist
        cdef float64_t right_dist

        # Compute the distance between the query and reference nodes
        if self.is_KDTree:
            node_dist = kdtree_min_rdist_dual(self.dist,
                                            node1, node2, self.node_bounds,
                                            self.num_features)
        else: #BallTree
            node_dist = balltree_min_dist_dual(node1_info.radius,
                                            node2_info.radius,
                                            node1, node2,
                                            self.centroid_distances)


        # If the distance between the nodes is less than the current bound for
        # the query and the nodes are not in the same component continue;
        # otherwise we get to prune this branch and return early.
        if node_dist < self.bounds_ptr[node1]:
            if (self.component_of_node_ptr[node1] ==
                self.component_of_node_ptr[node2] and
                    self.component_of_node_ptr[node1] >= 0):
                return 0
        else:
            return 0

        # Case 1: Both nodes are leaves
        #       for each pair of points in node1 x node2 we need
        #       to compute the distance and see if it better than
        #       the current nearest neighbor for the component of
        #       the point in the query node.
        #
        #       We get to take some shortcuts:
        #           - if the core distance for a point is larger than
        #             the distance to the nearst neighbor of the
        #             component of the point ... then we can't get
        #             a better mutual reachability distance and we
        #             can skip computing anything for that point
        #           - if the points are in the same component we
        #             don't have to compute the distance.
        #
        #       We also have some catches:
        #           - we need to compute mutual reachability distance
        #             not just the ordinary distance; this involves
        #             fiddling with core distances.
        #           - We need to scale distances according to alpha,
        #             but don't want to lose performance in the case
        #             that alpha is 1.0.
        #
        #       Finally we can compute new bounds for the query node
        #       based on the distances found here, so do that and
        #       propagate the results up the tree.
        if node1_info.is_leaf and node2_info.is_leaf:

            new_upper_bound = 0.0
            new_lower_bound = DBL_MAX

            point_indices1 = self.idx_array[node1_info.idx_start:
                                            node1_info.idx_end]
            point_indices2 = self.idx_array[node2_info.idx_start:
                                            node2_info.idx_end]

            for i in range(point_indices1.shape[0]):

                p = point_indices1[i]
                component1 = self.component_of_point_ptr[p]

                if (self.core_distance_ptr[p] >
                        self.candidate_distance_ptr[component1]):
                    continue

                for j in range(point_indices2.shape[0]):

                    q = point_indices2[j]
                    component2 = self.component_of_point_ptr[q]

                    if (self.core_distance_ptr[q] >
                            self.candidate_distance_ptr[component1]):
                        continue

                    if component1 != component2:
                        if self.is_KDTree:
                            d = self.dist.rdist(
                                &self.raw_data[self.num_features * p][0],
                                &self.raw_data[self.num_features * q][0],
                                self.num_features
                            )
                        else:
                            d = self.dist.dist(
                                &self.raw_data[self.num_features * p][0],
                                &self.raw_data[self.num_features * q][0],
                                self.num_features
                            ) * self.alpha
                        if self.alpha != 1.0:
                            mr_dist = max(
                                d / self.alpha,
                                self.core_distance_ptr[p],
                                self.core_distance_ptr[q]
                            )
                        else:
                            mr_dist = max(
                                d, self.core_distance_ptr[p],
                                self.core_distance_ptr[q]
                            )
                        if mr_dist < self.candidate_distance_ptr[component1]:
                            self.candidate_distance_ptr[component1] = mr_dist
                            self.candidate_neighbor_ptr[component1] = q
                            self.candidate_point_ptr[component1] = p

                new_upper_bound = max(
                    new_upper_bound,
                    self.candidate_distance_ptr[component1]
                )
                new_lower_bound = min(
                    new_lower_bound,
                    self.candidate_distance_ptr[component1]
                )

            # Compute new bounds for the query node, and
            # then propagate the results of that computation
            # up the tree.
            _radius = self.dist._dist_to_rdist(node1_info.radius) if self.is_KDTree else node1_info.radius
            new_bound = min(
                new_upper_bound,
                new_lower_bound + 2 * _radius
            )
            if new_bound < self.bounds_ptr[node1]:
                self.bounds_ptr[node1] = new_bound

                # Propagate bounds up the tree
                while node1 > 0:
                    parent = (node1 - 1) // 2
                    left = 2 * parent + 1
                    right = 2 * parent + 2

                    parent_info = self.node_data[parent]
                    left_info = self.node_data[left]
                    right_info = self.node_data[right]

                    bound_max = max(self.bounds_ptr[left],
                                    self.bounds_ptr[right])

                    if self.is_KDTree:
                        new_bound = bound_max
                    else:
                        bound_min = min(
                            self.bounds_ptr[left] + 2 * (parent_info.radius - left_info.radius),
                            self.bounds_ptr[right] + 2 * (parent_info.radius - right_info.radius)
                        )

                        if bound_min > 0:
                            new_bound = min(bound_max, bound_min)
                        else:
                            new_bound = bound_max
                    if new_bound < self.bounds_ptr[parent]:
                        self.bounds_ptr[parent] = new_bound
                        node1 = parent
                    else:
                        break

        # Case 2a: The query node is a leaf, or is smaller than
        #          the reference node.
        #
        #       We descend in the reference tree. We first
        #       compute distances between nodes to determine
        #       whether we should prioritise the left or
        #       right branch in the reference tree.
        elif node1_info.is_leaf or (not node2_info.is_leaf and
                                    node2_info.radius > node1_info.radius):

            left = 2 * node2 + 1
            right = 2 * node2 + 2

            if self.is_KDTree:
                left_dist = kdtree_min_rdist_dual(self.dist,
                                                node1, left,
                                                self.node_bounds,
                                                self.num_features)
                right_dist = kdtree_min_rdist_dual(self.dist,
                                                node1, right,
                                                self.node_bounds,
                                                self.num_features)
            else:
                node2_info = self.node_data[left]
                left_dist = balltree_min_dist_dual(node1_info.radius,
                                                node2_info.radius,
                                                node1, left,
                                                self.centroid_distances)
                node2_info = self.node_data[right]
                right_dist = balltree_min_dist_dual(node1_info.radius,
                                                    node2_info.radius,
                                                    node1, right,
                                                    self.centroid_distances)

            if left_dist < right_dist:
                self.dual_tree_traversal(node1, left)
                self.dual_tree_traversal(node1, right)
            else:
                self.dual_tree_traversal(node1, right)
                self.dual_tree_traversal(node1, left)

        # Case 2b: The reference node is a leaf, or is smaller than
        #          the query node.
        #
        #       We descend in the query tree. We first
        #       compute distances between nodes to determine
        #       whether we should prioritise the left or
        #       right branch in the query tree.
        else:
            left = 2 * node1 + 1
            right = 2 * node1 + 2
            if self.is_KDTree:
                left_dist = kdtree_min_rdist_dual(self.dist,
                                                left, node2,
                                                self.node_bounds,
                                                self.num_features)
                right_dist = kdtree_min_rdist_dual(self.dist,
                                                right, node2,
                                                self.node_bounds,
                                                self.num_features)
            else:
                node1_info = self.node_data[left]
                left_dist = balltree_min_dist_dual(node1_info.radius,
                                                node2_info.radius,
                                                left, node2,
                                                self.centroid_distances)
                node1_info = self.node_data[right]
                right_dist = balltree_min_dist_dual(node1_info.radius,
                                                    node2_info.radius,
                                                    right, node2,
                                                    self.centroid_distances)


            if left_dist < right_dist:
                self.dual_tree_traversal(left, node2)
                self.dual_tree_traversal(right, node2)
            else:
                self.dual_tree_traversal(right, node2)
                self.dual_tree_traversal(left, node2)

        return 0

    cpdef spanning_tree(self):
        """Compute the minimum spanning tree of the data held by
        the tree passed in at construction"""

        cdef intp_t num_components
        cdef intp_t num_nodes

        num_components = self.tree.data.shape[0]
        num_nodes = self.tree.node_data.shape[0]
        while num_components > 1:
            self.dual_tree_traversal(0, 0)
            num_components = self.update_components()

        return self.edges
