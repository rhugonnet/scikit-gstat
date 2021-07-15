from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import cKDTree
from scipy import sparse
import numpy as np
import time

def _sparse_dok_get(m, fill_value=np.NaN):
    """Like m.toarray(), but setting empty values to `fill_value`, by
    default `np.NaN`, rather than 0.0.

    Parameters
    ----------
    m : scipy.sparse.dok_matrix
    fill_value : float
    """
    mm = np.full(m.shape, fill_value)
    for (x, y), value in m.items():
        mm[x, y] = value
    return mm


class DistanceMethods(object):
    def find_closest(self, idx, max_dist=None, N=None):
        """find neighbors
        Find the (N) closest points (in the right set) to the point with
        index idx (in the left set).

        Parameters
        ----------
        idx : int
            Index of the point that the N closest neighbors
            are searched for.
        max_dist : float
            Maximum distance at which other points are searched
        N : int
            Number of points searched.

        Returns
        -------
        ridx : numpy.ndarray
            Indices of the N closeset points to idx

        """

        if max_dist is None:
            max_dist = self.max_dist
        else:
            if self.max_dist is not None and max_dist != self.max_dist:
                raise AttributeError(
                    "max_dist specified and max_dist != self.max_dist"
                )

        if isinstance(self.dists, sparse.spmatrix):
            dists = self.dists.getrow(idx)
        else:
            dists = self.dists[idx, :]
        if isinstance(dists, sparse.spmatrix):
            ridx = np.array([k[1] for k in dists.todok().keys()])
        elif max_dist is not None:
            ridx = np.where(dists <= max_dist)[0]
        else:
            ridx = np.arange(len(dists))
        if ridx.size > N:
            if isinstance(dists, sparse.spmatrix):
                selected_dists = dists[0, ridx].toarray()[0, :]
            else:
                selected_dists = dists[ridx]
            sorted_ridx = np.argsort(selected_dists, kind="stable")
            ridx = ridx[sorted_ridx][:N]
        return ridx


class MetricSpace(DistanceMethods):
    """
    A MetricSpace represents a point cloud together with a distance
    metric and possibly a maximum distance. It efficiently provides
    the distances between each point pair (when shorter than the
    maximum distance).

    Note: If a max_dist is specified a sparse matrix representation is
    used for the distances, which saves space and calculation time for
    large datasets, especially where max_dist << the size of the point
    cloud in space. However, it slows things down for small datasets.
    """

    def __init__(self, coords, dist_metric="euclidean", max_dist=None):
        """ProbabalisticMetricSpace class

        Parameters
        ----------
        coords : numpy.ndarray
            Coordinate array of shape (Npoints, Ndim)
        dist_metric : str
            Distance metric names as used by scipy.spatial.distance.pdist
        max_dist : float
            Maximum distance between points after which the distance
            is considered infinite and not calculated.
        """
        self.coords = coords.copy()
        self.dist_metric = dist_metric
        self.max_dist = max_dist
        self._tree = None
        self._dists = None

        # Check if self.dist_metric is valid
        try:
            pdist(self.coords[:1, :], metric=self.dist_metric)
        except ValueError as e:
            raise e

    @property
    def tree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of `self.coords`. Undefined otherwise."""
        # only Euclidean supported
        if self.dist_metric != "euclidean":
            raise ValueError((
                "A coordinate tree can only be constructed "
                "for an euclidean space"
            ))

        # if not cached - calculate
        if self._tree is None:
            self._tree = cKDTree(self.coords)

        # return
        return self._tree

    @property
    def dists(self):
        """A distance matrix of all point pairs. If `self.max_dist` is
        not `None` and `self.dist_metric` is set to `euclidean`, a
        `scipy.sparse.csr_matrix` sparse matrix is returned.
        """
        # calculate if not cached
        if self._dists is None:
            # check if max dist is given
            if self.max_dist is not None and self.dist_metric == "euclidean":
                self._dists = self.tree.sparse_distance_matrix(
                    self.tree,
                    self.max_dist,
                    output_type="coo_matrix"
                ).tocsr()

            # otherwise use pdist
            else:
                self._dists = squareform(
                    pdist(self.coords, metric=self.dist_metric)
                )

        # return
        return self._dists

    def diagonal(self, idx=None):
        """
        Return a diagonal matrix (as per
        :func:`squareform <scipy.spatial.distance.squareform>`),
        optionally for a subset of the points

        Parameters
        ----------
        idx : list
            list of indices that the diagonal matrix is calculated for.

        Returns
        -------
        diagonal : numpy.ndarray
            squareform matrix of the subset of coordinates

        """
        # get the dists
        dist_mat = self.dists

        # subset dists if requested
        if idx is not None:
            dist_mat = dist_mat[idx, :][:, idx]

        # handle sparse matrix
        if isinstance(self.dists, sparse.spmatrix):
            dist_mat = _sparse_dok_get(dist_mat.todok(), np.inf)
            np.fill_diagonal(dist_mat, 0)  # Normally set to inf

        return squareform(dist_mat)

    def __len__(self):
        return len(self.coords)


class MetricSpacePair(DistanceMethods):
    """
    A MetricSpacePair represents a set of point clouds (MetricSpaces).
    It efficiently provides the distances between each point in one
    point cloud and each point in the other point cloud (when shorter
    than the maximum distance). The two point clouds are required to
    have the same distance metric as well as maximum distance.
    """
    def __init__(self, ms1, ms2):
        """
        Parameters
        ----------
        ms1 : MetricSpace
        ms2 : MetricSpace

        Note: `ms1` and `ms2` need to have the same `max_dist` and
        `distance_metric`.
        """
        # check input data
        # same distance metrix
        if ms1.dist_metric != ms2.dist_metric:
            raise ValueError(
                "Both MetricSpaces need to have the same distance metric"
            )

        # same max_dist setting
        if ms1.max_dist != ms2.max_dist:
            raise ValueError(
                "Both MetricSpaces need to have the same max_dist"
            )
        self.ms1 = ms1
        self.ms2 = ms2
        self._dists = None

    @property
    def dist_metric(self):
        return self.ms1.dist_metric

    @property
    def max_dist(self):
        return self.ms1.max_dist

    @property
    def dists(self):
        """A distance matrix of all point pairs. If `self.max_dist` is
        not `None` and `self.dist_metric` is set to `euclidean`, a
        `scipy.sparse.csr_matrix` sparse matrix is returned.
        """
        # if not cached, calculate
        if self._dists is None:
            # handle euclidean with max_dist with Tree
            if self.max_dist is not None and self.dist_metric == "euclidean":
                self._dists = self.ms1.tree.sparse_distance_matrix(
                    self.ms2.tree,
                    self.max_dist,
                    output_type="coo_matrix"
                ).tocsr()

            # otherwise Tree not possible
            else:
                self._dists = cdist(
                    self.ms1.coords,
                    self.ms2.coords,
                    metric=self.ms1.dist_metric
                )

        # return
        return self._dists


class ProbabalisticMetricSpace(MetricSpace):
    """Like MetricSpace but samples the distance pairs only returning a
       `samples` sized subset. `samples` can either be a fraction of
       the total number of pairs (float < 1), or an integer count.
    """
    def __init__(
            self,
            coords,
            dist_metric="euclidean",
            max_dist=None,
            samples=0.5,
            rnd=None
        ):
        """ProbabalisticMetricSpace class

        Parameters
        ----------
        coords : numpy.ndarray
            Coordinate array of shape (Npoints, Ndim)
        dist_metric : str
            Distance metric names as used by scipy.spatial.distance.pdist
        max_dist : float
            Maximum distance between points after which the distance
            is considered infinite and not calculated.
        samples : float, int
            Number of samples (int) or fraction of coords to sample (float < 1).
        rnd : numpy.random.RandomState, int
            Random state to use for the sampling.
        """
        self.coords = coords.copy()
        self.dist_metric = dist_metric
        self.max_dist = max_dist
        self.samples = samples
        if rnd is None:
            self.rnd = np.random
        elif isinstance(rnd, np.random.RandomState):
            self.rnd = rnd
        else:
            self.rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(rnd)))

        self._lidx = None
        self._ridx = None
        self._ltree = None
        self._rtree = None
        self._dists = None
        # Do a very quick check to see throw exceptions 
        # if self.dist_metric is invalid...
        pdist(self.coords[:1, :], metric=self.dist_metric)

    @property
    def sample_count(self):
        if isinstance(self.samples, int):
            return self.samples
        return int(self.samples * len(self.coords))

    @property
    def lidx(self):
        """The sampled indices into `self.coords` for the left sample."""
        if self._lidx is None:
            self._lidx = self.rnd.choice(len(self.coords), size=self.sample_count, replace=False)
        return self._lidx

    @property
    def ridx(self):
        """The sampled indices into `self.coords` for the right sample."""
        if self._ridx is None:
            self._ridx = self.rnd.choice(len(self.coords), size=self.sample_count, replace=False)
        return self._ridx

    @property
    def ltree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of the left sample of `self.coords`. Undefined otherwise."""

        # only Euclidean supported
        if self.dist_metric != "euclidean":
            raise ValueError((
                "A coordinate tree can only be constructed "
                "for an euclidean space"
            ))

        if self._ltree is None:
            self._ltree = cKDTree(self.coords[self.lidx, :])
        return self._ltree

    @property
    def rtree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of the right sample of `self.coords`. Undefined otherwise."""

        # only Euclidean supported
        if self.dist_metric != "euclidean":
            raise ValueError((
                "A coordinate tree can only be constructed "
                "for an euclidean space"
            ))

        if self._rtree is None:
            self._rtree = cKDTree(self.coords[self.ridx, :])
        return self._rtree

    @property
    def dists(self):
        """A distance matrix of the sampled point pairs as a
        `scipy.sparse.csr_matrix` sparse matrix. """
        if self._dists is None:
            max_dist = self.max_dist
            if max_dist is None:
                max_dist = np.finfo(float).max
            dists = self.ltree.sparse_distance_matrix(
                self.rtree,
                max_dist,
                output_type="coo_matrix"
            ).tocsr()
            dists.resize((len(self.coords), len(self.coords)))
            dists.indices = self.ridx[dists.indices]
            dists = dists.tocsc()
            dists.indices = self.lidx[dists.indices]
            dists = dists.tocsr()
            self._dists = dists
        return self._dists

class RasterEquidistantMetricSpace(MetricSpace):
    """Like ProbabilisticMetricSpace but only applies to Raster data (2D gridded data) and
    samples iteratively an `equidistant` subset within distances to a 'center' subset.
    Subsets can either be a fraction of the total number of pairs (float < 1), or an integer count.
      """

    def __init__(
            self,
            coords,
            shape,
            extent,
            runs=None,
            dist_metric="euclidean",
            max_dist=None,
            samples=0.5,
            rnd=None
    ):
        """RasterEquidistantMetricSpace class

        Parameters
        ----------
        coords : numpy.ndarray
            Coordinate array of shape (Npoints, 2)
        shape: tuple[int, int]
            Shape of raster (X, Y)
        extent: tuple[float, float, float, float]
            Extent of raster (Xmin, Xmax, Ymin, Ymax)
        runs: int
            Number of subsamples to concatenate
        dist_metric : str
            Distance metric names as used by scipy.spatial.distance.pdist
        max_dist : float
            Maximum distance between points after which the distance
            is considered infinite and not calculated.
        samples : float, int
            Number of samples (int) or fraction of coords to sample (float < 1).
        rnd : numpy.random.RandomState, int
            Random state to use for the sampling.
        """
        self.coords = coords.copy()
        self.dist_metric = dist_metric
        self.shape = shape
        self.extent = extent
        self.res = np.sqrt(((extent[1] - extent[0])/shape[0])**2 + ((extent[3] - extent[2])/shape[1])**2)

        # TODO: if the number of runs is not specified, divide the grid in N center samples
        if runs is None:
            runs = 10

        self.runs = runs

        # if the maximum distance is not specified, find the maximum possible distance from the grid dimensions
        if max_dist is None:
            max_dist = np.max(self.shape) * self.res
        self.max_dist = max_dist

        self.samples = samples
        if rnd is None:
            self.rnd = np.random
        elif isinstance(rnd, np.random.RandomState):
            self.rnd = rnd
        else:
            self.rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence( )))

        # Index and KDTree of center sample
        self._cidx = None
        self._ctree = None

        # Index and KDTree of equidistant sample
        self._eqidx = None
        self._eqtree = None

        self._center = None
        self._center_radius = None
        self._dists = None
        # Do a very quick check to see throw exceptions
        # if self.dist_metric is invalid...
        pdist(self.coords[:1, :], metric=self.dist_metric)

    @property
    def sample_count(self):
        if isinstance(self.samples, int):
            return self.samples
        return int(self.samples * len(self.coords))

    @property
    def cidx(self):
        """The sampled indices into `self.coords` for the center sample."""

        if self._cidx is None:

            # First index: preselect samples in a disk of radius large enough to contain twice the sample_count samples
            dist_center = np.sqrt((self.coords[:, 0] - self._center[0]) ** 2 + (
                    self.coords[:, 1] - self._center[1]) ** 2)
            idx1 = dist_center < self._center_radius
            coords_center = self.coords[idx1, :]

            indices1 = np.argwhere(idx1)

            # Second index: randomly select half of the valid pixels, so that the other half can be used by the equidist
            # sample for low distances
            indices2 = self.rnd.choice(len(coords_center), size=int(len(coords_center) / 2), replace=False)

            self._cidx = indices1[indices2].squeeze()

        return self._cidx

    @property
    def ctree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of the center sample of `self.coords`. Undefined otherwise."""

        # only Euclidean supported
        if self.dist_metric != "euclidean":
            raise ValueError((
                "A coordinate tree can only be constructed "
                "for an euclidean space"
            ))

        if self._ctree is None:
            self._ctree = cKDTree(self.coords[self.cidx, :])
        return self._ctree


    @property
    def eqidx(self):
        """The sampled indices into `self.coords` for the equidistant sample."""

        # Hardcode exponential bins for now, see about providing more options later
        list_inout_radius = [0.]
        rad = self._center_radius
        increasing_rad = rad
        while increasing_rad < self.max_dist:
            list_inout_radius.append(increasing_rad)
            increasing_rad *= 1.5
        list_inout_radius.append(self.max_dist)

        dist_center = np.sqrt((self.coords[:, 0] - self._center[0]) ** 2 + (
                self.coords[:, 1] - self._center[1]) ** 2)

        if self._eqidx is None:

            # Loop over an iterative sampling in rings
            list_idx = []
            for i in range(len(list_inout_radius)-1):
                # First index: preselect samples in a ring of inside radius and outside radius
                idx1 = np.logical_and(dist_center < list_inout_radius[i+1], dist_center >= list_inout_radius[i])
                coords_equi = self.coords[idx1, :]

                indices1 = np.argwhere(idx1)

                # Second index: randomly select half of the valid pixels, so that the other half can be used by the equidist
                # sample for low distances
                indices2 = ~self.rnd.choice(len(coords_equi), size=min(len(coords_equi),self.sample_count), replace=False)
                subidx = indices1[indices2]
                if len(subidx)>1:
                    list_idx.append(subidx.squeeze())

            self._eqidx = np.concatenate(list_idx)

        return self._eqidx

    @property
    def eqtree(self):
        """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
        instance of the equidistant sample of `self.coords`. Undefined otherwise."""

        # only Euclidean supported
        if self.dist_metric != "euclidean":
            raise ValueError((
                "A coordinate tree can only be constructed "
                "for an euclidean space"
            ))

        if self._eqtree is None:
            self._eqtree = cKDTree(self.coords[self.eqidx, :])
        return self._eqtree

    @property
    def dists(self):
        """A distance matrix of the sampled point pairs as a
        `scipy.sparse.csr_matrix` sparse matrix. """

        if self._dists is None:

            list_dists, list_cidx, list_eqidx = ([] for i in range(3))

            idx_center = self.rnd.choice(len(self.coords), size=(2, self.runs), replace=False)

            t00 = time.time()

            for i in range(self.runs):

                t0 = time.time()
                # Each run has a different center
                self._center = (self.coords[idx_center[0, i], 0], self.coords[idx_center[1, i], 1])
                # Radius of center based on sample count
                self._center_radius = np.sqrt(self.sample_count / np.pi) * self.res

                #Derive explicitly the indexes
                tmp = self.cidx
                tmp2 = self.eqidx

                t01 = time.time()
                print('Inside loop (indexes): '+str(t01-t0))

                dists = self.ctree.sparse_distance_matrix(
                    self.eqtree,
                    self.max_dist,
                    output_type="coo_matrix"
                )

                t1 = time.time()
                print('Inside loop (pairwise): '+str(t1-t01))

                list_dists.append(dists.data)
                list_cidx.append(self.cidx[dists.row])
                list_eqidx.append(self.eqidx[dists.col])

                self._cidx = None
                self._ctree = None
                self._eqidx = None
                self._eqtree = None

            t11 = time.time()
            print('Loop full duration: '+str(t11 - t00))
            # concatenate the coo matrixes
            d = np.concatenate(list_dists)
            c = np.concatenate(list_cidx)
            eq = np.concatenate(list_eqidx)

            t2 = time.time()

            # remove possible duplicates (that would be summed by default)
            # from https://stackoverflow.com/questions/28677162/ignoring-duplicate-entries-in-sparse-matrix

            # Stable solution
            # c, eq, d = zip(*set(zip(c, eq, d)))
            # dists = sparse.csr_matrix((d, (c, eq)), shape=(len(self.coords), len(self.coords)))

            # Solution 5+ times faster than the preceding, but relies on _update() which might change in scipy (which
            # only has an implemented method for summing duplicates, and not ignoring them yet)
            dok = sparse.dok_matrix((len(self.coords), len(self.coords)))
            dok._update(zip(zip(c, eq), d))
            dists = dok.tocsr()

            t3 = time.time()

            print('Zipping: ' + str(t3 - t2))

            self._dists = dists

        return self._dists

    class RasterEquidistantMetricSpace2(MetricSpace):
        """Like ProbabilisticMetricSpace but only applies to Raster data (2D gridded data) and
        samples iteratively an `equidistant` subset within distances to a 'center' subset.
        Subsets can either be a fraction of the total number of pairs (float < 1), or an integer count.
          """

        def __init__(
                self,
                coords,
                shape,
                extent,
                runs=None,
                dist_metric="euclidean",
                max_dist=None,
                samples=0.5,
                rnd=None
        ):
            """RasterEquidistantMetricSpace class

            Parameters
            ----------
            coords : numpy.ndarray
                Coordinate array of shape (Npoints, 2)
            shape: tuple[int, int]
                Shape of raster (X, Y)
            extent: tuple[float, float, float, float]
                Extent of raster (Xmin, Xmax, Ymin, Ymax)
            runs: int
                Number of subsamples to concatenate
            dist_metric : str
                Distance metric names as used by scipy.spatial.distance.pdist
            max_dist : float
                Maximum distance between points after which the distance
                is considered infinite and not calculated.
            samples : float, int
                Number of samples (int) or fraction of coords to sample (float < 1).
            rnd : numpy.random.RandomState, int
                Random state to use for the sampling.
            """
            self.coords = coords.copy()
            self.dist_metric = dist_metric
            self.shape = shape
            self.extent = extent
            self.res = np.sqrt(((extent[1] - extent[0]) / shape[0]) ** 2 + ((extent[3] - extent[2]) / shape[1]) ** 2)

            # TODO: if the number of runs is not specified, divide the grid in N center samples
            if runs is None:
                runs = 10

            self.runs = runs

            # if the maximum distance is not specified, find the maximum possible distance from the grid dimensions
            if max_dist is None:
                max_dist = np.max(self.shape) * self.res
            self.max_dist = max_dist

            self.samples = samples
            if rnd is None:
                self.rnd = np.random
            elif isinstance(rnd, np.random.RandomState):
                self.rnd = rnd
            else:
                self.rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence()))

            # Index and KDTree of center sample
            self._cidx = None
            self._ctree = None

            # Index and KDTree of equidistant sample
            self._eqidx = None
            self._eqtree = None

            self._centers = None
            self._center_radius = None
            self._dists = None
            # Do a very quick check to see throw exceptions
            # if self.dist_metric is invalid...
            pdist(self.coords[:1, :], metric=self.dist_metric)

        @property
        def sample_count(self):
            if isinstance(self.samples, int):
                return self.samples
            return int(self.samples * len(self.coords))

        @property
        def cidx(self):
            """The sampled indices into `self.coords` for the center sample."""

            if self._cidx is None:
                # First index: preselect samples in a disk of radius large enough to contain twice the sample_count samples
                dist_center = np.sqrt((self.coords[None, :, 0] - self._centers[:, 0, None]) ** 2 + (
                        self.coords[None, :, 1] - self._centers[:, 0, None]) ** 2)
                idx1 = dist_center < self._center_radius

                # Cannot vectorize calculations here, as each subsample might have a different length
                cidxs = []
                for i in range(self.runs):
                    indices1 = np.argwhere(idx1[i, :])
                    count = np.count_nonzero(idx1[i, :])

                    # Second index: randomly select half of the valid pixels, so that the other half can be used by the equidist
                    # sample for low distances
                    indices2 = self.rnd.choice(count, size=int(count / 2), replace=False)
                    cidxs.append(indices1[indices2].squeeze())

                self._cidx = cidxs

            return self._cidx

        @property
        def ctree(self):
            """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
            instance of the center sample of `self.coords`. Undefined otherwise."""

            # only Euclidean supported
            if self.dist_metric != "euclidean":
                raise ValueError((
                    "A coordinate tree can only be constructed "
                    "for an euclidean space"
                ))

            if self._ctree is None:
                self._ctree = [cKDTree(self.coords[self.cidx[i], :]) for i in range(len(self.cidx))]
            return self._ctree

        @property
        def eqidx(self):
            """The sampled indices into `self.coords` for the equidistant sample."""

            if self._eqidx is None:

                # Hardcode exponential bins for now, see about providing more options later
                list_in_radius = [0.]
                rad = self._center_radius
                list_out_radius = [rad]

                increasing_rad = rad
                while increasing_rad < self.max_dist:
                    list_in_radius.append(increasing_rad)
                    increasing_rad *= 1.5
                    list_out_radius.append(increasing_rad)

                # Get distances from all centers
                dist_center = np.sqrt((self.coords[None, :, 0] - self._centers[:, 0, None]) ** 2 + (
                        self.coords[None, :, 1] - self._centers[:, 1, None]) ** 2)

                # Select samples in a ring of inside radius and outside radius for all runs
                idx = np.logical_and(dist_center[None, :] >= np.array(list_in_radius)[:, None, None],
                                     dist_center[None, :] < np.array(list_out_radius)[:, None, None])

                # Cannot vectorize with the random subsampling because the arrays are different sizes, looping
                eqidxs = []
                for i in range(self.runs):
                    indices1 = np.concatenate([np.argwhere(idx[j, i, :]).squeeze() for j in range(len(list_in_radius))])
                    counts = np.count_nonzero(idx[:, i, :], axis=1)

                    # Second index: randomly select half of the valid pixels, so that the other half can be used by the equidist
                    # sample for low distances
                    list_idx = []
                    for j in range(len(list_in_radius)):
                        indices2 = self.rnd.choice(counts[j], size=min(counts[j], self.sample_count), replace=False)
                        list_idx.append(indices1[indices2].squeeze())
                    eqidxs.append(np.concatenate(list_idx))

                self._eqidx = eqidxs

            return self._eqidx

        @property
        def eqtree(self):
            """If `self.dist_metric` is `euclidean`, a `scipy.spatial.cKDTree`
            instance of the equidistant sample of `self.coords`. Undefined otherwise."""

            # only Euclidean supported
            if self.dist_metric != "euclidean":
                raise ValueError((
                    "A coordinate tree can only be constructed "
                    "for an euclidean space"
                ))

            if self._eqtree is None:
                self._eqtree = [cKDTree(self.coords[self.eqidx[i], :]) for i in range(len(self.eqidx))]
            return self._eqtree

        @property
        def dists(self):
            """A distance matrix of the sampled point pairs as a
            `scipy.sparse.csr_matrix` sparse matrix. """

            if self._dists is None:

                list_dists, list_cidx, list_eqidx = ([] for i in range(3))

                # Each run has a different center
                idx_center = self.rnd.choice(len(self.coords), size=(self.runs), replace=False)
                self._centers = self.coords[idx_center]

                # Radius of center based on sample count
                self._center_radius = np.sqrt(self.sample_count / np.pi) * self.res

                t00 = time.time()

                # Derive explicitly the indexes
                tmp = self.cidx
                tmp2 = self.eqidx

                t01 = time.time()
                print('Outside loop (indexes): ' + str(t01 - t00))

                for i in range(self.runs):
                    t0 = time.time()

                    dists = self.ctree[i].sparse_distance_matrix(
                        self.eqtree[i],
                        self.max_dist,
                        output_type="coo_matrix"
                    )

                    t1 = time.time()
                    print('Inside loop (pairwise): ' + str(t1 - t0))

                    list_dists.append(dists.data)
                    list_cidx.append(self.cidx[i][dists.row])
                    list_eqidx.append(self.eqidx[i][dists.col])

                t11 = time.time()
                print('Loop full duration: ' + str(t11 - t00))

                # concatenate the coo matrixes
                d = np.concatenate(list_dists)
                c = np.concatenate(list_cidx)
                eq = np.concatenate(list_eqidx)

                t2 = time.time()

                # remove possible duplicates (that would be summed by default)
                # from https://stackoverflow.com/questions/28677162/ignoring-duplicate-entries-in-sparse-matrix

                # Stable solution
                # c, eq, d = zip(*set(zip(c, eq, d)))
                # dists = sparse.csr_matrix((d, (c, eq)), shape=(len(self.coords), len(self.coords)))

                # Solution 5+ times faster than the preceding, but relies on _update() which might change in scipy (which
                # only has an implemented method for summing duplicates, and not ignoring them yet)
                dok = sparse.dok_matrix((len(self.coords), len(self.coords)))
                dok._update(zip(zip(c, eq), d))
                dists = dok.tocsr()

                t3 = time.time()

                print('Zipping: ' + str(t3 - t2))

                self._dists = dists

            return self._dists