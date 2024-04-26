# Suppress warnings generated from doing the ffts for the spatial
# autocorrelogram
# see autoCorr2D and crossCorr2D
import warnings
from enum import Enum
from functools import cache
from collections import namedtuple

import numpy as np
import boost_histogram as bh
from astropy import convolution  # deals with nans unlike other convs
from scipy import signal
from scipy.spatial import distance
from shapely import MultiLineString, prepare
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.common.utils import blur_image, flatten_list

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in greater")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
np.seterr(divide="ignore", invalid="ignore")


class VariableToBin(Enum):
    XY = 1
    DIR = 2
    SPEED = 3
    XY_TIME = 4
    SPEED_DIR = 5
    EGO_BOUNDARY = 6


class MapType(Enum):
    RATE = 1
    POS = 2
    SPK = 3
    ADAPTIVE = 4


class RateMap(object):
    """
    Bins up positional data (xy, head direction etc) and produces rate maps
    of the relevant kind. This is a generic class meant to be independent of
    any particular recording format.

    Args:
        xy (ndarray): The xy data, usually given as a 2 x n sample numpy array.
        hdir (ndarray): The head direction data, usually a 1 x n sample numpy array.
        speed (ndarray): Similar to hdir.
        pos_weights (ndarray): A 1D numpy array n samples long which is used to weight a particular
            position sample when binning data. For example, if there were 5
            positions recorded and a cell spiked once in position 2 and 5 times
            in position 3 and nothing anywhere else then pos_weights looks like:
            [0 0 1 5 0]
            In the case of binning up position this will be an array of mostly 1's
            unless there are some positions you want excluded for some reason.
        ppm (int, optional): Pixels per metre. Specifies how many camera pixels per metre so this,
            in combination with cmsPerBin, will determine how many bins there are
            in the rate map. Defaults to None.
        xyInCms (bool, optional): Whether the positional data is in cms. Defaults to False.
        cmsPerBin (int, optional): How many cms on a side each bin is in a rate map OR the number of
            degrees per bin in the case of directional binning. Defaults to 3.
        smooth_sz (int, optional): The width of the smoothing kernel for smoothing rate maps. Defaults to 5.

    Notes:
        There are several instance variables you can set, see below.
    """

    def __init__(
        self,
        PosCalcs: PosCalcsGeneric,
        pos_weights: np.array = None,
        xyInCms: bool = False,
        binsize: int = 3,
        smooth_sz: int = 5,
    ):
        self.PosCalcs = PosCalcs
        self._pos_weights = pos_weights
        self._pos_time_splits = None
        self._spike_weights = None
        self._binsize = binsize
        self._binsize2d = None
        self._inCms = xyInCms
        self._binedges = None  # has setter and getter - see below
        self._x_lims = None
        self._y_lims = None
        self._smooth_sz = smooth_sz
        self._smoothingType = "gaussian"  # 'boxcar' or 'gaussian'
        self.whenToSmooth = "before"  # or 'after'
        self._var2Bin = VariableToBin.XY
        self._mapType = MapType.RATE
        self._calc_bin_edges()

    @property
    def xy(self):
        return self.PosCalcs.xy

    @property
    def dir(self):
        return self.PosCalcs.dir

    @property
    def speed(self):
        return self.PosCalcs.speed

    @property
    def pos_times(self):
        return self.PosCalcs.xyTS

    @property
    def inCms(self):
        # Whether the units are in cms or not
        return self._inCms

    @inCms.setter
    def inCms(self, value):
        self._inCms = value
        # will trigger a recalculation of position vars
        self.PosCalcs.convert2cm = value

    @property
    def ppm(self):
        # Get the current pixels per metre (ppm)
        return self.PosCalcs.ppm

    @ppm.setter
    def ppm(self, value):
        # will trigger a recalculation of position vars
        self._ppm = self.PosCalcs.ppm = value

    @property
    def var2Bin(self):
        return self._var2Bin

    @var2Bin.setter
    def var2Bin(self, value):
        self._var2Bin = value

    @property
    def mapType(self):
        return self._mapType

    @mapType.setter
    def mapType(self, value):
        self._mapType = value

    @property
    def binedges(self):
        return self._binedges

    @binedges.setter
    def binedges(self, value):
        self._binedges = value

    @property
    def x_lims(self):
        return self._x_lims

    @x_lims.setter
    def x_lims(self, value):
        self._x_lims = value

    @property
    def y_lims(self):
        return self._y_lims

    @y_lims.setter
    def y_lims(self, value):
        self._y_lims = value

    @property
    def pos_weights(self):
        """
        The 'weights' used as an argument to np.histogram* for binning up
        position
        Mostly this is just an array of 1's equal to the length of the pos
        data, but usefully can be adjusted when masking data in the trial
        by
        """
        if self._pos_weights is None:
            self._pos_weights = np.ones(self.xy.shape[1])
        return self._pos_weights

    @pos_weights.setter
    def pos_weights(self, value):
        self._pos_weights = value

    @property
    def spike_weights(self):
        return self._spike_weights

    @spike_weights.setter
    def spike_weights(self, value):
        self._spike_weights = value

    @property
    def binsize(self):
        # The number of cms per bin of the binned up map
        return self._binsize

    @binsize.setter
    def binsize(self, value):
        self._binsize = value
        self._binedges = self._calc_bin_edges(value)

    @property
    def smooth_sz(self):
        # The size of the smoothing window applied to the binned data
        return self._smooth_sz

    @smooth_sz.setter
    def smooth_sz(self, value):
        self._smooth_sz = value

    @property
    def smoothingType(self):
        # The type of smoothing to do - legal values are 'boxcar' or 'gaussian'
        return self._smoothingType

    @smoothingType.setter
    def smoothingType(self, value):
        self._smoothingType = value

    def apply_mask(self, mask):
        self.PosCalcs.apply_mask(mask)
        self.pos_weights.mask = mask

    def _getXYLimits(self):
        """
        Gets the min/max of the x/y data
        """
        x_lims = getattr(self, "x_lims", None)
        y_lims = getattr(self, "y_lims", None)
        if x_lims is None:
            x_lims = (np.nanmin(self.xy[0]), np.nanmax(self.xy[0]))
        if y_lims is None:
            y_lims = (np.nanmin(self.xy[1]), np.nanmax(self.xy[1]))
        self.x_lims = x_lims
        self.y_lims = y_lims
        return x_lims, y_lims

    def _calc_bin_dims(self):
        try:
            self._binDims = [len(b) for b in self._binedges]
        except TypeError:
            self._binDims = len(self._binedges)

    def _calc_bin_edges(self, binsize: int | tuple = 3) -> tuple:
        """
        Aims to get the right number of bins for the variable to be binned

        Args:
            binsize (int, optional): The number of cms per bin for XY OR degrees for DIR OR cm/s for SPEED. Defaults to 3.

        Returns:
            tuple: each member an array of bin edges
        """
        if self.var2Bin.value == VariableToBin.DIR.value:
            self.binedges = np.arange(0, 360 + binsize, binsize)
        elif self.var2Bin.value == VariableToBin.SPEED.value:
            maxspeed = np.nanmax(self.speed)
            # assume min speed = 0
            self.binedges = np.arange(0, maxspeed, binsize)
        elif self.var2Bin.value == VariableToBin.XY.value:
            x_lims, y_lims = self._getXYLimits()
            nxbins = int(np.ceil((x_lims[1] - x_lims[0]) / binsize))
            nybins = int(np.ceil((y_lims[1] - y_lims[0]) / binsize))
            _x = np.linspace(x_lims[0], x_lims[1], nxbins)
            _y = np.linspace(y_lims[0], y_lims[1], nybins)
            self.binedges = _y, _x
        elif self.var2Bin.value == VariableToBin.XY_TIME.value:
            if self._pos_time_splits is None:
                raise ValueError("Need pos times to bin up XY_TIME")
            x_lims, y_lims = self._getXYLimits()
            nxbins = int(np.ceil((x_lims[1] - x_lims[0]) / binsize))
            nybins = int(np.ceil((y_lims[1] - y_lims[0]) / binsize))
            _x = np.linspace(x_lims[0], x_lims[1], nxbins)
            _y = np.linspace(y_lims[0], y_lims[1], nybins)
            be = [_y, _x]
            be.append(self.pos_time_splits)
            self.binedges = be
        elif self.var2Bin.value == VariableToBin.SPEED_DIR.value:
            maxspeed = np.nanmax(self.speed)
            if isinstance(binsize, int):
                self.binedges = (
                    np.arange(0, maxspeed, binsize),
                    np.arange(0, 360 + binsize, binsize),
                )
            elif isinstance(binsize, tuple):
                self.binedges = (
                    np.arange(0, maxspeed, binsize[0]),
                    np.arange(0, 360 + binsize[1], binsize[1]),
                )
        elif self.var2Bin.value == VariableToBin.EGO_BOUNDARY.value:
            if isinstance(binsize, (float, int)):
                self.binedges = (
                    np.arange(0, 2 * np.pi, 2 * np.pi / 120),
                    np.arange(0, 50, 2.5),
                )
            elif isinstance(binsize, tuple):
                self.binedges = (
                    np.arange(0, 2 * np.pi, binsize[1]),
                    np.arange(0, 50, binsize[0]),
                )
        self._calc_bin_dims()
        return self.binedges

    def get_map(
        self,
        spk_weights,
        var_type=VariableToBin.XY,
        map_type=MapType.RATE,
        smoothing=True,
        **kwargs
    ):
        """
        Bins up the variable type var_type and returns a tuple of
        (rmap, binnedPositionDir) or
        (rmap, binnedPostionX, binnedPositionY)

        Args:
            spkWeights (array_like): Shape equal to number of positions samples captured and consists of
                position weights. For example, if there were 5 positions
                recorded and a cell spiked once in position 2 and 5 times in
                position 3 and nothing anywhere else then pos_weights looks
                like: [0 0 1 5 0]
                spkWeights can also be list-like where each entry in the list is a different set of
                weights - these are enumerated through in a list comp in the ._bin_data function. In
                this case the returned tuple will consist of a 2-tuple where the first entry is an
                array of the ratemaps (binned_spk / binned_pos) and the second part is the binned pos data (as it's common to all
                the spike weights)
            var_type (Enum value - see Variable2Bin defined at top of this file): The variable to bin up. Legal values are: XY, DIR and SPEED
            mapType (enum value - see MapType defined at top of this file): If RATE then the binned up spikes are divided by var_type.
                Otherwise return binned up position. Options are RATE or POS
            smoothing (bool, optional): Whether to smooth the data or not. Defaults to True.

        Returns:
            binned_data, binned_pos (tuple): This is either a 2-tuple or a 3-tuple depening on whether binned
                pos (mapType 'pos') or binned spikes (mapType 'rate') is asked
                for respectively
        """
        pos_weights = self.pos_weights
        if var_type.value == VariableToBin.DIR.value:
            sample = self.dir
        elif var_type.value == VariableToBin.SPEED.value:
            sample = self.speed
        elif var_type.value == VariableToBin.XY.value:
            sample = self.xy
        elif var_type.value == VariableToBin.XY_TIME.value:
            sample = np.concatenate(
                (np.atleast_2d(self.xy), np.atleast_2d(self.pos_times))
            )
        elif var_type.value == VariableToBin.SPEED_DIR.value:
            sample = np.concatenate(
                (np.atleast_2d(self.dir), np.atleast_2d(self.speed))
            )
        elif var_type.value == VariableToBin.EGO_BOUNDARY.value:
            arena_shape = kwargs.get("arena_shape", "circle")
            binsize = kwargs.get("binsize", 5)
            if isinstance(binsize, tuple):
                binsize = binsize[0]
            ego_angles, arena_xy = self._calc_ego_angles(arena_shape, binsize)
            ego_dists = distance.cdist(arena_xy, self.xy.T, "euclidean")
            sample = np.stack((np.ravel(ego_dists.T), np.ravel(ego_angles.T)))
            spk_weights = np.atleast_2d(spk_weights)
            spk_weights = np.repeat(spk_weights, arena_xy.shape[0], -1)
            pos_weights = np.repeat(self.pos_weights, arena_xy.shape[0])
            hist_range = ((0, 50), (0, 2 * np.pi))
            if "range" not in kwargs.keys():
                kwargs["range"] = hist_range
        else:
            raise ValueError("Unrecognized variable to bin.")

        assert sample is not None

        self.var2Bin = var_type
        binsize = kwargs.pop("binsize", self.binsize)
        hist_range = kwargs.pop("range", None)

        if hist_range is None:
            bin_edges = self._calc_bin_edges(binsize)
        else:
            bin_edges = None

        binned_pos, binned_pos_edges = self._bin_data(
            sample, bin_edges, pos_weights, hist_range
        )
        binned_pos = binned_pos / self.PosCalcs.sample_rate
        nanIdx = binned_pos == 0

        if (
            var_type.value == VariableToBin.DIR.value
            or var_type.value == VariableToBin.EGO_BOUNDARY
        ):
            boundary = "wrap"
        else:
            boundary = "extend"

        if map_type.value == MapType.POS.value:  # return binned up position
            if smoothing:
                binned_pos = blur_image(
                    binned_pos,
                    self.smooth_sz,
                    ftype=self.smoothingType,
                    boundary=boundary,
                    **kwargs
                )
            return binned_pos, binned_pos_edges

        binned_spk, _ = self._bin_data(sample, bin_edges, spk_weights, hist_range)

        if map_type.value == MapType.SPK.value:
            return binned_spk
        if map_type.value == MapType.ADAPTIVE.value:
            alpha = kwargs.pop("alpha", 4)
            if binned_spk.ndim == 3:
                smthd_rate = []
                for i in range(binned_spk.shape[0]):
                    smthd_rate.append(
                        self.getAdaptiveMap(binned_pos, binned_spk[i, ...], alpha)[0]
                    )
                return np.array(smthd_rate)
            smthd_rate, smthd_spk, smthd_pos = self.getAdaptiveMap(
                binned_pos, binned_spk, alpha
            )
            return smthd_rate

        # binned_spk is returned as a tuple of the binned data and the bin
        # edges
        if "after" in self.whenToSmooth:
            rmap = binned_spk / binned_pos
            rmap = blur_image(
                rmap,
                self.smooth_sz,
                ftype=self.smoothingType,
                boundary=boundary,
                **kwargs
            )
        else:  # default case
            if not smoothing:
                return binned_spk / binned_pos, binned_pos_edges
            binned_pos = blur_image(
                binned_pos,
                self.smooth_sz,
                ftype=self.smoothingType,
                boundary=boundary,
                **kwargs
            )
            binned_spk = blur_image(
                binned_spk,
                self.smooth_sz,
                ftype=self.smoothingType,
                boundary=boundary,
                **kwargs
            )
            rmap = binned_spk / binned_pos
            rmap[..., nanIdx] = np.nan

        return rmap, binned_pos_edges

    def _bin_data(self, var, bin_edges, weights, range=None):
        """
        Bins data taking account of possible multi-dimensionality

        Args:
            var (array_like): The variable to bin
            bin_edges (array_like): The edges of the data - see numpys histogramdd for more
            weights (array_like): The weights attributed to the samples in var
            good_indices (array_like): Valid indices (i.e. not nan and not infinite)

        Returns:
            ndhist (2-tuple): Think this always returns a two-tuple of the binned variable and
                the bin edges - need to check to be sure...

        Notes:
            This breaks compatability with numpys histogramdd
            In the 2d histogram case below I swap the axes around so that x and y
            are binned in the 'normal' format i.e. so x appears horizontally and y
            vertically.
            Multi-binning issue is dealt with awkwardly through checking
            the dimensionality of the weights array.
            'normally' this would be 1 dim but when multiple clusters are being
            binned it will be 2 dim.
            In that case np.apply_along_axis functionality is applied.
            The spike weights in that case might be created like so:

            >>> spk_W = np.zeros(shape=[len(trial.nClusters), trial.npos])
            >>> for i, cluster in enumerate(trial.clusters):
            >>>		x1 = trial.getClusterIdx(cluster)
            >>>		spk_W[i, :] = np.bincount(x1, minlength=trial.npos)

            This can then be fed into this fcn something like so:

            >>> rng = np.array((np.ma.min(
                trial.POS.xy, 1).data, np.ma.max(rial.POS.xy, 1).data))
            >>> h = _bin_data(
                var=trial.POS.xy, bin_edges=np.array([64, 64]),
                weights=spk_W, rng=rng)

            Returned will be a tuple containing the binned up data and
            the bin edges for x and y (obv this will be the same for all
            entries of h)
        """
        if weights is None:
            weights = np.ones_like(var)
        dims = weights.ndim
        if dims == 1 and var.ndim == 1:
            var = var[np.newaxis, :]
            if bin_edges is not None:
                bin_edges = bin_edges[np.newaxis, :]
        elif dims > 1 and var.ndim == 1:
            var = var[np.newaxis, :]
            if bin_edges is not None:
                bin_edges = bin_edges[np.newaxis, :]
        else:
            var = np.flipud(var)
        weights = np.atleast_2d(weights)  # needed for list comp below
        var = np.array(var.data.T)
        if bin_edges is None:
            ndhist = [
                bh.numpy.histogramdd(var, range=[range], weights=np.ravel(w))
                for w in weights
            ]
        else:
            ndhist = [
                bh.numpy.histogramdd(var, bins=bin_edges, weights=np.ravel(w))
                for w in weights
            ]
        if np.shape(weights)[0] == 1:
            return ndhist[0][0], ndhist[0][1]
        else:
            tmp = [d[0] for d in ndhist]
            tmp = np.array(tmp)
            return tmp, ndhist[1]

    def _circ_pad_smooth(self, var, n=3, ny=None):
        """
        Smooths a vector by convolving with a gaussian
        Mirror reflects the start and end of the vector to
        deal with edge effects

        Args:
            var (array_like): The vector to smooth
            n, ny (int): Size of the smoothing (sigma in gaussian)

        Returns:
            array_like: The smoothed vector with shape the same as var
        """

        tn = len(var)
        t2 = int(np.floor(tn / 2))
        var = np.concatenate((var[t2:tn], var, var[0:t2]))
        if ny is None:
            ny = n
        x, y = np.mgrid[-n : n + 1, 0 - ny : ny + 1]
        g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
        if np.ndim(var) == 1:
            g = g[n, :]
        g = g / g.sum()
        improc = convolution.convolve(var, g, normalize_kernel=False, boundary="wrap")
        improc = improc[tn - t2 : tn - t2 + tn]
        return improc

    def _circularStructure(self, radius):
        """
        Generates a circular binary structure for use with morphological
        operations such as ndimage.binary_dilation etc

        This is only used in this implementation for adaptively binning
        ratemaps for use with information theoretic measures (Skaggs etc)

        Args:
            radius (int): the size of the circular structure

        Returns:
            res (array_like): Binary structure with shape [(radius*2) + 1,(radius*2) + 1]

        See Also:
            RateMap.__adpativeMap
        """
        from skimage.morphology import disk

        return disk(radius)

    def getAdaptiveMap(self, pos_binned, spk_binned, alpha=4):
        """
        Produces a ratemap that has been adaptively binned according to the
        algorithm described in Skaggs et al., 1996) [1]_.

        Args:
            pos_binned (array_like): The binned positional data. For example that returned from get_map
                above with mapType as 'pos'
            spk_binned (array_like): The binned spikes
            alpha (int, optional): A scaling parameter determing the amount of occupancy to aim at
                in each bin. Defaults to 4. In the original paper this was set to 200.
                This is 4 here as the pos data is binned in seconds (the original data was in pos
                samples so this is a factor of 50 smaller than the original paper's value, given 50Hz sample rate)

        Returns:
            Returns adaptively binned spike and pos maps. Use to generate Skaggs
            information measure

        Notes:
            Positions with high rates mean proportionately less error than those
            with low rates, so this tries to even the playing field. This type
            of binning should be used for calculations of spatial info
            as with the skaggs_info method in the fieldcalcs class (see below)
            alpha is a scaling parameter that might need tweaking for different
            data sets.
            From the paper:
                The data [are] first binned
                into a 64 X 64 grid of spatial locations, and then the firing rate
                at each point in this grid was calculated by expanding a circle
                around the point until the following criterion was met:
                    Nspks > alpha / (Nocc^2 * r^2)
                where Nspks is the number of spikes emitted in a circle of radius
                r (in bins), Nocc is the number of occupancy samples, alpha is the
                scaling parameter
                The firing rate in the given bin is then calculated as:
                    sample_rate * (Nspks / Nocc)

        References:
            .. [1] W. E. Skaggs, B. L. McNaughton, K. M. Gothard & E. J. Markus
                "An Information-Theoretic Approach to Deciphering the Hippocampal
                Code"
                Neural Information Processing Systems, 1993.
        """
        #  assign output arrays
        smthdPos = np.zeros_like(pos_binned)
        smthdSpk = np.zeros_like(spk_binned)
        smthdRate = np.zeros_like(pos_binned)
        idx = pos_binned == 0
        pos_binned[idx] = np.nan
        spk_binned[idx] = np.nan
        visited = np.zeros_like(pos_binned)
        visited[pos_binned > 0] = 1
        # array to check which bins have made it
        binCheck = np.isnan(pos_binned)
        r = 1
        while np.any(~binCheck):
            # create the filter kernel
            h = self._circularStructure(r)
            h[h >= np.max(h) / 3.0] = 1
            h[h != 1] = 0
            if h.shape >= pos_binned.shape:
                break
            # filter the arrays using astropys convolution
            filtPos = convolution.convolve(pos_binned, h, boundary=None)
            filtSpk = convolution.convolve(spk_binned, h, boundary=None)
            filtVisited = convolution.convolve(visited, h, boundary=None)
            # get the bins which made it through this iteration
            trueBins = alpha / (np.sqrt(filtSpk) * filtPos) <= r
            trueBins = np.logical_and(trueBins, ~binCheck)
            # insert values where true
            smthdPos[trueBins] = filtPos[trueBins] / filtVisited[trueBins]
            smthdSpk[trueBins] = filtSpk[trueBins] / filtVisited[trueBins]
            binCheck[trueBins] = True
            r += 1
        smthdRate = smthdSpk / smthdPos
        smthdRate[idx] = np.nan
        smthdSpk[idx] = np.nan
        smthdPos[idx] = np.nan
        return smthdRate, smthdSpk, smthdPos

    def autoCorr2D(self, A, nodwell=None, tol=1e-10):
        """
        Performs a spatial autocorrelation on the array A

        Args:
            A (array_like): Either 2 or 3D. In the former it is simply the binned up ratemap
                where the two dimensions correspond to x and y.
                If 3D then the first two dimensions are x
                and y and the third (last dimension) is 'stack' of ratemaps
            nodwell (array_like): A boolean array corresponding the bins in the ratemap that
                weren't visited. See Notes below.
            tol (float, optional): Values below this are set to zero to deal with v small values
                thrown up by the fft. Default 1e-10

        Returns:
            sac (array_like): The spatial autocorrelation in the relevant dimensionality

        Notes:
            The nodwell input can usually be generated by:

            >>> nodwell = ~np.isfinite(A)
        """

        assert np.ndim(A) == 2
        m, n = np.shape(A)
        o = 1
        x = np.reshape(A, (m, n, o))
        if nodwell is None:
            nodwell = ~np.isfinite(A)
        nodwell = np.reshape(nodwell, (m, n, o))
        x[nodwell] = 0
        # [Step 1] Obtain FFTs of x, the sum of squares and bins visited
        Fx = np.fft.fft(np.fft.fft(x, 2 * m - 1, axis=0), 2 * n - 1, axis=1)
        FsumOfSquares_x = np.fft.fft(
            np.fft.fft(np.power(x, 2), 2 * m - 1, axis=0), 2 * n - 1, axis=1
        )
        Fn = np.fft.fft(
            np.fft.fft(np.invert(nodwell).astype(int), 2 * m - 1, axis=0),
            2 * n - 1,
            axis=1,
        )
        # [Step 2] Multiply the relevant transforms and invert to obtain the
        # equivalent convolutions
        rawCorr = np.fft.fftshift(
            np.real(np.fft.ifft(np.fft.ifft(Fx * np.conj(Fx), axis=1), axis=0)),
            axes=(0, 1),
        )
        sums_x = np.fft.fftshift(
            np.real(np.fft.ifft(np.fft.ifft(np.conj(Fx) * Fn, axis=1), axis=0)),
            axes=(0, 1),
        )
        sumOfSquares_x = np.fft.fftshift(
            np.real(
                np.fft.ifft(np.fft.ifft(Fn * np.conj(FsumOfSquares_x), axis=1), axis=0)
            ),
            axes=(0, 1),
        )
        N = np.fft.fftshift(
            np.real(np.fft.ifft(np.fft.ifft(Fn * np.conj(Fn), axis=1), axis=0)),
            axes=(0, 1),
        )
        # [Step 3] Account for rounding errors.
        rawCorr[np.abs(rawCorr) < tol] = 0
        sums_x[np.abs(sums_x) < tol] = 0
        sumOfSquares_x[np.abs(sumOfSquares_x) < tol] = 0
        N = np.round(N)
        N[N <= 1] = np.nan
        # [Step 4] Compute correlation matrix
        mapStd = np.sqrt((sumOfSquares_x * N) - sums_x**2)
        mapCovar = (rawCorr * N) - sums_x * sums_x[::-1, :, :][:, ::-1, :][:, :, :]

        return np.squeeze(mapCovar / mapStd / mapStd[::-1, :, :][:, ::-1, :][:, :, :])

    def crossCorr2D(self, A, B, A_nodwell, B_nodwell, tol=1e-10):
        """
        Performs a spatial crosscorrelation between the arrays A and B

        Args:
            A, B (array_like): Either 2 or 3D. In the former it is simply the binned up ratemap
                where the two dimensions correspond to x and y.
                If 3D then the first two dimensions are x
                and y and the third (last dimension) is 'stack' of ratemaps
            nodwell_A, nodwell_B (array_like): A boolean array corresponding the bins in the ratemap that
                weren't visited. See Notes below.
            tol (float, optional): Values below this are set to zero to deal with v small values
                thrown up by the fft. Default 1e-10

        Returns:
            sac (array_like): The spatial crosscorrelation in the relevant dimensionality

        Notes:
            The nodwell input can usually be generated by:

            >>> nodwell = ~np.isfinite(A)
        """
        if np.ndim(A) != np.ndim(B):
            raise ValueError("Both arrays must have the same dimensionality")
        assert np.ndim(A) == 2
        ma, na = np.shape(A)
        mb, nb = np.shape(B)
        oa = ob = 1
        A = np.reshape(A, (ma, na, oa))
        B = np.reshape(B, (mb, nb, ob))
        A_nodwell = np.reshape(A_nodwell, (ma, na, oa))
        B_nodwell = np.reshape(B_nodwell, (mb, nb, ob))
        A[A_nodwell] = 0
        B[B_nodwell] = 0
        # [Step 1] Obtain FFTs of x, the sum of squares and bins visited
        Fa = np.fft.fft(np.fft.fft(A, 2 * mb - 1, axis=0), 2 * nb - 1, axis=1)
        FsumOfSquares_a = np.fft.fft(
            np.fft.fft(np.power(A, 2), 2 * mb - 1, axis=0), 2 * nb - 1, axis=1
        )
        Fn_a = np.fft.fft(
            np.fft.fft(np.invert(A_nodwell).astype(int), 2 * mb - 1, axis=0),
            2 * nb - 1,
            axis=1,
        )
        Fb = np.fft.fft(np.fft.fft(B, 2 * ma - 1, axis=0), 2 * na - 1, axis=1)
        FsumOfSquares_b = np.fft.fft(
            np.fft.fft(np.power(B, 2), 2 * ma - 1, axis=0), 2 * na - 1, axis=1
        )
        Fn_b = np.fft.fft(
            np.fft.fft(np.invert(B_nodwell).astype(int), 2 * ma - 1, axis=0),
            2 * na - 1,
            axis=1,
        )
        # [Step 2] Multiply the relevant transforms and invert to obtain the
        # equivalent convolutions
        rawCorr = np.fft.fftshift(
            np.real(np.fft.ifft(np.fft.ifft(Fa * np.conj(Fb), axis=1), axis=0))
        )
        sums_a = np.fft.fftshift(
            np.real(np.fft.ifft(np.fft.ifft(Fa * np.conj(Fn_b), axis=1), axis=0))
        )
        sums_b = np.fft.fftshift(
            np.real(np.fft.ifft(np.fft.ifft(Fn_a * np.conj(Fb), axis=1), axis=0))
        )
        sumOfSquares_a = np.fft.fftshift(
            np.real(
                np.fft.ifft(
                    np.fft.ifft(FsumOfSquares_a * np.conj(Fn_b), axis=1), axis=0
                )
            )
        )
        sumOfSquares_b = np.fft.fftshift(
            np.real(
                np.fft.ifft(
                    np.fft.ifft(Fn_a * np.conj(FsumOfSquares_b), axis=1), axis=0
                )
            )
        )
        N = np.fft.fftshift(
            np.real(np.fft.ifft(np.fft.ifft(Fn_a * np.conj(Fn_b), axis=1), axis=0))
        )
        # [Step 3] Account for rounding errors.
        rawCorr[np.abs(rawCorr) < tol] = 0
        sums_a[np.abs(sums_a) < tol] = 0
        sums_b[np.abs(sums_b) < tol] = 0
        sumOfSquares_a[np.abs(sumOfSquares_a) < tol] = 0
        sumOfSquares_b[np.abs(sumOfSquares_b) < tol] = 0
        N = np.round(N)
        N[N <= 1] = np.nan
        # [Step 4] Compute correlation matrix
        mapStd_a = np.sqrt((sumOfSquares_a * N) - sums_a**2)
        mapStd_b = np.sqrt((sumOfSquares_b * N) - sums_b**2)
        mapCovar = (rawCorr * N) - sums_a * sums_b

        return np.squeeze(mapCovar / (mapStd_a * mapStd_b))

    def tWinSAC(
        self,
        xy,
        spkIdx,
        ppm=365,
        winSize=10,
        pos_sample_rate=50,
        nbins=71,
        boxcar=5,
        Pthresh=100,
        downsampfreq=50,
        plot=False,
    ):
        """
        Temporal windowed spatial autocorrelation.

        Args:
            xy (array_like): The position data
            spkIdx (array_like): The indices in xy where the cell fired
            ppm (int, optional): The camera pixels per metre. Default 365
            winSize (int, optional): The window size for the temporal search
            pos_sample_rate (int, optional): The rate at which position was sampled. Default 50
            nbins (int, optional): The number of bins for creating the resulting ratemap. Default 71
            boxcar (int, optional): The size of the smoothing kernel to smooth ratemaps. Default 5
            Pthresh (int, optional): The cut-off for values in the ratemap; values < Pthresh become nans. Default 100
            downsampfreq (int, optional): How much to downsample. Default 50
            plot (bool, optional): Whether to show a plot of the result. Default False

        Returns:
            H (array_like): The temporal windowed SAC
        """
        # [Stage 0] Get some numbers
        xy = xy / ppm * 100
        n_samps = xy.shape[1]
        n_spks = len(spkIdx)
        winSizeBins = np.min([winSize * pos_sample_rate, n_samps])
        # factor by which positions are downsampled
        downsample = np.ceil(pos_sample_rate / downsampfreq)
        Pthresh = Pthresh / downsample  # take account of downsampling

        # [Stage 1] Calculate number of spikes in the window for each spikeInd
        # (ignoring spike itself)
        # 1a. Loop preparation
        nSpikesInWin = np.zeros(n_spks, dtype=int)

        # 1b. Keep looping until we have dealt with all spikes
        for i, s in enumerate(spkIdx):
            t = np.searchsorted(spkIdx, (s, s + winSizeBins))
            nSpikesInWin[i] = len(spkIdx[t[0] : t[1]]) - 1  # ignore ith spike

        # [Stage 2] Prepare for main loop
        # 2a. Work out offset inidices to be used when storing spike data
        off_spike = np.cumsum([nSpikesInWin])
        off_spike = np.pad(off_spike, (1, 0), "constant", constant_values=(0))

        # 2b. Work out number of downsampled pos bins in window and
        # offset indices for storing data
        nPosInWindow = np.minimum(winSizeBins, n_samps - spkIdx)
        nDownsampInWin = np.floor((nPosInWindow - 1) / downsample) + 1

        off_dwell = np.cumsum(nDownsampInWin.astype(int))
        off_dwell = np.pad(off_dwell, (1, 0), "constant", constant_values=(0))

        # 2c. Pre-allocate dwell and spike arrays, singles for speed
        dwell = np.zeros((2, off_dwell[-1]), dtype=np.single) * np.nan
        spike = np.zeros((2, off_spike[-1]), dtype=np.single) * np.nan

        filled_pvals = 0
        filled_svals = 0

        for i in range(n_spks):
            # calculate dwell displacements
            winInd_dwell = np.arange(
                spkIdx[i] + 1,
                np.minimum(spkIdx[i] + winSizeBins, n_samps),
                downsample,
                dtype=int,
            )
            WL = len(winInd_dwell)
            dwell[:, filled_pvals : filled_pvals + WL] = np.rot90(
                np.array(np.rot90(xy[:, winInd_dwell]) - xy[:, spkIdx[i]])
            )
            filled_pvals = filled_pvals + WL
            # calculate spike displacements
            winInd_spks = (
                i + np.nonzero(spkIdx[i + 1 : n_spks] < spkIdx[i] + winSizeBins)[0]
            )
            WL = len(winInd_spks)
            spike[:, filled_svals : filled_svals + WL] = np.rot90(
                np.array(np.rot90(xy[:, spkIdx[winInd_spks]]) - xy[:, spkIdx[i]])
            )
            filled_svals = filled_svals + WL

        dwell = np.delete(dwell, np.isnan(dwell).nonzero()[1], axis=1)
        spike = np.delete(spike, np.isnan(spike).nonzero()[1], axis=1)

        dwell = np.hstack((dwell, -dwell))
        spike = np.hstack((spike, -spike))

        dwell_min = np.min(dwell, axis=1)
        dwell_max = np.max(dwell, axis=1)

        binsize = (dwell_max[1] - dwell_min[1]) / nbins

        dwell = np.round(
            (dwell - np.ones_like(dwell) * dwell_min[:, np.newaxis]) / binsize
        )
        spike = np.round(
            (spike - np.ones_like(spike) * dwell_min[:, np.newaxis]) / binsize
        )

        binsize = np.max(dwell, axis=1).astype(int)
        binedges = np.array(((-0.5, -0.5), binsize + 0.5)).T
        Hp = np.histogram2d(dwell[0, :], dwell[1, :], range=binedges, bins=binsize)[0]
        Hs = np.histogram2d(spike[0, :], spike[1, :], range=binedges, bins=binsize)[0]

        # reverse y,x order
        Hp = np.swapaxes(Hp, 1, 0)
        Hs = np.swapaxes(Hs, 1, 0)

        fHp = blur_image(Hp, boxcar)
        fHs = blur_image(Hs, boxcar)

        H = fHs / fHp
        H[Hp < Pthresh] = np.nan

        return H

    def _calc_ego_angles(self, arena_shape="circle", xy_binsize=2.5) -> None:
        """
        Calculate the angles between the segments of the arena wall
        and the positions of the animal throughout the trial.

        Returns the angles as well as the arena x-y coordinates.
        NB The angles are in radians
        """
        arena_width = np.ceil(np.mean(np.ptp(self.xy.data, 1) / 2))
        arena_width = arena_width.tolist()
        arena_centre = Point(np.nanmin(self.xy.data, 1) + arena_width)

        if "circle" in arena_shape:
            arena_boundary = arena_centre.buffer(arena_width).boundary
        elif "square" in arena_shape:
            arena_boundary = arena_centre.buffer(arena_width, cap_style=3).boundary
        arena_boundary = arena_boundary.segmentize(max_segment_length=xy_binsize)
        arena_xy = np.array(arena_boundary.xy).T
        animal_xy = self.xy
        dx = np.atleast_2d(animal_xy[0]) - np.atleast_2d(arena_xy[:, 0]).T
        dy = np.atleast_2d(animal_xy[1]) - np.atleast_2d(arena_xy[:, 1]).T
        # make sure angles are in range [0-2PI]
        angles = np.arctan2(dy, dx) + np.pi
        animal_hd = np.radians(self.dir)
        ego_angles = (angles - animal_hd) % (np.pi * 2)
        # ego_angles in range [0-2PI]
        # and with size arena_xy_ncoords x npos
        return ego_angles, arena_xy

    def get_egocentric_boundary_map(
        self,
        spk_weights,
        degs_per_bin: float = 3,
        xy_binsize: float = 2.5,
        arena_type: str = "circle",
        return_dists: bool = False,
        return_angles: bool = False,
        return_raw_spk: bool = False,
        return_raw_occ: bool = False,
        **kwargs
    ) -> namedtuple:
        assert self.dir is not None, "No direction data available"
        ego_angles, arena_xy = self._calc_ego_angles()
        ego_dists = distance.cdist(arena_xy, self.xy.T, "euclidean")
        occ = np.histogramdd(
            np.stack((np.ravel(ego_dists.T), np.ravel(ego_angles.T))).T,
            range=((0, 50), (0, 2 * np.pi)),
            bins=(int(50 / xy_binsize), int(360 / degs_per_bin)),
        )
        spk_weights = np.repeat(spk_weights, arena_xy.shape[0])
        spk = np.histogramdd(
            np.stack((np.ravel(ego_dists.T), np.ravel(ego_angles.T))).T,
            range=((0, 50), (0, 2 * np.pi)),
            bins=(int(50 / xy_binsize), int(360 / degs_per_bin)),
            weights=spk_weights,
        )
        rmap = spk[0] / occ[0]
        EgoMap = namedtuple(
            "EgoMap", ["rmap", "occ", "spk", "dists", "angles"], defaults=None
        )
        em = EgoMap(None, None, None, None, None)
        em = em._replace(rmap=rmap)
        if return_dists:
            em = em._replace(dists=ego_dists)
        if return_raw_occ:
            em = em._replace(occ=occ[0])
        if return_raw_spk:
            em = em._replace(spk=spk[0])
        if return_angles:
            em = em._replace(angles=ego_angles)
        return em

    def getAllSpikeWeights(
        self,
        spike_times: np.ndarray,
        spike_clusters: np.ndarray,
        pos_times: np.ndarray,
        **kwargs
    ):
        """
        Args:
            spike_times (np.ndarray): Spike times in seconds
            spike_clusters (np.ndarray): Cluster identity vector
            pos_times (np.ndarray): The times at which position was captured in seconds

        Returns:
            np.ndarray: The bincounts with respect to position for each cluster. Shape of returned array will be nClusters x npos
        """
        assert len(spike_clusters) == len(spike_times)
        clusters = np.unique(spike_clusters)
        npos = len(self.dir)
        idx = np.searchsorted(pos_times, spike_times) - 1
        weights = [
            np.bincount(idx[spike_clusters == c], minlength=npos) for c in clusters
        ]
        return np.array(weights)

    def _splitStackedCorrelations(self, binned_data: list) -> tuple:
        """
        Takes in the result of doStackedCorrelations() and splits into
        two arrays and returns these as a 2-tuple
        """
        result = [(s[0][:, :, 0], s[0][:, :, 1]) for s in binned_data]
        result = np.array(result)
        return np.squeeze(result[:, 0, :, :]), np.squeeze(result[:, 1, :, :])

    def doStackedCorrelations(
        self,
        spkW: np.ndarray,
        times: np.ndarray,
        splits: np.ndarray,
        var2bin: Enum = VariableToBin.XY,
        maptype: Enum = MapType.RATE,
        **kwargs
    ):
        """
        Returns a list of binned data where each item in the list
        is the result of running np.histogramdd on a spatial
        variable (xy, dir etc) and a temporal one at the same
        time. The idea is to split the spatial variable into two
        temporal halves based on the bin edges in 'splits' and
        then to run correlations between the two halves and
        furthermore to do this for all of the clusters that have
        spike weights in 'spkW'. 'spkW' should be the result of
        using getAllSpikeWeights().

        Args:
            spkW (np.ndarray): The result of calling getAllSpikeWeights()
            times (np.ndarray): Position times in seconds
            splits (np.ndarray): Where to split the data in seconds. Will
                typically take the form (0, 100, 200) for
                example which will give a split between 0-100
                and 100-200 seconds
            var2bin (Enum): The spatial variable to bin up
            maptype (Enum): The type of map to produce
        """
        if var2bin.value == VariableToBin.DIR.value:
            sample = self.dir
        elif var2bin.value == VariableToBin.SPEED.value:
            sample = self.speed
        elif var2bin.value == VariableToBin.XY.value:
            sample = self.xy
        else:
            raise ValueError("Unrecognized variable to bin.")
        assert sample is not None
        self.pos_time_splits = splits

        sample = np.concatenate((np.atleast_2d(sample), np.atleast_2d(times)))
        edges = [b for b in self._binedges][::-1]
        edges.append(splits)
        # bin pos
        bp, bpe = np.histogramdd(sample.T, bins=edges)
        map1_pos, map2_pos = np.squeeze(bp[:, :, 0]), np.squeeze(bp[:, :, 1])
        # smooth position
        map1_pos = blur_image(map1_pos, 7, ftype="gaussian")
        map2_pos = blur_image(map2_pos, 7, ftype="gaussian")
        # bin spk - ie the histogram is weighted by spike count
        # in bin i
        spk = [np.histogramdd(sample.T, bins=edges, weights=w) for w in spkW]
        map1_spk, map2_spk = self._splitStackedCorrelations(spk)
        map1_sm_spk = np.array([blur_image(m, 7, ftype="gaussian") for m in map1_spk])
        map2_sm_spk = np.array([blur_image(m, 7, ftype="gaussian") for m in map2_spk])
        map1_rmaps = map1_sm_spk / map1_pos
        map2_rmaps = map2_sm_spk / map2_pos
        return map1_rmaps, map2_rmaps
