import warnings

import numpy as np
import boost_histogram as bh
from shapely import Point
from astropy import convolution  # deals with nans unlike other convs
from scipy.spatial import distance
from skimage.morphology import disk
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.common.utils import (
    blur_image,
    BinnedData,
    VariableToBin,
    MapType,
)

# Suppress warnings generated from doing the ffts for the spatial
# autocorrelogram
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings(
    "ignore", message="invalid value encountered in subtract")
warnings.filterwarnings(
    "ignore", message="invalid value encountered in greater")
warnings.filterwarnings(
    "ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide")
warnings.filterwarnings(
    "ignore", message="divide by zero encountered in true_divide")
np.seterr(divide="ignore", invalid="ignore")


class RateMap(object):
    """
    Bins up positional data (xy, head direction etc) and produces rate maps
    of the relevant kind. This is a generic class meant to be independent of
    any particular recording format.

    Parameters
    ----------
    xy : np.ndarray
        The xy data as a 2 x n_samples numpy array.
    hdir : np.ndarray
        The head direction data a 1 x n_samples numpy array.
    speed : np.ndarray
        Similar to hdir.
    pos_weights : np.ndarray
        A 1 x n_samples numpy array used to weight a particular
        position samples when binning data. For example, if there were 5
        positions recorded and a cell spiked once in position 2 and 5 times
        in position 3 and nothing anywhere else then pos_weights looks like:
        [0 0 1 5 0]
        In the case of binning up position this will be an array of mostly 1's
        unless there are some positions you want excluded.
    ppm : int
        Optional. Pixels per metre. Specifies how many camera pixels per metre so this,
        in combination with cmsPerBin, will determine how many bins there are
        in the rate map. Defaults to None.
    xyInCms :bool
        Optional. Whether the positional data is in cms. Defaults to False.
    cmsPerBin : int
        Optional. How many cms on a side each bin is in a rate map OR the number of
        degrees per bin in the case of directional binning. Defaults to 3.
    smooth_sz : int
        Optional. The width of the smoothing kernel for smoothing rate maps. Defaults to 5.

    Attributes
    ----------
    xy : np.ndarray
        the x-y position as 2 x n_samples
    dir : np.ndarray
        directional bearing as n_samples vector. In degrees.
    speed : np.ndarray
        speed as n_samples vector
    pos_times : np.ndarray
        the times, in seconds, at which the position sample was recorded.
    inCms : bool
        whether the data has been converted to cms or not.
    ppm : int
        the number of pixels (camera coordinates) per metre
    var2Bin : Enum
        VariableToBin enum value. See ephysiopy.common.utils.VariableToBin
    mapType : Enum
        MapType enum value. See ephysiopy.common.utils.MapType
    binedges : tuple[np.ndarray,...]
        The edges of the binned data (can be n-dimensional)
    x_lims : tuple[float,...]
        The limits of the x data (min and max)
    y_lims : tuple[float,...]
        The limits of the y data (min and max)
    pos_weights : np.ndarray
        the weights assigned to position data when binning. n_samples vector
    spike_weights : np.ndarray
        same as pos_weights but for spikes
    binsize : tuple
        the size of each bin (could be x-y or degrees or some other unit)
    smooth_sz : int
        the size of the smoothing kernel for filtering rate maps
    smoothingType : str
        the kind of smoothing to apply. 'boxcar' or 'gaussian'

    """

    def __init__(
        self,
        PosCalcs: PosCalcsGeneric,
        pos_weights: np.ma.MaskedArray | None = None,
        xyInCms: bool = False,
        binsize: int = 3,
        smooth_sz: int = 5,
    ):
        self.PosCalcs = PosCalcs
        self.sample_to_bin = None
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
    def sample_to_bin(self):
        return self.sample

    @sample_to_bin.setter
    def sample_to_bin(self, val):
        self.sample = val

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
    def phi(self):
        return self.PosCalcs.phi

    @property
    def pos_times(self):
        return self.PosCalcs.xyTS

    @property
    def inCms(self):
        # Whether the units are in cms or not
        return self.PosCalcs.convert2cm

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
            self._pos_weights = np.ma.MaskedArray(np.ones(self.PosCalcs.npos))
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
        self.pos_weights.mask = mask

    def _getXYLimits(self):
        """
        Gets the min/max of the x/y data
        """
        xy = self.PosCalcs.xy.data
        x_lims = (np.nanmin(xy[0]), np.nanmax(xy[0]))
        y_lims = (np.nanmin(xy[1]), np.nanmax(xy[1]))
        self.x_lims = x_lims
        self.y_lims = y_lims
        return x_lims, y_lims

    def _calc_bin_dims(self):
        try:
            self._binDims = [len(b) for b in self._binedges]
        except TypeError:
            self._binDims = len(self._binedges)

    def _calc_bin_edges(self, binsize: int | tuple = 3) -> tuple[np.ndarray, ...]:
        """
        Aims to get the right number of bins for the variable to be binned

        Parameters
        ----------
        binsize : int | tuple
            Optional. The number of cms per bin for XY OR degrees for DIR OR cm/s for SPEED. Defaults to 3.

        Returns
        -------
        bins : tuple of np.ndarray
            each member an array of bin edges
        """
        if self.var2Bin.value == VariableToBin.DIR.value:
            self.binedges = np.linspace(0, 360, int(360 / binsize)).tolist()

        elif self.var2Bin.value == VariableToBin.SPEED.value:
            maxspeed = np.nanmax(self.speed)
            # assume min speed = 0
            self.binedges = np.linspace(
                0, maxspeed, int(maxspeed / binsize)).tolist()

        elif self.var2Bin.value == VariableToBin.X.value:
            x_lims = self.x_lims
            if x_lims is None:
                x_lims = self._getXYLimits()[0]
            nxbins = int(np.ceil((x_lims[1] - x_lims[0]) / binsize))
            self.binedges = np.linspace(x_lims[0], x_lims[1], nxbins)

        elif self.var2Bin.value == VariableToBin.Y.value:
            y_lims = self.y_lims
            if y_lims is None:
                y_lims = self._getXYLimits()[1]
            nybins = int(np.ceil((y_lims[1] - y_lims[0]) / binsize))
            self.binedges = np.linspace(y_lims[0], y_lims[1], nybins)

        elif self.var2Bin.value == VariableToBin.PHI.value:
            phi_max = np.nanmax(self.phi)
            nbins = int(np.ceil(phi_max / binsize))
            self.binedges = np.linspace(0, phi_max, nbins)

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
            self.binedges = _y, _x, self.pos_time_splits

        elif self.var2Bin.value == VariableToBin.SPEED_DIR.value:
            maxspeed = np.nanmax(self.speed)
            if isinstance(binsize, int):
                self.binedges = (
                    np.linspace(0, maxspeed, int(maxspeed / binsize)),
                    np.linspace(0, 360, int(360 / binsize)),
                )
            elif isinstance(binsize, tuple):
                self.binedges = (
                    np.linspace(0, maxspeed, int(maxspeed / binsize[0])),
                    np.linspace(0, 360, int(360 / binsize[1])),
                )

        elif self.var2Bin.value == VariableToBin.EGO_BOUNDARY.value:
            if isinstance(binsize, (float, int)):
                self.binedges = np.linspace(
                    0, 50, 20), np.linspace(0, 2 * np.pi, 120)
            elif isinstance(binsize, tuple):
                self.binedges = (
                    np.linspace(0, 50, int(50 / binsize[0])),
                    np.linspace(0, 2 * np.pi, int((2 * np.pi) / binsize[1])),
                )
        self._calc_bin_dims()
        return self.binedges

    def get_map(
        self,
        spk_weights,
        var_type=VariableToBin.XY,
        map_type=MapType.RATE,
        smoothing=True,
        **kwargs,
    ) -> BinnedData | None:
        """
        Bins up the variable type var_type and returns a tuple of
        (rmap, binnedPositionDir) or
        (rmap, binnedPostionX, binnedPositionY)

        Parameters
        ----------
        spk_weights : np.ndarray
            Shape equal to number of positions samples captured and consists of
            position weights. For example, if there were 5 positions
            recorded and a cell spiked once in position 2 and 5 times in
            position 3 and nothing anywhere else then pos_weights looks
            like: [0 0 1 5 0].
            spk_weights can also be list-like where each entry in the list is a different set of
            weights - these are enumerated through in a list comp in the ._bin_data function. In
            this case the returned tuple will consist of a 2-tuple where the first entry is an
            array of the ratemaps (binned_spk / binned_pos) and the second part is the binned pos data (as it's common to all
            the spike weights)

        var_type : Variable2Bin
            The variable to bin. See ephysiopy.common.utils for values.

        map_type : MapType
            The kind of map returned. See ephysiopy.common.utils for values.

        smoothing : bool
            Optional. Smooth the data or not. Default True.

        kwargs: dict
            Values: hist_range - currently only used by EGO_BOUNDARY var_type
                    bin_edges - bin_edges are supplied by calling function.
                                 Mostly for when filtering the sample data
                                 might lead to different bin ranges being
                                 used vs unfiltered data

        Returns
        -------
        BinnedData
            An instance of BinnedData containing the binned data, the bin edges, the variable binned and
            the map type. See ephysiopy.common.utils for details of the class.
        """
        boundary = "extend"
        pos_weights = np.invert(self.pos_weights.mask).astype(int)

        if var_type.value == VariableToBin.DIR.value:
            sample = self.dir
            self.sample_to_bin = sample
            boundary = "wrap"

        elif var_type.value == VariableToBin.SPEED.value:
            sample = self.speed

        elif var_type.value == VariableToBin.XY.value:
            sample = self.xy

        elif var_type.value == VariableToBin.X.value:
            sample = self.xy[0]

        elif var_type.value == VariableToBin.Y.value:
            sample = self.xy[1]

        elif var_type.value == VariableToBin.PHI.value:
            # TODO: this might be the best way to handle masking
            # ie to modify the pos_weights that are input to the
            # binning function
            sample = self.phi

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
            boundary = "wrap"
            binsize = kwargs.get("binsize", 5)
            if isinstance(binsize, tuple):
                binsize = binsize[0]
            # breakpoint()
            ego_angles, arena_xy = self._calc_ego_angles(arena_shape, binsize)
            ego_dists = distance.cdist(arena_xy, self.xy.T, "euclidean")
            sample = np.stack((np.ravel(ego_angles.T), np.ravel(ego_dists.T)))
            spk_weights = np.atleast_2d(spk_weights)
            spk_weights = np.tile(spk_weights, arena_xy.shape[0])
            pos_weights = np.tile(self.pos_weights, arena_xy.shape[0])
            hist_range = (0, 50), (0, 2 * np.pi)

            if "range" not in kwargs.keys():
                kwargs["range"] = hist_range
        else:
            raise ValueError("Unrecognized variable to bin.")

        assert sample is not None

        self.sample_to_bin = sample

        self.var2Bin = var_type
        binsize = kwargs.pop("binsize", self.binsize)
        hist_range = kwargs.pop("range", None)

        if hist_range is None:
            bin_edges = self._calc_bin_edges(binsize)
        else:
            bin_edges = None

        bin_edges = kwargs.get("bin_edges", bin_edges)

        binned_pos, binned_pos_edges = self._bin_data(
            sample, bin_edges, pos_weights)
        binned_pos = binned_pos / self.PosCalcs.sample_rate
        nanIdx = binned_pos == 0
        pos = BinnedData(var_type, MapType.POS, [binned_pos], binned_pos_edges)

        if map_type.value == MapType.POS.value:  # return binned up position
            if smoothing:
                sm_pos = blur_image(
                    pos,
                    self.smooth_sz,
                    ftype=self.smoothingType,
                    boundary=boundary,
                    **kwargs,
                )
                sm_pos.set_nan_indices(nanIdx)
                return sm_pos
            else:
                pos.set_nan_indices(nanIdx)
                return pos

        binned_spk, _ = self._bin_data(sample, bin_edges, spk_weights)
        if not isinstance(binned_spk, list):
            binned_spk = [binned_spk]
        spk = BinnedData(var_type, MapType.SPK, binned_spk, binned_pos_edges)

        if map_type.value == MapType.SPK.value:
            if smoothing:
                return blur_image(
                    spk,
                    self.smooth_sz,
                    ftype=self.smoothingType,
                    boundary=boundary,
                    **kwargs,
                )
            else:
                return spk
        if map_type.value == MapType.ADAPTIVE.value:
            alpha = kwargs.pop("alpha", 4)
            # deal with a stack of binned maps

            smthd_rate = []
            for bs in binned_spk:
                smthd_rate.append(self.getAdaptiveMap(
                    binned_pos, bs, alpha)[0])
            return BinnedData(var_type, map_type, smthd_rate, binned_pos_edges)

        if not smoothing:
            rmap = spk / pos
            rmap.map_type = MapType.RATE
            rmap.set_nan_indices(nanIdx)
            return rmap

        if "after" in self.whenToSmooth:
            rmap = spk / pos
            rmap = blur_image(
                rmap,
                self.smooth_sz,
                ftype=self.smoothingType,
                boundary=boundary,
                **kwargs,
            )
        else:  # default case
            sm_pos = blur_image(
                pos,
                self.smooth_sz,
                ftype=self.smoothingType,
                boundary=boundary,
                **kwargs,
            )
            sm_spk = blur_image(
                spk,
                self.smooth_sz,
                ftype=self.smoothingType,
                boundary=boundary,
                **kwargs,
            )
            rmap = sm_spk / sm_pos
        # breakpoint()
        rmap.set_nan_indices(nanIdx)
        return rmap

    def _bin_data(self, var, bin_edges, weights) -> tuple[np.ndarray, ...]:
        """
        Bins data taking account of possible multi-dimensionality

        Parameters
        ----------
        var : np.ndarray
            The variable to bin
        bin_edges : np.ndarray
            The edges of the data - see numpys histogramdd for more
        weights : np.ndarray
            The weights attributed to the samples in var

        Returns
        -------
        tuple of np.ndarray
            The binned variable and the bin edges

        Notes
        -----
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
            weights = np.ma.MaskedArray(np.ones_like(var))
        weights = np.ma.MaskedArray(weights)
        dims = weights.ndim
        if dims == 1 and var.ndim == 1:
            var = var[np.newaxis, :]
            if bin_edges is not None:
                bin_edges = np.atleast_2d(bin_edges)
        elif dims > 1 and var.ndim == 1:
            var = var[np.newaxis, :]
            if bin_edges is not None:
                bin_edges = np.atleast_2d(bin_edges)
        else:
            var = np.flipud(var)
        weights = np.atleast_2d(weights)  # needed for list comp below
        weights = np.ma.filled(weights, 0)
        var = np.array(var.T)
        if bin_edges is None:
            bin_edges = self.binedges
        if len(bin_edges) == 1:
            hist = bh.Histogram(
                bh.axis.Regular(len(bin_edges[0]),
                                bin_edges[0][0], bin_edges[0][-1])
            )
        else:
            hist = bh.Histogram(
                bh.axis.Regular(
                    len(bin_edges[0]), bin_edges[0][0], bin_edges[0][-1]),
                bh.axis.Regular(
                    len(bin_edges[1]), bin_edges[1][0], bin_edges[1][-1]),
            )
        ndhist = []
        for w in weights:
            hist.fill(*var.T, weight=w)
            ndhist.append(
                [hist.values().copy(), [np.ravel(e) for e in hist.axes.edges]]
            )
            hist.reset()
        if np.shape(weights)[0] == 1:
            return ndhist[0]
        else:
            tmp = [d[0] for d in ndhist]
            return tmp, ndhist[1]

    def getAdaptiveMap(self, pos_binned, spk_binned, alpha=4):
        """
        Produces a ratemap that has been adaptively binned according to the
        algorithm described in Skaggs et al., 1996) [1]_.

        Parameters
        ----------
        pos_binned : np.ndarray
            The binned positional data.
        spk_binned : np.ndarray
            The binned spikes
        alpha : int, optional
            A scaling parameter determing the amount of occupancy to aim at
            in each bin. Defaults to 4. In the original paper this was set to 200.
            This is 4 here as the pos data is binned in seconds (the original data was in pos
            samples so this is a factor of 50 smaller than the original paper's value, given 50Hz sample rate)

        Returns
        -------
        tuple of np.ndarray
            The adaptively binned spike and pos maps.
            Use this to generate Skaggs information measure

        Notes
        -----
        Positions with high rates mean proportionately less error than those
        with low rates, so this tries to even the playing field. This type
        of binning should be used for calculations of spatial info
        as with the skaggs_info method in the fieldcalcs class (see below)
        alpha is a scaling parameter that might need tweaking for different
        data sets.

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

        References
        ----------
        .. [1] W. E. Skaggs, B. L. McNaughton, K. M. Gothard & E. J. Markus
            "An Information-Theoretic Approach to Deciphering the Hippocampal
            Code"
            Neural Information Processing Systems, 1993.
        """
        #  assign output arrays
        smthdpos = np.zeros_like(pos_binned)
        smthdspk = np.zeros_like(spk_binned)
        smthdrate = np.zeros_like(pos_binned)
        idx = pos_binned == 0
        pos_binned[idx] = np.nan
        spk_binned[idx] = np.nan
        visited = np.zeros_like(pos_binned)
        visited[pos_binned > 0] = 1
        # array to check which bins have made it
        bincheck = np.isnan(pos_binned)
        r = 1
        while np.any(~bincheck):
            # create the filter kernel
            if pos_binned.ndim == 1:
                k = np.diag(disk(r))
            else:
                k = disk(r)
            if k.flags["WRITEABLE"] is False:
                k.setflags(write=True)
            k[k >= np.max(k) / 3.0] = 1
            k[k != 1] = 0
            if k.shape >= pos_binned.shape:
                break
            # filter the arrays using astropys convolution
            filtpos = convolution.convolve(pos_binned, k)
            filtspk = convolution.convolve(spk_binned, k)
            filtvisited = convolution.convolve(visited, k)
            # get the bins which made it through this iteration
            truebins = alpha / (np.sqrt(filtspk) * filtpos) <= r
            truebins = np.logical_and(truebins, ~bincheck)
            # insert values where true
            smthdpos[truebins] = filtpos[truebins] / filtvisited[truebins]
            smthdspk[truebins] = filtspk[truebins] / filtvisited[truebins]
            bincheck[truebins] = True
            r += 1
        smthdrate = smthdspk / smthdpos
        smthdrate[idx] = np.nan
        smthdspk[idx] = np.nan
        smthdpos[idx] = np.nan
        return smthdrate, smthdspk, smthdpos

    def autoCorr2D(
        self, A: BinnedData, nodwell: np.ndarray = None, tol: float = 1e-10, **kwargs
    ) -> BinnedData:
        """
        Performs autocorrelations on all the maps in an instance of BinnedData.

        Parameters
        ----------
        A : BinnedData
            The binned data
        nodwell : np.ndarray or None
            An array with NaNs where there was no position sampled. If None, it's created from A where there are NaNs.
        tol : float
            Tolerance below which values are set to 0.

        Returns
        -------
        BinnedData
            The data in A with the maps replaced by autocorrelograms
        """
        result = BinnedData(A.variable, MapType.AUTO_CORR)
        result.cluster_id = kwargs.get("cluster_id", [])
        for rmap in A.binned_data:
            if nodwell is None:
                nodwell = np.isnan(rmap)
            rr = self._autoCorr2D(rmap, nodwell, tol)
            result.binned_data += [rr]
        xlen = len(A.bin_edges[0]) - 1
        ylen = len(A.bin_edges[1]) - 1
        result.bin_edges = [
            np.arange(-xlen + 1, xlen + 1),
            np.arange(-ylen + 1, ylen + 1),
        ]
        return result

    def _autoCorr2D(self, A: np.ndarray, nodwell: np.ndarray, tol: float = 1e-10):
        """
        Performs a spatial autocorrelation on the array A

        Parameters
        ----------
        A : np.ndarray
            Either 2 or 3D. In the former it is simply the binned up ratemap
            where the two dimensions correspond to x and y.
            If 3D then the first two dimensions are x
            and y and the third (last dimension) is 'stack' of ratemaps
        nodwell : np.ndarray
            A boolean array corresponding the bins in the ratemap that
            weren't visited. See Notes.
        tol : float, optional
            Values below this are set to zero to deal with v small values
            thrown up by the fft. Default 1e-10

        Returns
        -------
        np.ndarray
            The spatial autocorrelation

        Notes
        -----
        The nodwell input can usually be generated by:

        >>> nodwell = ~np.isfinite(A)
        """
        assert np.ndim(A) == 2
        m, n = np.shape(A)
        o = 1
        x = np.reshape(A, (m, n, o))
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
                np.fft.ifft(np.fft.ifft(
                    Fn * np.conj(FsumOfSquares_x), axis=1), axis=0)
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
        mapCovar = (rawCorr * N) - sums_x * \
            sums_x[::-1, :, :][:, ::-1, :][:, :, :]

        return np.squeeze(mapCovar / mapStd / mapStd[::-1, :, :][:, ::-1, :][:, :, :])

    def crossCorr2D(
        self,
        A: BinnedData,
        B: BinnedData,
        A_nodwell: np.ndarray,
        B_nodwell: np.ndarray,
        tol: float = 1e-10,
    ) -> BinnedData:
        """
        Performs crosscorrelations between the maps in two instances of BinnedData, A and B.

        Parameters
        ----------
        A, B : BinnedData
            an instance of BinnedData
        A_nodwell, B_nodwell : np.ndarray
            an array with NaNs where there was no position sampled.
        tol : float
            values below this are set to 0.

        Returns
        -------
        BinnedData
            the data in A with the maps replaced by autocorrelograms
        """
        result = BinnedData(A.variable, MapType.CROSS_CORR)
        for rmap in A:
            for rmap2 in B:
                rr = self._crossCorr2D(rmap, rmap2, A_nodwell, B_nodwell, tol)
                result.binned_data += [rr]
        xlen = max(len(A.bin_edges[0]) - 1, len(B.bin_edges[0]) - 1)
        ylen = max(len(A.bin_edges[1]) - 1, len(B.bin_edges[1]) - 1)
        result.bin_edges = [
            np.arange(-xlen + 1, xlen + 1),
            np.arange(-ylen + 1, ylen + 1),
        ]
        return result

    def _crossCorr2D(
        self,
        A: BinnedData | np.ndarray,
        B: BinnedData | np.ndarray,
        A_nodwell: np.ndarray,
        B_nodwell: np.ndarray,
        tol: float = 1e-10,
    ):
        """
        Performs crosscorrelations between the maps in two instances of BinnedData, A and B.

        Parameters
        ----------
        A, B : BinnedData
            instance of BinnedData
        A_nodwell, B_nodwell : np.ndarray
            array with NaNs where there was no position sampled.
        tol : float
            values below this are set to 0.

        Returns
        -------
        BinnedData
            the data in A with the maps replaced by autocorrelograms
        """
        if isinstance(A, BinnedData):
            A = A.binned_data[0]
        if isinstance(B, BinnedData):
            B = B.binned_data[0]
        if np.ndim(A) != np.ndim(B):
            raise ValueError("Both arrays must have the same dimensionality")
        assert np.ndim(A) == 2
        ma, na = np.shape(A)
        mb, nb = np.shape(B)
        oa = ob = 1
        A = np.reshape(A, (ma, na, oa))
        B = np.reshape(B, (mb, nb, ob))
        if A_nodwell is None:
            A_nodwell = ~np.isfinite(A)
        if B_nodwell is None:
            B_nodwell = ~np.isfinite(B)
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
    ) -> BinnedData:
        """
        Performs a temporal windowed spatial autocorrelation.

        Parameters
        ----------
        xy : np.ndarray
            The position data
        spkIdx : np.ndarray
            The indices in xy where the cell fired
        ppm : int
            Optional. The camera pixels per metre. Default 365
        winSize : int
            Optional. The window size for the temporal search
        pos_sample_rate : int
            Optional. The rate at which position was sampled. Default 50
        nbins : int
            Optional. The number of bins for creating the resulting ratemap. Default 71
        boxcar :int
            Optional. The size of the smoothing kernel to smooth ratemaps. Default 5
        Pthresh : int
            Optional. The cut-off for values in the ratemap; values < Pthresh become nans. Default 100
        downsampfreq : int
            Optional. How much to downsample. Default 50

        Returns
        -------
        np.ndarray
            the temporal windowed SAC
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
            nSpikesInWin[i] = len(spkIdx[t[0]: t[1]]) - 1  # ignore ith spike

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
            dwell[:, filled_pvals: filled_pvals + WL] = np.rot90(
                np.array(np.rot90(xy[:, winInd_dwell]) - xy[:, spkIdx[i]])
            )
            filled_pvals = filled_pvals + WL
            # calculate spike displacements
            winInd_spks = (
                i + np.nonzero(spkIdx[i + 1: n_spks] <
                               spkIdx[i] + winSizeBins)[0]
            )
            WL = len(winInd_spks)
            spike[:, filled_svals: filled_svals + WL] = np.rot90(
                np.array(
                    np.rot90(xy[:, spkIdx[winInd_spks]]) - xy[:, spkIdx[i]])
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
        Hp, Hpe_y, Hpe_x = np.histogram2d(
            dwell[0, :], dwell[1, :], range=binedges, bins=binsize
        )
        Hs, Hse_y, Hse_x = np.histogram2d(
            spike[0, :], spike[1, :], range=binedges, bins=binsize
        )

        # reverse y,x order
        Hp = np.swapaxes(Hp, 1, 0)
        Hs = np.swapaxes(Hs, 1, 0)

        Hp = BinnedData(VariableToBin.XY, MapType.RATE, [Hp], [Hpe_x, Hpe_y])
        Hs = BinnedData(VariableToBin.XY, MapType.RATE, [Hs], [Hse_x, Hse_y])

        # smooth the maps
        fHp = blur_image(Hp, boxcar)
        fHs = blur_image(Hs, boxcar)

        H = fHs / fHp
        H.set_nan_indices(Hp.binned_data[0] < Pthresh)
        return H

    def _calc_ego_angles(
        self, arena_shape: str = "circle", xy_binsize: float = 2.5
    ) -> tuple[np.ndarray, ...]:
        """
        Calculate the angles between the segments of the arena wall
        and the positions of the animal throughout the trial.

        Parameters
        ----------
        arena_shape : str
            the shape of the arena, 'circle' or 'square'.
        xy_binsize : float
            the binsize

        Returns
        -------
        tupleof np.ndarray
            the angles as well as the arena x-y coordinates.

        Notes
        -----
        Angles are in radians.
        """
        arena_width = np.ceil(
            np.nanmean((np.nanmax(self.xy.data, 1) -
                       np.nanmin(self.xy.data, 1)) / 2)
        )
        arena_width = arena_width.tolist()
        arena_centre = Point(np.nanmin(self.xy.data, 1) + arena_width)

        if "circle" in arena_shape:
            arena_boundary = arena_centre.buffer(arena_width).boundary
        elif "square" in arena_shape:
            arena_boundary = arena_centre.buffer(
                arena_width, cap_style=3).boundary
        arena_boundary = arena_boundary.segmentize(
            max_segment_length=xy_binsize)
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

    def get_disperion_map(
        self, spk_times: np.ndarray, pos_times: np.ndarray
    ) -> BinnedData | None:
        """
        Attempt to write a faster version of creating an overdispersion
        map. A cell will sometimes fire too much or too little on a given
        run through its receptive field. This function quantifies that.

        This shows the amount of 'observed' variance in spiking around
        the mean spiking in a bin...

        Parameters
        ----------
        spk_times : np.ndarray
            a vector of spike times (in seconds)
        pos_times : np.ndarray
            vector of position times (in seconds)

        Returns
        -------
        BinnedData
            the overdispersion map in an instance of BinnedData

        """
        idx = np.searchsorted(pos_times, spk_times, side="right")
        spike_weights = np.bincount(idx, minlength=len(pos_times))
        expected_spikes = self.get_map(spike_weights, map_type=MapType.SPK)
        # bin_edges[1] is x, bin_edges[0] is y
        x_bins = np.digitize(self.xy[0], expected_spikes.bin_edges[1][:-1]) - 1
        y_bins = np.digitize(self.xy[1], expected_spikes.bin_edges[0][:-1]) - 1
        map_shape = np.shape(expected_spikes.binned_data[0])
        pos_bins_linear_idx = np.ravel_multi_index([y_bins, x_bins], map_shape)
        expected_spikes_xy = expected_spikes.binned_data[0][y_bins, x_bins]
        min_rate_threshold = np.nanmax(expected_spikes.binned_data[0]) * 0.25
        x_bins_with_firing = x_bins[spike_weights > 0]
        y_bins_with_firing = y_bins[spike_weights > 0]
        bins_with_firing_linear_idx = np.ravel_multi_index(
            [y_bins_with_firing, x_bins_with_firing], map_shape
        )

        observed_spikes = expected_spikes
        return observed_spikes
