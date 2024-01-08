# Suppress warnings generated from doing the ffts for the spatial
# autocorrelogram
# see autoCorr2D and crossCorr2D
import warnings
from enum import Enum
from functools import cache
from collections import namedtuple

import numpy as np
from astropy import convolution  # deals with nans unlike other convs
from scipy import signal
from shapely import MultiLineString, prepare
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point
from ephysiopy.common.utils import blurImage, flatten_list

warnings.filterwarnings(
    "ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings(
    "ignore", message="invalid value encountered in subtract")
warnings.filterwarnings(
    "ignore", message="invalid value encountered in greater")
warnings.filterwarnings(
    "ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings(
    "ignore", message="divide by zero encountered in true_divide")
np.seterr(divide="ignore", invalid="ignore")


class VariableToBin(Enum):
    XY = 1
    DIR = 2
    SPEED = 3
    XY_TIME = 4


class MapType(Enum):
    RATE = 1
    POS = 2
    SPK = 3


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
        xy: np.array = None,
        hdir: np.array = None,
        speed: np.array = None,
        pos_weights: np.array = None,
        pos_times: np.array = None,
        ppm: int = 430,
        xyInCms: bool = False,
        binsize: int = 3,
        smooth_sz: int = 5,
    ):
        self.xy = xy
        self.dir = hdir
        self.speed = speed
        self._pos_weights = pos_weights
        self._pos_times = pos_times
        self._pos_time_splits = None
        self._spike_weights = None
        self._ppm = ppm  # pixels per metre
        self._binsize = binsize
        self._inCms = xyInCms
        self._nBins = None
        self._binedges = None  # has setter and getter - see below
        self._x_lims = None
        self._y_lims = None
        self._smooth_sz = smooth_sz
        self._smoothingType = "gaussian"  # 'boxcar' or 'gaussian'
        self.whenToSmooth = "before"  # or 'after'
        self._var2Bin = VariableToBin.XY
        self._mapType = MapType.RATE
        self._calcBinEdges()

    @property
    def inCms(self):
        # Whether the units are in cms or not
        return self._inCms

    @inCms.setter
    def inCms(self, value):
        self._inCms = value

    @property
    def ppm(self):
        # Get the current pixels per metre (ppm)
        return self._ppm

    @ppm.setter
    def ppm(self, value):
        self._ppm = value
        # self._binedges = self._calcBinEdges(self.binsize)

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
    def nBins(self):
        '''
        The number of bins for each dim
        '''
        if self.binsize:
            return len(self._binedges[0]), len(self._binedges[1])
        else:
            return None

    @nBins.setter
    def nBins(self, value):
        '''
        Sets the number of bins
        '''
        if self.var2Bin == VariableToBin.XY:
            x_lims, y_lims = self._getXYLimits()
            if isinstance(value, int):
                value = [value]
            if len(value) == 1:
                _x, bs_x = np.linspace(x_lims[0],
                                       x_lims[1],
                                       int(value[0]),
                                       retstep=True)
                _y, bs_y = np.linspace(y_lims[0],
                                       y_lims[1],
                                       int(value[0]),
                                       retstep=True)
            elif len(value) == 2:
                _x, bs_x = np.linspace(x_lims[0],
                                       x_lims[1],
                                       int(value[0]),
                                       retstep=True)
                _y, bs_y = np.linspace(y_lims[0],
                                       y_lims[1],
                                       int(value[1]),
                                       retstep=True)
            self._binedges = _y, _x
            self.binsize = np.mean([bs_x, bs_y])
        elif self.var2Bin == VariableToBin.DIR:
            self._binedges, binsize = np.linspace(0,
                                                  360 + self.binsize,
                                                  value[0],
                                                  retstep=True)
            self.binsize = binsize
        elif self.var2Bin == VariableToBin.SPEED:
            maxspeed = np.max(self.speed)
            self._binedges, binsize = np.linspace(0,
                                                  maxspeed,
                                                  value[0],
                                                  retstep=True)
            self.binsize = binsize

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
    def pos_times(self):
        return self._pos_times

    @pos_times.setter
    def pos_times(self, value):
        self._pos_times = value

    @property
    def pos_time_splits(self):
        return self._pos_times

    @pos_time_splits.setter
    def pos_time_splits(self, value):
        self._pos_times = value

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
        self._binedges = self._calcBinEdges(value)

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

    def _getXYLimits(self):
        '''
        Gets the min/max of the x/y data
        '''
        x_lims = getattr(self, "x_lims", None)
        y_lims = getattr(self, "y_lims", None)
        if x_lims is None:
            x_lims = (np.nanmin(self.xy[0]), np.nanmax(self.xy[0]))
        if y_lims is None:
            y_lims = (np.nanmin(self.xy[1]), np.nanmax(self.xy[1]))
        self.x_lims = x_lims
        self.y_lims = y_lims
        return x_lims, y_lims

    def _calcBinDims(self):
        try:
            self._binDims = [len(b) for b in self._binedges]
        except TypeError:
            self._binDims = len(self._binedges)

    def _calcBinEdges(self, binsize: int = 3) -> tuple:
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
            maxspeed = np.max(self.speed)
            # assume min speed = 0
            self.binedges = np.arange(0, maxspeed, binsize)
        elif self.var2Bin == VariableToBin.XY:
            x_lims, y_lims = self._getXYLimits()
            nxbins = int(np.ceil((x_lims[1]-x_lims[0])/binsize))
            nybins = int(np.ceil((y_lims[1]-y_lims[0])/binsize))
            _x = np.linspace(x_lims[0], x_lims[1], nxbins)
            _y = np.linspace(y_lims[0], y_lims[1], nybins)
            self.binedges = _y, _x
        elif self.var2Bin == VariableToBin.XY_TIME:
            if self._pos_time_splits is None:
                raise ValueError("Need pos times to bin up XY_TIME")
            x_lims, y_lims = self._getXYLimits()
            nxbins = int(np.ceil((x_lims[1]-x_lims[0])/binsize))
            nybins = int(np.ceil((y_lims[1]-y_lims[0])/binsize))
            _x = np.linspace(x_lims[0], x_lims[1], nxbins)
            _y = np.linspace(y_lims[0], y_lims[1], nybins)
            be = [_y, _x]
            be.append(self.pos_time_splits)
            self.binedges = be
        self._calcBinDims()
        return self.binedges

    def getSpatialSparsity(self,
                           spkWeights,
                           sample_rate=50,
                           **kwargs):
        """
        Gets the spatial sparsity measure - closer to 1 means
        sparser firing field.

        References:
            Skaggs, W.E., McNaughton, B.L., Wilson, M.A. & Barnes, C.A.
            Theta phase precession in hippocampal neuronal populations
            and the compression of temporal sequences.
            Hippocampus 6, 149–172 (1996).
        """
        self.var2Bin = VariableToBin.XY
        self._calcBinEdges()
        sample = self.xy
        keep_these = np.isfinite(sample[0])
        pos, _ = self._binData(sample,
                               self._binedges,
                               self.pos_weights,
                               keep_these)
        npos = len(self.dir)
        p_i = np.count_nonzero(pos) / npos / sample_rate
        spk, _ = self._binData(sample,
                               self._binedges,
                               spkWeights,
                               keep_these)
        res = 1-(np.nansum(p_i*spk)**2) / np.nansum(p_i*spk**2)
        return res

    def getMap(self, spkWeights,
               varType=VariableToBin.XY,
               mapType=MapType.RATE,
               smoothing=True,
               **kwargs):
        """
        Bins up the variable type varType and returns a tuple of
        (rmap, binnedPositionDir) or
        (rmap, binnedPostionX, binnedPositionY)

        Args:
            spkWeights (array_like): Shape equal to number of positions samples captured and consists of
                position weights. For example, if there were 5 positions
                recorded and a cell spiked once in position 2 and 5 times in
                position 3 and nothing anywhere else then pos_weights looks
                like: [0 0 1 5 0]
            varType (Enum value - see Variable2Bin defined at top of this file): The variable to bin up. Legal values are: XY, DIR and SPEED
            mapType (enum value - see MapType defined at top of this file): If RATE then the binned up spikes are divided by varType.
                Otherwise return binned up position. Options are RATE or POS
            smoothing (bool, optional): Whether to smooth the data or not. Defaults to True.

        Returns:
            binned_data, binned_pos (tuple): This is either a 2-tuple or a 3-tuple depening on whether binned
                pos (mapType 'pos') or binned spikes (mapType 'rate') is asked
                for respectively
        """
        if varType.value == VariableToBin.DIR.value:
            sample = self.dir
            keep_these = np.isfinite(sample)
        elif varType.value == VariableToBin.SPEED.value:
            sample = self.speed
            keep_these = np.isfinite(sample)
        elif varType.value == VariableToBin.XY.value:
            sample = self.xy
            keep_these = np.isfinite(sample[0])
        elif varType.value == VariableToBin.XY_TIME.value:
            sample = np.concatenate((np.atleast_2d(self.xy),
                                    np.atleast_2d(self.pos_times)))
            keep_these = np.isfinite(self.xy[0])
        else:
            raise ValueError("Unrecognized variable to bin.")
        assert sample is not None

        self.var2Bin = varType
        self._spike_weights = spkWeights
        self._calcBinEdges(self.binsize)

        binned_pos, binned_pos_edges = self._binData(
            sample,
            self._binedges,
            self.pos_weights,
            keep_these)
        nanIdx = binned_pos == 0

        if mapType.value == MapType.POS.value:  # return binned up position
            if smoothing:
                if varType.value == VariableToBin.DIR.value:
                    binned_pos = self._circPadSmooth(
                        binned_pos, n=self.smooth_sz)
                else:
                    binned_pos = blurImage(binned_pos,
                                           self.smooth_sz,
                                           ftype=self.smoothingType,
                                           **kwargs)
            return binned_pos, binned_pos_edges

        binned_spk, _ = self._binData(
            sample, self._binedges, spkWeights, keep_these)
        if mapType.value == MapType.SPK:
            return binned_spk
        # binned_spk is returned as a tuple of the binned data and the bin
        # edges
        if "after" in self.whenToSmooth:
            rmap = binned_spk / binned_pos
            if varType.value == VariableToBin.DIR.value:
                rmap = self._circPadSmooth(rmap, self.smooth_sz)
            else:
                rmap = blurImage(rmap,
                                 self.smooth_sz,
                                 ftype=self.smoothingType,
                                 **kwargs)
        else:  # default case
            if not smoothing:
                return binned_spk / binned_pos, binned_pos_edges
            if varType.value == VariableToBin.DIR.value:
                binned_pos = self._circPadSmooth(binned_pos, self.smooth_sz)
                binned_spk = self._circPadSmooth(binned_spk, self.smooth_sz)
                rmap = binned_spk / binned_pos
            else:
                binned_pos = blurImage(binned_pos,
                                       self.smooth_sz,
                                       ftype=self.smoothingType,
                                       **kwargs)
                if binned_spk.ndim == 2:
                    pass
                elif binned_spk.ndim == 1:
                    binned_spk_tmp = np.zeros(
                        [binned_spk.shape[0], binned_spk.shape[0], 1]
                    )
                    for i in range(binned_spk.shape[0]):
                        binned_spk_tmp[i, :, :] = binned_spk[i]
                    binned_spk = binned_spk_tmp
                binned_spk = blurImage(
                    binned_spk,
                    self.smooth_sz,
                    ftype=self.smoothingType,
                    **kwargs)
                rmap = binned_spk / binned_pos
                if rmap.ndim <= 2:
                    rmap[nanIdx] = np.nan

        return rmap, binned_pos_edges

    def getSAC(self, spkWeights, **kwargs):
        '''
        Returns the SAC - convenience function
        '''
        rmap = self.getMap(spkWeights=spkWeights, **kwargs)
        nodwell = ~np.isfinite(rmap[0])
        return self.autoCorr2D(rmap[0], nodwell)

    def _binData(self, var, bin_edges, weights, good_indices=None):
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
            >>> h = _binData(
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
            # if self.var2Bin != VariableToBin.XY and len(bin_edges) != 1:
            #     bin_edges = self._calcBinEdges(self.binsize)
            bin_edges = bin_edges[np.newaxis, :]
        elif dims > 1 and var.ndim == 1:
            var = var[np.newaxis, :]
            bin_edges = bin_edges[np.newaxis, :]
        else:
            var = np.flipud(var)
        weights = np.atleast_2d(weights)  # needed for list comp below
        var = np.array(var.data.T.tolist())
        ndhist = [np.histogramdd(
            sample=var[good_indices],
            bins=bin_edges,
            weights=np.ravel(w[good_indices])) for w in weights]
        if np.shape(weights)[0] == 1:
            return ndhist[0][0], ndhist[0][1]
        else:
            tmp = [d[0] for d in ndhist]
            tmp = np.array(tmp)
            return tmp, ndhist[1]

    def _circPadSmooth(self, var, n=3, ny=None):
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
        x, y = np.mgrid[-n: n + 1, 0 - ny: ny + 1]
        g = np.exp(-(x**2 / float(n) + y**2 / float(ny)))
        if np.ndim(var) == 1:
            g = g[n, :]
        g = g / g.sum()
        improc = signal.convolve(var, g, mode="same")
        improc = improc[tn - t2: tn - t2 + tn]
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

    def getAdaptiveMap(self, pos_binned, spk_binned, alpha=200):
        """
        Produces a ratemap that has been adaptively binned according to the
        algorithm described in Skaggs et al., 1996) [1]_.

        Args:
            pos_binned (array_like): The binned positional data. For example that returned from getMap
                above with mapType as 'pos'
            spk_binned (array_like): The binned spikes
            alpha (int, optional): A scaling parameter determing the amount of occupancy to aim at
                in each bin. Defaults to 200.

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

    def autoCorr2D(self, A, nodwell, tol=1e-10):
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
            np.real(np.fft.ifft(
                np.fft.ifft(Fx * np.conj(Fx), axis=1), axis=0)),
            axes=(0, 1),
        )
        sums_x = np.fft.fftshift(
            np.real(np.fft.ifft(
                np.fft.ifft(np.conj(Fx) * Fn, axis=1), axis=0)),
            axes=(0, 1),
        )
        sumOfSquares_x = np.fft.fftshift(
            np.real(
                np.fft.ifft(
                    np.fft.ifft(Fn * np.conj(FsumOfSquares_x), axis=1), axis=0)
            ),
            axes=(0, 1),
        )
        N = np.fft.fftshift(
            np.real(np.fft.ifft(
                np.fft.ifft(Fn * np.conj(Fn), axis=1), axis=0)),
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

        return np.squeeze(
            mapCovar / mapStd / mapStd[::-1, :, :][:, ::-1, :][:, :, :])

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
            np.real(np.fft.ifft(np.fft.ifft(
                Fa * np.conj(Fn_b), axis=1), axis=0))
        )
        sums_b = np.fft.fftshift(
            np.real(np.fft.ifft(np.fft.ifft(
                Fn_a * np.conj(Fb), axis=1), axis=0))
        )
        sumOfSquares_a = np.fft.fftshift(
            np.real(
                np.fft.ifft(
                    np.fft.ifft(
                        FsumOfSquares_a * np.conj(Fn_b), axis=1), axis=0
                )
            )
        )
        sumOfSquares_b = np.fft.fftshift(
            np.real(
                np.fft.ifft(
                    np.fft.ifft(
                        Fn_a * np.conj(FsumOfSquares_b), axis=1), axis=0
                )
            )
        )
        N = np.fft.fftshift(
            np.real(np.fft.ifft(np.fft.ifft(
                Fn_a * np.conj(Fn_b), axis=1), axis=0))
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
        Hp = np.histogram2d(dwell[0, :], dwell[1, :],
                            range=binedges, bins=binsize)[0]
        Hs = np.histogram2d(spike[0, :], spike[1, :],
                            range=binedges, bins=binsize)[0]

        # reverse y,x order
        Hp = np.swapaxes(Hp, 1, 0)
        Hs = np.swapaxes(Hs, 1, 0)

        fHp = blurImage(Hp, boxcar)
        fHs = blurImage(Hs, boxcar)

        H = fHs / fHp
        H[Hp < Pthresh] = np.nan

        return H

    @cache
    def _create_boundary_distance_lookup(self,
                                         arena_boundary: MultiLineString,
                                         degs_per_bin: float,
                                         xy_binsize: float,
                                         **kwargs):
        # Now we generate lines radiating out from a point as a
        # multilinestring geometry collection - this looks
        # like a 360/degs_per_bin
        # star. We will move this to each valid location in the position map
        # and then calculate the distance to the nearest intersection with the
        # arena boundary.
        # get the arena boundaries to figure out the radius of the arena,
        # regardless of its actual shape
        x1, y1, x2, y2 = arena_boundary.bounds
        radius = max(x2-x1, y2-y1)/2
        startpoint = Point((x1+radius, y1+radius))
        endpoint = Point([x2, y1+radius])
        angles = np.arange(0, 360, degs_per_bin)
        lines = MultiLineString(
            [rotate(LineString([startpoint, endpoint]), ang, origin=startpoint)
             for ang in angles])
        prepare(lines)
        # arena centre
        cx = x1 + radius
        cy = y1 + radius
        # get the position map and the valid locations within it
        pos_map, (ybin_edges, xbin_edges) = self.getMap(np.ones_like(self.dir),
                                                        varType=VariableToBin.XY,
                                                        mapType=MapType.POS,
                                                        smoothing=False)
        yvalid, xvalid = np.nonzero(~np.isnan(pos_map))

        # preallocate the array to hold distances
        distances = np.full(
            (len(xbin_edges), len(ybin_edges), len(angles)), np.nan)

        # Now iterate through valid locations in the pos map and calculate the
        # distances and the indices of the lines that intersect with the
        # arena boundary. The indices are equivalent to the angle of the
        # line in the lines geometry collection. This iteration is a bit slow
        # but it will only need to be done once per session as it's creating
        # a lookup table for the distances
        for xi, yi in zip(xvalid, yvalid):
            i_point = Point((xbin_edges[xi]+xy_binsize,
                             ybin_edges[yi]+xy_binsize))
            ipx, ipy = i_point.xy
            new_point = Point(cx-ipx[0], cy-ipy[0])
            t_arena = translate(arena_boundary, -new_point.x, -new_point.y)
            prepare(t_arena)
            di = [(startpoint.distance(t_arena.intersection(line)), idx)
                  for idx, line in enumerate(lines.geoms) if
                  t_arena.intersects(line)]
            d, i = zip(*di)
            distances[xi, yi, i] = d
        return distances

    def get_egocentric_boundary_map(self,
                                    spk_weights,
                                    degs_per_bin: float = 3,
                                    xy_binsize: float = 2.5,
                                    arena_type: str = "circle",
                                    return_dists: bool = False,
                                    return_raw_spk: bool = False,
                                    return_raw_occ: bool = False) -> namedtuple:
        """
        Helps construct dwell time/spike counts maps with respect to boundaries at given egocentric directions and distances.

        Note:
            For the directional input, the 0 degree reference is horizontal pointing East and moves counter-clockwise.
        """
        assert self.dir is not None, "No direction data available"
        # initially do some binning to get valid locations
        # (some might be nans due to
        # arena shape and/or poor sampling) and then digitize
        # the x and y positions
        # and the angular positions
        self.binsize = xy_binsize  # this will trigger a
        # re-calculation of the bin edges

        angles = np.arange(0, 360, degs_per_bin)

        # Use the shaeply package to specify some geometry for the arena
        # boundary and the lines radiating out
        # from the current location of the animal. The geometry for the
        # arena should be user specified but for now I'll just use a circle
        if arena_type == "circle":
            radius = 50
            circle_centre = Point(
                np.nanmin(self.xy[0])+radius, np.nanmin(self.xy[1])+radius)
            arena_boundary = circle_centre.buffer(radius).boundary
        # now we have a circle with its centre at the centre of the arena
        # i.e. the circle defines the arena edges. Calling .boundary on the
        # circle geometry actually gives us a 65-gon polygon
        distances = self._create_boundary_distance_lookup(
            arena_boundary, degs_per_bin, xy_binsize)
        # iterate through the digitized locations (x/y and angular), using the
        # lookup table to get the distances to the arena boundary and then
        # increment the appropriate bin in the egocentric boundary map
        good_idx = np.isfinite(self.xy[0])
        xy_by_heading, _ = np.histogramdd([self.xy[0][good_idx],
                                           self.xy[1][good_idx],
                                           self.dir[good_idx]],
                                          bins=distances.shape,
                                          weights=self.pos_weights[good_idx])
        spk_xy_by_hd, _ = np.histogramdd([self.xy[0][good_idx],
                                          self.xy[1][good_idx],
                                          self.dir[good_idx]],
                                         bins=distances.shape,
                                         weights=spk_weights[good_idx])
        assert xy_by_heading.shape == distances.shape
        distlist = []
        anglist = []
        spkdists = []
        spkangs = []
        for i_bin in np.ndindex(distances.shape[:2]):
            i_dist = distances[i_bin]
            valid_dist = np.isfinite(i_dist)
            nonzero_bincounts = np.nonzero(xy_by_heading[i_bin])[0]
            nonzero_spkbins = np.nonzero(spk_xy_by_hd[i_bin])[0]
            for i_angle in nonzero_bincounts:
                ego_angles = np.roll(angles, i_angle)[valid_dist]
                n_repeats = xy_by_heading[i_bin][i_angle]
                ego_angles_repeats = np.repeat(ego_angles, n_repeats)
                dist_repeats = np.repeat(i_dist[valid_dist], n_repeats)
                distlist.append(dist_repeats)
                anglist.append(ego_angles_repeats)
                if i_angle in nonzero_spkbins:
                    n_repeats = spk_xy_by_hd[i_bin][i_angle]
                    ego_angles_repeats = np.repeat(ego_angles, n_repeats)
                    dist_repeats = np.repeat(i_dist[valid_dist], n_repeats)
                    spkdists.append(dist_repeats)
                    spkangs.append(ego_angles_repeats)
        flat_angs = flatten_list(anglist)
        flat_dists = flatten_list(distlist)
        flat_spk_dists = flatten_list(spkdists)
        flat_spk_angs = flatten_list(spkangs)
        bins = [int(radius/xy_binsize), len(angles)]
        ego_boundary_occ, _, _ = np.histogram2d(x=flat_dists, y=flat_angs,
                                                bins=bins)
        ego_boundary_spk, _, _ = np.histogram2d(x=flat_spk_dists,
                                                y=flat_spk_angs,
                                                bins=bins)
        kernel = convolution.Gaussian2DKernel(5, x_size=3, y_size=5)
        sm_occ = convolution.convolve(ego_boundary_occ,
                                      kernel,
                                      boundary='extend')
        sm_spk = convolution.convolve(ego_boundary_spk,
                                      kernel,
                                      boundary='extend')
        ego_boundary_map = sm_spk / sm_occ
        EgoMap = namedtuple("EgoMap", ['rmap', 'occ', 'spk', 'dists'],
                            defaults=None)
        em = EgoMap(None, None, None, None)
        em = em._replace(rmap=ego_boundary_map)
        if return_dists:
            em = em._replace(dists=distances)
        if return_raw_occ:
            em = em._replace(occ=ego_boundary_occ)
        if return_raw_spk:
            em = em._replace(spk=ego_boundary_spk)
        return em

    def getAllSpikeWeights(self,
                           spike_times: np.ndarray,
                           spike_clusters: np.ndarray,
                           pos_times: np.ndarray,
                           **kwargs):
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
        weights = [np.bincount(idx[spike_clusters == c], minlength=npos)
                   for c in clusters]
        return np.array(weights)

    def _splitStackedCorrelations(self, binned_data: list) -> tuple:
        '''
        Takes in the result of doStackedCorrelations() and splits into
        two arrays and returns these as a 2-tuple
        '''
        result = [(s[0][:, :, 0], s[0][:, :, 1]) for s in binned_data]
        result = np.array(result)
        return np.squeeze(result[:, 0, :, :]), np.squeeze(result[:, 1, :, :])

    def doStackedCorrelations(self,
                              spkW: np.ndarray,
                              times: np.ndarray,
                              splits: np.ndarray,
                              var2bin: Enum = VariableToBin.XY,
                              maptype: Enum = MapType.RATE,
                              **kwargs):
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

        sample = np.concatenate((np.atleast_2d(sample),
                                np.atleast_2d(times)))
        edges = [b for b in self._binedges][::-1]
        edges.append(splits)
        # bin pos
        bp, bpe = np.histogramdd(sample.T, bins=edges)
        map1_pos, map2_pos = np.squeeze(bp[:, :, 0]), np.squeeze(bp[:, :, 1])
        # smooth position
        map1_pos = blurImage(map1_pos, 7, ftype='gaussian')
        map2_pos = blurImage(map2_pos, 7, ftype='gaussian')
        # bin spk - ie the histogram is weighted by spike count
        # in bin i
        spk = [np.histogramdd(sample.T, bins=edges, weights=w)
               for w in spkW]
        map1_spk, map2_spk = self._splitStackedCorrelations(spk)
        map1_sm_spk = np.array([blurImage(m, 7, ftype='gaussian')
                                for m in map1_spk])
        map2_sm_spk = np.array([blurImage(m, 7, ftype='gaussian')
                                for m in map2_spk])
        map1_rmaps = map1_sm_spk / map1_pos
        map2_rmaps = map2_sm_spk / map2_pos
        return map1_rmaps, map2_rmaps
