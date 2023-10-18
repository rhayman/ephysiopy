# Suppress warnings generated from doing the ffts for the spatial
# autocorrelogram
# see autoCorr2D and crossCorr2D
import warnings
from enum import Enum

import numpy as np
from astropy import convolution  # deals with nans unlike other convs
from scipy import signal
from shapely import MultiLineString, prepare
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
from ephysiopy.common.utils import blurImage, memoized, flatten_list

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


class MapType(Enum):
    RATE = 1
    POS = 2


class RateMap(object):
    """
    Bins up positional data (xy, head direction etc) and produces rate maps
    of the relevant kind. This is a generic class meant to be independent of
    any particular recording format

    Parameters
    ----------
    xy : array_like, optional
        The xy data, usually given as a 2 x n sample numpy array
    hdir : array_like, optional
        The head direction data, usualy a 1 x n sample numpy array
    speed : array_like, optional
        Similar to hdir
    pos_weights : array_like, optional
        A 1D numpy array n samples long which is used to weight a particular
        position sample when binning data. For example, if there were 5
        positions recorded and a cell spiked once in position 2 and 5 times
        in position 3 and nothing anywhere else then pos_weights looks like:
        [0 0 1 5 0]
        In the case of binning up position this will be an array of mostly 1's
        unless there are some positions you want excluded for some reason
    ppm : int, optional
        Pixels per metre. Specifies how many camera pixels per metre so this,
        in combination with cmsPerBin, will determine how many bins there are
        in the rate map
    xyInCms : bool, optional, default False
        Whether the positional data is in cms
    cmsPerBin : int, optional, default 3
        How many cms on a side each bin is in a rate map OR the number of
        degrees per bin in the case of directional binning
    smooth_sz : int, optional, default = 5
        The width of the smoothing kernel for smoothing rate maps

    Notes
    ----
    There are several instance variables you can set, see below

    """

    def __init__(
        self,
        xy: np.array = None,
        hdir: np.array = None,
        speed: np.array = None,
        pos_weights: np.array = None,
        ppm: int = 430,
        xyInCms: bool = False,
        binsize: int = 3,
        smooth_sz: int = 5,
    ):
        self.xy = xy
        self.dir = hdir
        self.speed = speed
        self._pos_weights = pos_weights
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
        Aims to get the right number of bins for the variable to be
        binned

        Parameters
        ----------
        binsize : int, optional, default = 3
            The number of cms per bin for XY
            OR degrees for DIR
            OR cm/s for SPEED

        Returns
        -------
        tuple: each member an array of bin edges
        """
        if self.var2Bin.value == VariableToBin.DIR.value:
            self.binedges = np.arange(0, 360 + binsize, binsize)
        elif self.var2Bin.value == VariableToBin.SPEED.value:
            maxspeed = np.max(self.speed)
            # assume min speed = 0
            self.binedges = np.arange(0, maxspeed, binsize)
        else:  # self.var2Bin == VariableToBin.XY:
            x_lims, y_lims = self._getXYLimits()
            _x = np.arange(x_lims[0], x_lims[1], binsize)
            _y = np.arange(y_lims[0], y_lims[1], binsize)
            self.binedges = _y, _x
        self._calcBinDims()
        return self.binedges

    def getMap(self, spkWeights,
               varType=VariableToBin.XY,
               mapType=MapType.RATE,
               smoothing=True):
        """
        Bins up the variable type varType and returns a tuple of
        (rmap, binnedPositionDir) or
        (rmap, binnedPostionX, binnedPositionY)

        Parameters
        ----------
        spkWeights : array_like
            Shape equal to number of positions samples captured and consists of
            position weights. For example, if there were 5 positions
            recorded and a cell spiked once in position 2 and 5 times in
            position 3 and nothing anywhere else then pos_weights looks
            like: [0 0 1 5 0]
        varType : Enum value - see Variable2Bin defined at top of this file
            The variable to bin up. Legal values are: XY, DIR and SPEED
        mapType : enum value - see MapType defined at top of this file
            If RATE then the binned up spikes are divided by varType.
            Otherwise return binned up position. Options are RATE or POS
        smoothing : bool, optional, default True
            Whether to smooth the data or not

        Returns
        -------
        binned_data, binned_pos : tuple
            This is either a 2-tuple or a 3-tuple depening on whether binned
            pos (mapType 'pos') or binned spikes (mapType 'rate') is asked
            for respectively

        """
        if varType.value == VariableToBin.DIR.value:
            sample = self.dir
        elif varType.value == VariableToBin.SPEED.value:
            sample = self.speed
        elif varType.value == VariableToBin.XY.value:
            sample = self.xy
        else:
            raise ValueError("Unrecognized variable to bin.")
        assert sample is not None

        self.var2Bin = varType
        self._spike_weights = spkWeights
        self._calcBinEdges(self.binsize)

        binned_pos, binned_pos_edges = self._binData(
                                                     sample,
                                                     self._binedges,
                                                     self.pos_weights)
        nanIdx = binned_pos == 0

        if mapType.value == MapType.POS.value:  # return just binned up position
            if smoothing:
                if varType.value == VariableToBin.DIR.value:
                    binned_pos = self._circPadSmooth(
                        binned_pos, n=self.smooth_sz)
                else:
                    binned_pos = blurImage(
                        binned_pos, self.smooth_sz, ftype=self.smoothingType
                    )
            return binned_pos, binned_pos_edges

        binned_spk, _ = self._binData(sample, self._binedges, spkWeights)
        # binned_spk is returned as a tuple of the binned data and the bin
        # edges
        if "after" in self.whenToSmooth:
            rmap = binned_spk / binned_pos
            if varType.value == VariableToBin.DIR.value:
                rmap = self._circPadSmooth(rmap, self.smooth_sz)
            else:
                rmap = blurImage(
                    rmap, self.smooth_sz, ftype=self.smoothingType)
        else:  # default case
            if not smoothing:
                return binned_spk / binned_pos, binned_pos_edges
            if varType.value == VariableToBin.DIR.value:
                binned_pos = self._circPadSmooth(binned_pos, self.smooth_sz)
                binned_spk = self._circPadSmooth(binned_spk, self.smooth_sz)
                rmap = binned_spk / binned_pos
            else:
                binned_pos = blurImage(
                    binned_pos, self.smooth_sz, ftype=self.smoothingType
                )
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
                    np.squeeze(binned_spk),
                    self.smooth_sz,
                    ftype=self.smoothingType
                )
                rmap = binned_spk / binned_pos
                if rmap.ndim <= 2:
                    rmap[nanIdx] = np.nan

        return rmap, binned_pos_edges

    def _binData(self, var, bin_edges, weights):
        """
        Bins data taking account of possible multi-dimensionality

        Parameters
        ----------
        var : array_like
            The variable to bin
        bin_edges : array_like
            The edges of the data - see numpys histogramdd for more
        weights : array_like
            The weights attributed to the samples in var

        Returns
        -------
        ndhist : 2-tuple
            Think this always returns a two-tuple of the binned variable and
            the bin edges - need to check to be sure...

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
                sample=var,
                bins=bin_edges,
                weights=np.ravel(w)) for w in weights]
        if np.shape(weights)[0] == 1:
            return ndhist[0][0], ndhist[0][1]
        else:
            tmp = [d[0] for d in ndhist]
            return tmp, ndhist[1]

    def _circPadSmooth(self, var, n=3, ny=None):
        """
        Smooths a vector by convolving with a gaussian
        Mirror reflects the start and end of the vector to
        deal with edge effects

        Parameters
        ----------
        var : array_like
            The vector to smooth
        n, ny : int
            Size of the smoothing (sigma in gaussian)

        Returns
        -------
        res : array_like
            The smoothed vector with shape the same as var
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

        Parameters
        ----------
        radius : int
            the size of the circular structure

        Returns
        -------
        res : array_like
            Binary structure with shape [(radius*2) + 1,(radius*2) + 1]

        See Also
        --------
        RateMap.__adpativeMap
        """
        from skimage.morphology import disk

        return disk(radius)

    def getAdaptiveMap(self, pos_binned, spk_binned, alpha=200):
        """
        Produces a ratemap that has been adaptively binned according to the
        algorithm described in Skaggs et al., 1996) [1]_.

        Parameters
        ----------
        pos_binned : array_like
            The binned positional data. For example that returned from getMap
            above with mapType as 'pos'
        spk_binned : array_like
            The binned spikes
        alpha : int, optional, default = 200
            A scaling parameter determing the amount of occupancy to aim at
            in each bin

        Returns
        -------
        Returns adaptively binned spike and pos maps. Use to generate Skaggs
        information measure

        Notes
        -----
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

        References
        ----------
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

        Parameters
        ----------
        A : array_like
            Either 2 or 3D. In the former it is simply the binned up ratemap
            where the two dimensions correspond to x and y.
            If 3D then the first two dimensions are x
            and y and the third (last dimension) is 'stack' of ratemaps
        nodwell : array_like
            A boolean array corresponding the bins in the ratemap that
            weren't visited. See Notes below.
        tol : float, optional
            Values below this are set to zero to deal with v small values
            thrown up by the fft. Default 1e-10

        Returns
        -------
        sac : array_like
            The spatial autocorrelation in the relevant dimensionality

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

        Parameters
        ----------
        A, B : array_like
            Either 2 or 3D. In the former it is simply the binned up ratemap
            where the two dimensions correspond to x and y.
            If 3D then the first two dimensions are x
            and y and the third (last dimension) is 'stack' of ratemaps
        nodwell_A, nodwell_B : array_like
            A boolean array corresponding the bins in the ratemap that
            weren't visited. See Notes below.
        tol : float, optional
            Values below this are set to zero to deal with v small values
            thrown up by the fft. Default 1e-10

        Returns
        -------

        sac : array_like
            The spatial crosscorrelation in the relevant dimensionality

        Notes
        -----
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

        Parameters
        ----------
        xy : array_like
            The position data
        spkIdx : array_like
            The indices in xy where the cell fired
        ppm : int, optional
            The camera pixels per metre. Default 365
        winSize : int, optional
            The window size for the temporal search
        pos_sample_rate : int, optional
            The rate at which position was sampled. Default 50
        nbins : int, optional
            The number of bins for creating the resulting ratemap. Default 71
        boxcar : int, optional
            The size of the smoothing kernel to smooth ratemaps. Default 5
        Pthresh : int, optional
            The cut=off for values in the ratemap; values < Pthresh become
            nans.
            Default 100
        downsampfreq : int, optional
            How much to downsample. Default 50
        plot : bool, optional
            Whether to show a plot of the result. Default False

        Returns
        -------
        H : array_like
            The temporal windowed SAC

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

    @memoized
    def _create_boundary_distance_lookup(self,
                                         arena_boundary: MultiLineString,
                                         degs_per_bin: float,
                                         xy_binsize: float,
                                         **kwargs):
        # Now we generate lines radiating out from a point as a 
        # multilinestring geometry collection - this looks like a 360/degs_per_bin
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
        pos_map, (ybin_edges, xbin_edges) = self.getMap(np.ones_like(self.dir))
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
            i_point = Point((xbin_edges[xi]+xy_binsize, ybin_edges[yi]+xy_binsize))
            ipx, ipy = i_point.xy
            new_point = Point(cx-ipx[0], cy-ipy[0])
            t_arena = translate(arena_boundary, -new_point.x, -new_point.y)
            prepare(t_arena)
            di = [(startpoint.distance(t_arena.intersection(line)), idx) for idx, line in enumerate(lines.geoms) if t_arena.intersects(line)]
            d, i = zip(*di)
            distances[xi, yi, i] = d
        return distances

    def plot_geometries(self, arena, point=None, lines=None):
        fig, ax = plt.subplots()
        arena_x, arena_y = arena.xy
        ax.plot(arena_x, arena_y, 'r')
        if point is not None:
            px, py = point.xy
            ax.plot(px, py, 'go')
        if lines is not None:
            ips = list(arena.intersection(lines).geoms)
            for line in list(lines.geoms):
                lx, ly = line.xy
                ax.plot(lx, ly, 'k')
            for ip in ips:
                ax.plot(ip.x, ip.y, 'ro')
        plt.show()

    def get_egocentric_boundary_map(self,
                                    degs_per_bin: float = 3,
                                    xy_binsize: float = 2.5,
                                    arena_type: str = "circle",
                                    method: int = 1,
                                    pos_weights=None) -> np.ndarray:
        '''
        Supposed to help construct dwell time/spike counts
        maps wrt boundaries at given egocentric directions
        and distances
        NB: for the directional input
        the 0 degree reference is horiztonal pointing
        East and moves counter-clockwise
        '''
        assert self.dir is not None, "No direction data available"
        import time
        # initially do some binning to get valid locations (some might be nans due to
        # arena shape and/or poor sampling) and then digitize the x and y positions
        # and the angular positions
        self.binsize = xy_binsize  # this will trigger a re-calculation of the bin edges

        # digitize the x and y positions and the angular positions
        x_binedges = self.binedges[1]
        y_binedges = self.binedges[0]
        xinds = np.digitize(self.xy[0], x_binedges)
        yinds = np.digitize(self.xy[1], y_binedges)
        angles = np.arange(0, 360, degs_per_bin)
        # pos_weights is used to increment the occupancy/ spike counts in the
        # final loop
        if pos_weights is None:
            pos_weights = np.ones_like(xinds)
        # Use the shaeply package to specify some geometry for the arena
        # boundary and the lines radiating out
        # from the current location of the animal. The geometry for the 
        # arena should be user specified but for now I'll just use a circle
        if arena_type == "circle":
            radius = 50
            circle_centre = Point(
                np.min(self.xy[0])+radius, np.min(self.xy[1])+radius)
            arena_boundary = circle_centre.buffer(radius).boundary
        # now we have a circle with its centre at the centre of the arena
        # i.e. the circle defines the arena edges. Calling .boundary on the
        # circle geometry actually gives us a 65-gon polygon
        start = time.time()
        distances = self._create_boundary_distance_lookup(
            arena_boundary, degs_per_bin, xy_binsize)
        end = time.time()
        print(f"Time to create lookup map: {end-start}")
        # iterate through the digitized locations (x/y and angular), using the
        # lookup table to get the distances to the arena boundary and then
        # increment the appropriate bin in the egocentric boundary map
        start = time.time()
        if method == 1:
            # pre-allocate the egocentric boundary map - made a bit bigger to
            # capture  the half open bin at the end
            ego_boundary_map = np.zeros([int(radius / xy_binsize)+1, len(angles)])
            for xi, yi, head_direction in zip(xinds-1, yinds-1, self.dir):
                dists = distances[xi, yi]
                valid_idx = np.isfinite(dists)
                walls_at_these_angles = np.roll(angles, int(head_direction/degs_per_bin))[valid_idx]
                dists_to_walls = dists[valid_idx]
                dist_idx_to_map = np.floor(dists_to_walls/xy_binsize).astype(int)
                ang_idx_to_map = np.floor(walls_at_these_angles/degs_per_bin).astype(int)
                linear_idx = np.ravel_multi_index([yi, xi], self.nBins)
                ego_boundary_map[dist_idx_to_map, ang_idx_to_map] += pos_weights[linear_idx]
        elif method == 2:
            xy_by_heading, _ = np.histogramdd([self.xy[0], self.xy[1], self.dir], bins=distances.shape, weights=pos_weights)
            assert xy_by_heading.shape == distances.shape
            distlist = []
            anglist = []
            for i_bin in np.ndindex(distances.shape[:2]):
                i_distances = distances[i_bin]
                valid_distances = np.isfinite(i_distances)
                nonzero_bincounts = np.nonzero(xy_by_heading[i_bin])[0]
                for i_angle in nonzero_bincounts:
                    ego_angles = np.roll(angles, i_angle)[valid_distances]
                    n_repeats = xy_by_heading[i_bin][i_angle]
                    ego_angles_repeats = np.repeat(ego_angles, n_repeats)
                    distances_repeats = np.repeat(i_distances[valid_distances], n_repeats)
                    distlist.append(distances_repeats)
                    anglist.append(ego_angles_repeats)
            flat_angles = flatten_list(anglist)
            flat_dists = flatten_list(distlist)
            ego_boundary_map, _, _ = np.histogram2d(x=flat_dists, y=flat_angles,
                                                 bins=[int(radius/xy_binsize), len(angles)])
        end = time.time()
        print(f"Time to get egocentric boundary map: {end-start}")
        return ego_boundary_map, distances

    def plot_egocentric_boundary_map(self, ego_map_occ: np.ndarray,
                                     ego_map_spk: np.ndarray,
                                     ax=None, **kwargs):
        '''
        Plots the egocentric boundary map
        '''
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        theta = np.arange(0, 2*np.pi, 2*np.pi/ego_map_occ.shape[1])
        phi = np.arange(0, ego_map_occ.shape[0]*2.5, 2.5)
        X, Y = np.meshgrid(theta, phi)
        occ_sm = blurImage(ego_map_occ, 5, 3, ftype='gaussian')
        spk_sm = blurImage(ego_map_spk, 5, 3, ftype='gaussian')
        ax.pcolormesh(X, Y, spk_sm/occ_sm, **kwargs)
        ax.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
        # ax.set_xticklabels(np.arange(0, 2*np.pi, np.pi/4))
        ax.set_yticks(np.arange(0, 50, 10))
        ax.set_yticklabels(np.arange(0, 50, 10))
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Distance (cm)')
        return ax
