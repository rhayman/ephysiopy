from ephysiopy.common.binning import RateMap
from ephysiopy.common import fieldcalcs


class SAC(object):
    """
    Spatial AutoCorrelation (SAC) class
    """
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
        In order to maintain backward compatibility I've kept this method
        here as a
        wrapper into ephysiopy.common.binning.RateMap.autoCorr2D()

        See Also
        --------
        ephysiopy.common.binning.RateMap.autoCorr2D()

        """
        R = RateMap()
        return R.autoCorr2D(A, nodwell, tol)

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
        In order to maintain backward compatibility I've kept this method here
        as a
        wrapper into ephysiopy.common.binning.RateMap.autoCorr2D()

        """
        R = RateMap()
        return R.crossCorr2D(A, B, A_nodwell, B_nodwell, tol)

    def t_win_SAC(
            self, xy, spkIdx, ppm=365, winSize=10, pos_sample_rate=50,
            nbins=71, boxcar=5, Pthresh=100, downsampfreq=50, plot=False):
        """
        Temporal windowed spatial autocorrelation.
        For rationale see Notes below

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

        Notes
        -----
        In order to maintain backward compatibility I've kept this method here
        as a
        wrapper into ephysiopy.common.binning.RateMap.crossCorr2D()

        """
        R = RateMap()
        return R.tWinSAC(
            xy, spkIdx, ppm, winSize, pos_sample_rate, nbins,
            boxcar, Pthresh, downsampfreq, plot)

    def getMeasures(
            self, A, maxima='centroid',
            allProps=True, **kwargs):
        """
        Extracts various measures from a spatial autocorrelogram

        Parameters
        ----------
        A : array_like
            The spatial autocorrelogram (SAC)
        maxima : str, optional
            The method used to detect the peaks in the SAC.
            Legal values are 'single' and 'centroid'. Default 'centroid'
        field_extent_method : int, optional
            The method used to delimit the regions of interest in the SAC
            Legal values:
            * 1 - uses the half height of the ROI peak to limit field extent
            * 2 - uses a watershed method to limit field extent
            Default 2
        allProps : bool, optional
            Whether to return a dictionary that contains the attempt to fit an
            ellipse around the edges of the central size peaks. See below
            Default True

        Returns
        -------
        props : dict
            A dictionary containing measures of the SAC. The keys include:
            * gridness score
            * scale
            * orientation
            * the coordinates of the peaks (nominally 6) closest to  SAC centre
            * a binary mask that defines the extent of the 6 central fields
            * values of the rotation procedure used to calculate gridness
            * ellipse axes and angle (if allProps is True and the it worked)

        Notes
        -----
        In order to maintain backward comaptibility this is a wrapper for
        ephysiopy.common.ephys_generic.FieldCalcs.grid_field_props()

        See Also
        --------
        ephysiopy.common.ephys_generic.FieldCalcs.grid_field_props()

        """
        return fieldcalcs.grid_field_props(
            A, maxima, allProps, **kwargs)

    def show(self, A, inDict, ax=None, **kwargs):
        """
        Displays the result of performing a spatial autocorrelation (SAC)
        on a grid cell.

        Uses the dictionary containing measures of the grid cell SAC to
        make a pretty picture

        Parameters
        ----------
        A : array_like
            The spatial autocorrelogram
        inDict : dict
            The dictionary calculated in getmeasures
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            If given the plot will get drawn in these axes. Default None

        Returns
        -------
        fig : matplotlib.Figure instance
            The Figure on which the SAC is shown

        See Also
        --------
        ephysiopy.common.binning.RateMap.autoCorr2D()
        ephysiopy.common.ephys_generic.FieldCalcs.getMeaures()
        """
        from ephysiopy.visualise.plotting import FigureMaker
        F = FigureMaker()
        F.show_SAC(A, inDict, ax, **kwargs)
