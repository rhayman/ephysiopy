from ephysiopy.common import fieldcalcs


class SAC(object):
    """
    Spatial AutoCorrelation (SAC) class
    """

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
