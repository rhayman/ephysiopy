import numpy as np
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
    
    def get_basic_gridscore(self, A: np.ndarray, step: int=30, **kwargs):
        '''
        Rotates the image A in step amounts, correlated each rotated image
        with the original. The maximum of the values at 30, 90 and 150 degrees
        is the subtracted from the minimum of the values at 60, 120
        and 180 degrees to give the grid score.
        '''
        from ephysiopy.common.fieldcalcs import gridness
        return gridness(A, step)[0]
    
    def get_expanding_circle_gridscore(self, A: np.ndarray, **kwargs):
        '''
        Calculates the gridscore for each circular sub-region of image A
        where the circles are centred on the image centre and expanded to
        the edge of the image. The maximum of the get_basic_gridscore() for
        each of these circular sub-regions is returned as the gridscore
        '''

        from ephysiopy.common.fieldcalcs import get_circular_regions
        images = get_circular_regions(A, **kwargs)
        gridscores = [self.get_basic_gridscore(im) for im in images]
        return max(gridscores)
    
    def get_deformed_sac_gridscore(self, A: np.ndarray, **kwargs):
        '''
        Deforms a non-circular SAC into a circular SAC (circular meaning
        the ellipse drawn around the edges of the 6 nearest peaks to the
        SAC centre) and returns get_basic_griscore() calculated on the 
        deformed (or re-formed?!) SAC
        '''
        from ephysiopy.common.fieldcalcs import deform_SAC
        deformed_SAC, _ = deform_SAC(A)
        return self.get_basic_gridscore(deformed_SAC)

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
