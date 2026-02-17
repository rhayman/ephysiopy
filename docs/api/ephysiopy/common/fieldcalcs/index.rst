ephysiopy.common.fieldcalcs
===========================

.. py:module:: ephysiopy.common.fieldcalcs


Functions
---------

.. autoapisummary::

   ephysiopy.common.fieldcalcs.__deform_SAC
   ephysiopy.common.fieldcalcs.__get_circular_regions
   ephysiopy.common.fieldcalcs.__grid_orientation
   ephysiopy.common.fieldcalcs._get_field_labels
   ephysiopy.common.fieldcalcs.border_score
   ephysiopy.common.fieldcalcs.calc_angs
   ephysiopy.common.fieldcalcs.coherence
   ephysiopy.common.fieldcalcs.fancy_partition
   ephysiopy.common.fieldcalcs.fast_overdispersion
   ephysiopy.common.fieldcalcs.field_lims
   ephysiopy.common.fieldcalcs.field_props
   ephysiopy.common.fieldcalcs.filter_for_speed
   ephysiopy.common.fieldcalcs.filter_runs
   ephysiopy.common.fieldcalcs.get_all_phase
   ephysiopy.common.fieldcalcs.get_basic_gridscore
   ephysiopy.common.fieldcalcs.get_deformed_sac_gridscore
   ephysiopy.common.fieldcalcs.get_expanding_circle_gridscore
   ephysiopy.common.fieldcalcs.get_mean_resultant
   ephysiopy.common.fieldcalcs.get_mean_resultant_angle
   ephysiopy.common.fieldcalcs.get_mean_resultant_length
   ephysiopy.common.fieldcalcs.get_peak_coords
   ephysiopy.common.fieldcalcs.get_run
   ephysiopy.common.fieldcalcs.get_run_times
   ephysiopy.common.fieldcalcs.get_thigmotaxis_score
   ephysiopy.common.fieldcalcs.grid_field_props
   ephysiopy.common.fieldcalcs.gridness
   ephysiopy.common.fieldcalcs.infill_ratemap
   ephysiopy.common.fieldcalcs.kl_spatial_sparsity
   ephysiopy.common.fieldcalcs.kldiv
   ephysiopy.common.fieldcalcs.kldiv_dir
   ephysiopy.common.fieldcalcs.limit_to_one
   ephysiopy.common.fieldcalcs.simple_partition
   ephysiopy.common.fieldcalcs.skaggs_info
   ephysiopy.common.fieldcalcs.sort_fields_by_attr
   ephysiopy.common.fieldcalcs.spatial_sparsity


Module Contents
---------------

.. py:function:: __deform_SAC(A, circleXY=None, ellipseXY=None)

   
   Deforms an elliptical SAC to be circular

   :param A: The SAC
   :type A: np.ndarray
   :param circleXY: The xy coordinates defining a circle.
   :type circleXY: np.ndarray, default=None
   :param ellipseXY: The xy coordinates defining an ellipse.
   :type ellipseXY: np.ndarray, default=None

   :returns: The SAC deformed to be more circular
   :rtype: np.ndarray

   .. seealso:: :py:obj:`ephysiopy.common.ephys_generic.FieldCalcs.grid_field_props`, :py:obj:`skimage.transform.AffineTransform`, :py:obj:`skimage.transform.warp`, :py:obj:`skimage.exposure.rescale_intensity`















   ..
       !! processed by numpydoc !!

.. py:function:: __get_circular_regions(A, **kwargs)

   
   Returns a list of images which are expanding circular
   regions centred on the middle of the image out to the
   image edge and the radii used to create them.
   Used for calculating the grid score of each
   image to find the one with the max grid score.

   :param A: The SAC
   :type A: np.ndarray
   :param \*\*kwargs: min_radius (int): The smallest radius circle to start with

   :returns: A list of images which are circular sub-regions of the
             original SAC and a list of the radii used to create them
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: __grid_orientation(peakCoords, closestPeakIdx)

   
   Calculates the orientation angle of a grid field.

   The orientation angle is the angle of the first peak working
   counter-clockwise from 3 o'clock

   :param peakCoords: The peak coordinates as pairs of xy
   :type peakCoords: np.ndarray
   :param closestPeakIdx: A 1D array of the indices in peakCoords
                          of the peaks closest to the centre of the SAC
   :type closestPeakIdx: np.ndarray

   :returns: The first value in an array of the angles of
             the peaks in the SAC working counter-clockwise from a line
             extending from the middle of the SAC to 3 o'clock.
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: _get_field_labels(A, **kwargs)

   
   Returns a labeled version of A after finding the peaks
   in A and finding the watershed basins from the markers
   found from those peaks. Used in field_props() and
   grid_field_props()

   :param A: The array to process
   :type A: np.ndarray
   :param \*\*kwargs: min_distance (float, optional): The distance in bins between fields to
                      separate the regions of the image
                      clear_border (bool, optional): Input to skimage.feature.peak_local_max.
                      The number of pixels to ignore at the edge of the image















   ..
       !! processed by numpydoc !!

.. py:function:: border_score(A, B=None, shape='square', fieldThresh=0.3, circumPrc=0.2, binSize=3.0, minArea=200)

   
   Calculates a border score totally dis-similar to that calculated in
   Solstad et al (2008)

   :param A: the ratemap
   :type A: np.ndarray
   :param B: This should be a boolean mask where True (1)
             is equivalent to the presence of a border and False (0)
             is equivalent to 'open space'. Naievely this will be the
             edges of the ratemap but could be used to take account of
             boundary insertions/ creations to check tuning to multiple
             environmental boundaries. Default None: when the mask is
             None then a mask is created that has 1's at the edges of the
             ratemap i.e. it is assumed that occupancy = environmental
             shape
   :type B: np.ndarray, default None
   :param shape: description of environment shape. Currently
                 only 'square' or 'circle' accepted. Used to calculate the
                 proportion of the environmental boundaries to examine for
                 firing
   :type shape: str, default 'square'
   :param fieldThresh: Between 0 and 1 this is the percentage
                       amount of the maximum firing rate
                       to remove from the ratemap (i.e. to remove noise)
   :type fieldThresh: float, default 0.3
   :param circumPrc: The percentage amount of the circumference
                     of the environment that the field needs to be to count
                     as long enough to make it through
   :type circumPrc: float, default 0.2
   :param binSize: bin size in cm
   :type binSize: float, default 3.0
   :param minArea: min area for a field to be considered
   :type minArea: float, default 200

   :returns: the border score
   :rtype: float

   .. rubric:: Notes

   If the cell is a border cell (BVC) then we know that it should
   fire at a fixed distance from a given boundary (possibly more
   than one). In essence this algorithm estimates the amount of
   variance in this distance i.e. if the cell is a border cell this
   number should be small. This is achieved by first doing a bunch of
   morphological operations to isolate individual fields in the
   ratemap (similar to the code used in phasePrecession.py - see
   the fancy_partition method therein). These partitioned fields are then
   thinned out (using skimage's skeletonize) to a single pixel
   wide field which will lie more or less in the middle of the
   (highly smoothed) sub-field. It is the variance in distance from the
   nearest boundary along this pseudo-iso-line that is the boundary
   measure

   Other things to note are that the pixel-wide field has to have some
   minimum length. In the case of a circular environment this is set to
   20% of the circumference; in the case of a square environment markers
   this is at least half the length of the longest side















   ..
       !! processed by numpydoc !!

.. py:function:: calc_angs(points)

   
   Calculates the angles for all triangles in a delaunay tesselation of
   the peak points in the ratemap
















   ..
       !! processed by numpydoc !!

.. py:function:: coherence(smthd_rate, unsmthd_rate)

   
   Calculates the coherence of receptive field via correlation of smoothed
   and unsmoothed ratemaps

   :param smthd_rate: The smoothed rate map
   :type smthd_rate: np.ndarray
   :param unsmthd_rate: The unsmoothed rate map
   :type unsmthd_rate: np.ndarray

   :returns: The coherence of the rate maps
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: fancy_partition(binned_data, field_threshold_percent = 50, area_threshold_percent = 10)

   
   Another partitioning method

   :param binned_data - BinnedData:
   :param field_threshold_percent - int | float: pixels below this are set to zero and ignored
   :param area_threshold_percent - float: the expected minimum size of a receptive field















   ..
       !! processed by numpydoc !!

.. py:function:: fast_overdispersion(rmap, xy, spikes, **kws)

   
   Calculates the overdispersion of a spatial ratemap

   :param rmap: The spatial ratemap
   :type rmap: BinnedData
   :param xy: The xy data
   :type xy: np.ndarray
   :param spikes: The spike times binned into in samples. Length
                  will be equal to duration in seconds * sample_rate
   :type spikes: np.ndarray
   :param \*\*kws:
                   sample_rate : int, default=50
                       The sample rate of the position data
                   window : int, default=5
                       The window in seconds over which to calculate the overdispersion

   :returns: The overdispersion of the spatial ratemap
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: field_lims(A)

   
   Returns a labelled matrix of the ratemap A.
   Uses anything greater than the half peak rate to select as a field.
   Data is heavily smoothed.

   :param A: A BinnedData instance containing the ratemap
   :type A: BinnedData

   :returns: The labelled ratemap
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: field_props(A, min_dist=5, neighbours=2, prc=50, plot=False, ax=None, tri=False, verbose=True, **kwargs)

   
   Returns a dictionary of properties of the field(s) in a ratemap A

   :param A: a ratemap (but could be any image)
   :type A: array_like
   :param min_dist: the separation (in bins) between fields for measures
                    such as field distance to make sense. Used to
                    partition the image into separate fields in the call to
                    feature.peak_local_max
   :type min_dist: float
   :param neighbours: the number of fields to consider as neighbours to
                      any given field. Defaults to 2
   :type neighbours: int
   :param prc: percent of fields to consider
   :type prc: float
   :param ax: user supplied axis. If None a new figure window
   :type ax: matplotlib.Axes
   :param is created:
   :param tri: whether to do Delaunay triangulation between fields
               and add to plot
   :type tri: bool
   :param verbose: dumps the properties to the console
   :type verbose: bool
   :param plot: whether to plot some output - currently consists of the
                ratemap A, the fields of which are outline in a black
                contour. Default False
   :type plot: bool

   :returns: The properties of the field(s) in the input ratemap A
   :rtype: result (dict)















   ..
       !! processed by numpydoc !!

.. py:function:: filter_for_speed(field_props, min_speed)

   
   Mask for low speeds across the list of fields / runs

   :param field_props: The field properties to filter
   :type field_props: list of FieldProps
   :param min_speed: The minimum speed to keep a run
   :type min_speed: float

   :returns: The filtered field properties
   :rtype: list of FieldProps















   ..
       !! processed by numpydoc !!

.. py:function:: filter_runs(field_props, attributes, ops, vals, **kwargs)

   
   Filter out runs that are too short, too slow or have too few spikes

   :param field_props:
   :type field_props: list of FieldProps
   :param attributes: attributes of RunProps to filter on
   :type attributes: list of str
   :param ops: operations to use for filtering. Supported operations are
               np.less and np.greater
   :type ops: list of str
   :param vals: values to filter on
   :type vals: list of float

   :rtype: list of FieldProps

   .. rubric:: Notes

   this modifies the input list

   .. rubric:: Example

   >> field_props = filter_runs(field_props, ['n_spikes'], [np.greater], [5])

   field_props now only contains runs with more than 5 spikes















   ..
       !! processed by numpydoc !!

.. py:function:: get_all_phase(field_props)

   
   Get all the phases from the field properties

   :param field_props: The field properties to search through
   :type field_props: list of FieldProps

   :returns: An array of all the phases from all runs in all fields
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: get_basic_gridscore(A, **kwargs)

   
   Calculates the grid score of a spatial autocorrelogram

   :param A: The spatial autocorrelogram
   :type A: np.ndarray

   :returns: The grid score of the SAC
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: get_deformed_sac_gridscore(A)

   
   Deforms a non-circular SAC into a circular SAC (circular meaning
   the ellipse drawn around the edges of the 6 nearest peaks to the
   SAC centre) and returns get_basic_griscore() calculated on the
   deformed (or re-formed?!) SAC

   :param A: The SAC
   :type A: np.ndarray

   :returns: The gridscore of the deformed SAC
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: get_expanding_circle_gridscore(A, **kwargs)

   
   Calculates the gridscore for each circular sub-region of image A
   where the circles are centred on the image centre and expanded to
   the edge of the image. The maximum of the get_basic_gridscore() for
   each of these circular sub-regions is returned as the gridscore

   :param A: The SAC
   :type A: np.ndarray

   :returns: The maximum grid score of the circular sub
             regions of the SAC
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: get_mean_resultant(ego_boundary_map)

   
   Calculates the mean resultant vector of a boundary map in egocentric coordinates

   :param ego_boundary_map: The egocentric boundary map
   :type ego_boundary_map: np.ndarray

   :returns: The mean resultant vector of the egocentric boundary map
   :rtype: float

   .. rubric:: Notes

   See Hinman et al., 2019 for more details















   ..
       !! processed by numpydoc !!

.. py:function:: get_mean_resultant_angle(ego_boundary_map, **kwargs)

   
   Calculates the angle of the mean resultant vector of a
   boundary map in egocentric coordinates

   :param ego_boundary_map: The egocentric boundary map
   :type ego_boundary_map: np.ndarray

   :returns: The angle mean resultant vector of the egocentric boundary map
   :rtype: float

   .. rubric:: Notes

   See Hinman et al., 2019 for more details















   ..
       !! processed by numpydoc !!

.. py:function:: get_mean_resultant_length(ego_boundary_map, **kwargs)

   
   Calculates the length of the mean resultant vector of a
   boundary map in egocentric coordinates

   :param ego_boundary_map: The egocentric boundary map
   :type ego_boundary_map: np.ndarray

   :returns: The length of the mean resultant vector of the egocentric boundary map
   :rtype: float

   .. rubric:: Notes

   See Hinman et al., 2019 for more details















   ..
       !! processed by numpydoc !!

.. py:function:: get_peak_coords(rmap, labels)

   
   Get the peak coordinates of the firing fields in the ratemap
















   ..
       !! processed by numpydoc !!

.. py:function:: get_run(field_props, run_num)

   
   Get a specific run from the field properties

   :param field_props: The field properties to search through
   :type field_props: list of FieldProps
   :param run_num: The run number to search for
   :type run_num: int

   :returns: The run properties for the specified run number
   :rtype: RunProps















   ..
       !! processed by numpydoc !!

.. py:function:: get_run_times(field_props)

   
   Get the run start and stop times in seconds for all runs
   through all fields in the field_props list
















   ..
       !! processed by numpydoc !!

.. py:function:: get_thigmotaxis_score(xy, shape = 'circle')

   
   Returns a score which is the ratio of the time spent in the inner
   portion of an environment to the time spent in the outer portion.
   The portions are allocated so that they have equal area.

   :param xy: The xy coordinates of the animal's position. 2 x nsamples
   :type xy: np.ndarray
   :param shape: The shape of the environment. Legal values are 'circle' and 'square'
   :type shape: str, default='circle'

   :returns: Values closer to 1 mean more time was spent in the inner portion of the environment.
             Values closer to -1 mean more time in the outer portion of the environment.
             A value of 0 indicates the animal spent equal time in both portions of the
             environment.
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: grid_field_props(A, maxima='centroid', allProps=True, **kwargs)

   
   Extracts various measures from a spatial autocorrelogram

   :param A:
             object containing the spatial autocorrelogram (SAC) in
                 A.binned_data[0]
   :type A: BinnedData
   :param maxima (str: Legal values are 'single' and 'centroid'. Default 'centroid'
   :type maxima (str: The method used to detect the peaks in the SAC.
   :param optional): Legal values are 'single' and 'centroid'. Default 'centroid'
   :type optional): The method used to detect the peaks in the SAC.
   :param allProps: Whether to return a dictionary that contains the attempt to fit
                    an ellipse around the edges of the central size peaks. See below
   :type allProps: bool default=True

   :returns: Measures of the SAC.
             Keys include:
                 * gridness score
                 * scale
                 * orientation
                 * coordinates of the peaks (nominally 6) closest to SAC centre
                 * a binary mask around the extent of the 6 central fields
                 * values of the rotation procedure used to calculate gridness
                 * ellipse axes and angle (if allProps is True and the it worked)
   :rtype: dict

   .. rubric:: Notes

   The output from this method can be used as input to the show() method
   of this class.
   When it is the plot produced will display a lot more informative.
   The coordinate system internally used is centred on the image centre.

   .. seealso:: :py:obj:`ephysiopy.common.binning.autoCorr2D`















   ..
       !! processed by numpydoc !!

.. py:function:: gridness(image, step=30)

   
   Calculates the gridness score in a grid cell SAC.

   The data in `image` is rotated in `step` amounts and
   each rotated array is correlated with the original.
   The maximum of the values at 30, 90 and 150 degrees
   is the subtracted from the minimum of the values at 60, 120
   and 180 degrees to give the grid score.

   :param image: The spatial autocorrelogram
   :type image: np.ndarray
   :param step: The amount to rotate the SAC in each step of the
                rotational correlation procedure
   :type step: int, default=30

   :returns: The gridscore, the correlation values at each
             `step` and the rotational array
   :rtype: 3-tuple

   .. rubric:: Notes

   The correlation performed is a Pearsons R. Some rescaling of the
   values in `image` is performed following rotation.

   .. seealso::

      :py:obj:`skimage.transform.rotate`
          for how the rotation of `image` is done

      :py:obj:`skimage.exposure.rescale_intensity`
          for the resscaling following

      :py:obj:`rotation`















   ..
       !! processed by numpydoc !!

.. py:function:: infill_ratemap(rmap)

   
   The ratemaps used in the phasePrecession2D class are a) super smoothed and
   b) very large i.e. the bins per cm is low. This
   results in firing fields that have lots of holes (nans) in them. We want to
   smooth over these holes so we can construct measures such as the expected
   rate in a given bin whilst also preserving whatever 'geometry' of the
   environment exists in the ratemap as a result of where position has been
   sampled. That is, if non-sampled positions are designated with nans, we
   want to smooth over those that in theory could have been sampled and keep
   those that never could have been.

   :param rmap: The ratemap to be filled
   :type rmap: np.ndarray

   :returns: The filled ratemap
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: kl_spatial_sparsity(pos_map)

   
   Calculates the spatial sampling of an arena by comparing the
   observed spatial sampling to an expected uniform one using kl divergence

   Data in pos_map should be unsmoothed (not checked) and the MapType should
   be POS (checked)

   :param pos_map: The position map
   :type pos_map: BinnedData

   :returns: The spatial sparsity of the position map
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: kldiv(X, pvect1, pvect2, variant = '')

   
   Calculates the Kullback-Leibler or Jensen-Shannon divergence between
   two distributions.

   :param X: Vector of M variable values
   :type X: np.ndarray
   :param P1: Length-M vectors of probabilities representing distribution 1 and 2
   :type P1: np.ndarray
   :param P2: Length-M vectors of probabilities representing distribution 1 and 2
   :type P2: np.ndarray
   :param variant: If 'sym', returns a symmetric variant of the
                   Kullback-Leibler divergence, given by [KL(P1,P2)+KL(P2,P1)]/2
                   If 'js', returns the Jensen-Shannon divergence, given by
                   [KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2
   :type variant: str, default 'sym'

   :returns: The Kullback-Leibler divergence or Jensen-Shannon divergence
   :rtype: float

   .. rubric:: Notes

   The Kullback-Leibler divergence is given by:

   .. math:: KL(P1(x),P2(x)) = sum_[P1(x).log(P1(x)/P2(x))]

   If X contains duplicate values, there will be an warning message,
   and these values will be treated as distinct values.  (I.e., the
   actual values do not enter into the computation, but the probabilities
   for the two duplicate values will be considered as probabilities
   corresponding to two unique values.).
   The elements of probability vectors P1 and P2 must
   each sum to 1 +/- .00001.

   This function is taken from one on the Mathworks file exchange

   .. seealso::

      Cover, T.M. and J.A. Thomas. "Elements of Information Theory," Wiley,
      1991.

      https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence















   ..
       !! processed by numpydoc !!

.. py:function:: kldiv_dir(polarPlot)

   
   Returns a kl divergence for directional firing: measure of directionality.
   Calculates kl diveregence between a smoothed ratemap (probably should be
   smoothed otherwise information theoretic measures
   don't 'care' about position of bins relative to one another) and a
   pure circular distribution.
   The larger the divergence the more tendancy the cell has to fire when the
   animal faces a specific direction.

   :param polarPlot np.ndarray: The binned and smoothed directional ratemap

   :returns: The divergence from circular of the 1D-array
             from a uniform circular distribution
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: limit_to_one(A, prc=50, min_dist=5)

   
   Processes a multi-peaked ratemap and returns a matrix
   where the multi-peaked ratemap consist of a single peaked field that is
   a) not connected to the border and b) close to the middle of the
   ratemap

   :param A: The ratemap
   :type A: np.ndarray
   :param prc: The percentage of the peak rate to threshold the ratemap at
   :type prc: int
   :param min_dist: The minimum distance between peaks
   :type min_dist: int

   :returns: RegionProperties of the fields (list of RegionProperties)
             The single peaked ratemap (np.ndarray)
             The index of the field (int)
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: simple_partition(binned_data, rate_threshold_prc = 200, **kwargs)

   
   Simple partitioning of fields based on mean firing rate. Only
   returns a single field (the highest firing rate field) per
   binned_data instance

   The default is to limit to fields that have a mean firing rate
   greater than twice the mean firing rate of the entire
   ratemap

   :param binned_data: an instance of ephysiopy.common.utils.BinnedData
   :type binned_data: BinnedData
   :param rate_threshold_prc: removes pixels in a field that fall below this percent of
                              the mean firing rate
   :type rate_threshold_prc: int

   :returns: peaksXY - The xy coordinates of the peak rates in
             the highest firing field
             peaksRate - The peak rates in peaksXY
             labels - An array of the labels corresponding to the highest firing field
             rmap_filled - The filled ratemap of the tetrode / cluster
   :rtype: tuple of np.ndarray

   .. rubric:: Notes

   This is a simple method to partition fields that only returns
   a single field - the one with the highest mean firing rate.















   ..
       !! processed by numpydoc !!

.. py:function:: skaggs_info(ratemap, dwelltimes, **kwargs)

   
   Calculates Skaggs information measure

   :param ratemap: The binned up ratemap and dwelltimes. Must be the same size
   :type ratemap: np.ndarray
   :param dwelltimes: The binned up ratemap and dwelltimes. Must be the same size
   :type dwelltimes: np.ndarray

   :returns: Skaggs information score in bits spike
   :rtype: float

   .. rubric:: Notes

   The ratemap data should have undergone adaptive binning as per
   the original paper. See getAdaptiveMap() in binning class

   The estimate of spatial information in bits per spike:

   .. math:: I = sum_{x} p(x).r(x).log(r(x)/r)















   ..
       !! processed by numpydoc !!

.. py:function:: sort_fields_by_attr(field_props, attr='area', reverse=True)

   
   Sorts the fields in the list by attribute

   .. rubric:: Notes

   In the default case will sort by area, largest first















   ..
       !! processed by numpydoc !!

.. py:function:: spatial_sparsity(rate_map, pos_map)

   
   Calculates the spatial sparsity of a rate map as defined by
   Markus et al (1994)

   For example, a sparsity score of 0.10 indicates that the cell fired on
   10% of the maze surface

   :param rate_map: The rate map
   :type rate_map: np.ndarray
   :param pos_map: The occupancy map
   :type pos_map: np.ndarray

   :returns: The spatial sparsity of the rate map
   :rtype: float

   .. rubric:: References

   Markus, E.J., Barnes, C.A., McNaughton, B.L., Gladden, V.L. &
   Skaggs, W.E. Spatial information content and reliability of
   hippocampal CA1 neurons: effects of visual input. Hippocampus
   4, 410–421 (1994).















   ..
       !! processed by numpydoc !!

