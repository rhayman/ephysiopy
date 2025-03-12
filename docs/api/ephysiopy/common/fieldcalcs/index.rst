ephysiopy.common.fieldcalcs
===========================

.. py:module:: ephysiopy.common.fieldcalcs


Attributes
----------

.. autoapisummary::

   ephysiopy.common.fieldcalcs.PROPS
   ephysiopy.common.fieldcalcs.PROP_VALS


Classes
-------

.. autoapisummary::

   ephysiopy.common.fieldcalcs.FieldProps
   ephysiopy.common.fieldcalcs.LFPSegment
   ephysiopy.common.fieldcalcs.RunProps


Functions
---------

.. autoapisummary::

   ephysiopy.common.fieldcalcs._get_field_labels
   ephysiopy.common.fieldcalcs.border_score
   ephysiopy.common.fieldcalcs.calc_angs
   ephysiopy.common.fieldcalcs.coherence
   ephysiopy.common.fieldcalcs.deform_SAC
   ephysiopy.common.fieldcalcs.field_lims
   ephysiopy.common.fieldcalcs.field_props
   ephysiopy.common.fieldcalcs.fieldprops
   ephysiopy.common.fieldcalcs.get_basic_gridscore
   ephysiopy.common.fieldcalcs.get_circular_regions
   ephysiopy.common.fieldcalcs.get_deformed_sac_gridscore
   ephysiopy.common.fieldcalcs.get_expanding_circle_gridscore
   ephysiopy.common.fieldcalcs.get_mean_resultant
   ephysiopy.common.fieldcalcs.get_mean_resultant_angle
   ephysiopy.common.fieldcalcs.get_mean_resultant_length
   ephysiopy.common.fieldcalcs.get_thigmotaxis_score
   ephysiopy.common.fieldcalcs.global_threshold
   ephysiopy.common.fieldcalcs.grid_field_props
   ephysiopy.common.fieldcalcs.grid_orientation
   ephysiopy.common.fieldcalcs.gridness
   ephysiopy.common.fieldcalcs.infill_ratemap
   ephysiopy.common.fieldcalcs.kl_spatial_sparsity
   ephysiopy.common.fieldcalcs.kldiv
   ephysiopy.common.fieldcalcs.kldiv_dir
   ephysiopy.common.fieldcalcs.limit_to_one
   ephysiopy.common.fieldcalcs.local_threshold
   ephysiopy.common.fieldcalcs.partitionFields
   ephysiopy.common.fieldcalcs.reduce_labels
   ephysiopy.common.fieldcalcs.skaggs_info
   ephysiopy.common.fieldcalcs.spatial_sparsity


Module Contents
---------------

.. py:class:: FieldProps(slice, label, label_image, binned_data, cache, *, extra_properties, spacing, offset, index=0)

   Bases: :py:obj:`skimage.measure._regionprops.RegionProperties`


   
   Please refer to `skimage.measure.regionprops` for more information
   on the available region properties.
















   ..
       !! processed by numpydoc !!

   .. py:method:: __getattr__(attr)


   .. py:method:: __str__()

      
      Override the string representation printed to the console
















      ..
          !! processed by numpydoc !!


   .. py:method:: overdispersion(spike_train, sample_rate = 50)


   .. py:method:: perimeter_minus_field_max()


   .. py:method:: r_per_run()


   .. py:method:: runs_expected_spikes(expected_rate_at_pos, sample_rate = 50)


   .. py:method:: smooth_runs(k, spatial_lp_cut, sample_rate)

      
      Smooth in x and y in preparation for converting the smoothed cartesian
      coordinates to polar ones

      :param k (float) - smoothing constant for the instantaneous firing rate:
      :param spatial_lp_cut (int) - spatial lowpass cut off:
      :param sample_rate (int) - position sample rate in Hz:















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _runs
      :value: []



   .. py:property:: bin_coords
      :type: numpy.ndarray



   .. py:attribute:: binned_data


   .. py:property:: bw_perim
      :type: numpy.ndarray



   .. py:property:: cumulative_distance
      :type: numpy.ndarray



   .. py:property:: cumulative_time
      :type: numpy.ndarray



   .. py:property:: current_direction
      :type: numpy.ndarray



   .. py:property:: global_perimeter_coords
      :type: numpy.ndarray



   .. py:property:: intensity_max
      :type: float



   .. py:property:: intensity_mean
      :type: float



   .. py:property:: intensity_min
      :type: float



   .. py:property:: intensity_std
      :type: float



   .. py:property:: max_index
      :type: numpy.ndarray



   .. py:property:: num_runs
      :type: int



   .. py:property:: perimeter_angle_from_peak
      :type: numpy.ndarray



   .. py:property:: perimeter_coords
      :type: tuple



   .. py:property:: perimeter_dist_from_peak
      :type: numpy.ndarray



   .. py:property:: phi
      :type: numpy.ndarray


      
      Calculate the angular distance between the mean direction of each run and
      each position samples direction to the field centre
















      ..
          !! processed by numpydoc !!


   .. py:property:: pos_phi
      :type: numpy.ndarray


      
      Calculate the angular distance between the mean direction of each run and
      each position samples direction to the field centre
















      ..
          !! processed by numpydoc !!


   .. py:property:: pos_r
      :type: numpy.ndarray


      
      Calculate the ratio of the distance from the field peak to the position sample
      and the distance from the field peak to the point on the perimeter that is most
      colinear with the position sample

      NB The values just before being returned can be >= 1 so these are capped to 1















      ..
          !! processed by numpydoc !!


   .. py:property:: pos_xy
      :type: numpy.ndarray



   .. py:property:: projected_direction
      :type: numpy.ndarray


      
      direction projected onto the mean run direction is just the x-coord
      when cartesian x and y is converted to from polar rho and phi
















      ..
          !! processed by numpydoc !!


   .. py:property:: r_and_phi_to_x_and_y
      :type: numpy.ndarray



   .. py:property:: rho
      :type: numpy.ndarray



   .. py:property:: run_labels


   .. py:property:: run_slices


   .. py:property:: runs


   .. py:property:: runs_observed_spikes
      :type: numpy.ndarray



   .. py:property:: runs_speed
      :type: numpy.ndarray



   .. py:property:: spike_position_index


   .. py:property:: xy_angle_to_peak
      :type: numpy.ndarray



   .. py:property:: xy_at_peak
      :type: numpy.ndarray



   .. py:property:: xy_coords
      :type: numpy.ndarray



   .. py:property:: xy_dist_to_peak
      :type: numpy.ndarray



   .. py:property:: xy_relative_to_peak
      :type: numpy.ndarray



.. py:class:: LFPSegment(field_label, run_label, slice, spike_times, signal, filtered_signal, phase, amplitude, sample_rate, filter_band)

   Bases: :py:obj:`object`


   
   A custom class for dealing with segments of an LFP signal and how
   they relate to specific runs (see RunProps below) through a
   receptive field (see FieldProps below)

   .. attribute:: field_label

      :type: int

   .. attribute:: run_label

      :type: int

   .. attribute:: slice

      :type: slice

   .. attribute:: spike_times

      :type: np.ndarray

   .. attribute:: signal

      :type: np.ndarray

   .. attribute:: filtered_signal

      :type: np.ndarray

   .. attribute:: phase

      :type: np.ndarray

   .. attribute:: amplitude

      :type: np.ndarray

   .. attribute:: sample_rate

      :type: float, int

   .. attribute:: filter_band

      :type: tuple[int,int]















   ..
       !! processed by numpydoc !!

   .. py:attribute:: amplitude


   .. py:attribute:: field_label


   .. py:attribute:: filter_band


   .. py:attribute:: filtered_signal


   .. py:attribute:: phase


   .. py:attribute:: run_label


   .. py:attribute:: sample_rate


   .. py:attribute:: signal


   .. py:attribute:: slice


   .. py:attribute:: spike_times


.. py:class:: RunProps(label, slice, xy_coords, spike_count, speed, peak_xy, max_index, perimeter_coords)

   Bases: :py:obj:`object`


   
   A custom class for holding information about runs through a receptive field

   Each run needs to have some information about the field to which it belongs
   so the constructor takes in the peak x-y coordinate of the field and its index
   as well as the coordinates of the perimeter of the field

   .. attribute:: label

      :type: int

   .. attribute:: slice

      :type: slice

   .. attribute:: xy_coords

      :type: np.ndarray

   .. attribute:: spike_count

      :type: np.ndarray

   .. attribute:: speed

      :type: np.ndarray

   .. attribute:: peak_xy

      :type: tuple[float, float]

   .. attribute:: max_index

      :type: int

   .. attribute:: perimeter_coords

      :type: np.ndarray

   .. attribute:: hdir

      the heading direction

      :type: np.ndarray

   .. attribute:: min_speed

      :type: float

   .. attribute:: cumulative_time

      :type: np.ndarray

   .. attribute:: duration

      :type: int

   .. attribute:: n_spikes

      :type: int

   .. attribute:: run_start

      :type: int

   .. attribute:: run_stop

      :type: int

   .. attribute:: mean_direction

      :type: float

   .. attribute:: current_direction

      :type: np.ndarray

   .. attribute:: cumulative_distance

      :type: np.ndarray

   .. attribute:: spike_position_index

      :type: np.ndarray

   .. attribute:: observed_spikes

      :type: np.ndarray

   .. attribute:: xy_angle_to_peak

      :type: np.ndarray

   .. attribute:: xy_dist_to_peak

      :type: np.ndarray

   .. attribute:: xy_dist_to_peak_normed

      :type: np.ndarray

   .. attribute:: pos_xy

      :type: np.ndarray

   .. attribute:: pos_phi

      :type: np.ndarray

   .. attribute:: rho

      :type: np.ndarray

   .. attribute:: phi

      :type: np.ndarray

   .. attribute:: r_and_phi_to_x_and_y

      :type: np.ndarray

   .. attribute:: tortuosity

      :type: np.ndarray

   .. attribute:: xy_is_smoothed

      :type: bool















   ..
       !! processed by numpydoc !!

   .. py:method:: __len__()


   .. py:method:: __str__()


   .. py:method:: expected_spikes(expected_rate_at_pos, sample_rate = 50)

      
      Calculates the expected number of spikes along this run given the
      whole ratemap.

      :param expected_rate_at_pos: the rate seen at each xy position of the whole trial
      :type expected_rate_at_pos: np.ndarray

      :returns: **expected_rate** -- the expected rate at each xy position of this run
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: overdispersion(spike_train, sample_rate = 50)

      
      The overdispersion map for this run

      :param spike_train: the spike train (spikes binned up by position) for the whole trial. Same
                          length as the trial n_samples
      :type spike_train: np.mdarray
      :param sample_rate:
      :type sample_rate: int















      ..
          !! processed by numpydoc !!


   .. py:method:: perimeter_angle_from_peak()


   .. py:method:: perimeter_minus_field_max()


   .. py:method:: smooth_xy(k, spatial_lp_cut, sample_rate)

      
      Smooth in x and y in preparation for converting the smoothed cartesian
      coordinates to polar ones

      :param k: smoothing constant for the instantaneous firing rate
      :type k: float
      :param spatial_lp_cut: spatial lowpass cut off
      :type spatial_lp_cut: int
      :param sample_rate: position sample rate in Hz
      :type sample_rate: int















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _max_index


   .. py:attribute:: _peak_xy


   .. py:attribute:: _perimeter_coords


   .. py:attribute:: _slice


   .. py:attribute:: _speed


   .. py:attribute:: _spike_count


   .. py:attribute:: _xy_coords


   .. py:property:: cumulative_distance


   .. py:property:: cumulative_time
      :type: numpy.ndarray



   .. py:property:: current_direction


   .. py:property:: duration


   .. py:property:: hdir


   .. py:attribute:: label


   .. py:property:: mean_direction


   .. py:property:: min_speed


   .. py:property:: n_spikes


   .. py:property:: observed_spikes


   .. py:property:: phi


   .. py:property:: pos_phi


   .. py:property:: pos_r


   .. py:property:: pos_xy


   .. py:property:: r_and_phi_to_x_and_y


   .. py:property:: rho


   .. py:property:: run_start


   .. py:property:: run_stop


   .. py:property:: spike_position_index


   .. py:property:: tortuosity


   .. py:property:: xy


   .. py:property:: xy_angle_to_peak


   .. py:property:: xy_dist_to_peak


   .. py:property:: xy_dist_to_peak_normed


   .. py:attribute:: xy_is_smoothed
      :value: False



.. py:function:: _get_field_labels(A, **kwargs)

   
   Returns a labeled version of A after finding the peaks
   in A and finding the watershed basins from the markers
   found from those peaks. Used in field_props() and
   grid_field_props()

   :param A: The array to process
   :type A: np.ndarray
   :param min_distance: The distance in bins between fields to
   :type min_distance: float, optional
   :param separate the regions of the image:
   :param clear_border: Input to skimage.feature.peak_local_max.
   :type clear_border: bool, optional
   :param The number of: pixels to ignore at the edge of the image















   ..
       !! processed by numpydoc !!

.. py:function:: border_score(A, B=None, shape='square', fieldThresh=0.3, circumPrc=0.2, binSize=3.0, minArea=200)

   
   Calculates a border score totally dis-similar to that calculated in
   Solstad et al (2008)

   :param A: Should be the ratemap
   :type A: array_like
   :param B: This should be a boolean mask where True (1)
             is equivalent to the presence of a border and False (0)
             is equivalent to 'open space'. Naievely this will be the
             edges of the ratemap but could be used to take account of
             boundary insertions/ creations to check tuning to multiple
             environmental boundaries. Default None: when the mask is
             None then a mask is created that has 1's at the edges of the
             ratemap i.e. it is assumed that occupancy = environmental
             shape
   :type B: array_like
   :param shape: description of environment shape. Currently
                 only 'square' or 'circle' accepted. Used to calculate the
                 proportion of the environmental boundaries to examine for
                 firing
   :type shape: str
   :param fieldThresh: Between 0 and 1 this is the percentage
                       amount of the maximum firing rate
                       to remove from the ratemap (i.e. to remove noise)
   :type fieldThresh: float
   :param smthKernSig: the sigma value used in smoothing the ratemap
                       (again!) with a gaussian kernel
   :type smthKernSig: float
   :param circumPrc: The percentage amount of the circumference
                     of the environment that the field needs to be to count
                     as long enough to make it through
   :type circumPrc: float
   :param binSize: bin size in cm
   :type binSize: float
   :param minArea: min area for a field to be considered
   :type minArea: float
   :param debug: If True then some plots and text will be output
   :type debug: bool

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
   the partitionFields method therein). These partitioned fields are then
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

   
   calculates coherence of receptive field via correlation of smoothed
   and unsmoothed ratemaps
















   ..
       !! processed by numpydoc !!

.. py:function:: deform_SAC(A, circleXY=None, ellipseXY=None)

   
   Deforms a SAC that is non-circular to be more circular

   Basically a blatant attempt to improve grid scores, possibly
   introduced in a paper by Matt Nolan...

   :param A: The SAC
   :type A: array_like
   :param circleXY: The xy coordinates defining a circle.
   :type circleXY: array_like, optional
   :param Default None.:
   :param ellipseXY: The xy coordinates defining an
   :type ellipseXY: array_like, optional
   :param ellipse. Default None.:

   :returns: The SAC deformed to be more circular
   :rtype: deformed_sac (array_like)

   .. seealso::

      ephysiopy.common.ephys_generic.FieldCalcs.grid_field_props
      skimage.transform.AffineTransform
      skimage.transform.warp
      skimage.exposure.rescale_intensity















   ..
       !! processed by numpydoc !!

.. py:function:: field_lims(A)

   
   Returns a labelled matrix of the ratemap A.
   Uses anything greater than the half peak rate to select as a field.
   Data is heavily smoothed.

   :param A: A BinnedData instance containing the ratemap
   :type A: BinnedData

   :returns: The labelled ratemap
   :rtype: label (np.array)















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

.. py:function:: fieldprops(label_image, binned_data, xy, spikes_per_pos, cache=True, *, extra_properties=None, spacing=None, offset=None, **kwargs)

   
   Measure properties of labeled image regions.

   :param label_image: Labeled input image. Labels with value 0 are ignored.

                       .. versionchanged:: 0.14.1
                           Previously, ``label_image`` was processed by ``numpy.squeeze`` and
                           so any number of singleton dimensions was allowed. This resulted in
                           inconsistent handling of images with singleton dimensions. To
                           recover the old behaviour, use
                           ``regionprops(np.squeeze(label_image), ...)``.
   :type label_image: (M, N[, P]) ndarray
   :param xy: The x-y coordinates for all runs through the field corresponding to
              a particular label
   :type xy: (2 x N) ndarray
   :param binned_data:
   :type binned_data: BinnedData instance from ephysiopy.common.utils
   :param cache: Determine whether to cache calculated properties. The computation is
                 much faster for cached properties, whereas the memory consumption
                 increases.
   :type cache: bool, optional
   :param extra_properties: Add extra property computation functions that are not included with
                            skimage. The name of the property is derived from the function name,
                            the dtype is inferred by calling the function on a small sample.
                            If the name of an extra property clashes with the name of an existing
                            property the extra property will not be visible and a UserWarning is
                            issued. A property computation function must take a region mask as its
                            first argument. If the property requires an intensity image, it must
                            accept the intensity image as the second argument.
   :type extra_properties: Iterable of callables
   :param spacing: The pixel spacing along each axis of the image.
   :type spacing: tuple of float, shape (ndim,)
   :param offset: Coordinates of the origin ("top-left" corner) of the label image.
                  Normally this is ([0, ]0, 0), but it might be different if one wants
                  to obtain regionprops of subvolumes within a larger volume.
   :type offset: array-like of int, shape `(label_image.ndim,)`, optional

   :returns: **properties** -- Each item describes one labeled region, and can be accessed using the
             attributes listed below.
   :rtype: list of RegionProperties

   .. rubric:: Notes

   The following properties can be accessed as attributes or keys:

   **area** : float
       Area of the region i.e. number of pixels of the region scaled by pixel-area.
   **area_bbox** : float
       Area of the bounding box i.e. number of pixels of bounding box scaled by pixel-area.
   **area_convex** : float
       Area of the convex hull image, which is the smallest convex
       polygon that encloses the region.
   **area_filled** : float
       Area of the region with all the holes filled in.
   **axis_major_length** : float
       The length of the major axis of the ellipse that has the same
       normalized second central moments as the region.
   **axis_minor_length** : float
       The length of the minor axis of the ellipse that has the same
       normalized second central moments as the region.
   **bbox** : tuple
       Bounding box ``(min_row, min_col, max_row, max_col)``.
       Pixels belonging to the bounding box are in the half-open interval
       ``[min_row; max_row)`` and ``[min_col; max_col)``.
   **centroid** : array
       Centroid coordinate tuple ``(row, col)``.
   **centroid_local** : array
       Centroid coordinate tuple ``(row, col)``, relative to region bounding
       box.
   **centroid_weighted** : array
       Centroid coordinate tuple ``(row, col)`` weighted with intensity
       image.
   **centroid_weighted_local** : array
       Centroid coordinate tuple ``(row, col)``, relative to region bounding
       box, weighted with intensity image.
   **coords_scaled** : (K, 2) ndarray
       Coordinate list ``(row, col)`` of the region scaled by ``spacing``.
   **coords** : (K, 2) ndarray
       Coordinate list ``(row, col)`` of the region.
   **eccentricity** : float
       Eccentricity of the ellipse that has the same second-moments as the
       region. The eccentricity is the ratio of the focal distance
       (distance between focal points) over the major axis length.
       The value is in the interval [0, 1).
       When it is 0, the ellipse becomes a circle.
   **equivalent_diameter_area** : float
       The diameter of a circle with the same area as the region.
   **euler_number** : int
       Euler characteristic of the set of non-zero pixels.
       Computed as number of connected components subtracted by number of
       holes (input.ndim connectivity). In 3D, number of connected
       components plus number of holes subtracted by number of tunnels.
   **extent** : float
       Ratio of pixels in the region to pixels in the total bounding box.
       Computed as ``area / (rows * cols)``
   **feret_diameter_max** : float
       Maximum Feret's diameter computed as the longest distance between
       points around a region's convex hull contour as determined by
       ``find_contours``. [R80f53045c2a3-5]_
   **image** : (H, J) ndarray
       Sliced binary region image which has the same size as bounding box.
   **image_convex** : (H, J) ndarray
       Binary convex hull image which has the same size as bounding box.
   **image_filled** : (H, J) ndarray
       Binary region image with filled holes which has the same size as
       bounding box.
   **image_intensity** : ndarray
       Image inside region bounding box.
   **inertia_tensor** : ndarray
       Inertia tensor of the region for the rotation around its mass.
   **inertia_tensor_eigvals** : tuple
       The eigenvalues of the inertia tensor in decreasing order.
   **intensity_max** : float
       Value with the greatest intensity in the region.
   **intensity_mean** : float
       Value with the mean intensity in the region.
   **intensity_min** : float
       Value with the least intensity in the region.
   **intensity_std** : float
       Standard deviation of the intensity in the region.
   **label** : int
       The label in the labeled input image.
   **moments** : (3, 3) ndarray
       Spatial moments up to 3rd order::

           m_ij = sum{ array(row, col) * row^i * col^j }

       where the sum is over the `row`, `col` coordinates of the region.
   **moments_central** : (3, 3) ndarray
       Central moments (translation invariant) up to 3rd order::

           mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

       where the sum is over the `row`, `col` coordinates of the region,
       and `row_c` and `col_c` are the coordinates of the region's centroid.
   **moments_hu** : tuple
       Hu moments (translation, scale and rotation invariant).
   **moments_normalized** : (3, 3) ndarray
       Normalized moments (translation and scale invariant) up to 3rd order::

           nu_ij = mu_ij / m_00^[(i+j)/2 + 1]

       where `m_00` is the zeroth spatial moment.
   **moments_weighted** : (3, 3) ndarray
       Spatial moments of intensity image up to 3rd order::

           wm_ij = sum{ array(row, col) * row^i * col^j }

       where the sum is over the `row`, `col` coordinates of the region.
   **moments_weighted_central** : (3, 3) ndarray
       Central moments (translation invariant) of intensity image up to
       3rd order::

           wmu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

       where the sum is over the `row`, `col` coordinates of the region,
       and `row_c` and `col_c` are the coordinates of the region's weighted
       centroid.
   **moments_weighted_hu** : tuple
       Hu moments (translation, scale and rotation invariant) of intensity
       image.
   **moments_weighted_normalized** : (3, 3) ndarray
       Normalized moments (translation and scale invariant) of intensity
       image up to 3rd order::

           wnu_ij = wmu_ij / wm_00^[(i+j)/2 + 1]

       where ``wm_00`` is the zeroth spatial moment (intensity-weighted area).
   **num_pixels** : int
       Number of foreground pixels.
   **orientation** : float
       Angle between the 0th axis (rows) and the major
       axis of the ellipse that has the same second moments as the region,
       ranging from `-pi/2` to `pi/2` counter-clockwise.
   **perimeter** : float
       Perimeter of object which approximates the contour as a line
       through the centers of border pixels using a 4-connectivity.
   **perimeter_crofton** : float
       Perimeter of object approximated by the Crofton formula in 4
       directions.
   **slice** : tuple of slices
       A slice to extract the object from the source image.
   **solidity** : float
       Ratio of pixels in the region to pixels of the convex hull image.

   Each region also supports iteration, so that you can do::

     for prop in region:
         print(prop, region[prop])

   .. seealso:: :obj:`label`

   .. rubric:: References

   .. [R80f53045c2a3-1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
          Core Algorithms. Springer-Verlag, London, 2009.
   .. [R80f53045c2a3-2] B. Jähne. Digital Image Processing. Springer-Verlag,
          Berlin-Heidelberg, 6. edition, 2005.
   .. [R80f53045c2a3-3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
          Features, from Lecture notes in computer science, p. 676. Springer,
          Berlin, 1993.
   .. [R80f53045c2a3-4] https://en.wikipedia.org/wiki/Image_moment
   .. [R80f53045c2a3-5] W. Pabst, E. Gregorová. Characterization of particles and particle
          systems, pp. 27-28. ICT Prague, 2007.
          https://old.vscht.cz/sil/keramika/Characterization_of_particles/CPPS%20_English%20version_.pdf

   .. rubric:: Examples

   >>> from skimage import data, util
   >>> from skimage.measure import label, regionprops
   >>> img = util.img_as_ubyte(data.coins()) > 110
   >>> label_img = label(img, connectivity=img.ndim)
   >>> props = regionprops(label_img)
   >>> # centroid of first labeled object
   >>> props[0].centroid
   (22.72987986048314, 81.91228523446583)
   >>> # centroid of first labeled object
   >>> props[0]['centroid']
   (22.72987986048314, 81.91228523446583)

   Add custom measurements by passing functions as ``extra_properties``

   >>> from skimage import data, util
   >>> from skimage.measure import label, regionprops
   >>> import numpy as np
   >>> img = util.img_as_ubyte(data.coins()) > 110
   >>> label_img = label(img, connectivity=img.ndim)
   >>> def pixelcount(regionmask):
   ...     return np.sum(regionmask)
   >>> props = regionprops(label_img, extra_properties=(pixelcount,))
   >>> props[0].pixelcount
   7741
   >>> props[1]['pixelcount']
   42















   ..
       !! processed by numpydoc !!

.. py:function:: get_basic_gridscore(A, **kwargs)

.. py:function:: get_circular_regions(A, **kwargs)

   
   Returns a list of images which are expanding circular
   regions centred on the middle of the image out to the
   image edge. Used for calculating the grid score of each
   image to find the one with the max grid score. Based on
   some Moser paper I can't recall.

   :param A: The SAC
   :type A: np.ndarray

   :keyword min_radius: The smallest radius circle to start with
   :kwtype min_radius: int















   ..
       !! processed by numpydoc !!

.. py:function:: get_deformed_sac_gridscore(A)

   
   Deforms a non-circular SAC into a circular SAC (circular meaning
   the ellipse drawn around the edges of the 6 nearest peaks to the
   SAC centre) and returns get_basic_griscore() calculated on the
   deformed (or re-formed?!) SAC
















   ..
       !! processed by numpydoc !!

.. py:function:: get_expanding_circle_gridscore(A, **kwargs)

   
   Calculates the gridscore for each circular sub-region of image A
   where the circles are centred on the image centre and expanded to
   the edge of the image. The maximum of the get_basic_gridscore() for
   each of these circular sub-regions is returned as the gridscore
















   ..
       !! processed by numpydoc !!

.. py:function:: get_mean_resultant(ego_boundary_map)

   
   Calculates the mean resultant vector of a boundary map in egocentric coordinates

   See Hinman et al., 2019 for more details

   :param ego_boundary_map: The egocentric boundary map
   :type ego_boundary_map: np.ndarray

   :returns: The mean resultant vector of the egocentric boundary map
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: get_mean_resultant_angle(ego_boundary_map, **kwargs)

.. py:function:: get_mean_resultant_length(ego_boundary_map, **kwargs)

.. py:function:: get_thigmotaxis_score(xy, shape = 'circle')

   
   Returns a score which is the ratio of the time spent in the inner
   portion of an environment to the time spent in the outer portion.
   The portions are allocated so that they have equal area.

   :param xy: The xy coordinates of the animal's position. 2 x nsamples
   :type xy: np.ndarray
   :param shape: The shape of the environment. Legal values are 'circle'
   :type shape: str
   :param and 'square'. Default 'circle':

   Returns:
   thigmoxtaxis_score (float): Values closer to 1 indicate the
   animal spent more time in the inner portion of the environment. Values closer to -1
   indicates the animal spent more time in the outer portion of the environment.
   A value of 0 indicates the animal spent equal time in both portions of the
   environment.















   ..
       !! processed by numpydoc !!

.. py:function:: global_threshold(A, prc=50, min_dist=5)

   
   Globally thresholds a ratemap and counts number of fields found
















   ..
       !! processed by numpydoc !!

.. py:function:: grid_field_props(A, maxima='centroid', allProps=True, **kwargs)

   
   Extracts various measures from a spatial autocorrelogram

   :param A: BinnedData object containing the spatial autocorrelogram (SAC) in
             A.binned_data[0]
   :param maxima: The method used to detect the peaks in the SAC.
                  Legal values are 'single' and 'centroid'. Default 'centroid'
   :type maxima: str, optional
   :param allProps: Whether to return a dictionary that
   :type allProps: bool, optional
   :param contains the attempt to fit an ellipse around the edges of the:
   :param central size peaks. See below: Default True

   :returns: A dictionary containing measures of the SAC.
             Keys include:
                 * gridness score
                 * scale
                 * orientation
                 * coordinates of the peaks (nominally 6) closest to SAC centre
                 * a binary mask around the extent of the 6 central fields
                 * values of the rotation procedure used to calculate gridness
                 * ellipse axes and angle (if allProps is True and the it worked)
   :rtype: props (dict)

   .. rubric:: Notes

   The output from this method can be used as input to the show() method
   of this class.
   When it is the plot produced will display a lot more informative.
   The coordinate system internally used is centred on the image centre.

   .. seealso:: ephysiopy.common.binning.autoCorr2D()















   ..
       !! processed by numpydoc !!

.. py:function:: grid_orientation(peakCoords, closestPeakIdx)

   
   Calculates the orientation angle of a grid field.

   The orientation angle is the angle of the first peak working
   counter-clockwise from 3 o'clock

   :param peakCoords: The peak coordinates as pairs of xy
   :type peakCoords: array_like
   :param closestPeakIdx: A 1D array of the indices in peakCoords
   :type closestPeakIdx: array_like
   :param of the peaks closest to the centre of the SAC:

   :returns: The first value in an array of the angles of
             the peaks in the SAC working counter-clockwise from a line
             extending from the middle of the SAC to 3 o'clock.
   :rtype: peak_orientation (float)















   ..
       !! processed by numpydoc !!

.. py:function:: gridness(image, step=30)

   
   Calculates the gridness score in a grid cell SAC.

   Briefly, the data in `image` is rotated in `step` amounts and
   each rotated array is correlated with the original.
   The maximum of the values at 30, 90 and 150 degrees
   is the subtracted from the minimum of the values at 60, 120
   and 180 degrees to give the grid score.

   :param image: The spatial autocorrelogram
   :type image: array_like
   :param step: The amount to rotate the SAC in each step of the
   :type step: int, optional
   :param rotational correlation procedure:

   :returns: The gridscore, the correlation values at each
             `step` and the rotational array
   :rtype: gridmeasures (3-tuple)

   .. rubric:: Notes

   The correlation performed is a Pearsons R. Some rescaling of the
   values in `image` is performed following rotation.

   .. seealso::

      skimage.transform.rotate : for how the rotation of `image` is done
      skimage.exposure.rescale_intensity : for the resscaling following
      rotation















   ..
       !! processed by numpydoc !!

.. py:function:: infill_ratemap(rmap)

.. py:function:: kl_spatial_sparsity(pos_map)

   
   Calculates a measure of spatial sampling of an arena by comparing the
   given spatial sampling to a uniform one using kl divergence

   Data in pos_map should be unsmoothed (not checked) and the MapType should
   be POS (checked)















   ..
       !! processed by numpydoc !!

.. py:function:: kldiv(X, pvect1, pvect2, variant = '')

   
   Calculates the Kullback-Leibler or Jensen-Shannon divergence between
   two distributions.

   :param X: Vector of M variable values
   :type X: array_like
   :param P1: Length-M vector of probabilities representing
   :type P1: array_like
   :param distribution 1:
   :param P2: Length-M vector of probabilities representing
   :type P2: array_like
   :param distribution 2:
   :param sym: If 'sym', returns a symmetric variant of the
               Kullback-Leibler divergence, given by [KL(P1,P2)+KL(P2,P1)]/2
   :type sym: str, optional
   :param js: If 'js', returns the Jensen-Shannon divergence,
   :type js: str, optional
   :param given by: [KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2

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

   :param polarPlot: The binned and smoothed directional ratemap
   :type polarPlot: 1D-array

   :returns: The divergence from circular of the 1D-array
             from a uniform circular distribution
   :rtype: klDivergence (float)















   ..
       !! processed by numpydoc !!

.. py:function:: limit_to_one(A, prc=50, min_dist=5)

   
   Processes a multi-peaked ratemap (ie grid cell) and returns a matrix
   where the multi-peaked ratemap consist of a single peaked field that is
   a) not connected to the border and b) close to the middle of the
   ratemap
















   ..
       !! processed by numpydoc !!

.. py:function:: local_threshold(A, prc=50, min_dist=5)

   
   Locally thresholds a ratemap to take only the surrounding prc amount
   around any local peak
















   ..
       !! processed by numpydoc !!

.. py:function:: partitionFields(binned_data, field_threshold_percent = 50, field_rate_threshold = 0.5, area_threshold=0.01)

   
   Partitions fields.

   Partitions spikes into fields by finding the watersheds around the
   peaks of a super-smoothed ratemap

   :param binned_data (BinnedData) - an instance of ephysiopy.common.utils.BinnedData:
   :param field_threshold_percent (int) - removes pixels in a field that fall below this percent: of the maximum firing rate in the field
   :param field_rate_threshold (float) - anything below this firing rate in Hz threshold is set to 0:
   :param area_threshold (float) - defines the minimum field size as a proportion of the: environment size. Default of 0.01 says a field has to be at
                                                                                          least 1% of the size of the environment i.e.
                                                                                          binned_area_width * binned_area_height to be counted as a field

   :returns: peaksXY (array_like): The xy coordinates of the peak rates in
             each field
             peaksRate (array_like): The peak rates in peaksXY
             labels (numpy.ndarray): An array of the labels corresponding to
             each field (starting  1)
             rmap (numpy.ndarray): The ratemap of the tetrode / cluster
   :rtype: tuple[np.ndarray] - including















   ..
       !! processed by numpydoc !!

.. py:function:: reduce_labels(A, labels, reduce_by = 50)

.. py:function:: skaggs_info(ratemap, dwelltimes, **kwargs)

   
   Calculates Skaggs information measure

   :param ratemap: The binned up ratemap
   :type ratemap: array_like
   :param dwelltimes: Must be same size as ratemap
   :type dwelltimes: array_like

   :returns: Skaggs information score
   :rtype: bits_per_spike (float)

   .. rubric:: Notes

   THIS DATA SHOULD UNDERGO ADAPTIVE BINNING
   See getAdaptiveMap() in binning class

   Returns Skaggs et al's estimate of spatial information
   in bits per spike:

   .. math:: I = sum_{x} p(x).r(x).log(r(x)/r)















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

.. py:data:: PROPS

.. py:data:: PROP_VALS

