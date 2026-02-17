ephysiopy.common.fieldproperties
================================

.. py:module:: ephysiopy.common.fieldproperties


Attributes
----------

.. autoapisummary::

   ephysiopy.common.fieldproperties.PROPS
   ephysiopy.common.fieldproperties.PROP_VALS


Classes
-------

.. autoapisummary::

   ephysiopy.common.fieldproperties.FieldProps
   ephysiopy.common.fieldproperties.LFPSegment
   ephysiopy.common.fieldproperties.RunProps
   ephysiopy.common.fieldproperties.SpikeTimes
   ephysiopy.common.fieldproperties.SpikingProperty


Functions
---------

.. autoapisummary::

   ephysiopy.common.fieldproperties.fieldprops
   ephysiopy.common.fieldproperties.flatten_output
   ephysiopy.common.fieldproperties.mask_array_with_dynamic_mask
   ephysiopy.common.fieldproperties.spike_count
   ephysiopy.common.fieldproperties.spike_times


Module Contents
---------------

.. py:class:: FieldProps(slice, label, label_image, binned_data, cache, *, extra_properties, spacing, offset, index=0)

   Bases: :py:obj:`skimage.measure._regionprops.RegionProperties`


   
   Describes various properties of a receptive field.

   .. attribute:: slice

      The slice of the field in the binned data (x slice, y slice)

      :type: tuple of slice

   .. attribute:: label

      The label of the field

      :type: int

   .. attribute:: image_intensity

      The intensity image of the field (in Hz)

      :type: np.ndarray

   .. attribute:: runs

      The runs through the field

      :type: list of RunProps

   .. attribute:: run_slices

      The slices of the runs through the field (slices are position indices)

      :type: list of slice

   .. attribute:: run_labels

      The labels of the runs

      :type: np.ndarray

   .. attribute:: max_index

      The index of the maximum intensity in the field

      :type: np.ndarray

   .. attribute:: num_runs

      The number of runs through the field

      :type: int

   .. attribute:: cumulative_time

      The cumulative time spent on the field for each run through the field

      :type: list of np.ndarray

   .. attribute:: cumulative_distance

      The cumulative time spent on the field for each run through the field

      :type: list of np.ndarray

   .. attribute:: runs_speed

      The speed of each run through the field

      :type: list of np.ndarray

   .. attribute:: runs_observed_spikes

      The observed spikes for each run through the field

      :type: np.ndarray

   .. attribute:: spike_index

      The index of the spikes in the position data

      :type: np.ndarray

   .. attribute:: xy_at_peak

      The x-y coordinate of the field max

      :type: np.ndarray

   .. attribute:: xy

      The x-y coordinates of the field for all runs

      :type: np.ndarray

   .. attribute:: xy_relative_to_peak

      The x-y coordinates of the field zeroed with respect to the peak

      :type: np.ndarray

   .. attribute:: xy_angle_to_peak

      The angle each x-y coordinate makes to the field peak

      :type: np.ndarray

   .. attribute:: xy_dist_to_peak

      The distance of each x-y coordinate to the field peak

      :type: np.ndarray

   .. attribute:: bw_perim

      The perimeter of the field as an array of bool

      :type: np.ndarray

   .. attribute:: perimeter_coords

      The x-y coordinates of the field perimeter

      :type: tuple

   .. attribute:: global_perimeter_coords

      The global x-y coordinates of the field perimeter

      :type: np.ndarray

   .. attribute:: perimeter_minus_field_max

      The x-y coordinates of the field perimeter minus the field max

      :type: np.ndarray

   .. attribute:: perimeter_angle_from_peak

      The angle each point on the perimeter makes to the field peak

      :type: np.ndarray

   .. attribute:: perimeter_dist_from_peak

      The distance of each point on the perimeter to the field peak

      :type: np.ndarray

   .. attribute:: bin_coords

      The x-y coordinates of the field in the binned data

      :type: np.ndarray

   .. attribute:: phi

      The angular distance between the mean direction of each run and
      each position samples direction to the field centre

      :type: np.ndarray

   .. attribute:: rho

      The distance of each position sample to the field max (1 is furthest)

      :type: np.ndarray

   .. attribute:: pos_xy

      The cartesian x-y coordinates of each position sample

      :type: np.ndarray

   .. attribute:: pos_phi

      The angular distance between the mean direction of each run and
      each position samples direction to the field centre

      :type: np.ndarray

   .. attribute:: pos_r

      The ratio of the distance from the field peak to the position sample
      and the distance from the field peak to the point on the perimeter
      that is most colinear with the position sample

      :type: np.ndarray

   .. attribute:: r_and_phi_to_x_and_y

      Converts rho and phi to x and y coordinates

      :type: np.ndarray

   .. attribute:: r_per_run

      The polar radial distance for each run

      :type: np.ndarray

   .. attribute:: current_direction

      The direction projected onto the mean run direction

      :type: np.ndarray

   .. attribute:: cumulative_distance

      The cumulative distance for each run

      :type: list of np.ndarray

   .. attribute:: projected_direction

      The direction projected onto the mean run direction

      :type: np.ndarray

   .. attribute:: intensity_max

      The maximum intensity of the field (i.e. field peak rate)

      :type: float

   .. attribute:: intensity_mean

      The mean intensity of the field

      :type: float

   .. attribute:: intensity_min

      The minimum intensity of the field

      :type: float

   .. attribute:: intensity_std

      The standard deviation of the field intensity

      :type: float















   ..
       !! processed by numpydoc !!

   .. py:method:: __getattr__(attr)


   .. py:method:: __str__()

      
      Override the string representation printed to the console
















      ..
          !! processed by numpydoc !!


   .. py:method:: mean_spiking_var(var='current_direction')

      
      Get the mean value of a variable at the posiition of
      the spikes for all runs through this field when multiple spikes
      occur in a single theta cycle

      :param var: the variable to get the mean value of at the position of spikes
      :type var: str

      :returns: the mean value of the variable at the position of spikes
                for all runs through this field
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: overdispersion(spikes, fs = 50)

      
      Calculate the overdispersion for each run through the field

      :param spike_train: the spike train (spikes binned up by position) for the whole trial.
                          Same length as the trial n_samples
      :type spike_train: np.ndarray
      :param fs: the sample rate of the position data
      :type fs: int

      :returns: the overdispersion for each run through the field
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: runs_expected_spikes(expected_rate, sample_rate = 50)

      
      Calculate the expected number of spikes along each run given the
      whole ratemap.

      :param expected_rate: the rate seen at each xy position of the whole trial
      :type expected_rate: np.ndarray
      :param sample_rate: the sample rate of the position data
      :type sample_rate: int

      :returns: the expected rate at each xy position for each run
      :rtype: np.ndarray

      .. rubric:: Notes

      The expected spikes should be calculated from the smoothed
      ratemap and the xy position data using np.digitize:

      >> xbins = np.digitize(xy[0], binned_data.bin_edges[1][:-1]) - 1
      >> ybins = np.digitize(xy[1], binned_data.bin_edges[0][:-1]) - 1
      >> expected_rate_at_pos = binned_data.binned_data[0][ybins, xbins]
      >> exptd_spks = fieldprops.runs_expected_spikes(expected_rate_at_pos)















      ..
          !! processed by numpydoc !!


   .. py:method:: smooth_runs(k, spatial_lp_cut, sample_rate)

      
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


   .. py:method:: spiking_var(var='current_direction')

      
      Get the value of a variable at the position of
      spikes for all runs through this field

      :param var: the variable to get the value of at the position of spikes
      :type var: str

      :returns: the value of the variable at the position of spikes
                for all runs through this field
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _label_image


   .. py:attribute:: _runs
      :value: []



   .. py:property:: bin_coords
      :type: numpy.ndarray



   .. py:attribute:: binned_data


   .. py:property:: bw_perim
      :type: numpy.ndarray



   .. py:property:: compressed_phase
      :type: numpy.ndarray


      
      The phases of the LFP signal for all runs through this field
      compressed into a single array
















      ..
          !! processed by numpydoc !!


   .. py:property:: cumulative_distance
      :type: list



   .. py:property:: cumulative_time
      :type: list



   .. py:property:: current_direction
      :type: list



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



   .. py:property:: label_image


   .. py:property:: max_index
      :type: numpy.ndarray



   .. py:property:: mean_spike_phase


   .. py:property:: n_spikes
      :type: int


      
      The total number of spikes emitted on all runs through the field
















      ..
          !! processed by numpydoc !!


   .. py:property:: normalized_position
      :type: list


      
      Only makes sense to run this on linear track data unless
      we want to pass the unit circle distance or something...

      Get the normalized position for each run through the field.

      Normalized position is the position of the run relative to the
      start of the field (0) to the end of the field (1).















      ..
          !! processed by numpydoc !!


   .. py:property:: num_runs
      :type: int



   .. py:property:: observed_spikes
      :type: numpy.ndarray



   .. py:property:: perimeter_angle_from_peak
      :type: numpy.ndarray



   .. py:property:: perimeter_coords
      :type: tuple



   .. py:property:: perimeter_dist_from_peak
      :type: numpy.ndarray



   .. py:property:: perimeter_minus_field_max
      :type: numpy.ndarray



   .. py:property:: phase
      :type: list


      
      The phases of the LFP signal for all runs through this field
















      ..
          !! processed by numpydoc !!


   .. py:property:: phi
      :type: numpy.ndarray


      
      Calculate the angular distance between the mean direction of each run
      and each position samples direction to the field centre
















      ..
          !! processed by numpydoc !!


   .. py:property:: pos_phi
      :type: numpy.ndarray


      
      Calculate the angular distance between the mean direction of each run
      and each position samples direction to the field centre
















      ..
          !! processed by numpydoc !!


   .. py:property:: pos_r
      :type: numpy.ndarray


      
      Calculate the ratio of the distance from the field peak to the position
      sample and the distance from the field peak to the point on the
      perimeter that is most colinear with the position sample

      NB The values just before being returned can be >= 1 so these are
      capped to 1















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



   .. py:property:: r_per_run
      :type: numpy.ndarray



   .. py:property:: rho
      :type: numpy.ndarray



   .. py:property:: run_labels


   .. py:property:: run_slices


   .. py:property:: runs


   .. py:property:: speed
      :type: list



   .. py:property:: spike_index


   .. py:property:: spike_phase


   .. py:property:: spike_run_labels


   .. py:property:: xy
      :type: numpy.ndarray



   .. py:property:: xy_angle_to_peak
      :type: numpy.ndarray



   .. py:property:: xy_at_peak
      :type: numpy.ndarray



   .. py:property:: xy_dist_to_peak
      :type: numpy.ndarray



   .. py:property:: xy_relative_to_peak
      :type: numpy.ndarray



.. py:class:: LFPSegment(parent, field_label, run_label, slice, spike_times, mask, signal, filtered_signal, phase, cycle_label, sample_rate)

   Bases: :py:obj:`SpikingProperty`, :py:obj:`object`


   
   A custom class for dealing with segments of an LFP signal and how
   they relate to specific runs through a receptive field
   (see RunProps and FieldProps below)

   .. attribute:: field_label

      The field id

      :type: int

   .. attribute:: run_label

      The run id

      :type: int

   .. attribute:: slice

      slice into the LFP data for a segment

      :type: slice

   .. attribute:: spike_times

      the times in seconds spikes occurred for a segment

      :type: np.ndarray

   .. attribute:: spike_count

      spikes binned into lfp samples for a segment

      :type: np.ndarray

   .. attribute:: signal

      raw signal for a segment

      :type: np.ndarray

   .. attribute:: filtered_signal

      bandpass filtered signal for a segment

      :type: np.ndarray

   .. attribute:: phase

      phase data for a segment

      :type: np.ndarray

   .. attribute:: amplitude

      amplitude for a segment

      :type: np.ndarray

   .. attribute:: sample_rate

      sample rate for the LFP segment

      :type: float, int

   .. attribute:: filter_band

      the bandpass filter values

      :type: tuple[int,int]















   ..
       !! processed by numpydoc !!

   .. py:method:: mean_spiking_var(var='phase')

      
      Get the mean value of a variable at the posiition of
      the spikes for all runs through this field when multiple spikes
      occur in a single theta cycle
















      ..
          !! processed by numpydoc !!


   .. py:method:: spiking_var(var='phase')

      
      Get the value of a variable at the position of
      spikes for this run
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _cycle_label


   .. py:attribute:: _filtered_signal


   .. py:attribute:: _phase


   .. py:attribute:: _signal


   .. py:property:: cycle_label


   .. py:attribute:: field_label


   .. py:property:: filtered_signal


   .. py:property:: phase


   .. py:attribute:: run_label


   .. py:attribute:: sample_rate


   .. py:property:: signal


   .. py:attribute:: slice


.. py:class:: RunProps(parent, label, slice, spike_times, mask, xy_coords, speed, peak_xy, max_index, perimeter_coords, sample_rate = 50)

   Bases: :py:obj:`SpikingProperty`, :py:obj:`object`


   
   A custom class for holding information about runs through a receptive field

   Each run needs to have some information about the field to which it belongs
   so the constructor takes in the peak x-y coordinate of the field and its
   index as well as the coordinates of the perimeter of the field

   .. attribute:: label

      the run id

      :type: int

   .. attribute:: slice

      the slice of the position data for a run

      :type: slice

   .. attribute:: xy

      the x-y coordinates for a run (global coordinates)

      :type: np.ndarray

   .. attribute:: speed

      the speed at each xy coordinate

      :type: np.ndarray

   .. attribute:: peak_xy

      the fields max rate xy location

      :type: tuple[float, float]

   .. attribute:: max_index

      the index into the arrays of the field max

      :type: int

   .. attribute:: perimeter_coords

      xy coordinates of the field perimeter

      :type: np.ndarray

   .. attribute:: hdir

      the heading direction

      :type: np.ndarray

   .. attribute:: min_speed

      the minimum speed

      :type: float

   .. attribute:: cumulative_time

      the cumulative time spent on a run

      :type: np.ndarray

   .. attribute:: duration

      the total duration of a run in seconds

      :type: float

   .. attribute:: n_spikes

      the total number of spikes emitted on a run

      :type: int

   .. attribute:: run_start

      the position index of the run start

      :type: int

   .. attribute:: run_stop

      the position index of the run stop

      :type: int

   .. attribute:: mean_direction

      the mean direction of a run

      :type: float

   .. attribute:: current_direction

      the current direction of a run

      :type: np.ndarray

   .. attribute:: cumulative_distance

      the cumulative distance covered in a run

      :type: np.ndarray

   .. attribute:: spike_index

      the index into the position data of the spikes on a run

      :type: np.ndarray

   .. attribute:: observed_spikes

      the observed spikes on a run (binned by position samples)

      :type: np.ndarray

   .. attribute:: xy_angle_to_peak

      the xy angle to the peak (radians)

      :type: np.ndarray

   .. attribute:: xy_dist_to_peak

      the distance to the field max

      :type: np.ndarray

   .. attribute:: xy_dist_to_peak_normed

      normalised distance to field max

      :type: np.ndarray

   .. attribute:: pos_xy

      cartesian xy coordinates but normalised on a unit circle

      :type: np.ndarray

   .. attribute:: pos_phi

      the angular distance between a runs main direction and the
      direction to the peak for each position sample

      :type: np.ndarray

   .. attribute:: rho

      the polar radial distance (1 = field edge)

      :type: np.ndarray

   .. attribute:: phi

      the polar angle (radians)

      :type: np.ndarray

   .. attribute:: r_and_phi_to_x_and_y

      converts rho and phi to x and y coordinates (range = -1 -> +1)

      :type: np.ndarray

   .. attribute:: tortuosity

      the tortuosity for a run (closer to 1 = a straighter run)

      :type: np.ndarray

   .. attribute:: xy_is_smoothed

      whether the xy data has been smoothed

      :type: bool















   ..
       !! processed by numpydoc !!

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


   .. py:method:: mean_spiking_var(var='current_direction')

      
      Get the mean value of a variable at the posiition of
      the spikes for all runs through this field when multiple spikes
      occur in a single theta cycle
















      ..
          !! processed by numpydoc !!


   .. py:method:: overdispersion(spike_train, fs = 50)

      
      The overdispersion map for this run

      :param spike_train: the spike train (spikes binned up by position) for the whole trial.
                          Same length as the trial n_samples
      :type spike_train: np.mdarray
      :param fs: the sample rate of the position data
      :type fs: int















      ..
          !! processed by numpydoc !!


   .. py:method:: perimeter_angle_from_peak()


   .. py:method:: perimeter_minus_field_max()


   .. py:method:: smooth_xy(k, spatial_lp, sample_rate)

      
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


   .. py:method:: spiking_var(var='current_direction')

      
      Get the value of a variable at the position of
      spikes for this run
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _hdir
      :value: None



   .. py:attribute:: _max_index


   .. py:attribute:: _peak_xy


   .. py:attribute:: _perimeter_coords


   .. py:attribute:: _slice


   .. py:attribute:: _speed


   .. py:attribute:: _xy_coords


   .. py:property:: cumulative_distance
      :type: numpy.ma.MaskedArray



   .. py:property:: cumulative_time
      :type: numpy.ma.MaskedArray



   .. py:property:: current_direction
      :type: numpy.ma.MaskedArray


      
      Supposed to calculate current direction wrt to field centre?
















      ..
          !! processed by numpydoc !!


   .. py:property:: duration
      :type: float



   .. py:property:: hdir
      :type: numpy.ma.MaskedArray



   .. py:attribute:: label


   .. py:property:: mean_direction
      :type: float



   .. py:property:: min_speed
      :type: float



   .. py:property:: ndim

      
      Return the dimensionality of the data

      For 1 x n linear track data dimensionality = 1
      for 2 x n open field (or other) data dimensionality = 2















      ..
          !! processed by numpydoc !!


   .. py:property:: normed_x
      :type: numpy.ndarray


      
      Normalise the x data to lie between -1 and 1
      with respect to the field limits
      of the parent field and the run direction such
      that -1 is entry and +1 is exit
















      ..
          !! processed by numpydoc !!


   .. py:property:: phi
      :type: numpy.ma.MaskedArray


      
      Values lie between 0 and 2pi
















      ..
          !! processed by numpydoc !!


   .. py:property:: pos_phi
      :type: numpy.ma.MaskedArray


      
      Values lie between 0 and 2pi
















      ..
          !! processed by numpydoc !!


   .. py:property:: pos_r
      :type: numpy.ma.MaskedArray


      
      Values lie between 0 and 1
















      ..
          !! processed by numpydoc !!


   .. py:property:: pos_xy
      :type: numpy.ma.MaskedArray



   .. py:property:: r_and_phi_to_x_and_y
      :type: numpy.ma.MaskedArray



   .. py:property:: rho
      :type: numpy.ma.MaskedArray


      
      Values lie between 0 and 1
















      ..
          !! processed by numpydoc !!


   .. py:property:: run_start
      :type: int



   .. py:property:: run_stop
      :type: int



   .. py:attribute:: sample_rate
      :value: 50



   .. py:property:: slice
      :type: slice



   .. py:property:: speed
      :type: numpy.ma.MaskedArray



   .. py:property:: spike_num_in_run


   .. py:property:: tortuosity
      :type: numpy.ma.MaskedArray



   .. py:property:: total_distance
      :type: float



   .. py:property:: xy
      :type: numpy.ma.MaskedArray



   .. py:property:: xy_angle_to_peak
      :type: numpy.ma.MaskedArray



   .. py:property:: xy_dist_to_peak
      :type: numpy.ma.MaskedArray



   .. py:property:: xy_dist_to_peak_normed
      :type: numpy.ma.MaskedArray


      
      Values lie between 0 and 1
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: xy_is_smoothed
      :value: False



.. py:class:: SpikeTimes

   Bases: :py:obj:`object`


   
   Descriptor for getting spike times that fall within both
   the LFP segment and the run segment dealing correctly with
   masked time stamps in both cases
















   ..
       !! processed by numpydoc !!

   .. py:method:: __get__(obj, objtype=None)


   .. py:method:: __set__(obj, value)


   .. py:method:: __set_name__(owner, name)


.. py:class:: SpikingProperty(parent, times, mask = None)

   Bases: :py:obj:`object`


   
   Interface for getting attributes by using spike times to
   retrieve their values. Spike times can be masked to
   indicate invalid values (e.g. spikes that occurred
   outside of valid LFP segments, when run speed was too low
   etc)
















   ..
       !! processed by numpydoc !!

   .. py:method:: __len__()


   .. py:attribute:: _all_spike_times


   .. py:attribute:: _mask
      :value: None



   .. py:property:: index


   .. py:property:: mask


   .. py:property:: n_spikes
      :type: int



   .. py:property:: observed_spikes
      :type: numpy.ndarray



   .. py:attribute:: parent


   .. py:property:: raw_spike_times

      
      Return the spike times that fall within the slice without masking
















      ..
          !! processed by numpydoc !!


   .. py:property:: spike_count


   .. py:property:: spike_index
      :type: numpy.ndarray


      
      Get the index into the LFP data of the spikes for this segment

      :returns: the index into the LFP data of the spikes for this segment
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: spike_times


   .. py:property:: time


.. py:function:: fieldprops(label_image, binned_data, spike_times, xy, method='field', cache=True, *, extra_properties=None, spacing=None, offset=None, **kwargs)

   
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
   :type xy: (2 x n_samples) np.ndarray
   :param binned_data:
   :type binned_data: BinnedData instance from ephysiopy.common.utils
   :param spike_times: The spike times for the neuron being analysed
   :type spike_times: np.ndarray
   :param method: Method used to calculate region properties:

                  - 'field': Standard method using discrete pixel counts based
                      on a segmentation of the rate map into labeled regions (fields).
                      This method
                      is faster, but can be inaccurate for small regions and will not
                      work well for positional data that has been masked for direction of
                      running say (ie linear track)
                  - 'clump_runs': Exact method which accounts for filtered data better by
                      looking for contiguous areas of the positional data that are NOT
                      masked (uses np.ma.clump_unmasked)
                  cache : bool, optional
                  Determine whether to cache calculated properties. The computation is
                  much faster for cached properties, whereas the memory consumption
                  increases.
   :type method: {'field', 'clump_runs'}, optional
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
   :param \*\*kwargs: Additional arguments passed to the FieldProps constructor.
                      Legal arguments are:
                          pos_sample_rate : int
                          min_run_length : int
   :type \*\*kwargs: keyword arguments

       .. versionadded:: 0.14.1

   :returns: **properties** -- Each item describes one labeled region, and can be accessed using the
             attributes listed below.
   :rtype: list of RegionProperties

   .. rubric:: Notes

   The following properties can be accessed as attributes or keys:

   **area** : float
       Area of the region i.e. number of pixels of the region scaled
       by pixel-area.
   **area_bbox** : float
       Area of the bounding box i.e. number of pixels of bounding box scaled
       by pixel-area.
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

   .. seealso:: :py:obj:`label`

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

.. py:function:: flatten_output(func)

   
   Decorator to flatten the output of a function that would
   otherwise return a list from a list comprehension and return
   as a np.ndarray
















   ..
       !! processed by numpydoc !!

.. py:function:: mask_array_with_dynamic_mask(func)

   
   Decorator to convert a numpy array to a masked array using a dynamically
   generated mask defined by an instance method or variable of the decorated
   function's class.
















   ..
       !! processed by numpydoc !!

.. py:function:: spike_count(ts, slice, sample_rate, length)

.. py:function:: spike_times(obj)

.. py:data:: PROPS

.. py:data:: PROP_VALS

