ephysiopy.common.utils
======================

.. py:module:: ephysiopy.common.utils


Classes
-------

.. autoapisummary::

   ephysiopy.common.utils.BinnedData
   ephysiopy.common.utils.ClusterID
   ephysiopy.common.utils.MapType
   ephysiopy.common.utils.TrialFilter
   ephysiopy.common.utils.VariableToBin


Functions
---------

.. autoapisummary::

   ephysiopy.common.utils.applyFilter2Labels
   ephysiopy.common.utils.blur_image
   ephysiopy.common.utils.bwperim
   ephysiopy.common.utils.cart2pol
   ephysiopy.common.utils.circ_abs
   ephysiopy.common.utils.clean_kwargs
   ephysiopy.common.utils.cluster_intersection
   ephysiopy.common.utils.corr_maps
   ephysiopy.common.utils.count_runs_and_unique_numbers
   ephysiopy.common.utils.count_to
   ephysiopy.common.utils.fileContainsString
   ephysiopy.common.utils.filter_data
   ephysiopy.common.utils.filter_trial_by_time
   ephysiopy.common.utils.find_runs
   ephysiopy.common.utils.fixAngle
   ephysiopy.common.utils.flatten_list
   ephysiopy.common.utils.getLabelEnds
   ephysiopy.common.utils.getLabelStarts
   ephysiopy.common.utils.get_z_score
   ephysiopy.common.utils.labelContigNonZeroRuns
   ephysiopy.common.utils.labelledCumSum
   ephysiopy.common.utils.mean_norm
   ephysiopy.common.utils.memmapBinaryFile
   ephysiopy.common.utils.min_max_norm
   ephysiopy.common.utils.pol2cart
   ephysiopy.common.utils.polar
   ephysiopy.common.utils.rect
   ephysiopy.common.utils.remap_to_range
   ephysiopy.common.utils.repeat_ind
   ephysiopy.common.utils.shift_vector
   ephysiopy.common.utils.smooth
   ephysiopy.common.utils.window_rms


Module Contents
---------------

.. py:class:: BinnedData(variable, map_type, binned_data, bin_edges, cluster_id = ClusterID(0, 0))

   
   A dataclass to store binned data. The binned data is stored in a list of
   numpy arrays. The bin edges are stored in a list of numpy arrays. The
   variable to bin is stored as an instance of the VariableToBin enum.
   The map type is stored as an instance of the MapType enum.
   The binned data and bin edges are initialized as
   empty lists. bin_units is how to conver the binned data
   to "real" units e.g. for XY it might be how to convert to cms,
   for time to seconds etc. You multiply the binned data by that
   number to get the real values. Note that this might not make sense
   / be obvious for some binning (i.e. SPEED_DIR)

   The BinnedData class is the output of the main binning function in the
   ephysiopy.common.binning.RateMap class. It is used to store the binned data
   as a convenience mostly for easily iterating over the binned data and
   using the bin_edges to plot the data.
   As such, it is used as a convenience for plotting as the bin edges
   are used when calling pcolormesh in the plotting functions.















   ..
       !! processed by numpydoc !!

   .. py:method:: T()


   .. py:method:: __add__(other)

      
      Adds the binned_data of another BinnedData instance
      to the binned_data of this instance.

      :param other: The instance to add to the current one
      :type other: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: __assert_equal_bin_edges__(other)


   .. py:method:: __eq__(other)

      
      Checks for equality of two instances of BinnedData
















      ..
          !! processed by numpydoc !!


   .. py:method:: __getitem__(i)

      
      Returns a specified index of the binned_data as a BinnedData instance.
      The data in binned_data is a deep copy of the original so can be
      modified without affecting the original.

      :param i: The index of binned_data to return
      :type i: int















      ..
          !! processed by numpydoc !!


   .. py:method:: __iter__()


   .. py:method:: __len__()


   .. py:method:: __truediv__(other)

      
      Divides the binned data by the binned data of
      another BinnedData instance i.e. spike data / pos data to get
      a rate map.

      :param other: the denominator
      :type other: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: correlate(other=None, as_matrix=False)

      
      This method is used to correlate the binned data of this BinnedData
      instance with the binned data of another BinnedData instance.

      :param other: The other BinnedData instance to correlate with.
                    If None, then correlations are performed between all the data held
                    in the list self.binned_data
      :type other: BinnedData
      :param as_matrix: If True will return the full correlation matrix for
                        all of the correlations in the list of data in self.binned_data. If
                        False, a list of the unique correlations for the comparisons in
                        self.binned_data are returned.
      :type as_matrix: bool

      :returns: A new BinnedData instance with the correlation of the
                binned data of this instance and the other instance.
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: get_cluster(id)

      
      Returns the binned data for the specified cluster id

      :param id: The cluster id to return
      :type id: ClusterID

      :returns: A new BinnedData instance with the binned data for
                the specified cluster id
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: set_nan_indices(indices)

      
      Sets the values of the binned data at the specified indices to NaN.

      :param indices: The indices to convert to NaN
      :type indices: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: bin_edges
      :type:  list[numpy.ndarray]
      :value: []



   .. py:attribute:: binned_data
      :type:  list[numpy.ma.MaskedArray]
      :value: []



   .. py:attribute:: cluster_id
      :type:  list[ClusterID]
      :value: []



   .. py:attribute:: map_type
      :type:  MapType


   .. py:attribute:: variable
      :type:  VariableToBin


.. py:class:: ClusterID

   Bases: :py:obj:`tuple`


   .. py:attribute:: Channel


   .. py:attribute:: Cluster


.. py:class:: MapType(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   
   A human readable representation of the map type
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: ADAPTIVE
      :value: 4



   .. py:attribute:: AUTO_CORR
      :value: 5



   .. py:attribute:: CROSS_CORR
      :value: 6



   .. py:attribute:: POS
      :value: 2



   .. py:attribute:: RATE
      :value: 1



   .. py:attribute:: SPK
      :value: 3



.. py:class:: TrialFilter(name, start, end = None)

   
   A basic dataclass for holding filter values

   Units:
   time: seconds
   dir: degrees
   speed: cm/s
   xrange/ yrange: cm















   ..
       !! processed by numpydoc !!

   .. py:attribute:: end
      :type:  float | str


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: start
      :type:  float | str


.. py:class:: VariableToBin(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   
   Holds a human readable representation of the variable being binned
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: DIR
      :value: 2



   .. py:attribute:: EGO_BOUNDARY
      :value: 6



   .. py:attribute:: PHI
      :value: 10



   .. py:attribute:: SPEED
      :value: 3



   .. py:attribute:: SPEED_DIR
      :value: 5



   .. py:attribute:: TIME
      :value: 7



   .. py:attribute:: X
      :value: 8



   .. py:attribute:: XY
      :value: 1



   .. py:attribute:: XY_TIME
      :value: 4



   .. py:attribute:: Y
      :value: 9



.. py:function:: applyFilter2Labels(M, x)

   
   M is a logical mask specifying which label numbers to keep
   x is an array of positive integer labels

   This method sets the undesired labels to 0 and renumbers the remaining
   labels 1 to n when n is the number of trues in M















   ..
       !! processed by numpydoc !!

.. py:function:: blur_image(im, n, ny = 0, ftype = 'boxcar', **kwargs)

   
   Smooths all the binned_data in an instance of BinnedData
   by convolving with a filter.

   :param im: Contains the array to smooth.
   :type im: BinnedData
   :param n: The size of the smoothing kernel.
   :type n: int
   :param ny: The size of the smoothing kernel.
   :type ny: int
   :param ftype: The type of smoothing kernel. Either 'boxcar' or 'gaussian'.
   :type ftype: str

   :returns: BinnedData instance with the smoothed data.
   :rtype: BinnedData

   .. rubric:: Notes

   This essentially does the smoothing in-place















   ..
       !! processed by numpydoc !!

.. py:function:: bwperim(bw, n=4)

   
   Finds the perimeter of objects in binary images.

   A pixel is part of an object perimeter if its value is one and there
   is at least one zero-valued pixel in its neighborhood.

   By default, the neighborhood of a pixel is 4 nearest pixels, but
   if `n` is set to 8, the 8 nearest pixels will be considered.

   :param bw: A black-and-white image.
   :type bw: array_like
   :param n: Connectivity. Must be 4 or 8. Default is 4.
   :type n: int, optional

   :returns: **perim** -- A boolean image.
   :rtype: array_like















   ..
       !! processed by numpydoc !!

.. py:function:: cart2pol(x, y)

   
   Convert Cartesian coordinates to polar coordinates.

   :param x: X coordinate(s).
   :type x: float or np.ndarray
   :param y: Y coordinate(s).
   :type y: float or np.ndarray

   :returns: * **r** (*float or np.ndarray*) -- Radial coordinate(s).
             * **th** (*float or np.ndarray*) -- Angular coordinate(s) in radians.















   ..
       !! processed by numpydoc !!

.. py:function:: circ_abs(x)

   
   Calculate the absolute value of an angle in radians,
   normalized to the range [-pi, pi].

   :param x: Angle(s) in radians.
   :type x: float or np.ndarray

   :returns: Absolute value of the angle(s) normalized to the range [-pi, pi].
   :rtype: float or np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: clean_kwargs(func, kwargs)

   
   This function is used to remove any keyword arguments that are not
   accepted by the function. It is useful for passing keyword arguments
   to other functions without having to worry about whether they are
   accepted by the function or not.

   :param func: The function to check for keyword arguments.
   :type func: function
   :param \*\*kwargs: The keyword arguments to check.

   :returns: A dictionary containing only the keyword arguments that are
             accepted by the function.
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: cluster_intersection(A, B)

   
   Gets the intersection of clusters between two instances
   of BinnedData.

   :param A: The two instances
   :type A: BinnedData
   :param B: The two instances
   :type B: BinnedData

   :returns: **A, B** -- The modified instances with only the overlapping clusters
             present in both
   :rtype: BinnedData















   ..
       !! processed by numpydoc !!

.. py:function:: corr_maps(map1, map2, maptype='normal')

   
   Correlates two rate maps together, ignoring areas that have zero sampling.

   :param map1: The first rate map to correlate.
   :type map1: np.ndarray
   :param map2: The second rate map to correlate.
   :type map2: np.ndarray
   :param maptype: The type of correlation to perform. Options are "normal" and "grid".
                   Default is "normal".
   :type maptype: str, optional

   :returns: The correlation coefficient between the two rate maps.
   :rtype: float

   .. rubric:: Notes

   If the shapes of the input maps are different, the smaller map will be
   resized to match the shape of the larger map using reflection mode.

   The "normal" maptype considers non-zero and non-NaN values for correlation,
   while the "grid" maptype considers only finite values.















   ..
       !! processed by numpydoc !!

.. py:function:: count_runs_and_unique_numbers(arr)

   
   Counts the number of continuous runs of numbers in a 1D numpy array.

   :param arr: The input 1D numpy array of numbers.
   :type arr: np.ndarray

   :returns: A tuple containing:
             - dict: A dictionary with the count of runs for each unique number.
             - set: The set of unique numbers in the array.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: count_to(n)

   
   This function is equivalent to hstack((arange(n_i) for n_i in n)).
   It seems to be faster for some possible inputs and encapsulates
   a task in a function.

   .. rubric:: Examples

   >>> n = np.array([0, 0, 3, 0, 0, 2, 0, 2, 1])
   >>> count_to(n)
   array([0, 1, 2, 0, 1, 0, 1, 0])















   ..
       !! processed by numpydoc !!

.. py:function:: fileContainsString(pname, searchStr)

   
   Checks if the search string is contained in a file

   :param pname: The file to look in
   :type pname: str
   :param searchStr: The string to look for
   :type searchStr: str

   :returns: Whether the string was found or not
   :rtype: bool















   ..
       !! processed by numpydoc !!

.. py:function:: filter_data(data, f)

   
   Filters the input data based on the specified TrialFilter.

   :param data: The data to filter.
   :type data: np.ndarray
   :param f: The filter to apply.
   :type f: TrialFilter

   :returns: A boolean array where Trues are the 'to-be' masked values.
   :rtype: np.ndarray

   .. rubric:: Notes

   When calculating the filters, be sure to do the calculations on the
   'data' property of the masked arrays so you get access to the
   underlying data without the mask.

   This function is used in io.recording to filter the data















   ..
       !! processed by numpydoc !!

.. py:function:: filter_trial_by_time(duration, how = 'in_half')

   
   Filters the data in trial by time

   :param duration - the duration of the trial in seconds:
   :param how (str) - how to split the trial.: Legal values: "in_half" or "odd_even"
                                               "in_half" filters for first n seconds and last n second
                                               "odd_even" filters for odd vs even minutes

   :returns: A tuple of TrialFilter instances, one for each half or odd/even minutes
   :rtype: tuple of TrialFilter















   ..
       !! processed by numpydoc !!

.. py:function:: find_runs(x)

   
   Find runs of consecutive items in an array.

   :param x: The array to search for runs in
   :type x: np.ndarray, list

   :returns: * **run_values** (*np.ndarray*) -- the values of each run
             * **run_starts** (*np.ndarray*) -- the indices into x at which each run starts
             * **run_lengths** (*np.ndarray*) -- The length of each run

   .. rubric:: Examples

   >>> n = np.array([0, 0, 3, 3, 0, 2, 0,0, 1])
   >>> find_runs(n)
   (array([0, 3, 0, 2, 0, 1]),
   array([0, 2, 4, 5, 6, 8]),
   array([2, 2, 1, 1, 2, 1]))

   .. rubric:: Notes

   Taken from:
   https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065















   ..
       !! processed by numpydoc !!

.. py:function:: fixAngle(a)

   
   Ensure angles lie between -pi and pi.

   :param a: Angle(s) in radians.
   :type a: float or np.ndarray

   :returns: Angle(s) normalized to the range [-pi, pi].
   :rtype: float or np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: flatten_list(list_to_flatten)

   
   Flattens a list of lists

   :param list_to_flatten: the list to flatten
   :type list_to_flatten: list

   :returns: The flattened list
   :rtype: list















   ..
       !! processed by numpydoc !!

.. py:function:: getLabelEnds(x)

   
   Get the indices of the end of contiguous runs of non-zero values
   in a 1D numpy array.

   :param x: The input 1D numpy array.
   :type x: np.ndarray

   :returns: An array of indices marking the end of each contiguous run of
             non-zero values.
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: getLabelStarts(x)

   
   Get the indices of the start of contiguous runs of non-zero values in a
   1D numpy array.

   :param x: The input 1D numpy array.
   :type x: np.ndarray

   :returns: An array of indices marking the start of each contiguous run of
             non-zero values.
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: get_z_score(x, mean=None, sd=None, axis = 0)

   
   Calculate the z-scores for array x based on the mean
   and standard deviation in that sample, unless stated

   :param x: The array to z-score
   :type x: np.ndarray
   :param mean: The mean of x. Calculated from x if not provided
   :type mean: float, optional
   :param sd: The standard deviation of x. Calculated from x if not provided
   :type sd: float, optional
   :param axis: The axis along which to operate
   :type axis: int

   :returns: The z-scored version of the input array x
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: labelContigNonZeroRuns(x)

   
   Label contiguous non-zero runs in a 1D numpy array.

   :param x: The input 1D numpy array.
   :type x: np.ndarray

   :returns: An array where each element is labeled with an integer representing
             the contiguous non-zero run it belongs to.
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: labelledCumSum(X, L)

   
   Compute the cumulative sum of an array with labels, resetting the
   sum at label changes.

   :param X: Input array to compute the cumulative sum.
   :type X: np.ndarray
   :param L: Label array indicating where to reset the cumulative sum.
   :type L: np.ndarray

   :returns: The cumulative sum array with resets at label changes, masked
             appropriately.
   :rtype: np.ma.MaskedArray















   ..
       !! processed by numpydoc !!

.. py:function:: mean_norm(x, mn=None, axis = 0)

   
   Mean normalise an input array

   :param x: The array t normalise
   :type x: np.ndarray
   :param mn: The mean of x
   :type mn: float, optional
   :param axis: The axis along which to operate
   :type axis: int

   :returns: The mean normalised version of the input array
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: memmapBinaryFile(path2file, n_channels=384, **kwargs)

   
   Returns a numpy memmap of the int16 data in the
   file path2file, if present

   :param path2file: The location of the file to be mapped
   :type path2file: Path
   :param n_channels: the number of channels (size of the second dimension)
   :type n_channels: int
   :param \*\*kwargs:
                      'data_type' : np.dtype, default np.int16
                          The data type of the file to be mapped.

   :returns: The memory mapped data file
   :rtype: np.memmap















   ..
       !! processed by numpydoc !!

.. py:function:: min_max_norm(x, min=None, max=None, axis = 0)

   
   Normalise the input array x to lie between min and max

   :param x: the array to normalise
   :type x: np.ndarray
   :param min: the minimun value in the returned array
   :type min: float
   :param max: the maximum value in the returned array
   :type max: float
   :param axis: the axis along which to operate. Default 0
   :type axis: int

   :returns: the normalised array
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: pol2cart(r, theta)

   
   Convert polar coordinates to Cartesian coordinates.

   :param r: Radial coordinate(s).
   :type r: float or np.ndarray
   :param theta: Angular coordinate(s) in radians.
   :type theta: float or np.ndarray

   :returns: * **x** (*float or np.ndarray*) -- X coordinate(s).
             * **y** (*float or np.ndarray*) -- Y coordinate(s).















   ..
       !! processed by numpydoc !!

.. py:function:: polar(x, y, deg=False)

   
   Converts from rectangular coordinates to polar ones.

   :param x: The x coordinates.
   :type x: array_like
   :param y: The y coordinates.
   :type y: array_like
   :param deg: If True, returns the angle in degrees. Default is False (radians).
   :type deg: bool, optional

   :returns: * **r** (*array_like*) -- The radial coordinates.
             * **theta** (*array_like*) -- The angular coordinates.















   ..
       !! processed by numpydoc !!

.. py:function:: rect(r, w, deg=False)

   
   Convert from polar (r, w) to rectangular (x, y) coordinates.

   :param r: Radial coordinate(s).
   :type r: float or np.ndarray
   :param w: Angular coordinate(s).
   :type w: float or np.ndarray
   :param deg: If True, `w` is in degrees. Default is False (radians).
   :type deg: bool, optional

   :returns: A tuple containing:
             - x : float or np.ndarray
                 X coordinate(s).
             - y : float or np.ndarray
                 Y coordinate(s).
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: remap_to_range(x, new_min=0, new_max=1, axis=0)

   
   Remap the values of x to the range [new_min, new_max].

   :param x: the array to remap
   :type x: np.ndarray
   :param new_min: the minimun value in the returned array
   :type new_min: float
   :param max: the maximum value in the returned array
   :type max: float

   :returns: The remapped values
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: repeat_ind(n)

   
   Repeat a given index a specified number of times.

   The input specifies how many times to repeat the given index.
   It is equivalent to something like this:

   hstack((zeros(n_i,dtype=int)+i for i, n_i in enumerate(n)))

   But this version seems to be faster, and probably scales better.
   At any rate, it encapsulates a task in a function.

   :param n: A 1D array where each element specifies the number of times to repeat its index.
   :type n: np.ndarray

   :returns: A 1D array with indices repeated according to the input array.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> n = np.array([0, 0, 3, 0, 0, 2, 0, 2, 1])
   >>> repeat_ind(n)
   array([2, 2, 2, 5, 5, 7, 7, 8])















   ..
       !! processed by numpydoc !!

.. py:function:: shift_vector(v, shift, maxlen=None)

   
   Shifts the elements of a vector by a given amount.
   A bit like numpys roll function but when the shift goes
   beyond some limit that limit is subtracted from the shift.
   The result is then sorted and returned.

   :param v: The input vector.
   :type v: array_like
   :param shift: The amount to shift the elements.
   :type shift: int
   :param fill_value: The value to fill the empty spaces.
   :type fill_value: int

   :returns: The shifted vector.
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: smooth(x, window_len=9, window='hanning')

   
   Smooth the data using a window with requested size.

   This method is based on the convolution of a scaled window with the signal.
   The signal is prepared by introducing reflected copies of the signal
   (with the window size) in both ends so that transient parts are minimized
   in the beginning and end part of the output signal.

   :param x: The input signal.
   :type x: np.ndarray
   :param window_len: The length of the smoothing window.
   :type window_len: int
   :param window: The type of window from 'flat', 'hanning', 'hamming',
                  'bartlett', 'blackman'. 'flat' window will produce a moving average
                  smoothing.
   :type window: str

   :returns: The smoothed signal.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> t=linspace(-2,2,0.1)
   >>> x=sin(t)+randn(len(t))*0.1
   >>> y=smooth(x)

   .. seealso:: :py:obj:`numpy.hanning`, :py:obj:`numpy.hamming`, :py:obj:`numpy.bartlett`, :py:obj:`numpy.blackman`, :py:obj:`numpy.convolve`, :py:obj:`scipy.signal.lfilter`

   .. rubric:: Notes

   The window parameter could be the window itself if an array instead of
   a string.















   ..
       !! processed by numpydoc !!

.. py:function:: window_rms(a, window_size)

   
   Calculates the root mean square of the input a over a window of
   size window_size

   :param a: The input array
   :type a: np.ndarray
   :param window_size: The size of the smoothing window
   :type window_size: int, float

   :returns: The rms'd result
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

