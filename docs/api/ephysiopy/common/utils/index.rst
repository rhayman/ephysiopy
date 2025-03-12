ephysiopy.common.utils
======================

.. py:module:: ephysiopy.common.utils


Classes
-------

.. autoapisummary::

   ephysiopy.common.utils.BinnedData
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
   ephysiopy.common.utils.corr_maps
   ephysiopy.common.utils.count_runs_and_unique_numbers
   ephysiopy.common.utils.count_to
   ephysiopy.common.utils.fileContainsString
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

.. py:class:: BinnedData

   .. py:method:: T()


   .. py:method:: __add__(other)


   .. py:method:: __assert_equal_bin_edges__(other)


   .. py:method:: __eq__(other)


   .. py:method:: __getitem__(i)


   .. py:method:: __iter__()


   .. py:method:: __len__()


   .. py:method:: __truediv__(other)


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

      :returns:

                A new BinnedData instance with the correlation of the
                    binned data of this instance and the other instance.
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: set_nan_indices(indices)


   .. py:attribute:: bin_edges
      :type:  list[numpy.ndarray]
      :value: []



   .. py:attribute:: binned_data
      :type:  list[numpy.ndarray]
      :value: []



   .. py:attribute:: map_type
      :type:  MapType


   .. py:attribute:: variable
      :type:  VariableToBin


.. py:class:: MapType

   Bases: :py:obj:`enum.Enum`


   
   Generic enumeration.

   Derive from this class to define new enumerations.















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



.. py:class:: TrialFilter(name, start, end)

   .. py:attribute:: end
      :type:  float | str


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: start
      :type:  float | str


.. py:class:: VariableToBin

   Bases: :py:obj:`enum.Enum`


   
   Generic enumeration.

   Derive from this class to define new enumerations.















   ..
       !! processed by numpydoc !!

   .. py:attribute:: DIR
      :value: 2



   .. py:attribute:: EGO_BOUNDARY
      :value: 6



   .. py:attribute:: SPEED
      :value: 3



   .. py:attribute:: SPEED_DIR
      :value: 5



   .. py:attribute:: TIME
      :value: 7



   .. py:attribute:: XY
      :value: 1



   .. py:attribute:: XY_TIME
      :value: 4



.. py:function:: applyFilter2Labels(M, x)

   
   M is a logical mask specifying which label numbers to keep
   x is an array of positive integer labels

   This method sets the undesired labels to 0 and renumbers the remaining
   labels 1 to n when n is the number of trues in M















   ..
       !! processed by numpydoc !!

.. py:function:: blur_image(im, n, ny = 0, ftype = 'boxcar', **kwargs)

   
   Smooths a 2D image by convolving with a filter.

   :param im: Contains the array to smooth.
   :type im: BinnedData
   :param n: The size of the smoothing kernel.
   :type n: int
   :param ny: The size of the smoothing kernel.
   :type ny: int
   :param ftype: The type of smoothing kernel.
                 Either 'boxcar' or 'gaussian'.
   :type ftype: str

   :returns: BinnedData instance with the smoothed data.
   :rtype: res (BinnedData)

   .. rubric:: Notes

   This essentially does the smoothing in-place















   ..
       !! processed by numpydoc !!

.. py:function:: bwperim(bw, n=4)

   
   Finds the perimeter of objects in binary images.

   A pixel is part of an object perimeter if its value is one and there
   is at least one zero-valued pixel in its neighborhood.

   By default the neighborhood of a pixel is 4 nearest pixels, but
   if `n` is set to 8 the 8 nearest pixels will be considered.

   :param bw: A black-and-white image.
   :type bw: array_like
   :param n: Connectivity. Must be 4 or 8. Default is 8.
   :type n: int, optional

   :returns: A boolean image.
   :rtype: perim (array_like)















   ..
       !! processed by numpydoc !!

.. py:function:: cart2pol(x, y)

.. py:function:: circ_abs(x)

.. py:function:: clean_kwargs(func, kwargs)

   
   This function is used to remove any keyword arguments that are not
   accepted by the function. It is useful for passing keyword arguments
   to other functions without having to worry about whether they are
   accepted by the function or not.

   :param func: The function to check for keyword arguments.
   :type func: function
   :param kwargs: The keyword arguments to check.
   :type kwargs: dict

   :returns: A dictionary containing only the keyword arguments that are
             accepted by the function.
   :rtype: dict















   ..
       !! processed by numpydoc !!

.. py:function:: corr_maps(map1, map2, maptype='normal')

   
   correlates two ratemaps together ignoring areas that have zero sampling
















   ..
       !! processed by numpydoc !!

.. py:function:: count_runs_and_unique_numbers(arr)

   
   Counts the number of continuous runs of numbers in a 1D numpy array
   and returns the count of runs for each unique number and the unique
   numbers.

   :param arr: The input 1D numpy array of numbers.
   :type arr: np.ndarray

   :returns: A tuple containing a dictionary with the count of runs for
             each unique number and the set of unique numbers in the array.
   :rtype: tuple















   ..
       !! processed by numpydoc !!

.. py:function:: count_to(n)

   
   This function is equivalent to hstack((arange(n_i) for n_i in n)).
   It seems to be faster for some possible inputs and encapsulates
   a task in a function.

   .. rubric:: Example

   Given n = [0, 0, 3, 0, 0, 2, 0, 2, 1],
   the result would be [0, 1, 2, 0, 1, 0, 1, 0].















   ..
       !! processed by numpydoc !!

.. py:function:: fileContainsString(pname, searchStr)

.. py:function:: find_runs(x)

   
   Find runs of consecutive items in an array.

   Taken from:
   https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065















   ..
       !! processed by numpydoc !!

.. py:function:: fixAngle(a)

   
   Ensure angles lie between -pi and pi
   a must be in radians
















   ..
       !! processed by numpydoc !!

.. py:function:: flatten_list(list_to_flatten)

.. py:function:: getLabelEnds(x)

.. py:function:: getLabelStarts(x)

.. py:function:: get_z_score(x, mean=None, sd=None, axis=0)

   
   Calculate the z-scores for array x based on the mean
   and standard deviation in that sample, unless stated
















   ..
       !! processed by numpydoc !!

.. py:function:: labelContigNonZeroRuns(x)

.. py:function:: labelledCumSum(X, L)

.. py:function:: mean_norm(x, mn=None, axis=0)

.. py:function:: memmapBinaryFile(path2file, n_channels=384, **kwargs)

   
   Returns a numpy memmap of the int16 data in the
   file path2file, if present
















   ..
       !! processed by numpydoc !!

.. py:function:: min_max_norm(x, min=None, max=None, axis=0)

   
   Normalise the input array x to lie between min and max

   :param x (np.ndarray) - the array to normalise:
   :param min (float) - the minimun value in the returned array:
   :param max (float) - the maximum value in the returned array:
   :param axis - the axis along which to operate. Default 0:

   :rtype: out (np.ndarray) - the normalised array















   ..
       !! processed by numpydoc !!

.. py:function:: pol2cart(r, theta)

.. py:function:: polar(x, y, deg=False)

   
   Converts from rectangular coordinates to polar ones.

   :param x: The x and y coordinates.
   :type x: array_like, list_like
   :param y: The x and y coordinates.
   :type y: array_like, list_like
   :param deg: Radian if deg=0; degree if deg=1.
   :type deg: int

   :returns: The polar version of x and y.
   :rtype: p (array_like)















   ..
       !! processed by numpydoc !!

.. py:function:: rect(r, w, deg=False)

   
   Convert from polar (r,w) to rectangular (x,y)
   x = r cos(w)
   y = r sin(w)
















   ..
       !! processed by numpydoc !!

.. py:function:: remap_to_range(x, new_min=0, new_max=1, axis=0)

   
   Remap the values of x to the range [new_min, new_max].
















   ..
       !! processed by numpydoc !!

.. py:function:: repeat_ind(n)

   
   .. rubric:: Examples

   >>> n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
   >>> res = repeat_ind(n)
   >>> res = [2, 2, 2, 5, 5, 7, 7, 8]

   The input specifies how many times to repeat the given index.
   It is equivalent to something like this:

       hstack((zeros(n_i,dtype=int)+i for i, n_i in enumerate(n)))

   But this version seems to be faster, and probably scales better.
   At any rate, it encapsulates a task in a function.















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
   :rtype: array_like















   ..
       !! processed by numpydoc !!

.. py:function:: smooth(x, window_len=9, window='hanning')

   
   Smooth the data using a window with requested size.

   This method is based on the convolution of a scaled window with the signal.
   The signal is prepared by introducing reflected copies of the signal
   (with the window size) in both ends so that transient parts are minimized
   in the beginning and end part of the output signal.

   :param x: The input signal.
   :type x: array_like
   :param window_len: The length of the smoothing window.
   :type window_len: int
   :param window: The type of window from 'flat', 'hanning', 'hamming',
                  'bartlett', 'blackman'. 'flat' window will produce a moving average
                  smoothing.
   :type window: str

   :returns: The smoothed signal.
   :rtype: out (array_like)

   .. rubric:: Example

   >>> t=linspace(-2,2,0.1)
   >>> x=sin(t)+randn(len(t))*0.1
   >>> y=smooth(x)

   .. seealso::

      numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
      numpy.convolve, scipy.signal.lfilter

   .. rubric:: Notes

   The window parameter could be the window itself if an array instead of
   a string.















   ..
       !! processed by numpydoc !!

.. py:function:: window_rms(a, window_size)

   
   Returns the root mean square of the input a over a window of
   size window_size
















   ..
       !! processed by numpydoc !!

