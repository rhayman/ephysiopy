ephysiopy.common.ephys_generic
==============================

.. py:module:: ephysiopy.common.ephys_generic

.. autoapi-nested-parse::

   The classes contained in this module are supposed to be agnostic to recording
   format and encapsulate some generic mechanisms for producing
   things like spike timing autocorrelograms, power spectrum calculation and so on

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   ephysiopy.common.ephys_generic.EEGCalcsGeneric
   ephysiopy.common.ephys_generic.EventsGeneric
   ephysiopy.common.ephys_generic.PosCalcsGeneric


Functions
---------

.. autoapisummary::

   ephysiopy.common.ephys_generic.calculate_rms_and_std
   ephysiopy.common.ephys_generic.downsample_aux
   ephysiopy.common.ephys_generic.find_high_amp_long_duration


Module Contents
---------------

.. py:class:: EEGCalcsGeneric(sig, fs)

   Bases: :py:obj:`object`


   
   Generic class for processing and analysis of EEG data

   :param sig: The signal (of the LFP data)
   :type sig: np.ndarray
   :param fs: The sample rate
   :type fs: float















   ..
       !! processed by numpydoc !!

   .. py:method:: _nextpow2(val)

      
      Calculates the next power of 2 that will hold val
















      ..
          !! processed by numpydoc !!


   .. py:method:: apply_mask(mask)

      
      Applies a mask to the signal

      :param mask: The mask to be applied. For use with np.ma.MaskedArray's mask attribute
      :type mask: np.ndarray

      .. rubric:: Notes

      If mask is empty, the mask is removed
      The mask should be a list of tuples, each tuple containing
      the start and end times of the mask i.e. [(start1, end1), (start2, end2)]
      everything inside of these times is masked















      ..
          !! processed by numpydoc !!


   .. py:method:: butterFilter(low, high, order = 5)

      
       Filters self.sig with a butterworth filter with a bandpass filter
       defined by low and high

      :param low: the lower and upper bounds of the bandpass filter
      :type low: float
      :param high: the lower and upper bounds of the bandpass filter
      :type high: float
      :param order: the order of the filter
      :type order: int
      :param Returns:
      :param -------:
      :param filt: the filtered signal
      :type filt: np.ndarray
      :param Notes:
      :param -----:
      :param the signal is filtered in both the forward and: reverse directions (scipy.signal.filtfilt)















      ..
          !! processed by numpydoc !!


   .. py:method:: calcEEGPowerSpectrum(**kwargs)

      
      Calculates the power spectrum of self.sig

      :returns: * **psd** (*tuple[np.ndarray, float,...]*)
                * *A 5-tuple of the following and sets a bunch of member variables*
                * **freqs (array_like)** (*The frequencies at which the spectrogram*)
                * *was calculated*
                * **power (array_like)** (*The power at the frequencies defined above*)
                * **sm_power (array_like)** (*The smoothed power*)
                * **bandmaxpower (float)** (*The maximum power in the theta band*)
                * **freqatbandmaxpower (float)** (*The frequency at which the power*)
                * *is maximum*















      ..
          !! processed by numpydoc !!


   .. py:method:: ifftFilter(sig, freqs, fs=250)

      
      Calculates the dft of signal and filters out the frequencies in
      freqs from the result and reconstructs the original signal using
      the inverse fft without those frequencies

      :param sig: the LFP signal to be filtered
      :type sig: np.ndarray
      :param freqs: the frequencies to be filtered out
      :type freqs: list
      :param fs: the sampling frequency of sig
      :type fs: int

      :returns: **fftRes** -- the filtered LFP signal
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: fs


   .. py:attribute:: maxFreq
      :value: 125



   .. py:attribute:: maxPow
      :value: None



   .. py:attribute:: outsideRange
      :value: [3, 125]



   .. py:attribute:: sig


   .. py:attribute:: smthKernelSigma
      :value: 0.1875



   .. py:attribute:: smthKernelWidth
      :value: 2



   .. py:attribute:: sn2Width
      :value: 2



   .. py:attribute:: thetaRange
      :value: [6, 12]



.. py:class:: EventsGeneric

   Bases: :py:obj:`object`


   
   Holds records of events, specifically for now, TTL events produced
   by either the Axona recording system or an Arduino-based plugin I
   (RH) wrote for the open-ephys recording system.

   Idea is to present a generic interface to other classes/ functions
   regardless of how the events were created.

   As a starting point lets base this on the axona STM class which extends
   dict() and axona.axonaIO.IO().

   For a fairly complete description of the nomenclature used for the
   timulation / event parameters see the STM property of the
   axonaIO.Stim() class

   Once a .stm file is loaded the keys for STM are:

   .. attribute:: on

      time in samples of the event

      :type: np.array

   .. attribute:: trial_date

      :type: str

   .. attribute:: trial_time

      :type: str

   .. attribute:: experimenter

      :type: str

   .. attribute:: comments

      :type: str

   .. attribute:: duration

      :type: str

   .. attribute:: sw_version

      :type: str

   .. attribute:: num_chans

      :type: str

   .. attribute:: timebase

      :type: str

   .. attribute:: bytes_per_timestamp

      :type: str

   .. attribute:: data_format

      :type: str

   .. attribute:: num_stm_samples

      :type: str

   .. attribute:: posSampRate

      :type: int

   .. attribute:: eegSampRate

      :type: int

   .. attribute:: egfSampRate

      :type: int

   .. attribute:: off

      :type: np.ndarray

   .. attribute:: stim_params

      This has keys:
          Phase_1 : str
          Phase_2 : str
          Phase_3 : str
          etc
          Each of these keys is also a dict with keys:
              startTime: None
              duration: int
                  in seconds
              name: str
              pulseWidth: int
                  microseconds
              pulseRatio: None
              pulsePause: int
                  microseconds

      :type: OrderedDict

   .. attribute:: The most important entries are the on and off numpy arrays and pulseWidth,

   .. attribute:: the last mostly for plotting purposes.

   .. attribute:: Let's emulate that dict generically so it can be co-opted for use with

   .. attribute:: the various types of open-ephys recordings using the Arduino-based plugin

   .. attribute:: (called StimControl - see https

      :type: //github.com/rhayman/StimControl)















   ..
       !! processed by numpydoc !!

   .. py:attribute:: _event_dict


.. py:class:: PosCalcsGeneric(x, y, ppm, convert2cm = True, jumpmax = 100, **kwargs)

   Bases: :py:obj:`object`


   
   Generic class for post-processing of position data
   Uses numpys masked arrays for dealing with bad positions, filtering etc

   :param x: the x and y positions
   :type x: np.ndarray
   :param y: the x and y positions
   :type y: np.ndarray
   :param ppm: Pixels per metre
   :type ppm: int
   :param convert2cm: Whether everything is converted into cms or not
   :type convert2cm: bool
   :param jumpmax: Jumps in position (pixel coords) > than this are bad
   :type jumpmax: int
   :param \*\*kwargs: a dict[str, float] called 'tracker_params' is used to limit
                      the range of valid xy positions - 'bad' positions are masked out
                      and interpolated over

   .. attribute:: orig_xy

      the original xy coordinates, never modified directly

      :type: np.ndarray

   .. attribute:: npos

      the number of position samples

      :type: int

   .. attribute:: xy

      2 x npos array

      :type: np.ndarray

   .. attribute:: convert2cm

      whether to convert the xy position data to cms or not

      :type: bool

   .. attribute:: duration

      the trial duration in seconds

      :type: float

   .. attribute:: xyTS

      the timestamps the position data was recorded at. npos long vector

      :type: np.ndarray

   .. attribute:: dir

      the directional data. In degrees

      :type: np.ndarray

   .. attribute:: ppm

      the number of pixels per metre

      :type: float

   .. attribute:: jumpmax

      the minimum jump between consecutive positions before a jump is considered 'bad'
      and smoothed over

      :type: float

   .. attribute:: speed

      the speed data, extracted from a difference of xy positions. npos long vector

      :type: np.ndarray

   .. attribute:: sample_rate

      the sample rate of the position data

      :type: int

   .. rubric:: Notes

   The positional data (x,y) is turned into a numpy masked array once this
   class is initialised - that mask is then modified through various
   functions (postprocesspos being the main one).















   ..
       !! processed by numpydoc !!

   .. py:method:: apply_mask(mask)

      
      Applies a mask to the position data

      :param mask: The mask to be applied. For use with np.ma.MaskedArray's mask attribute
      :type mask: np.ndarray

      .. rubric:: Notes

      If mask is empty, the mask is removed
      The mask should be a list of tuples, each tuple containing
      the start and end times of the mask i.e. [(start1, end1), (start2, end2)]
      everything inside of these times is masked















      ..
          !! processed by numpydoc !!


   .. py:method:: calcHeadDirection(xy)

      
      Calculates the head direction from the xy data

      :param xy: The xy data
      :type xy: np.ma.MaskedArray

      :returns: The head direction data
      :rtype: np.ma.MaskedArray















      ..
          !! processed by numpydoc !!


   .. py:method:: calcSpeed(xy)

      
      Calculates speed

      :param xy: The xy positional data
      :type xy: np.ma.MaskedArray















      ..
          !! processed by numpydoc !!


   .. py:method:: interpnans(xy)

      
      Interpolates over bad values in the xy data

      :param xy:
      :type xy: np.ma.MaskedArray

      :returns: The interpolated xy data
      :rtype: np.ma.MaskedArray















      ..
          !! processed by numpydoc !!


   .. py:method:: postprocesspos(tracker_params = {})

      
      Post-process position data

      :param tracker_params: Same dict as created in OESettings.Settings.parse
                             (from module openephys2py)
      :type tracker_params: dict

      .. rubric:: Notes

      Several internal functions are called here: speedfilter,
      interpnans, smoothPos and calcSpeed.
      Some internal state/ instance variables are set as well. The
      mask of the positional data (an instance of numpy masked array)
      is modified throughout this method.















      ..
          !! processed by numpydoc !!


   .. py:method:: smoothPos(xy)

      
      Smooths position data

      :param xy: The xy data
      :type xy: np.ma.MaskedArray

      :returns: **xy** -- The smoothed positional data
      :rtype: array_like















      ..
          !! processed by numpydoc !!


   .. py:method:: smooth_speed(speed, window_len = 21)

      
      Smooth speed data with a window a little bit bigger than the usual
      400ms window used for smoothing position data

      NB Uses a box car filter as with Axona















      ..
          !! processed by numpydoc !!


   .. py:method:: speedfilter(xy)

      
      Filters speed

      :param xy: The xy data
      :type xy: np.ma.MaskedArray

      :returns: The xy data with speeds >
                self.jumpmax masked
      :rtype: xy (np.ma.MaskedArray)















      ..
          !! processed by numpydoc !!


   .. py:method:: upsamplePos(xy, upsample_rate = 50)

      
      Upsamples position data from 30 to upsample_rate

      :param xy: The xy positional data
      :type xy: np.ma.MaskedArray
      :param upsample_rate: The rate to upsample to
      :type upsample_rate: int

      :returns: The upsampled xy positional data
      :rtype: np.ma.MaskedArray

      .. rubric:: Notes

      This is mostly to get pos data recorded using PosTracker at 30Hz
      into Axona format 50Hz data















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _convert2cm
      :value: True



   .. py:attribute:: _dir


   .. py:attribute:: _jumpmax
      :value: 100



   .. py:attribute:: _ppm


   .. py:attribute:: _sample_rate
      :value: 30



   .. py:attribute:: _speed


   .. py:attribute:: _xy


   .. py:attribute:: _xyTS
      :value: None



   .. py:property:: convert2cm
      :type: bool



   .. py:property:: dir
      :type: numpy.ma.MaskedArray



   .. py:property:: duration
      :type: float



   .. py:property:: jumpmax
      :type: float



   .. py:attribute:: nleds


   .. py:property:: npos


   .. py:attribute:: orig_xy
      :type:  numpy.ma.MaskedArray


   .. py:property:: ppm
      :type: float



   .. py:property:: sample_rate
      :type: int



   .. py:property:: speed
      :type: numpy.ma.MaskedArray



   .. py:property:: xy
      :type: numpy.ma.MaskedArray



   .. py:property:: xyTS
      :type: numpy.ma.MaskedArray | None



.. py:function:: calculate_rms_and_std(sig, time_window = [0, 10], fs = 50)

   
   Calculate the root mean square value for time_window (in seconds)

   :param sig: the downsampled AUX data (single channel)
   :type sig: np.ndarray
   :param time_window: the range of times in seconds to calculate the RMS for
   :type time_window: list
   :param fs: the sampling frequency of sig
   :type fs: int

   :returns: the RMS and standard deviation of the signal
   :rtype: tuple of np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: downsample_aux(data, source_freq = 30000, target_freq = 50, axis=-1)

   
   Downsamples the default 30000Hz AUX signal to a default of 500Hz

   :param data: the source data
   :type data: np.ndarray
   :param source_freq: the sampling frequency of data
   :type source_freq: int
   :param target_freq: the desired output frequency of the data
   :type target_freq: int
   :param axis: the axis along which to apply the resampling
   :type axis: int

   :returns: the downsampled data
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: find_high_amp_long_duration(raw_signal, fs, amp_std = 3, duration_range = [0.03, 0.11], duration_std = 1)

   
   Find periods of high amplitude and long duration in the ripple bandpass
   filtered signal.

   :param raw_signal: the raw LFP signal which will be filtered here
   :type raw_signal: np.ndarray
   :param fs: the sampliing frequency of the raw signal
   :type fs: int
   :param amp_std: the signal needs to be this many standard deviations above the mean
   :type amp_std: int
   :param duration: the minimum and maximum durations in seconds for the ripples
   :type duration: list of int
   :param duration_std: how many standard deviations above the mean the ripples should
                        be for 'duration' ms
   :type duration_std: int

   :returns: the bandpass filtered LFP that has been masked outside of epochs that don't meet the above thresholds
   :rtype: np.ma.MaskedArray

   .. rubric:: Notes

   From Todorova & Zugaro (supp info):

   "To detect ripple events, we first detrended the LFP signals and used the Hilbert transform
   to compute the ripple band (100–250 Hz) amplitude for each channel recorded from the
   CA1 pyramidal layer. We then averaged these amplitudes, yielding the mean instanta-
   neous ripple amplitude. To exclude events of high spectral power not specific to the ripple
   band, we then subtracted the mean high-frequency (300–500 Hz) amplitude (if the differ-
   ence was negative, we set it to 0). Finally, we z-scored this signal, yielding a corrected
   and normalized ripple amplitude R(t). Ripples were defined as events where R(t) crossed
   a threshold of 3 s.d. and remained above 1 s.d. for 30 to 110 ms."

   .. rubric:: References

   Todorova & Zugaro, 2019. Isolated cortical computations during delta waves support memory consolidation. 366: 6463
   doi: 10.1126/science.aay0616















   ..
       !! processed by numpydoc !!

