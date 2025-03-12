ephysiopy.common.spikecalcs
===========================

.. py:module:: ephysiopy.common.spikecalcs


Classes
-------

.. autoapisummary::

   ephysiopy.common.spikecalcs.KSMeta
   ephysiopy.common.spikecalcs.SpikeCalcsAxona
   ephysiopy.common.spikecalcs.SpikeCalcsGeneric
   ephysiopy.common.spikecalcs.SpikeCalcsOpenEphys
   ephysiopy.common.spikecalcs.SpikeCalcsProbe


Functions
---------

.. autoapisummary::

   ephysiopy.common.spikecalcs.cluster_quality
   ephysiopy.common.spikecalcs.contamination_percent
   ephysiopy.common.spikecalcs.fit_smoothed_curve_to_xcorr
   ephysiopy.common.spikecalcs.get_param
   ephysiopy.common.spikecalcs.mahal
   ephysiopy.common.spikecalcs.xcorr


Module Contents
---------------

.. py:class:: KSMeta

   Bases: :py:obj:`tuple`


   .. py:attribute:: Amplitude


   .. py:attribute:: ContamPct


   .. py:attribute:: KSLabel


   .. py:attribute:: group


.. py:class:: SpikeCalcsAxona(spike_times, cluster, waveforms = None, **kwargs)

   Bases: :py:obj:`SpikeCalcsGeneric`


   
   Replaces SpikeCalcs from ephysiopy.axona.spikecalcs
















   ..
       !! processed by numpydoc !!

   .. py:method:: half_amp_dur(waveforms)

      
      Calculates the half amplitude duration of a spike.

      :param A: An nSpikes x nElectrodes x nSamples array.
      :type A: ndarray

      :returns:

                The half-amplitude duration for the channel
                    (electrode) that has the strongest (highest amplitude)
                    signal. Units are ms.
      :rtype: had (float)















      ..
          !! processed by numpydoc !!


   .. py:method:: p2t_time(waveforms)

      
      The peak to trough time of a spike in ms

      :param cluster: The cluster whose waveforms are to be analysed
      :type cluster: int

      :returns:

                The mean peak-to-trough time for the channel
                    (electrode) that has the strongest (highest amplitude) signal.
                    Units are ms.
      :rtype: p2t (float)















      ..
          !! processed by numpydoc !!


   .. py:method:: plotClusterSpace(waveforms, param='Amp', clusts = None, cluster_vec = None, **kwargs)

      
      Assumes the waveform data is signed 8-bit ints

      NB THe above assumption is mostly broken as waveforms by default are now
      in volts so you need to construct the trial object (AxonaTrial, OpenEphysBase
      etc) with volts=False (works for Axona, less sure about OE)
      TODO: aspect of plot boxes in ImageGrid not right as scaled by range of
      values now

      :param waveforms (np.ndarray) - the array of waveform data. For Axona recordings this: is nSpikes x nChannels x nSamplesPerWaveform
      :param param (str) - the parameter to plot. See get_param at the top of this file: for valid args
      :param clusts (optional - int or list) - which clusters to colour in:
      :param cluster_vec (optional - np.ndarray or list) - the cluster identity of each spike in waveforms: Must be nSpikes long















      ..
          !! processed by numpydoc !!


.. py:class:: SpikeCalcsGeneric(spike_times, cluster, waveforms = None, **kwargs)

   Bases: :py:obj:`object`


   
   Deals with the processing and analysis of spike data.
   There should be one instance of this class per cluster in the
   recording session. NB this differs from previous versions of this
   class where there was one instance per recording session and clusters
   were selected by passing in the cluster id to the methods.

   :param spike_times: The times of spikes in the trial in seconds
   :type spike_times: array_like
   :param waveforms: An nSpikes x nChannels x nSamples array
   :type waveforms: np.array, optional















   ..
       !! processed by numpydoc !!

   .. py:method:: acorr(Trange = np.array([-0.5, 0.5]), **kwargs)

      
      Calculates the autocorrelogram of a spike train

      :param ts: The spike times
      :type ts: np.ndarray
      :param Trange: The range of times to calculate the
                     autocorrelogram over
      :type Trange: np.ndarray

      Returns:
      result: (BinnedData): Container for the binned data















      ..
          !! processed by numpydoc !!


   .. py:method:: apply_filter(*trial_filter)

      
      Applies a mask to the spike times

      Args
          mask (list or tuple): The mask to apply to the spike times















      ..
          !! processed by numpydoc !!


   .. py:method:: contamination_percent(**kwargs)


   .. py:method:: get_ifr(spike_times, n_samples, **kwargs)

      
      Returns the instantaneous firing rate of the cluster

      :param ts: The times in seconds at which the cluster fired.
      :type ts: np.array
      :param n_samples: The number of samples to use in the calculation.
                        Practically this should be the number of position
                        samples in the recording.
      :type n_samples: int

      :returns: The instantaneous firing rate of the cluster
      :rtype: ifr (np.array)















      ..
          !! processed by numpydoc !!


   .. py:method:: get_ifr_power_spectrum(**kwargs)

      
      Returns the power spectrum of the instantaneous firing rate of a cell

      This is what is used to calculate the theta_mod_idxV3 score above















      ..
          !! processed by numpydoc !!


   .. py:method:: get_shuffled_ifr_sp_corr(ts, speed, nShuffles = 100, **kwargs)


   .. py:method:: ifr_sp_corr(ts, speed, minSpeed=2.0, maxSpeed=40.0, sigma=3, nShuffles=100, plot=False, **kwargs)

      
      Calculates the correlation between the instantaneous firing rate and
      speed.

      :param ts: The times in seconds at which the cluster fired.
      :type ts: np.array
      :param speed: Instantaneous speed (1 x nSamples).
      :type speed: np.array
      :param minSpeed: Speeds below this value are ignored.
                       Defaults to 2.0 cm/s as with Kropff et al., 2015.
      :type minSpeed: float, optional
      :param maxSpeed: Speeds above this value are ignored.
                       Defaults to 40.0 cm/s.
      :type maxSpeed: float, optional
      :param sigma: The standard deviation of the gaussian used
                    to smooth the spike train. Defaults to 3.
      :type sigma: int, optional
      :param nShuffles: The number of resamples to feed into
                        the permutation test. Defaults to 9999.
                        See scipy.stats.PermutationMethod.
      :type nShuffles: int, optional
      :param plot: Whether to plot the result.
                   Defaults to False.
      :type plot: bool, optional

      kwargs:
          method: how the significance of the speed vs firing rate correlation
                  is calculated - see the documentation for scipy.stats.PermutationMethod

                  An example of how I was calculating this is:

                  >> rng = np.random.default_rng()
                  >> method = stats.PermutationMethod(n_resamples=nShuffles, random_state=rng)















      ..
          !! processed by numpydoc !!


   .. py:method:: mean_isi_range(isi_range)

      
      Calculates the mean of the autocorrelation from 0 to n milliseconds
      Used to help classify a neurons type (principal, interneuron etc)

      :param isi_range: The range in ms to calculate the mean over
      :type isi_range: int

      :returns: The mean of the autocorrelogram between 0 and n milliseconds
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:method:: mean_waveform(channel_id = None)

      
      Returns the mean waveform and sem for a given spike train on a
      particular channel

      :param cluster_id: The cluster to get the mean waveform for
      :type cluster_id: int

      :returns:

                The mean waveforms, usually 4x50 for tetrode
                                    recordings
                std_wvs (ndarray): The standard deviations of the waveforms,
                                    usually 4x50 for tetrode recordings
      :rtype: mn_wvs (ndarray)















      ..
          !! processed by numpydoc !!


   .. py:method:: psch(bin_width_secs)

      
      Calculate the peri-stimulus *count* histogram of a cell's spiking
      against event times.

      :param bin_width_secs: The width of each bin in seconds.
      :type bin_width_secs: float

      :returns: Rows are counts of spikes per bin_width_secs.
                Size of columns ranges from self.event_window[0] to
                self.event_window[1] with bin_width_secs steps;
                so x is count, y is "event".
      :rtype: result (np.ndarray)















      ..
          !! processed by numpydoc !!


   .. py:method:: psth()

      
      Calculate the PSTH of event_ts against the spiking of a cell

      :param cluster_id: The cluster for which to calculate the psth
      :type cluster_id: int

      :returns:

                The list of time differences between the spikes of
                                the cluster and the events (x) and the trials (y)
      :rtype: x, y (list)















      ..
          !! processed by numpydoc !!


   .. py:method:: responds_to_stimulus(threshold, min_contiguous, return_activity = False, return_magnitude = False, **kwargs)

      
      Checks whether a cluster responds to a laser stimulus.

      :param cluster: The cluster to check.
      :type cluster: int
      :param threshold: The amount of activity the cluster needs to go
                        beyond to be classified as a responder (1.5 = 50% more or less
                        than the baseline activity).
      :type threshold: float
      :param min_contiguous: The number of contiguous samples in the
                             post-stimulus period for which the cluster needs to be active
                             beyond the threshold value to be classed as a responder.
      :type min_contiguous: int
      :param return_activity: Whether to return the mean reponse curve.
      :type return_activity: bool
      :param return_magnitude: Whether to return the magnitude of the
                               response. NB this is either +1 for excited or -1 for inhibited.
      :type return_magnitude: int

      :returns: Whether the cell responds or not.
                OR
                tuple: responds (bool), normed_response_curve (np.ndarray).
                OR
                tuple: responds (bool), normed_response_curve (np.ndarray),
                    response_magnitude (np.ndarray).
      :rtype: responds (bool)















      ..
          !! processed by numpydoc !!


   .. py:method:: smooth_spike_train(npos, sigma=3.0, shuffle=None)

      
      Returns a spike train the same length as num pos samples that has been
      smoothed in time with a gaussian kernel M in width and standard
      deviation equal to sigma.

      :param x1: The pos indices the spikes occurred at.
      :type x1: np.array
      :param npos: The number of position samples captured.
      :type npos: int
      :param sigma: The standard deviation of the gaussian used to
                    smooth the spike train.
      :type sigma: float
      :param shuffle: The number of seconds to shift the spike
                      train by. Default is None.
      :type shuffle: int, optional

      :returns: The smoothed spike train.
      :rtype: smoothed_spikes (np.array)















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_band_max_freq()

      
      Calculates the frequency with the maximum power in the theta band (6-12Hz)
      of a spike train's autocorrelogram.

      This function is used to look for differences in theta frequency in
      different running directions as per Blair.
      See Welday paper - https://doi.org/10.1523/jneurosci.0712-11.2011

      :param x1: The spike train for which the autocorrelogram will be
                 calculated.
      :type x1: np.ndarray

      :returns: The frequency with the maximum power in the theta band.
      :rtype: float

      :raises ValueError: If the input spike train is not valid.















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_mod_idx(**kwargs)

      
      Calculates a theta modulation index of a spike train based on the cells
      autocorrelogram.

      The difference of the mean power in the theta frequency band (6-11 Hz) and
      the mean power in the 1-50 Hz frequency band is divided by their sum to give
      a metric that lives between 0 and 1

      :param x1: The spike time-series.
      :type x1: np.array

      :returns: The difference of the values at the first peak
                and trough of the autocorrelogram.
      :rtype: thetaMod (float)

      NB This is a fairly skewed metric with a distribution strongly biased
      to -1 (although more evenly distributed than theta_mod_idxV2 below)















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_mod_idxV2()

      
      This is a simpler alternative to the theta_mod_idx method in that it
      calculates the difference between the normalized temporal
      autocorrelogram at the trough between 50-70ms and the
      peak between 100-140ms over their sum (data is binned into 5ms bins)

      Measure used in Cacucci et al., 2004 and Kropff et al 2015















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_mod_idxV3(**kwargs)

      
      Another theta modulation index score this time based on the method used
      by Kornienko et al., (2024) (Kevin Allens lab)
      see https://doi.org/10.7554/eLife.35949.001

      Basically uses the binned spike train instead of the autocorrelogram as
      the input to the periodogram function (they use pwelch in R; periodogram is a
      simplified call to welch in scipy.signal)

      The resulting metric is similar to the one in theta_mod_idx above except
      that the frequency bands compared to the theta band are narrower and
      exclusive of the theta band

      Produces a fairly normally distributed looking score with a mean and median
      pretty close to 0















      ..
          !! processed by numpydoc !!


   .. py:method:: trial_mean_fr()


   .. py:method:: update_KSMeta(value)

      
      Takes in a TemplateModel instance from a phy session and
      parses out the relevant metrics for the cluster and places
      into the namedtuple KSMeta
















      ..
          !! processed by numpydoc !!


   .. py:method:: waveforms(channel_id = None)


   .. py:property:: KSMeta
      :type: KSMetaTuple



   .. py:attribute:: _duration
      :value: None



   .. py:attribute:: _event_ts
      :value: None



   .. py:attribute:: _event_window


   .. py:attribute:: _ksmeta


   .. py:attribute:: _pos_sample_rate
      :value: 50



   .. py:attribute:: _post_spike_samples
      :value: 34



   .. py:attribute:: _pre_spike_samples
      :value: 16



   .. py:attribute:: _sample_rate
      :value: 30000



   .. py:attribute:: _secs_per_bin
      :value: 0.001



   .. py:attribute:: _stim_width
      :value: None



   .. py:attribute:: cluster


   .. py:property:: duration
      :type: float | int | None



   .. py:property:: event_ts
      :type: numpy.ndarray



   .. py:property:: event_window
      :type: numpy.ndarray



   .. py:property:: n_spikes

      
      Returns the number of spikes in the cluster

      :returns: The number of spikes in the cluster
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:property:: pos_sample_rate
      :type: int | float



   .. py:property:: post_spike_samples
      :type: int



   .. py:property:: pre_spike_samples
      :type: int



   .. py:property:: sample_rate
      :type: int | float



   .. py:property:: secs_per_bin
      :type: float | int



   .. py:attribute:: spike_times


   .. py:property:: stim_width
      :type: int | float | None



.. py:class:: SpikeCalcsOpenEphys(spike_times, cluster, waveforms=None, **kwargs)

   Bases: :py:obj:`SpikeCalcsGeneric`


   
   Deals with the processing and analysis of spike data.
   There should be one instance of this class per cluster in the
   recording session. NB this differs from previous versions of this
   class where there was one instance per recording session and clusters
   were selected by passing in the cluster id to the methods.

   :param spike_times: The times of spikes in the trial in seconds
   :type spike_times: array_like
   :param waveforms: An nSpikes x nChannels x nSamples array
   :type waveforms: np.array, optional















   ..
       !! processed by numpydoc !!

   .. py:method:: get_channel_depth_from_templates(pname)

      
      Determine depth of template as well as closest channel. Adopted from
      'templatePositionsAmplitudes' by N. Steinmetz
      (https://github.com/cortex-lab/spikes)
















      ..
          !! processed by numpydoc !!


   .. py:method:: get_template_id_for_cluster(pname, cluster)

      
      Determine the best channel (one with highest amplitude spikes)
      for a given cluster.
















      ..
          !! processed by numpydoc !!


   .. py:method:: get_waveforms(cluster, cluster_data, n_waveforms = 2000, n_channels = 64, channel_range=None, **kwargs)

      
      Returns waveforms for a cluster.

      :param cluster: The cluster to return the waveforms for.
      :type cluster: int
      :param cluster_data: The KiloSortSession object for the
                           session that contains the cluster.
      :type cluster_data: KiloSortSession
      :param n_waveforms: The number of waveforms to return.
                          Defaults to 2000.
      :type n_waveforms: int, optional
      :param n_channels: The number of channels in the
                         recording. Defaults to 64.
      :type n_channels: int, optional















      ..
          !! processed by numpydoc !!


   .. py:attribute:: TemplateModel
      :value: None



   .. py:attribute:: n_samples


.. py:class:: SpikeCalcsProbe

   Bases: :py:obj:`SpikeCalcsGeneric`


   
   Encapsulates methods specific to probe-based recordings
















   ..
       !! processed by numpydoc !!

.. py:function:: cluster_quality(waveforms = None, spike_clusters = None, cluster_id = None, fet = 1)

   
   Returns the L-ratio and Isolation Distance measures calculated
   on the principal components of the energy in a spike matrix.

   :param waveforms: The waveforms to be processed.
                     If None, the function will return None.
   :type waveforms: np.ndarray, optional
   :param spike_clusters: The spike clusters to be
                          processed.
   :type spike_clusters: np.ndarray, optional
   :param cluster_id: The ID of the cluster to be processed.
   :type cluster_id: int, optional
   :param fet: The feature to be used in the PCA calculation.
   :type fet: int, default=1

   :returns:

             A tuple containing the L-ratio and Isolation Distance of the
                 cluster.
   :rtype: tuple

   :raises Exception: If an error occurs during the calculation of the L-ratio or
       Isolation Distance.















   ..
       !! processed by numpydoc !!

.. py:function:: contamination_percent(x1, x2 = None, **kwargs)

   
   Computes the cross-correlogram between two sets of spikes and
   estimates how refractory the cross-correlogram is.

   :param st1: The first set of spikes.
   :type st1: np.array
   :param st2: The second set of spikes.
   :type st2: np.array

   kwargs:
       Anything that can be fed into xcorr above

   :returns: a measure of refractoriness
             R (float): a second measure of refractoriness
                     (kicks in for very low firing rates)
   :rtype: Q (float)

   .. rubric:: Notes

   Taken from KiloSorts ccg.m

   The contamination metrics are calculated based on
   an analysis of the 'shoulders' of the cross-correlogram.
   Specifically, the spike counts in the ranges +/-5-25ms and
   +/-250-500ms are compared for refractoriness















   ..
       !! processed by numpydoc !!

.. py:function:: fit_smoothed_curve_to_xcorr(xc, **kwargs)

   
   Idea is to smooth out the result of an auto- or cross-correlogram with
   a view to correlating the result with another auto- or cross-correlogram
   to see how similar two of these things are.
















   ..
       !! processed by numpydoc !!

.. py:function:: get_param(waveforms, param='Amp', t=200, fet=1)

   
   Returns the requested parameter from a spike train as a numpy array

   :param waveforms: Shape of array can be nSpikes x nSamples
                     OR
                     a nSpikes x nElectrodes x nSamples
   :type waveforms: numpy array
   :param param: Valid values are:
                 'Amp' - peak-to-trough amplitude (default)
                 'P' - height of peak
                 'T' - depth of trough
                 'Vt' height at time t
                 'tP' - time of peak (in seconds)
                 'tT' - time of trough (in seconds)
                 'PCA' - first n fet principal components (defaults to 1)
   :type param: str
   :param t: The time used for Vt
   :type t: int
   :param fet: The number of principal components
               (use with param 'PCA')
   :type fet: int















   ..
       !! processed by numpydoc !!

.. py:function:: mahal(u, v)

   
   Returns the L-ratio and Isolation Distance measures calculated on the
   principal components of the energy in a spike matrix.

   :param waveforms: The waveforms to be processed. If
                     None, the function will return None.
   :type waveforms: np.ndarray, optional
   :param spike_clusters: The spike clusters to be
                          processed.
   :type spike_clusters: np.ndarray, optional
   :param cluster_id: The ID of the cluster to be processed.
   :type cluster_id: int, optional
   :param fet: The feature to be used in the PCA calculation.
   :type fet: int, default=1

   :returns:

             A tuple containing the L-ratio and Isolation Distance of the
                 cluster.
   :rtype: tuple

   :raises Exception: If an error occurs during the calculation of the L-ratio or
       Isolation Distance.















   ..
       !! processed by numpydoc !!

.. py:function:: xcorr(x1, x2 = None, Trange = np.array([-0.5, 0.5]), binsize = 0.001, normed=False, **kwargs)

   
   Calculates the ISIs in x1 or x1 vs x2 within a given range

   :param x1: The times of the spikes emitted by the
              cluster(s) in seconds
   :type x1: array_like
   :param x2: The times of the spikes emitted by the
              cluster(s) in seconds
   :type x2: array_like
   :param Trange: Range of times to bin up in seconds
                  Defaults to [-0.5, +0.5]
   :type Trange: array_like
   :param binsize: The size of the bins in seconds
   :type binsize: float
   :param normed: Whether to divide the counts by the total
                  number of spikes to give a probabilty
   :type normed: bool
   :param \*\*kwargs - just there to suck up spare parameters:

   :returns:

             A BinnedData object containing the binned data and the
                         bin edges
   :rtype: BinnedData















   ..
       !! processed by numpydoc !!

