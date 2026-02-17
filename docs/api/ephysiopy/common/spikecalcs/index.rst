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
   ephysiopy.common.spikecalcs.get_burstiness
   ephysiopy.common.spikecalcs.get_param
   ephysiopy.common.spikecalcs.get_peak_to_trough_time
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


.. py:class:: SpikeCalcsAxona(spike_times, cluster, waveforms = None, *args, **kwargs)

   Bases: :py:obj:`SpikeCalcsGeneric`


   
   Replaces SpikeCalcs from ephysiopy.axona.spikecalcs
















   ..
       !! processed by numpydoc !!

   .. py:method:: half_amp_dur(waveforms)

      
      Calculates the half amplitude duration of a spike.

      :param A: An nSpikes x nElectrodes x nSamples array.
      :type A: np.ndarray

      :returns: The half-amplitude duration for the channel
                (electrode) that has the strongest (highest amplitude)
                signal. Units are ms.
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:method:: p2t_time(waveforms)

      
      The peak to trough time of a spike in ms

      :param cluster: The cluster whose waveforms are to be analysed
      :type cluster: int

      :returns: The mean peak-to-trough time for the channel
                (electrode) that has the strongest (highest amplitude) signal.
                Units are ms.
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:method:: plotClusterSpace(waveforms, param='Amp', clusts = None, cluster_vec = None, **kwargs)

      
      Assumes the waveform data is signed 8-bit ints

      NB THe above assumption is mostly broken as waveforms by default are now
      in volts so you need to construct the trial object (AxonaTrial, OpenEphysBase
      etc) with volts=False (works for Axona, less sure about OE)
      TODO: aspect of plot boxes in ImageGrid not right as scaled by range of
      values now

      :param waveforms: the array of waveform data. For Axona recordings this
                        is nSpikes x nChannels x nSamplesPerWaveform
      :type waveforms: np.ndarray
      :param param: the parameter to plot. See get_param at the top of this file
                    for valid args
      :type param: str
      :param clusts: which clusters to colour in
      :type clusts: int, list or None, default None
      :param cluster_vec: the cluster identity of each spike in waveforms must be nSpikes long
      :type cluster_vec: np.ndarray, list or None, default None
      :param \*\*kwargs: passed into ImageGrid















      ..
          !! processed by numpydoc !!


.. py:class:: SpikeCalcsGeneric(spike_times, cluster, waveforms = None, **kwargs)

   Bases: :py:obj:`object`


   
   Deals with the processing and analysis of spike data.
   There should be one instance of this class per cluster in the
   recording session. NB this differs from previous versions of this
   class where there was one instance per recording session and clusters
   were selected by passing in the cluster id to the methods.

   NB Axona waveforms are nSpikes x nChannels x nSamples - this boils
   down to nSpikes x 4 x 50
   NB KiloSort waveforms are nSpikes x nSamples x nChannels - these are ordered
   by 'best' channel first and then the rest of the channels. This boils
   down to nSpikes x 61 x 12 SO THIS NEEDS TO BE CHANGED to
   nSpikes x nChannels x nSamples

   :param spike_times: The times of spikes in the trial in seconds.
   :type spike_times: np.ndarray
   :param cluster: The cluster ID.
   :type cluster: int
   :param waveforms: An nSpikes x nChannels x nSamples array.
   :type waveforms: np.ndarray, optional
   :param \*\*kwargs: Additional keyword arguments.
   :type \*\*kwargs: dict

   .. attribute:: spike_times

      The times of spikes in the trial in seconds.

      :type: np.ma.MaskedArray

   .. attribute:: _waves

      The waveforms of the spikes.

      :type: np.ma.MaskedArray or None

   .. attribute:: cluster

      The cluster ID.

      :type: int

   .. attribute:: n_spikes

      the total number of spikes for the current cluster

      :type: int

   .. attribute:: duration

      total duration of the trial in seconds

      :type: float, int

   .. attribute:: event_ts

      The times that events occurred in seconds.

      :type: np.ndarray or None

   .. attribute:: event_window

      The window, in seconds, either side of the stimulus, to examine.

      :type: np.ndarray

   .. attribute:: stim_width

      The width, in ms, of the stimulus.

      :type: float or None

   .. attribute:: secs_per_bin

      The size of bins in PSTH.

      :type: float

   .. attribute:: sample_rate

      The sample rate of the recording.

      :type: int

   .. attribute:: pos_sample_rate

      The sample rate of the position data.

      :type: int

   .. attribute:: pre_spike_samples

      The number of samples before the spike.

      :type: int

   .. attribute:: post_spike_samples

      The number of samples after the spike.

      :type: int

   .. attribute:: KSMeta

      The metadata from KiloSort.

      :type: KSMetaTuple















   ..
       !! processed by numpydoc !!

   .. py:method:: acorr(Trange = np.array([-0.5, 0.5]), **kwargs)

      
      Calculates the autocorrelogram of a spike train.

      :param Trange: The range of times to calculate the autocorrelogram over (default is [-0.5, 0.5]).
      :type Trange: np.ndarray, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: Container for the binned data.
      :rtype: BinnedData















      ..
          !! processed by numpydoc !!


   .. py:method:: apply_filter(*trial_filter)

      
      Applies a mask to the spike times.

      :param trial_filter: The filter
      :type trial_filter: TrialFilter















      ..
          !! processed by numpydoc !!


   .. py:method:: contamination_percent(**kwargs)

      
      Returns the contamination percentage of a spike train.

      :param \*\*kwargs: Passed into the contamination_percent function.

      :returns: Q - A measure of refractoriness.
                R - A second measure of refractoriness (kicks in for very low firing rates).
      :rtype: tuple of float















      ..
          !! processed by numpydoc !!


   .. py:method:: estimate_AHP()

      
      Estimate the decay time for the AHP of the waveform of the
      best channel for the current cluster.

      :returns: The estimated AHP decay time in microseconds,
                or None if no waveforms are available.
      :rtype: float | None















      ..
          !! processed by numpydoc !!


   .. py:method:: get_best_channel()

      
      Returns the channel with the highest mean amplitude of the waveforms.

      :returns: The index of the channel with the highest mean amplitude,
                or None if no waveforms are available.
      :rtype: int | None















      ..
          !! processed by numpydoc !!


   .. py:method:: get_ifr(spike_times, n_samples, **kwargs)

      
      Returns the instantaneous firing rate of the cluster

      :param ts: The times in seconds at which the cluster fired.
      :type ts: np.ndarray
      :param n_samples: The number of samples to use in the calculation.
                        Practically this should be the number of position
                        samples in the recording.
      :type n_samples: int

      :returns: The instantaneous firing rate of the cluster
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_ifr_power_spectrum()

      
      Returns the power spectrum of the instantaneous firing rate of a cell

      Used to calculate the theta_mod_idxV3 score above

      :returns: The frequency and power of the instantaneous firing rate
      :rtype: tuple of np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: get_shuffled_ifr_sp_corr(ts, speed, nShuffles = 100, **kwargs)

      
      Returns an nShuffles x nSamples sized array of shuffled
      instantaneous firing rate x speed correlations

      :param ts: the times in seconds at which the cluster fired
      :type ts: np.ndarray
      :param speed: the speed vector
      :type speed: np.ndarray
      :param nShuffles: the number of times to shuffle the timestamp vector 'ts'
      :type nShuffles: int
      :param \*\*kwargs: Passed into ifr_sp_corr

      :returns: A nShuffles x nSamples sized array of the shuffled firing rate vs
                speed correlations.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: ifr_sp_corr(ts, speed, minSpeed=2.0, maxSpeed=40.0, sigma=3, nShuffles=100, **kwargs)

      
      Calculates the correlation between the instantaneous firing rate and
      speed.

      :param ts: The times in seconds at which the cluster fired.
      :type ts: np.ndarray
      :param speed: Instantaneous speed (nSamples lenght vector).
      :type speed: np.ndarray
      :param minSpeed: Speeds below this value are ignored.
      :type minSpeed: float, default=2.0
      :param maxSpeed: Speeds above this value are ignored.
      :type maxSpeed: float, default=40.0
      :param sigma: The standard deviation of the gaussian used
                    to smooth the spike train.
      :type sigma: int, default=3
      :param nShuffles: The number of resamples to feed into
                        the permutation test.
      :type nShuffles: int, default=100
      :param \*\*kwargs:
                         method: how the significance of the speed vs firing rate correlation
                                 is calculated

      .. rubric:: Examples

      An example of how I was calculating this is:

      >> rng = np.random.default_rng()
      >> method = stats.PermutationMethod(n_resamples=nShuffles, random_state=rng)

      .. seealso:: :py:obj:`See`















      ..
          !! processed by numpydoc !!


   .. py:method:: mean_isi_range(isi_range)

      
      Calculates the mean of the autocorrelation from 0 to n seconds.
      Used to help classify a neuron's type (principal, interneuron, etc).

      :param isi_range: The range in seconds to calculate the mean over.
      :type isi_range: int

      :returns: The mean of the autocorrelogram between 0 and n seconds.
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:method:: mean_waveform(channel_id = None)

      
      Returns the mean waveform and standard error of the mean (SEM) for a
      given spike train on a particular channel.

      :param channel_id: The channel IDs to return the mean waveform for. If None, returns
                         mean waveforms for all channels.
      :type channel_id: Sequence, optional

      :returns: A tuple containing:
                - mn_wvs (np.ndarray): The mean waveforms, usually 4x50 for tetrode recordings.
                - std_wvs (np.ndarray): The standard deviations of the waveforms, usually 4x50 for tetrode recordings.
      :rtype: tuple















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_waveforms(n_waveforms = 2000, n_channels = 4)

      
      Plots the waveforms of the cluster.

      :param n_waveforms: The number of waveforms to plot.
      :type n_waveforms: int, optional
      :param n_channels: The number of channels to plot.
      :type n_channels: int, optional

      :rtype: None















      ..
          !! processed by numpydoc !!


   .. py:method:: psch(bin_width_secs)

      
      Calculate the peri-stimulus *count* histogram of a cell's spiking
      against event times.

      :param bin_width_secs: The width of each bin in seconds.
      :type bin_width_secs: float

      :returns: **result** -- Rows are counts of spikes per bin_width_secs.
                Size of columns ranges from self.event_window[0] to
                self.event_window[1] with bin_width_secs steps;
                so x is count, y is "event".
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: psth()

      
      Calculate the PSTH of event_ts against the spiking of a cell

      :returns: * **x, y** (*list*)
                * *The list of time differences between the spikes of the cluster*
                * *and the events (x) and the trials (y)*















      ..
          !! processed by numpydoc !!


   .. py:method:: responds_to_stimulus(threshold, min_contiguous, return_activity = False, return_magnitude = False, **kwargs)

      
      Checks whether a cluster responds to a laser stimulus.

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

      :returns: With named fields "responds" (bool), "normed_response_curve" (np.ndarray),
                "response_magnitude" (np.ndarray)
      :rtype: namedtuple















      ..
          !! processed by numpydoc !!


   .. py:method:: smooth_spike_train(npos, sigma=3.0, shuffle=None)

      
      Returns a spike train the same length as num pos samples that has been
      smoothed in time with a gaussian kernel M in width and standard
      deviation equal to sigma.

      :param npos: The number of position samples captured.
      :type npos: int
      :param sigma: The standard deviation of the gaussian used to
                    smooth the spike train.
      :type sigma: float, default=3.0
      :param shuffle: The number of seconds to shift the spike
                      train by. Default is None.
      :type shuffle: int, default=None

      :returns: The smoothed spike train.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_band_max_freq()

      
      Calculates the frequency with the maximum power in the theta band (6-12Hz)
      of a spike train's autocorrelogram.

      This function is used to look for differences in theta frequency in
      different running directions as per Blair.
      See Welday paper - https://doi.org/10.1523/jneurosci.0712-11.2011

      :returns: The frequency with the maximum power in the theta band.
      :rtype: float

      :raises ValueError: If the input spike train is not valid.















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_mod_idx(**kwargs)

      
      Calculates a theta modulation index of a spike train based on the cells
      autocorrelogram.

      The difference of the mean power in the theta band (6-11 Hz) and
      the mean power in the 1-50 Hz band is divided by their sum to give
      a metric that lives between 0 and 1

      :param x1: The spike time-series.
      :type x1: np.ndarray

      :returns: The difference of the values at the first peak
                and trough of the autocorrelogram.
      :rtype: float

      .. rubric:: Notes

      This is a fairly skewed metric with a distribution strongly biased
      to -1 (although more evenly distributed than theta_mod_idxV2 below)















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_mod_idxV2()

      
      This is a simpler alternative to the theta_mod_idx method in that it
      calculates the difference between the normalized temporal
      autocorrelogram at the trough between 50-70ms and the
      peak between 100-140ms over their sum (data is binned into 5ms bins)

      :returns: The difference of the values at the first peak
                and trough of the autocorrelogram.
      :rtype: float

      .. rubric:: Notes

      Measure used in Cacucci et al., 2004 and Kropff et al 2015















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_mod_idxV3(**kwargs)

      
      Another theta modulation index score this time based on the method used
      by Kornienko et al., (2024) (Kevin Allens lab)
      see https://doi.org/10.7554/eLife.35949.001

      Uses the binned spike train instead of the autocorrelogram as
      the input to the periodogram function (they use pwelch in R;
      periodogram is a simplified call to welch in scipy.signal)

      The resulting metric is similar to that in theta_mod_idx above except
      that the frequency bands compared to the theta band are narrower and
      exclusive of the theta band

      Produces a fairly normally distributed score with a mean and median
      pretty close to 0

      :param \*\*kwargs: Passed into get_ifr_power_spectrum

      :returns: The difference of the values at the first peak
                and trough of the autocorrelogram.
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:method:: trial_mean_fr()


   .. py:method:: update_KSMeta(value)

      
      Takes in a TemplateModel instance from a phy session and
      parses out the relevant metrics for the cluster and places
      into the namedtuple KSMeta.

      :param value: A dictionary containing the relevant metrics for the cluster.
      :type value: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: waveforms(channel_id = None)

      
      Returns the waveforms of the cluster.

      :param channel_id: The channel IDs to return the waveforms for.
                         If None, returns waveforms for all channels.
      :type channel_id: Sequence, optional

      :returns: The waveforms of the cluster,
                or None if no waveforms are available.
      :rtype: np.ndarray | None















      ..
          !! processed by numpydoc !!


   .. py:property:: KSMeta
      :type: KSMetaTuple



   .. py:attribute:: _duration
      :value: None



   .. py:attribute:: _event_ts
      :value: None



   .. py:attribute:: _event_window


   .. py:attribute:: _invert_waveforms
      :value: False



   .. py:attribute:: _ksmeta


   .. py:attribute:: _pos_sample_rate
      :value: 50



   .. py:attribute:: _post_spike_samples
      :value: 40



   .. py:attribute:: _pre_spike_samples
      :value: 10



   .. py:attribute:: _sample_rate
      :value: 50000



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



   .. py:property:: invert_waveforms
      :type: bool



   .. py:property:: n_channels
      :type: int | None


      
      Returns the number of channels in the waveforms.

      :returns: The number of channels in the waveforms,
                or None if no waveforms are available.
      :rtype: int | None















      ..
          !! processed by numpydoc !!


   .. py:property:: n_samples
      :type: int | None


      
      Returns the number of samples in the waveforms.

      :returns: The number of samples in the waveforms,
                or None if no waveforms are available.
      :rtype: int | None















      ..
          !! processed by numpydoc !!


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

   NB Axona waveforms are nSpikes x nChannels x nSamples - this boils
   down to nSpikes x 4 x 50
   NB KiloSort waveforms are nSpikes x nSamples x nChannels - these are ordered
   by 'best' channel first and then the rest of the channels. This boils
   down to nSpikes x 61 x 12 SO THIS NEEDS TO BE CHANGED to
   nSpikes x nChannels x nSamples

   :param spike_times: The times of spikes in the trial in seconds.
   :type spike_times: np.ndarray
   :param cluster: The cluster ID.
   :type cluster: int
   :param waveforms: An nSpikes x nChannels x nSamples array.
   :type waveforms: np.ndarray, optional
   :param \*\*kwargs: Additional keyword arguments.
   :type \*\*kwargs: dict

   .. attribute:: spike_times

      The times of spikes in the trial in seconds.

      :type: np.ma.MaskedArray

   .. attribute:: _waves

      The waveforms of the spikes.

      :type: np.ma.MaskedArray or None

   .. attribute:: cluster

      The cluster ID.

      :type: int

   .. attribute:: n_spikes

      the total number of spikes for the current cluster

      :type: int

   .. attribute:: duration

      total duration of the trial in seconds

      :type: float, int

   .. attribute:: event_ts

      The times that events occurred in seconds.

      :type: np.ndarray or None

   .. attribute:: event_window

      The window, in seconds, either side of the stimulus, to examine.

      :type: np.ndarray

   .. attribute:: stim_width

      The width, in ms, of the stimulus.

      :type: float or None

   .. attribute:: secs_per_bin

      The size of bins in PSTH.

      :type: float

   .. attribute:: sample_rate

      The sample rate of the recording.

      :type: int

   .. attribute:: pos_sample_rate

      The sample rate of the position data.

      :type: int

   .. attribute:: pre_spike_samples

      The number of samples before the spike.

      :type: int

   .. attribute:: post_spike_samples

      The number of samples after the spike.

      :type: int

   .. attribute:: KSMeta

      The metadata from KiloSort.

      :type: KSMetaTuple















   ..
       !! processed by numpydoc !!

   .. py:method:: get_channel_depth_from_templates(pname)

      
      Determine depth of template as well as closest channel.

      :param pname: The path to the directory containing the KiloSort results.
      :type pname: Path

      :returns: The depth of the template and the index of the closest channel.
      :rtype: tuple of np.ndarray

      .. rubric:: Notes

      Adopted from
      'templatePositionsAmplitudes' by N. Steinmetz
      (https://github.com/cortex-lab/spikes)















      ..
          !! processed by numpydoc !!


   .. py:method:: get_template_id_for_cluster(pname, cluster)

      
      Determine the best channel (one with highest amplitude spikes)
      for a given cluster.

      :param pname: The path to the directory containing the KiloSort results.
      :type pname: Path
      :param cluster: The cluster to get the template ID for.
      :type cluster: int

      :returns: The template ID for the cluster.
      :rtype: int















      ..
          !! processed by numpydoc !!


   .. py:method:: get_waveforms(cluster, cluster_data, n_waveforms = 2000, n_channels = 64, channel_range=None)

      
      Returns waveforms for a cluster.

      :param cluster: The cluster to return the waveforms for.
      :type cluster: int
      :param cluster_data: The KiloSortSession object for the
                           session that contains the cluster.
      :type cluster_data: KiloSortSession
      :param n_waveforms: The number of waveforms to return.
      :type n_waveforms: int, default=2000
      :param n_channels: The number of channels in the recording.
      :type n_channels: int, default=64

      :returns: The waveforms for the cluster.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:attribute:: TemplateModel
      :value: None



   .. py:attribute:: n_samples

      
      Returns the number of samples in the waveforms.

      :returns: The number of samples in the waveforms,
                or None if no waveforms are available.
      :rtype: int | None















      ..
          !! processed by numpydoc !!


.. py:class:: SpikeCalcsProbe

   Bases: :py:obj:`SpikeCalcsGeneric`


   
   Encapsulates methods specific to probe-based recordings
















   ..
       !! processed by numpydoc !!

.. py:function:: cluster_quality(waveforms = None, spike_clusters = None, cluster_id = None, fet = 1)

   
   Returns the L-ratio and Isolation Distance measures calculated
   on the principal components of the energy in a spike matrix.

   :param waveforms: The waveforms to be processed. If None, the function will return None.
   :type waveforms: np.ndarray, optional
   :param spike_clusters: The spike clusters to be processed.
   :type spike_clusters: np.ndarray, optional
   :param cluster_id: The ID of the cluster to be processed.
   :type cluster_id: int, optional
   :param fet: The feature to be used in the PCA calculation (default is 1).
   :type fet: int, optional

   :returns: A tuple containing the L-ratio and Isolation Distance of the cluster.
   :rtype: tuple

   :raises Exception: If an error occurs during the calculation of the L-ratio or Isolation Distance.















   ..
       !! processed by numpydoc !!

.. py:function:: contamination_percent(x1, x2 = None, **kwargs)

   
   Computes the cross-correlogram between two sets of spikes and
   estimates how refractory the cross-correlogram is.

   :param x1: The first set of spikes.
   :type x1: np.ndarray
   :param x2: The second set of spikes. If None, x1 is used.
   :type x2: np.ndarray, optional
   :param \*\*kwargs: Additional keyword arguments that can be fed into xcorr.
   :type \*\*kwargs: dict

   :returns: A tuple containing:
             - Q (float): A measure of refractoriness.
             - R (float): A second measure of refractoriness (kicks in for very low firing rates).
   :rtype: tuple

   .. rubric:: Notes

   Taken from KiloSorts ccg.m

   The contamination metrics are calculated based on
   an analysis of the 'shoulders' of the cross-correlogram.
   Specifically, the spike counts in the ranges +/-5-25ms and















   ..
       !! processed by numpydoc !!

.. py:function:: fit_smoothed_curve_to_xcorr(xc, **kwargs)

   
   Idea is to smooth out the result of an auto- or cross-correlogram with
   a view to correlating the result with another auto- or cross-correlogram
   to see how similar two of these things are.

   Check Brandon et al., 2011?2012?















   ..
       !! processed by numpydoc !!

.. py:function:: get_burstiness(isi_matrix, whiten = False, plot_pcs = False)

   
   Returns the burstiness of a waveform.

   :param isi_matrix: A matrix of normalized interspike intervals (ISIs) for the neurons.
                      Rows are neurons, columns are ISI time bins.
   :type isi_matrix: np.ndarray

   :rtype: np.ndarray

   .. rubric:: Notes

   Algorithm:

   1) The interspike intervals between 0 and 60ms were binned into 2ms bins,
   and the area of the histogram was normalised to 1 to produce a
   probability distribution histogram for each neuron

   2) A principal components analysis (PCA) is performed on the matrix of
   the ISI probability distributions of all neurons

   3) Neurons were then assigned to two clusters using a k-means clustering
   algorithm on the first three principal components

   4) a linear discriminant analysis performed in MATLAB (‘classify’) was
   undertaken to determine the optimal linear discriminant (Fishers Linear
   Discriminant) i.e., the plane which best separated the two clusters in a
   three-dimensional scatter plot of the principal components.

   Training on 80% of the data and testing on the remaining 20% resulted in a
   good separation of the two clusters.

   5) A burstiness score was assigned to each neuron which was calculated by
   computing the shortest distance between the plotted point for each neuron
   in the three-dimensional cluster space (principal components 1,2 and 3),
   and the plane separating the two clusters (i.e., the optimal linear
   discriminant).

   6) To ensure the distribution of these burstiness scores was bimodal,
   reflecting the presence of two classes of neuron (‘bursty’ versus
   ‘non-bursty’), probability density functions for Gaussian mixture models
   with between one and four underlying Gaussian curves were fitted and the
   fit of each compared using the Akaike information criterion (AIC)

   7) Optionally plot the principal components and the centres of the
   kmeans results















   ..
       !! processed by numpydoc !!

.. py:function:: get_param(waveforms, param='Amp', t=200, fet=1)

   
   Returns the requested parameter from a spike train as a numpy array.

   :param waveforms: Shape of array can be nSpikes x nSamples OR nSpikes x nElectrodes x nSamples.
   :type waveforms: np.ndarray
   :param param: Valid values are:
                 - 'Amp': peak-to-trough amplitude
                 - 'P': height of peak
                 - 'T': depth of trough
                 - 'Vt': height at time t
                 - 'tP': time of peak (in seconds)
                 - 'tT': time of trough (in seconds)
                 - 'PCA': first n fet principal components (defaults to 1)
   :type param: str, default='Amp'
   :param t: The time used for Vt
   :type t: int, default=200
   :param fet: The number of principal components (use with param 'PCA').
   :type fet: int, default=1

   :returns: The requested parameter as a numpy array.
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: get_peak_to_trough_time(waveforms)

   
   Returns the time in seconds of the peak to trough in a waveform.

   :param waveforms: The waveforms to calculate the peak to trough time for.
   :type waveforms: np.ndarray

   :returns: The time of the peak to trough in seconds.
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: mahal(u, v)

   
   Returns the L-ratio and Isolation Distance measures calculated on the
   principal components of the energy in a spike matrix.

   :param u: The first set of waveforms.
   :type u: np.ndarray
   :param v: The second set of waveforms.
   :type v: np.ndarray

   :returns: The Mahalanobis distances.
   :rtype: np.ndarray

   :raises Warning: If input size mismatch, too few rows, or complex inputs are detected.















   ..
       !! processed by numpydoc !!

.. py:function:: xcorr(x1, x2 = None, Trange = np.array([-0.5, 0.5]), binsize = 0.001, normed=False, **kwargs)

   
   Calculates the ISIs in x1 or x1 vs x2 within a given range.

   :param x1: The times of the spikes emitted by the first cluster in seconds.
   :type x1: np.ndarray
   :param x2: The times of the spikes emitted by the second cluster in seconds. If None, x1 is used.
   :type x2: np.ndarray, optional
   :param Trange: Range of times to bin up in seconds (default is [-0.5, 0.5]).
   :type Trange: np.ndarray or list, optional
   :param binsize: The size of the bins in seconds (default is 0.001).
   :type binsize: float, optional
   :param normed: Whether to divide the counts by the total number of spikes to give a probability (default is False).
   :type normed: bool, optional
   :param \*\*kwargs: Additional keyword arguments.
   :type \*\*kwargs: dict

   :returns: A BinnedData object containing the binned data and the bin edges.
   :rtype: BinnedData















   ..
       !! processed by numpydoc !!

