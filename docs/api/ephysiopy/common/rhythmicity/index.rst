ephysiopy.common.rhythmicity
============================

.. py:module:: ephysiopy.common.rhythmicity


Classes
-------

.. autoapisummary::

   ephysiopy.common.rhythmicity.CosineDirectionalTuning
   ephysiopy.common.rhythmicity.LFPOscillations
   ephysiopy.common.rhythmicity.Rippler


Module Contents
---------------

.. py:class:: CosineDirectionalTuning(spike_times, pos_times, spk_clusters, x, y, tracker_params={})

   Bases: :py:obj:`object`


   
   Produces output to do with Welday et al (2011) like analysis
   of rhythmic firing a la oscialltory interference model
















   ..
       !! processed by numpydoc !!

   .. py:method:: _rolling_window(a, window)

      
      Totally nabbed from SO:
      https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
















      ..
          !! processed by numpydoc !!


   .. py:method:: getClusterPosIndices(clust)


   .. py:method:: getClusterSpikeTimes(cluster)


   .. py:method:: getDirectionalBinForCluster(cluster)


   .. py:method:: getDirectionalBinPerPosition(binwidth)

      
      Direction is in degrees as that what is created by me in some of the
      other bits of this package.

      :param binwidth: The bin width in degrees
      :type binwidth: int

      :returns: A digitization of which directional bin each pos sample belongs to















      ..
          !! processed by numpydoc !!


   .. py:method:: getPosIndices()


   .. py:method:: getRunsOfMinLength()

      
      Identifies runs of at least self.min_runlength seconds long,
      which at 30Hz pos sampling rate equals 12 samples, and
      returns the start and end indices at which
      the run was occurred and the directional bin that run belongs to

      :returns:

                The start and end indices into pos samples of the run
                          and the directional bin to which it belongs
      :rtype: np.array















      ..
          !! processed by numpydoc !!


   .. py:method:: intrinsic_freq_autoCorr(spkTimes=None, posMask=None, maxFreq=25, acBinSize=0.002, acWindow=0.5, plot=True, **kwargs)

      
      This is taken and adapted from ephysiopy.common.eegcalcs.EEGCalcs

      :param spkTimes: Times in seconds of the cells firing
      :type spkTimes: np.array
      :param posMask: Boolean array corresponding to the length of
                      spkTimes where True is stuff to keep
      :type posMask: np.array
      :param maxFreq: The maximum frequency to do the power spectrum
                      out to
      :type maxFreq: float
      :param acBinSize: The bin size of the autocorrelogram in seconds
      :type acBinSize: float
      :param acWindow: The range of the autocorr in seconds
      :type acWindow: float

      .. note:: Make sure all times are in seconds















      ..
          !! processed by numpydoc !!


   .. py:method:: power_spectrum(eeg, plot=True, binWidthSecs=None, maxFreq=25, pad2pow=None, ymax=None, **kwargs)

      
      Method used by eeg_power_spectra and intrinsic_freq_autoCorr
      Signal in must be mean normalised already
















      ..
          !! processed by numpydoc !!


   .. py:method:: speedFilterRuns(runs, minspeed=5.0)

      
      Given the runs identified in getRunsOfMinLength, filter for speed
      and return runs that meet the min speed criteria.

      The function goes over the runs with a moving window of length equal
      to self.min_runlength in samples and sees if any of those segments
      meet the speed criteria and splits them out into separate runs if true.

      NB For now this means the same spikes might get included in the
      autocorrelation procedure later as the
      moving window will use overlapping periods - can be modified later.

      :param runs: Generated from getRunsOfMinLength
      :type runs: 3 x nRuns np.array
      :param minspeed: Min running speed in cm/s for an epoch (minimum
                       epoch length defined previously
                       in getRunsOfMinLength as minlength, usually 0.4s)
      :type minspeed: float

      :returns: A modified version of the "runs" input variable
      :rtype: 3 x nRuns np.array















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _hdir


   .. py:attribute:: _min_runlength
      :value: 0.4



   .. py:attribute:: _pos_sample_rate
      :value: 30



   .. py:attribute:: _pos_samples_for_spike
      :value: None



   .. py:attribute:: _speed


   .. py:attribute:: _spk_sample_rate
      :value: 30000.0



   .. py:attribute:: _xy


   .. py:property:: hdir


   .. py:property:: min_runlength


   .. py:attribute:: posCalcs


   .. py:property:: pos_sample_rate


   .. py:property:: pos_samples_for_spike


   .. py:attribute:: pos_times


   .. py:attribute:: smthKernelSigma
      :value: 0.1875



   .. py:attribute:: smthKernelWidth
      :value: 2



   .. py:attribute:: sn2Width
      :value: 2



   .. py:property:: speed


   .. py:attribute:: spikeCalcs


   .. py:attribute:: spike_times


   .. py:attribute:: spk_clusters

      
      There can be more spikes than pos samples in terms of sampling as the
      open-ephys buffer probably needs to finish writing and the camera has
      already stopped, so cut of any cluster indices and spike times
      that exceed the length of the pos indices
















      ..
          !! processed by numpydoc !!


   .. py:property:: spk_sample_rate


   .. py:attribute:: thetaRange
      :value: [7, 11]



   .. py:attribute:: xmax
      :value: 11



   .. py:property:: xy


.. py:class:: LFPOscillations(sig, fs, **kwargs)

   Bases: :py:obj:`object`


   
   Does stuff with the LFP such as looking at nested oscillations
   (theta/ gamma coupling), the modulation index of such phenomena,
   filtering out certain frequencies in the LFP, getting the instantaneous
   phase and amplitude and so on
















   ..
       !! processed by numpydoc !!

   .. py:method:: filterForLaser(sig=None, width=0.125, dip=15.0, stimFreq=6.66)

      
      Attempts to filter out frequencies from optogenetic experiments where
      the frequency of laser stimulation was at 6.66Hz.

      .. note::

         This method needs tweaking for each trial as the power in the signal
         is variable across trials / animals etc. A potential improvement could be using mean
         power or a similar metric.















      ..
          !! processed by numpydoc !!


   .. py:method:: getFreqPhase(sig, band2filter, ford=3)

      
      Uses the Hilbert transform to calculate the instantaneous phase and
      amplitude of the time series in sig.

      :param sig: The signal to be analysed
      :type sig: np.array
      :param ford: The order for the Butterworth filter
      :type ford: int
      :param band2filter: The two frequencies to be filtered for
      :type band2filter: list















      ..
          !! processed by numpydoc !!


   .. py:method:: get_theta_phase(cluster_times, **kwargs)

      
      Calculates the phase of theta at which a cluster emitted spikes
      and returns a fit to a vonmises distribution

      :param cluster_times (np.ndarray) - the times the cluster emitted spikes in: seconds

      .. rubric:: Notes

      kwargs can include:
          low_theta (int) - low end for bandpass filter
          high_theta (int) - high end for bandpass filter















      ..
          !! processed by numpydoc !!


   .. py:method:: modulationindex(sig=None, nbins=20, forder=2, thetaband=[4, 8], gammaband=[30, 80], plot=True)

      
      Calculates the modulation index of theta and gamma oscillations.
      Specifically this is the circular correlation between the phase of
      theta and the power of theta.

      :param sig: The LFP signal
      :type sig: np.array
      :param nbins: The number of bins in the circular range 0 to 2*pi
      :type nbins: int
      :param forder: The order of the butterworth filter
      :type forder: int
      :param thetaband: The lower/upper bands of the theta freq range
      :type thetaband: list
      :param gammaband: The lower/upper bands of the gamma freq range
      :type gammaband: list
      :param plot: Show some pics or not
      :type plot: bool















      ..
          !! processed by numpydoc !!


   .. py:method:: plv(sig=None, forder=2, thetaband=[4, 8], gammaband=[30, 80], plot=True, **kwargs)

      
      Computes the phase-amplitude coupling (PAC) of nested oscillations.
      More specifically this is the phase-locking value (PLV) between two
      nested oscillations in EEG data, in this case theta (default 4-8Hz)
      and gamma (defaults to 30-80Hz). A PLV of unity indicates perfect phase
      locking (here PAC) and a value of zero indicates no locking (no PAC)

      :param eeg: The eeg data itself. This is a 1-d array which
      :type eeg: numpy array
      :param can be masked or not:
      :param forder: The order of the filter(s) applied to the eeg data
      :type forder: int
      :param thetaband: The range of values to bandpass
      :type thetaband: list/array
      :param gammaband: The range of values to bandpass
      :type gammaband: list/array
      :param filter for for the theta and gamma ranges:
      :param plot: Whether to plot the resulting binned up
      :type plot: bool, optional
      :param polar plot which shows the amplitude of the gamma oscillation:
      :param found at different phases of the theta oscillation.:
      :param Default is True.:

      :returns: The value of the phase-amplitude coupling
      :rtype: plv (float)















      ..
          !! processed by numpydoc !!


   .. py:method:: spike_xy_phase_plot(cluster, pos_data, phy_data, lfp_data)

      
      Produces a plot of the phase of theta at which each spike was
      emitted. Each spike is plotted according to the x-y location the
      animal was in when it was fired and the colour of the marker
      corresponds to the phase of theta at which it fired.
















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_running(pos_data, lfp_data, **kwargs)

      
      Returns metrics to do with the theta frequency/ power and running speed/ acceleration
















      ..
          !! processed by numpydoc !!


   .. py:attribute:: fs


   .. py:attribute:: sig


.. py:class:: Rippler(trial_root, signal, fs)

   Bases: :py:obj:`object`


   
   Does some spectrographic analysis and plots of LFP data
   looking specifically at the ripple band

   Until I modified the Ripple Detector plugin the duration of the TTL
   pulses was variable with a more or less bimodal distribution which
   is why there is a separate treatment of short and long duration TTL pulses below















   ..
       !! processed by numpydoc !!

   .. py:method:: _calc_ripple_chunks_duration_power(ttl_type='no_laser')

      
      Find the indices and durations of the events that have sufficient
      duration and power to be considered ripples.

      :param ttl_type (str) - which bit of the trial to do the calculation for: Either 'no_laser' or 'laser'

      :returns: **tuple**
      :rtype: the run indices to keep and the run durations in ms















      ..
          !! processed by numpydoc !!


   .. py:method:: _find_high_power_periods(n = 3, t = 10)

      
      Find periods where the power in the ripple band is above n standard deviations
      for t samples. Meant to recapitulate the algorithm from the Ripple Detector
      plugin
















      ..
          !! processed by numpydoc !!


   .. py:method:: _find_path_to_continuous(trial_root, **kwargs)

      
      Iterates through a directory tree and finds the path to the
      Ripple Detector plugin data and returns its location
















      ..
          !! processed by numpydoc !!


   .. py:method:: _find_path_to_ripple_ttl(trial_root, **kwargs)

      
      Iterates through a directory tree and finds the path to the
      Ripple Detector plugin data and returns its location
















      ..
          !! processed by numpydoc !!


   .. py:method:: _load_start_time(path_to_sync_message_file)

      
      Returns the start time contained in a sync file from OE
















      ..
          !! processed by numpydoc !!


   .. py:method:: _plot_ripple_lfp_with_ttl(i_time, **kwargs)


   .. py:method:: filter_timestamps_for_real_ripples()

      
      Filter out low power and short duration events from the list of timestamps
















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spectrogram(start_time, end_time, plot=False)


   .. py:method:: plot_and_save_ripple_band_lfp_with_ttl(**kwargs)


   .. py:method:: plot_filtered_lfp_chunk(start_time, end_time, **kwargs)


   .. py:method:: plot_mean_rippleband_power(**kwargs)

      
      Plots the mean power in the ripple band for the laser on and no laser
      conditions
















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_mean_spectrogram(laser_on = False, ax=None, **kwargs)

      
      Plots the mean spectrogram for either 'long' or 'short' ttl events
















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_mean_spectrograms(**kwargs)


   .. py:method:: plot_rasters(laser_on)


   .. py:method:: update_bandpass(low=None, high=None)


   .. py:attribute:: LFP


   .. py:attribute:: all_on_ts


   .. py:attribute:: all_ts


   .. py:attribute:: bit_volts
      :value: 0.1949999928474426



   .. py:attribute:: eeg_time


   .. py:attribute:: filtered_eeg


   .. py:attribute:: fs


   .. py:attribute:: gaussian_std
      :value: 5



   .. py:attribute:: gaussian_window
      :value: 12



   .. py:attribute:: high_band
      :value: 250



   .. py:attribute:: laser_off_ts


   .. py:attribute:: laser_on_ts


   .. py:attribute:: lfp_plotting_scale
      :value: 500



   .. py:attribute:: low_band
      :value: 120



   .. py:attribute:: min_ttl_duration
      :value: 0.01



   .. py:attribute:: n_channels
      :value: 64



   .. py:attribute:: no_laser_on_ts


   .. py:attribute:: orig_sig


   .. py:attribute:: pname_for_trial


   .. py:attribute:: post_ttl
      :value: 0.2



   .. py:attribute:: pre_ttl
      :value: 0.05



   .. py:attribute:: ripple_min_duration_ms
      :value: 20



   .. py:attribute:: ripple_std_dev
      :value: 2



   .. py:attribute:: settings


   .. py:attribute:: ttl_all_line
      :value: 4



   .. py:attribute:: ttl_duration
      :value: 0.05



   .. py:attribute:: ttl_out_line
      :value: 1



   .. py:attribute:: ttl_percent
      :value: 100



   .. py:attribute:: ttl_states


