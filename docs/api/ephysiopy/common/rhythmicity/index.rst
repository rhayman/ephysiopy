ephysiopy.common.rhythmicity
============================

.. py:module:: ephysiopy.common.rhythmicity


Classes
-------

.. autoapisummary::

   ephysiopy.common.rhythmicity.CosineDirectionalTuning
   ephysiopy.common.rhythmicity.FreqPhase
   ephysiopy.common.rhythmicity.LFPOscillations
   ephysiopy.common.rhythmicity.PowerSpectrumParams
   ephysiopy.common.rhythmicity.Rippler


Functions
---------

.. autoapisummary::

   ephysiopy.common.rhythmicity.power_spectrum


Module Contents
---------------

.. py:class:: CosineDirectionalTuning(trial, channel)

   Bases: :py:obj:`object`


   
   Produces output to do with Welday et al (2011) like analysis
   of rhythmic firing a la oscialltory interference model
















   ..
       !! processed by numpydoc !!

   .. py:method:: _rolling_window(a, window)

      
      Returns a view of the array a using a window length of window

      :param a: The array to be windowed
      :type a: np.array
      :param window: The window length
      :type window: int

      :returns: The windowed array
      :rtype: np.array

      .. rubric:: Notes

      Taken from:
      https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy















      ..
          !! processed by numpydoc !!


   .. py:method:: getClusterPosIndices(clust)


   .. py:method:: getClusterSpikeTimes(cluster)


   .. py:method:: getDirectionalBinForCluster(cluster)


   .. py:method:: getDirectionalBinPerPosition(binwidth)

      
      Digitizes the directional bin each position sample belongs to.

      Direction is in degrees as that what is created by me in some of the
      other bits of this package.

      :param binwidth: The bin width in degrees.
      :type binwidth: int

      :returns: A digitization of which directional bin each position sample belongs to.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: getPosIndices()


   .. py:method:: getRunsOfMinLength()

      
      Identifies runs of at least self.min_runlength seconds long,
      which at 30Hz pos sampling rate equals 12 samples, and
      returns the start and end indices at which
      the run was occurred and the directional bin that run belongs to.

      :returns: The start and end indices into pos samples of the run
                and the directional bin to which it belongs.
      :rtype: np.array















      ..
          !! processed by numpydoc !!


   .. py:method:: intrinsic_freq_autoCorr(spkTimes=None, posMask=None, maxFreq=25, acBinSize=0.002, acWindow=0.5, plot=True, **kwargs)

      
      Taken and adapted from ephysiopy.common.eegcalcs.EEGCalcs

      :param spkTimes: Times in seconds of the cells firing
      :type spkTimes: np.array
      :param posMask: Boolean array corresponding to the length of spkTimes where True is stuff to keep
      :type posMask: np.array
      :param maxFreq: The maximum frequency to do the power spectrum out to
      :type maxFreq: float
      :param acBinSize: The bin size of the autocorrelogram in seconds
      :type acBinSize: float
      :param acWindow: The range of the autocorr in seconds
      :type acWindow: float
      :param plot: Whether to plot the resulting autocorrelogram and power spectrum
      :type plot: bool

      :returns: A dictionary containing the power spectrum and other related metrics
      :rtype: dict

      .. rubric:: Notes

      Make sure all times are in seconds















      ..
          !! processed by numpydoc !!


   .. py:method:: speedFilterRuns(runs, minspeed=5.0)

      
      Given the runs identified in getRunsOfMinLength, filter for speed
      and return runs that meet the min speed criteria.

      The function goes over the runs with a moving window of length equal
      to self.min_runlength in samples and sees if any of those segments
      meet the speed criteria and splits them out into separate runs if true.

      .. rubric:: Notes

      For now this means the same spikes might get included in the
      autocorrelation procedure later as the moving window will use
      overlapping periods - can be modified later.

      :param runs: Generated from getRunsOfMinLength, shape (3, nRuns)
      :type runs: np.array
      :param minspeed: Min running speed in cm/s for an epoch (minimum epoch length
                       defined previously in getRunsOfMinLength as minlength, usually 0.4s)
      :type minspeed: float

      :returns: A modified version of the "runs" input variable, shape (3, nRuns)
      :rtype: np.array















      ..
          !! processed by numpydoc !!


   .. py:attribute:: _hdir


   .. py:attribute:: _min_runlength
      :value: 0.4



   .. py:attribute:: _pos_sample_rate


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


   .. py:property:: speed


   .. py:attribute:: spike_times


   .. py:property:: spk_sample_rate


   .. py:attribute:: trial


   .. py:property:: xy


.. py:class:: FreqPhase

   .. py:attribute:: amplitude
      :type:  numpy.ndarray


   .. py:attribute:: amplitude_filtered
      :type:  numpy.ndarray


   .. py:attribute:: filt_sig
      :type:  numpy.ndarray


   .. py:attribute:: inst_freq
      :type:  numpy.ndarray


   .. py:attribute:: phase
      :type:  numpy.ndarray


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

      :param sig: The signal to be filtered. If None, uses the signal provided during initialization.
      :type sig: np.array, optional
      :param width: The width of the filter (default is 0.125).
      :type width: float, optional
      :param dip: The dip of the filter (default is 15.0).
      :type dip: float, optional
      :param stimFreq: The frequency of the laser stimulation (default is 6.66Hz).
      :type stimFreq: float, optional

      :returns: The filtered signal.
      :rtype: np.array















      ..
          !! processed by numpydoc !!


   .. py:method:: getFreqPhase(sig, band2filter, ford=3)

      
      Uses the Hilbert transform to calculate the instantaneous phase and
      amplitude of the time series in sig.

      :param sig: The signal to be analysed.
      :type sig: np.array
      :param band2filter: The two frequencies to be filtered for.
      :type band2filter: list
      :param ford: The order for the Butterworth filter (default is 3).
      :type ford: int, optional

      :returns: A tuple containing the filtered signal, phase, amplitude,
                amplitude filtered, and instantaneous frequency.
      :rtype: tuple















      ..
          !! processed by numpydoc !!


   .. py:method:: get_comodulogram(low_freq_band=[1, 12], **kwargs)

      
      Computes the comodulogram of phase-amplitude coupling
      between different frequency bands.

      :param low_freq_band: The low frequency band - what the pactools module calls
                            the "driver" frequency
      :type low_freq_band: list
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The computed comodulogram.
      :rtype: np.ndarray

      .. rubric:: Notes

      This method is a placeholder and needs to be implemented.















      ..
          !! processed by numpydoc !!


   .. py:method:: get_mean_resultant_vector(spike_times, **kws)

      
      Calculates the mean phase at which the cluster emitted spikes
      and the length of the mean resultant vector.

      :param lfp_data (np.ndarray) - the LFP signal:
      :param fs (float) - the sample rate of the LFP signal:

      :returns: mean resultant direction
      :rtype: tuple (float, float) - the mean resultant vector length and mean

      .. rubric:: Notes

      For similar approach see Boccara et al., 2010.
      doi: 10.1038/nn.2602















      ..
          !! processed by numpydoc !!


   .. py:method:: get_oscillatory_epochs(out_window_size = 0.4, FREQ_BAND=(20, 90), **kwargs)

      
      Uses the continuous wavelet transform to find epochs
      of high oscillatory power in the LFP

      :param out_window_size: The size of the output window in seconds (default is 0.4).
      :type out_window_size: float, optional

      :returns: A dictionary where keys are the center time of the oscillatory
                window and values are the LFP signal in that window.
      :rtype: dict

      .. rubric:: Notes

      Uses a similar method to jun et al., but expands the window
      for candidate oscillatory windows in a better way

      .. rubric:: References

      Jun et al., 2020, Neuron 107, 1095–1112
      https://doi.org/10.1016/j.neuron.2020.06.023















      ..
          !! processed by numpydoc !!


   .. py:method:: get_theta_phase(cluster_times, **kwargs)

      
      Calculates the phase of theta at which a cluster emitted spikes
      and returns a fit to a vonmises distribution.

      :param cluster_times: The times the cluster emitted spikes in seconds.
      :type cluster_times: np.ndarray

      .. rubric:: Notes

      kwargs can include:
          low_theta : int
              Low end for bandpass filter.
          high_theta : int
              High end for bandpass filter.

      :returns: A tuple containing the phase of theta at which the cluster
                emitted spikes, the x values for the vonmises distribution,
                and the y values for the vonmises distribution.
      :rtype: tuple















      ..
          !! processed by numpydoc !!


   .. py:method:: modulationindex(sig=None, nbins=20, forder=2, thetaband=[6, 12], gammaband=[20, 90], plot=False)

      
      Calculates the modulation index of theta and gamma oscillations.
      Specifically, this is the circular correlation between the phase of
      theta and the power of gamma.

      :param sig: The LFP signal. If None, uses the signal provided during
                  initialization.
      :type sig: np.array, optional
      :param nbins: The number of bins in the circular range 0 to 2*pi (default is 20).
      :type nbins: int, optional
      :param forder: The order of the Butterworth filter (default is 2).
      :type forder: int, optional
      :param thetaband: The lower and upper bands of the theta frequency range
                        (default is [6, 12]).
      :type thetaband: list, optional
      :param gammaband: The lower and upper bands of the gamma frequency range
                        (default is [20, 90]).
      :type gammaband: list, optional
      :param plot: Whether to plot the results (default is True).
      :type plot: bool, optional

      :returns: The modulation index.
      :rtype: float

      .. rubric:: Notes

      The modulation index is a measure of the strength of phase-amplitude
      coupling between theta and gamma oscillations.















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_cwt(sig, Pos, start, stop, FREQ_BAND=(20, 90), **kwargs)

      
      Plots the continuous wavelet transform of the signal

      :param sig: The signal to be analysed.
      :type sig: np.ndarray
      :param Pos: The position object containing speed and time information.
      :type Pos: PosCalcsGeneric
      :param start: The start time for the plot (in seconds).
      :type start: float
      :param stop: The stop time for the plot (in seconds).
      :type stop: float
      :param FREQ_BAND: The frequency band to be highlighted (default is (20, 90)).
      :type FREQ_BAND: tuple, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: plv(sig=None, forder=2, thetaband=[4, 8], gammaband=[30, 80], plot=True, **kwargs)

      
      Computes the phase-amplitude coupling (PAC) of nested oscillations.
      More specifically this is the phase-locking value (PLV) between two
      nested oscillations in EEG data, in this case theta (default 4-8Hz)
      and gamma (defaults to 30-80Hz). A PLV of unity indicates perfect phase
      locking (here PAC) and a value of zero indicates no locking (no PAC).

      :param sig: The LFP signal. If None, uses the signal provided during initialization.
      :type sig: np.array, optional
      :param forder: The order of the Butterworth filter (default is 2).
      :type forder: int, optional
      :param thetaband: The lower and upper bands of the theta frequency range (default is [4, 8]).
      :type thetaband: list, optional
      :param gammaband: The lower and upper bands of the gamma frequency range (default is [30, 80]).
      :type gammaband: list, optional
      :param plot: Whether to plot the resulting binned up polar plot which shows the amplitude
                   of the gamma oscillation found at different phases of the theta oscillation
                   (default is True).
      :type plot: bool, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The value of the phase-amplitude coupling (PLV).
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:method:: power_spectrum(eeg=None, plot=True, binWidthSecs=1 / 250, maxFreq=25, pad2pow=None, ymax=None, **kwargs)

      
      Method used by eeg_power_spectra and intrinsic_freq_autoCorr.
      Signal in must be mean normalized already.

      :param eeg: The EEG signal to analyze.
      :type eeg: np.ndarray
      :param plot: Whether to plot the resulting power spectrum (default is True).
      :type plot: bool, optional
      :param binWidthSecs: The bin width in seconds for the power spectrum.
      :type binWidthSecs: float, optional
      :param maxFreq: The upper limit of the power spectrum frequency range
                      (default is 25).
      :type maxFreq: float, optional
      :param pad2pow: The power of 2 to pad the signal to (default is None).
      :type pad2pow: int, optional
      :param ymax: The maximum y-axis value for the plot (default is None).
      :type ymax: float, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: A dictionary containing the power spectrum and other
                related metrics.
                    "maxFreq", (float) - frequency at which max power in theta band
                                         occurs
                    "Power", (np.ndarray) - smoothed power values
                    "Freqs", (np.ndarray) - frequencies corresponding to power
                                            values
                    "s2n", - signal to noise ratio
                    "Power_raw", (np.ndarray) - raw power values
                    "k", (np.ndarray) - smoothing kernel
                    "kernelLen", (float) - length of smoothing kernel
                    "kernelSig", (float) - sigma of smoothing kernel
                    "binsPerHz", (float) - bins per Hz in the power spectrum
                    "kernelLen", (float) - length of the smoothing kernel
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: spike_xy_phase_plot(cluster, pos_data, lfp_data, cluster_times)

      
      Produces a plot of the phase of theta at which each spike was
      emitted. Each spike is plotted according to the x-y location the
      animal was in when it was fired and the colour of the marker
      corresponds to the phase of theta at which it fired.

      :param cluster: The cluster number.
      :type cluster: int
      :param pos_data: Position data object containing position and speed information.
      :type pos_data: PosCalcsGeneric
      :param phy_data: Phy data object containing spike times and clusters.
      :type phy_data: TemplateModel
      :param lfp_data: LFP data object containing the LFP signal and sampling rate.
      :type lfp_data: EEGCalcsGeneric

      :returns: The matplotlib axes object with the plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: theta_running(pos_data, lfp_data, plot = True, **kwargs)

      
      Returns metrics to do with the theta frequency/power and
      running speed/acceleration.

      :param pos_data: Position data object containing position and speed information.
      :type pos_data: PosCalcsGeneric
      :param lfp_data: LFP data object containing the LFP signal and sampling rate.
      :type lfp_data: EEGCalcsGeneric
      :param plot: Whether to plot the results (default is True).
      :type plot: bool
      :param \*\*kwargs:
                         Additional keyword arguments:
                             low_theta : float
                                 Lower bound of theta frequency (default is 6).
                             high_theta : float
                                 Upper bound of theta frequency (defaultt is 12).
                             low_speed : float
                                 Lower bound of running speed (data is masked
                                 below this value)
                             high_speed : float
                                 Upper bound of running speed (data is masked
                                 above this value)
                             nbins : int
                                 Number of bins into which to bin data (Same
                                 number for both speed and theta)
      :type \*\*kwargs: dict

      :returns: A tuple containing masked arrays for speed and theta frequency.
      :rtype: tuple[np.ma.MaskedArray, ...]

      .. rubric:: Notes

      The function calculates the instantaneous frequency of the theta band
      and interpolates the running speed to match the LFP data. It then
      creates a 2D histogram of theta frequency vs. running speed and
      overlays the mean points for each speed bin. The function also
      performs a linear regression to find the correlation between
      speed and theta frequency.















      ..
          !! processed by numpydoc !!


   .. py:attribute:: fs


   .. py:attribute:: sig


   .. py:attribute:: smthKernelSigma
      :value: 0.1875



   .. py:attribute:: smthKernelWidth
      :value: 2



   .. py:attribute:: sn2Width
      :value: 2



   .. py:attribute:: thetaRange
      :value: [6, 12]



   .. py:attribute:: xmax
      :value: 11



.. py:class:: PowerSpectrumParams

   
   Dataclass for holding the parameters for calculating a power
   spectrum as this was being used in several classes and needed
   refactoring out into a standalone function
















   ..
       !! processed by numpydoc !!

   .. py:attribute:: bin_width_in_secs
      :type:  float
      :value: 0.004



   .. py:attribute:: max_frequency
      :type:  float
      :value: 25



   .. py:attribute:: pad_to_power
      :type:  int


   .. py:attribute:: signal
      :type:  numpy.ndarray


   .. py:attribute:: signal_to_noise_width
      :type:  float
      :value: 2



   .. py:attribute:: smoothing_kernel_sigma
      :type:  float
      :value: 0.1875



   .. py:attribute:: smoothing_kernel_width
      :type:  float
      :value: 2



   .. py:attribute:: theta_range
      :type:  List
      :value: [6, 12]



.. py:class:: Rippler(trial_root, signal, fs)

   Bases: :py:obj:`object`


   
   Does some spectrographic analysis and plots of LFP data
   looking specifically at the ripple band

   NB This is tied pretty specifically to an experiment that
   uses TTL pulses to trigger some 'event' / 'events'...

   Until I modified the Ripple Detector plugin the duration of the TTL
   pulses was variable with a more or less bimodal distribution which
   is why there is a separate treatment of short and long duration TTL pulses below















   ..
       !! processed by numpydoc !!

   .. py:method:: _calc_ripple_chunks_duration_power(ttl_type='no_laser')

      
      Find the indices and durations of the events that have sufficient
      duration and power to be considered ripples.

      :param ttl_type: which bit of the trial to do the calculation for
                       Either 'no_laser' or 'laser'
      :type ttl_type: str, default='no_laser'

      :returns: the run indices to keep and the run durations in ms
      :rtype: tuple















      ..
          !! processed by numpydoc !!


   .. py:method:: _find_high_power_periods(n = 3, t = 10)

      
      Find periods where the power in the ripple band is above n standard deviations
      for t samples. Meant to recapitulate the algorithm from the Ripple Detector
      plugin.

      :param n: The number of standard deviations above the mean power to consider as high power (default is 3).
      :type n: int, optional
      :param t: The number of samples for which the power must be above the threshold (default is 10).
      :type t: int, optional

      :returns: An array of indices where the power is above the threshold for the specified duration.
      :rtype: np.ndarray















      ..
          !! processed by numpydoc !!


   .. py:method:: _find_path_to_continuous(trial_root, **kwargs)

      
      Iterates through a directory tree and finds the path to the
      Ripple Detector plugin data and returns its location.

      :param trial_root: The root directory of the trial.
      :type trial_root: Path
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The path to the continuous data.
      :rtype: Path















      ..
          !! processed by numpydoc !!


   .. py:method:: _find_path_to_ripple_ttl(trial_root, **kwargs)

      
      Iterates through a directory tree and finds the path to the
      Ripple Detector plugin data and returns its location.

      :param trial_root: The root directory of the trial.
      :type trial_root: Path
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The path to the ripple TTL data.
      :rtype: Path















      ..
          !! processed by numpydoc !!


   .. py:method:: _load_start_time(path_to_sync_message_file)

      
      Returns the start time contained in a sync file from OE.

      :param path_to_sync_message_file: Path to the sync message file.
      :type path_to_sync_message_file: Path

      :returns: The start time in seconds.
      :rtype: float















      ..
          !! processed by numpydoc !!


   .. py:method:: _plot_ripple_lfp_with_ttl(i_time, **kwargs)


   .. py:method:: filter_timestamps_for_real_ripples()

      
      Filter out low power and short duration events from the list of timestamps
















      ..
          !! processed by numpydoc !!


   .. py:method:: get_spectrogram(start_time, end_time, plot=False)

      
      Computes the spectrogram of the filtered LFP signal between the specified start and end times.

      :param start_time: The start time of the chunk to analyze, in seconds.
      :type start_time: float
      :param end_time: The end time of the chunk to analyze, in seconds.
      :type end_time: float
      :param plot: Whether to plot the resulting spectrogram (default is False).
      :type plot: bool, optional

      :returns: A tuple containing the ShortTimeFFT object, the number of samples, and the spectrogram array.
      :rtype: tuple















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_and_save_ripple_band_lfp_with_ttl(**kwargs)

      
      Plots and saves the ripple band LFP signal with TTL events.

      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_filtered_lfp_chunk(start_time, end_time, **kwargs)

      
      Plots a chunk of the filtered LFP signal between the specified start and end times.

      :param start_time: The start time of the chunk to plot, in seconds.
      :type start_time: float
      :param end_time: The end time of the chunk to plot, in seconds.
      :type end_time: float
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The matplotlib axes object with the plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_mean_rippleband_power(**kwargs)

      
      Plots the mean power in the ripple band for the laser on and no laser conditions.

      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The matplotlib axes object with the plot, or None if no data is available.
      :rtype: plt.Axes | None















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_mean_spectrogram(laser_on = False, ax=None, **kwargs)

      
      Plots the mean spectrograms for both laser on and laser off conditions.

      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The matplotlib figure object with the plots.
      :rtype: plt.Figure















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_mean_spectrograms(**kwargs)

      
      Plots the spectrograms of the LFP signal for both laser on
      and laser off conditions.
















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_rasters(laser_on)

      
      Plots raster plots for the given laser condition.

      :param laser_on: If True, plots rasters for laser on condition. If False, plots rasters for no laser condition.
      :type laser_on: bool















      ..
          !! processed by numpydoc !!


   .. py:method:: update_bandpass(low=None, high=None)

      
      Updates the bandpass filter settings.

      :param low: The low frequency for the bandpass filter.
      :type low: int, optional
      :param high: The high frequency for the bandpass filter.
      :type high: int, optional















      ..
          !! processed by numpydoc !!


   .. py:attribute:: LFP


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



.. py:function:: power_spectrum(params, plot=True, pad2pow=None, ymax=None, **kwargs)

   
   Method used by eeg_power_spectra and intrinsic_freq_autoCorr.
   Signal in must be mean normalized already.

   :param eeg: The EEG signal to analyze.
   :type eeg: np.ndarray
   :param plot: Whether to plot the resulting power spectrum (default is True).
   :type plot: bool, optional
   :param binWidthSecs: The bin width in seconds for the power spectrum.
   :type binWidthSecs: float, optional
   :param maxFreq: The upper limit of the power spectrum frequency range
                   (default is 25).
   :type maxFreq: float, optional
   :param pad2pow: The power of 2 to pad the signal to (default is None).
   :type pad2pow: int, optional
   :param ymax: The maximum y-axis value for the plot (default is None).
   :type ymax: float, optional
   :param \*\*kwargs: Additional keyword arguments.
   :type \*\*kwargs: dict

   :returns: A dictionary containing the power spectrum and other
             related metrics.
                 "maxFreq", (float) - frequency at which max power in theta band
                                         occurs
                 "Power", (np.ndarray) - smoothed power values
                 "Freqs", (np.ndarray) - frequencies corresponding to power
                                         values
                 "s2n", - signal to noise ratio
                 "Power_raw", (np.ndarray) - raw power values
                 "k", (np.ndarray) - smoothing kernel
                 "kernelLen", (float) - length of smoothing kernel
                 "kernelSig", (float) - sigma of smoothing kernel
                 "binsPerHz", (float) - bins per Hz in the power spectrum
                 "kernelLen", (float) - length of the smoothing kernel
   :rtype: dict















   ..
       !! processed by numpydoc !!

