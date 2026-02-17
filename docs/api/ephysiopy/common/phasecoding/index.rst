ephysiopy.common.phasecoding
============================

.. py:module:: ephysiopy.common.phasecoding


Attributes
----------

.. autoapisummary::

   ephysiopy.common.phasecoding.cbar_fontsize
   ephysiopy.common.phasecoding.cbar_tick_fontsize
   ephysiopy.common.phasecoding.jet_cmap


Functions
---------

.. autoapisummary::

   ephysiopy.common.phasecoding.detect_oscillation_episodes
   ephysiopy.common.phasecoding.get_bad_cycles
   ephysiopy.common.phasecoding.get_cross_wavelet
   ephysiopy.common.phasecoding.get_cycle_labels
   ephysiopy.common.phasecoding.get_phase_of_min_spiking
   ephysiopy.common.phasecoding.get_theta_cycle_spectogram
   ephysiopy.common.phasecoding.theta_filter_lfp


Module Contents
---------------

.. py:function:: detect_oscillation_episodes(lfp, fs)

.. py:function:: get_bad_cycles(filtered_eeg, negative_freqs, cycle_labels, min_power_percent_threshold, min_theta, max_theta, lfp_fs)

   
   Get the cycles that are bad based on their length and power

   :param filtered_eeg: The filtered EEG signal
   :type filtered_eeg: np.ndarray
   :param negative_freqs: A boolean array indicating negative frequencies
   :type negative_freqs: np.ndarray
   :param cycle_labels: The cycle labels for the phase array
   :type cycle_labels: np.ndarray
   :param min_power_percent_threshold: The minimum power percent threshold for rejecting cycles
   :type min_power_percent_threshold: float
   :param min_theta: The minimum theta frequency
   :type min_theta: float
   :param max_theta: The maximum theta frequency
   :type max_theta: float
   :param lfp_fs: The sampling frequency of the LFP signal
   :type lfp_fs: float

   :returns: A boolean array indicating bad cycles
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: get_cross_wavelet(theta_phase, theta_lfp, gamma_lfp, fs, **kwargs)

   
   Get the cross wavelet transform between the theta and gamma LFP signals
















   ..
       !! processed by numpydoc !!

.. py:function:: get_cycle_labels(spike_phase, min_allowed_min_spike_phase)

   
   Get the cycle labels for a given phase array

   :param phase: The phases at which the spikes were fired.
   :type phase: np.ndarray
   :param min_allowed_min_spike_phase: The minimum allowed phase for cycles to start.
   :type min_allowed_min_spike_phase: float

   :returns: The cycle labels for the phase array
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: get_phase_of_min_spiking(spkPhase)

   
   Returns the phase at which the minimum number of spikes are fired

   :param spkPhase: The phase of the spikes
   :type spkPhase: np.ndarray

   :returns: The phase in degrees at which the minimum number of spikes are fired
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: get_theta_cycle_spectogram(phase, cycle_label, filt_lfp, lfp, fs, **kwargs)

   
   Get a spectrogram of the theta cycles in the LFP
















   ..
       !! processed by numpydoc !!

.. py:function:: theta_filter_lfp(lfp, fs, **kwargs)

   
   Processes an LFP signal for theta cycles, filtering
   out bad cycles (low power, too long/ short etc) and
   applying labels to each cycle etc
















   ..
       !! processed by numpydoc !!

.. py:data:: cbar_fontsize
   :value: 8


.. py:data:: cbar_tick_fontsize
   :value: 6


.. py:data:: jet_cmap

