ephysiopy.visualise.plotting
============================

.. py:module:: ephysiopy.visualise.plotting


Attributes
----------

.. autoapisummary::

   ephysiopy.visualise.plotting.grey_cmap
   ephysiopy.visualise.plotting.jet_cmap


Classes
-------

.. autoapisummary::

   ephysiopy.visualise.plotting.FigureMaker


Functions
---------

.. autoapisummary::

   ephysiopy.visualise.plotting.addClusterChannelToAxes
   ephysiopy.visualise.plotting.savePlot
   ephysiopy.visualise.plotting.stripAxes


Module Contents
---------------

.. py:class:: FigureMaker

   Bases: :py:obj:`object`


   
   A mixin class for TrialInterface that deals solely with producing graphical output.
















   ..
       !! processed by numpydoc !!

   .. py:method:: _getPowerSpectrumPlot(freqs, power, sm_power, band_max_power, freq_at_band_max_power, max_freq = 50, theta_range = [6, 12], ax = None, **kwargs)

      
      Gets the power spectrum. The parameters can be obtained from
      calcEEGPowerSpectrum() in the EEGCalcsGeneric class.

      :param freqs: The frequencies.
      :type freqs: np.ndarray
      :param power: The power values.
      :type power: np.ndarray
      :param sm_power: The smoothed power values.
      :type sm_power: np.ndarray
      :param band_max_power: The maximum power in the band.
      :type band_max_power: float
      :param freq_at_band_max_power: The frequency at which the maximum power in the band occurs.
      :type freq_at_band_max_power: float
      :param max_freq: The maximum frequency. Defaults to 50.
      :type max_freq: int, optional
      :param theta_range: The theta range. Defaults to [6, 12].
      :type theta_range: tuple, optional
      :param ax: The axes to plot on. If None, new axes are created.
      :type ax: plt.Axes, optional
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes with the plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: _getRasterPlot(spk_times, dt=(-0.05, 0.1), ax = None, cluster=0, secs_per_bin = 0.001, **kwargs)

      
      Plots a raster plot for a specified tetrode/ cluster.

      :param spk_times: The spike times in seconds.
      :type spk_times: np.ndarray
      :param dt: The window of time in ms to examine zeroed on the event of interest.
                 Defaults to (-0.05, 0.1).
      :type dt: tuple, optional
      :param ax: The axes to plot into. If not provided, a new figure is created.
                 Defaults to None.
      :type ax: matplotlib.axes, optional
      :param cluster: The cluster number. Defaults to 0.
      :type cluster: int, optional
      :param secs_per_bin: The number of seconds in each bin of the raster plot. Defaults to 0.001.
      :type secs_per_bin: int, optional
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes with the plot, or None if no spikes were fired in the period.
      :rtype: plt.Axes or None















      ..
          !! processed by numpydoc !!


   .. py:method:: _getXCorrPlot(spk_times, ax = None, **kwargs)

      
      Returns an axis containing the autocorrelogram of the spike
      times provided over the range +/-500ms.

      :param spk_times: Spike times in seconds.
      :type spk_times: np.array
      :param ax: The axes to plot into. If None, new axes are created.
      :type ax: matplotlib.axes, optional
      :param \*\*kwargs: Additional keyword arguments for the function, including:
                         binsize : int, optional
                             The size of the bins in ms. Gets passed to SpikeCalcsGeneric.xcorr().
                             Defaults to 1.
      :type \*\*kwargs: dict

      :returns: The axes with the plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: _plotWaves(waves, ax, **kwargs)


   .. py:method:: _plot_multiple_clusters(func, clusters, channel, **kwargs)

      
      Plots multiple clusters.

      :param func: The function to apply to each cluster.
      :type func: function
      :param clusters: The list of clusters to plot.
      :type clusters: list
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The figure containing the plots.
      :rtype: matplotlib.figure.Figure















      ..
          !! processed by numpydoc !!


   .. py:method:: plotSpectrogramByDepth(nchannels = 384, nseconds = 100, maxFreq = 125, channels = [], frequencies = [], frequencyIncrement = 1, **kwargs)

      
      Plots a heat map spectrogram of the LFP for each channel.
      Line plots of power per frequency band and power on a subset of
      channels are also displayed to the right and above the main plot.

      :param nchannels: The number of channels on the probe.
      :type nchannels: int
      :param nseconds: How long in seconds from the start of the trial to do the spectrogram for (for speed).
                       Default is 100.
      :type nseconds: int, optional
      :param maxFreq: The maximum frequency in Hz to plot the spectrogram out to. Maximum is 1250. Default is 125.
      :type maxFreq: int
      :param channels: The channels to plot separately on the top plot.
      :type channels: list
      :param frequencies: The specific frequencies to examine across all channels. The mean from frequency:
                          frequency+frequencyIncrement is calculated and plotted on the left hand side of the plot.
      :type frequencies: list
      :param frequencyIncrement: The amount to add to each value of the frequencies list above.
      :type frequencyIncrement: int
      :param \*\*kwargs:
                         Additional keyword arguments for the function. Valid key value pairs:
                             "saveas" - save the figure to this location, needs absolute path and filename.
      :type \*\*kwargs: dict

      .. rubric:: Notes

      Should also allow kwargs to specify exactly which channels and / or frequency bands to do the line plots for.















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_acorr(cluster, channel, **kwargs)

      
      Plots the autocorrelogram for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the autocorrelogram for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the autocorrelogram plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_clusters_theta_phase(cluster, channel, **kwargs)

      
      Plots the theta phase for the specified cluster and channel.

      :param cluster: The cluster to get the theta phase for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the theta phase plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_eb_map(cluster, channel, **kwargs)

      
      Plots the ego-centric boundary map for the specified cluster(s) and
      channel.

      :param cluster: The cluster(s) to get the ego-centric boundary map for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the ego-centric boundary map plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_eb_spikes(cluster, channel, **kwargs)

      
      Plots the ego-centric boundary spikes for the specified cluster(s)
      and channel.

      :param cluster: The cluster(s) to get the ego-centric boundary spikes for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the ego-centric boundary spikes plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_hd_map(cluster, channel, **kwargs)

      
      Gets the head direction map for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the head direction map for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the head direction map plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_power_spectrum(**kwargs)

      
      Plots the power spectrum.

      :param \*\*kwargs: Additional keyword arguments passed to _getPowerSpectrumPlot















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_raster(cluster, channel, **kwargs)

      
      Plots the raster plot for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the raster plot for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function, including:
                         dt : list
                             The range in seconds to plot data over either side of the TTL pulse.
                         seconds_per_bin : float
                             The number of seconds per bin.
      :type \*\*kwargs: dict

      :returns: The axes containing the raster plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_rate_map(cluster, channel, **kwargs)

      
      Plots the rate map for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the rate map for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the rate map plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_sac(cluster, channel, **kwargs)

      
      Plots the spatial autocorrelation for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the spatial autocorrelation for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the spatial autocorrelation plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_speed_v_hd(cluster, channel, **kwargs)

      
      Plots the speed versus head direction plot for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the speed versus head direction plot for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the speed versus head direction plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_speed_v_rate(cluster, channel, **kwargs)

      
      Plots the speed versus rate plot for the specified cluster(s) and
      channel.

      By default the distribution of speeds will be plotted as a twin
      axis. To disable set add_speed_hist = False

      :param cluster: The cluster(s) to get the speed versus rate plot for.
      :type cluster: int
      :param channel: The channel number.
      :type channel: int
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the speed versus rate plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_spike_path(cluster=None, channel=None, **kwargs)

      
      Plots the spikes on the path for the specified cluster(s) and channel.

      :param cluster: The cluster(s) to get the spike path for.
      :type cluster: int or None
      :param channel: The channel number.
      :type channel: int or None
      :param \*\*kwargs: Additional keyword arguments for the function.
      :type \*\*kwargs: dict

      :returns: The axes containing the spike path plot.
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_theta_vs_running_speed(**kwargs)

      
      Plots theta frequency versus running speed.

      :param \*\*kwargs: Additional keyword arguments for the function, including:
                         low_theta : float
                             The lower bound of the theta frequency range (default is 6).
                         high_theta : float
                             The upper bound of the theta frequency range (default is 12).
                         low_speed : float
                             The lower bound of the running speed range (default is 2).
                         high_speed : float
                             The upper bound of the running speed range (default is 50).
      :type \*\*kwargs: dict

      :returns: The QuadMesh object containing the plot.
      :rtype: QuadMesh















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_xcorr(cluster_a, channel_a, cluster_b, channel_b, **kwargs)

      
      Plots the temporal cross-correlogram between cluster_a and cluster_b

      :param cluster_a: first cluster
      :type cluster_a: int
      :param channel_a: first channel
      :type channel_a: int
      :param cluster_b: second cluster
      :type cluster_b: int
      :param channel_b: second channel
      :type channel_b: int

      :returns: The axes containing the cross-correlogram plot
      :rtype: plt.Axes















      ..
          !! processed by numpydoc !!


   .. py:attribute:: PosCalcs
      :value: None


      
      Initializes the FigureMaker object with data from PosCalcs.
















      ..
          !! processed by numpydoc !!


.. py:function:: addClusterChannelToAxes(func)

   
   Decorator to add cluster and channel information to the axes of a plot.

   :param func: The function that generates the plot.
   :type func: callable

   :returns: The wrapped function that adds cluster and channel information to the axes.
   :rtype: callable















   ..
       !! processed by numpydoc !!

.. py:function:: savePlot(func)

   
   Decorator to save a plot generated by a function.

   :param func: The function that generates the plot.
   :type func: callable

   :returns: The wrapped function that saves the plot if 'save_as' is provided in kwargs.
   :rtype: callable















   ..
       !! processed by numpydoc !!

.. py:function:: stripAxes(func)

   
   Decorator to strip the axes from a plot generated by a function.

   :param func: The function that generates the plot.
   :type func: callable

   :returns: The wrapped function that strips the axes from the plot.
   :rtype: callable















   ..
       !! processed by numpydoc !!

.. py:data:: grey_cmap

.. py:data:: jet_cmap

