"""
A lot of the functionality here has been more generically implemented
in the ephys_generic.ephys_generic.SpikeCalcsGeneric class
"""
import numpy as np
import warnings
from scipy import signal		
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from ephysiopy.common.utils import blur_image

"""
NB AS OF 19/10/20 I MOVED MOST OF THE METHODS OF THE CLASSSPIKECALCS
INTO EPHYSIOPY.COMMON.SPIKECALCS.SPIKECALCSGENERIC

THE ONES REMAINING BELOW INVOLVE COMBINING SPIKES AND POSITION
INFORMATION SO SHOULD BE IN SOME OTHER PLACE...
"""

class SpikeCalcs(object):
	"""
	Mix-in class for use with Tetrode class below.
	
	Extends Tetrodes functionality by adding methods for analysis of spikes/
	spike trains
				
	Note lots of the methods here are native to dacq2py.axonaIO.Tetrode
	
	Note that units are in milliseconds
	"""
	
	def ifr_sp_corr(self, clusterA, speed, minSpeed=2.0, maxSpeed=40.0, sigma=3, 
					shuffle=False, nShuffles=100, minTime=30, plot=False):
		"""
		clusterA: int
			the cluster to do the correlation with speed
		speed: np.array (1 x nSamples)
			instantaneous speed 
		minSpeed: int
			speeds below this value are ignored - defaults to 2cm/s as with
			Kropff et al., 2015
		"""
		if clusterA not in self.clusters:
			warnings.warn('Cluster not available. Try again!')
		else:
			speed = speed.ravel()
			posSampRate = 50
			nSamples = len(speed)
			x1 = self.getClustIdx(clusterA)
			# position is sampled at 50Hz and so is 'automatically' binned into
			# 20ms bins
			spk_hist = np.bincount(x1, minlength=nSamples)
			# smooth the spk_hist (which is a temporal histogram) with a 250ms
			# gaussian as with Kropff et al., 2015
			h = signal.gaussian(13, sigma)
			h = h / float(np.sum(h))
			#filter for low speeds
			lowSpeedIdx = speed < minSpeed
			highSpeedIdx = speed > maxSpeed
			speed_filt = speed[~np.logical_or(lowSpeedIdx, highSpeedIdx)]
			spk_hist_filt = spk_hist[~np.logical_or(lowSpeedIdx, highSpeedIdx)]
			spk_sm = signal.filtfilt(h.ravel(), 1, spk_hist_filt)
			sm_spk_rate = spk_sm * posSampRate
			
			res = stats.pearsonr(sm_spk_rate, speed_filt)
			if plot:            
				# do some fancy plotting stuff
				speed_binned, sp_bin_edges = np.histogram(speed_filt, bins=50)
				sp_dig = np.digitize(speed_filt, sp_bin_edges, right=True)
				spks_per_sp_bin = [spk_hist_filt[sp_dig==i] for i in range(len(sp_bin_edges))] 
				rate_per_sp_bin = []
				for x in spks_per_sp_bin:
					rate_per_sp_bin.append(np.mean(x) * posSampRate)
				rate_filter = signal.gaussian(5, 1.0)
				rate_filter = rate_filter / np.sum(rate_filter)
				binned_spk_rate = signal.filtfilt(rate_filter, 1, rate_per_sp_bin)
				# instead of plotting a scatter plot of the firing rate at each 
				# speed bin, plot a log normalised heatmap and overlay results on it
				
				spk_binning_edges = np.linspace(np.min(sm_spk_rate), np.max(sm_spk_rate),
												len(sp_bin_edges))
				speed_mesh, spk_mesh = np.meshgrid(sp_bin_edges, spk_binning_edges)
				binned_rate, _, _ = np.histogram2d(speed_filt, sm_spk_rate, bins=[sp_bin_edges,
												   spk_binning_edges])
				#blur the binned rate a bit to make it look nicer
				sm_binned_rate = blur_image(binned_rate, 5)
				plt.figure()
				plt.pcolormesh(speed_mesh, spk_mesh, sm_binned_rate, norm=colors.LogNorm(), alpha=0.5, shading='flat', edgecolors='None')
				#overlay the smoothed binned rate against speed
				plt.hold(True)
				plt.plot(sp_bin_edges, binned_spk_rate, 'r')
				#do the linear regression and plot the fit too
				# TODO: linear regression is broken ie not regressing the correct variables
				lr = stats.linregress(speed_filt, sm_spk_rate)
				end_point = lr.intercept + ((sp_bin_edges[-1] - sp_bin_edges[0]) * lr.slope)
				plt.plot([np.min(sp_bin_edges), np.max(sp_bin_edges)], [lr.intercept, end_point], 'r--')
				ax = plt.gca()
				ax.set_xlim(np.min(sp_bin_edges), np.max(sp_bin_edges[-2]))
				ax.set_ylim(0, np.nanmax(binned_spk_rate) * 1.1)
				ax.set_ylabel('Firing rate(Hz)')
				ax.set_xlabel('Running speed(cm/s)')
				ax.set_title('Intercept: {0:.3f}    Slope: {1:.5f}\nPearson: {2:.5f}'.format(lr.intercept, lr.slope, lr.rvalue))
			#do some shuffling of the data to see if the result is signficant            
			if shuffle:                
				# shift spikes by at least 30 seconds after trial start and
				# 30 seconds before trial end
				timeSteps = np.random.randint(30 * posSampRate, nSamples - (30 * posSampRate),
													  nShuffles)
				shuffled_results = []            
				for t in timeSteps:
					spk_count = np.roll(spk_hist, t)
					spk_count_filt = spk_count[~lowSpeedIdx]
					spk_count_sm = signal.filtfilt(h.ravel(), 1, spk_count_filt)
					shuffled_results.append(stats.pearsonr(spk_count_sm, speed_filt)[0])
				if plot:
					plt.figure()
					ax = plt.gca()
					ax.hist(np.abs(shuffled_results), 20)
					ylims = ax.get_ylim()
					ax.vlines(res, ylims[0], ylims[1], 'r')
				
			print("PPMC: {0}".format(res[0]))


	def smoothSpikePosCount(self, x1, npos, sigma=3.0, shuffle=None):
		"""
		Returns a spike train the same length as num pos samples that has been
		smoothed in time with a gaussian kernel M in width and standard deviation
		equal to sigma
		
		Parameters
		--------------
		x1 : np.array
			The pos indices the spikes occured at
		npos : int
			The number of position samples captured
		sigma : float
			the standard deviation of the gaussian used to smooth the spike
			train
		shuffle: int
			The number of seconds to shift the spike train by. Default None
		
		Returns
		-----------
		smoothed_spikes : np.array
			The smoothed spike train
		"""
		spk_hist = np.bincount(x1, minlength=npos)
		if shuffle is not None:
			spk_hist = np.roll(spk_hist, int(shuffle * 50))
		# smooth the spk_hist (which is a temporal histogram) with a 250ms
		# gaussian as with Kropff et al., 2015
		h = signal.gaussian(13, sigma)
		h = h / float(np.sum(h))
		return signal.filtfilt(h.ravel(), 1, spk_hist)
	