import numpy as np
import matplotlib
import matplotlib.pylab as plt
from itertools import groupby
from operator import itemgetter
from ephysiopy.common.ephys_generic import PosCalcsGeneric, SpikeCalcsGeneric

class CosineDirectionalTuning(object):
	"""
	Produces output to do with Welday et al (2011) like analysis
	of rhythmic firing a la oscialltory interference model
	"""

	@staticmethod
	def bisection(array,value):
		'''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
		and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
		to indicate that ``value`` is out of range below and above respectively.
		
		NB From SO:
		https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
		'''
		n = len(array)
		if (value < array[0]):
			return -1
		elif (value > array[n-1]):
			return n
		jl = 0# Initialize lower
		ju = n-1# and upper limits.
		while (ju-jl > 1):# If we are not yet done,
			jm=(ju+jl) >> 1# compute a midpoint with a bitshift
			if (value >= array[jm]):
				jl=jm# and replace either the lower limit
			else:
				ju=jm# or the upper limit, as appropriate.
			# Repeat until the test condition is satisfied.
		if (value == array[0]):# edge cases at bottom
			return 0
		elif (value == array[n-1]):# and top
			return n-1
		else:
			return jl

	def __init__(self, spike_times: np.array, pos_times: np.array, spk_clusters: np.array, x: np.array, y: np.array, tracker_params: dict):
		"""
		Parameters
		----------
		spike_times - 1d np.array
		pos_times - 1d np.array
		spk_clusters - 1d np.array
		pos_xy - 1d np.array
		tracker_params - dict - from the PosTracker as created in OEKiloPhy.Settings.parsePos

		NB All timestamps should be given in sub-millisecond accurate seconds and pos_xy in cms
		"""
		self.spike_times = spike_times
		self.pos_times = pos_times
		self.spk_clusters = spk_clusters
		'''
		There can be more spikes than pos samples in terms of sampling as the
		open-ephys buffer probably needs to finish writing and the camera has
		already stopped, so cut of any cluster indices and spike times
		that exceed the length of the pos indices
		'''
		idx_to_keep = self.spike_times < self.pos_times[-1]
		self.spike_times = self.spike_times[idx_to_keep]
		self.spk_clusters = self.spk_clusters[idx_to_keep]
		self._pos_sample_rate = 30
		self._spk_sample_rate = 3e4
		self._pos_samples_for_spike = None
		self._min_runlength = 0.4 # in seconds
		self.posCalcs = PosCalcsGeneric(x, y, 230, cm=True, jumpmax=100)
		self.spikeCalcs = SpikeCalcsGeneric(spike_times)
		self.spikeCalcs.spk_clusters = spk_clusters
		xy, hdir = self.posCalcs.postprocesspos(tracker_params)
		self.posCalcs.calcSpeed(xy)
		self._xy = xy
		self._hdir = hdir
		self._speed = self.posCalcs.speed
		# TEMPORARY FOR POWER SPECTRUM STUFF
		self.smthKernelWidth = 2
		self.smthKernelSigma = 0.1875
		self.sn2Width = 2
		self.thetaRange = [7,11]
		self.xmax = 11
		
	@property
	def spk_sample_rate(self):
		return self._spk_sample_rate

	@spk_sample_rate.setter
	def spk_sample_rate(self, value):
		self._spk_sample_rate = value
	
	@property
	def pos_sample_rate(self):
		return self._pos_sample_rate
	
	@pos_sample_rate.setter
	def pos_sample_rate(self, value):
		self._pos_sample_rate = value

	@property
	def min_runlength(self):
		return self._min_runlength
	
	@min_runlength.setter
	def min_runlength(self, value):
		self._min_runlength = value

	@property
	def xy(self):
		return self._xy

	@xy.setter
	def xy(self, value):
		self._xy = value

	@property
	def hdir(self):
		return self._hdir

	@hdir.setter
	def hdir(self, value):
		self._hdir = value

	@property
	def speed(self):
		return self._speed
	
	@speed.setter
	def speed(self, value):
		self._speed = value

	@property
	def pos_samples_for_spike(self):
		return self._pos_samples_for_spike

	@pos_samples_for_spike.setter
	def pos_samples_for_spike(self, value):
		self._pos_samples_for_spike = value

	def _rolling_window(self, a: np.array, window: int):
			"""
			Totally nabbed from SO:
			https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
			"""
			shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
			strides = a.strides + (a.strides[-1],)
			return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
	
	def getPosIndices(self):
		self.pos_samples_for_spike = np.floor(self.spike_times * self.pos_sample_rate).astype(int)
	
	def getClusterPosIndices(self, cluster: int)->np.array:
		if self.pos_samples_for_spike is None:
			self.getPosIndices()
		cluster_pos_indices = self.pos_samples_for_spike[self.spk_clusters==cluster]
		cluster_pos_indices[cluster_pos_indices>=len(self.pos_times)] = len(self.pos_times)-1
		return cluster_pos_indices

	def getClusterSpikeTimes(self, cluster: int):
		ts = self.spike_times[self.spk_clusters==cluster]
		if self.pos_samples_for_spike is None:
			self.getPosIndices()
		# cluster_pos_indices = self.pos_samples_for_spike[self.spk_clusters==cluster]
		# idx_to_keep = cluster_pos_indices < len(self.pos_times)
		return ts#[idx_to_keep]
	
	def getDirectionalBinPerPosition(self, binwidth: int):
		"""
		Direction is in degrees as that what is created by me in some of the
		other bits of this package.

		Parameters
		----------
		binwidth : int - binsizethe bin width in degrees

		Outputs
		-------
		A digitization of which directional bin each position sample belongs to
		"""
		
		bins = np.arange(0, 360, binwidth)
		return np.digitize(self.hdir, bins)

	def getDirectionalBinForCluster(self, cluster: int):
		b = self.getDirectionalBinPerPosition(45)
		cluster_pos = self.getClusterPosIndices(cluster)
		# idx_to_keep = cluster_pos < len(self.pos_times)
		# cluster_pos = cluster_pos[idx_to_keep]
		return b[cluster_pos]

	def getRunsOfMinLength(self):
		"""
		Identifies runs of at least self.min_runlength seconds long, which at 30Hz pos
		sampling rate equals 12 samples, and returns the start and end indices at which
		the run was occurred and the directional bin that run belongs to

		Returns
		-------
		np.array - the start and end indices into position samples of the run and the
					directional bin to which it belongs
		"""

		b = self.getDirectionalBinPerPosition(45)
		# nabbed from SO
		from itertools import groupby
		grouped_runs = [(k,sum(1 for i in g)) for k,g in groupby(b)]
		grouped_runs_array = np.array(grouped_runs)
		run_start_indices = np.cumsum(grouped_runs_array[:,1]) - grouped_runs_array[:,1]
		minlength_in_samples = int(self.pos_sample_rate * self.min_runlength)
		runs_at_least_minlength_to_keep_mask = grouped_runs_array[:, 1] >= minlength_in_samples
		ret = np.array([run_start_indices[runs_at_least_minlength_to_keep_mask], grouped_runs_array[runs_at_least_minlength_to_keep_mask,1]]).T
		ret = np.insert(ret, 1, np.sum(ret, 1), 1) # contains run length as last column
		ret = np.insert(ret, 2, grouped_runs_array[runs_at_least_minlength_to_keep_mask, 0], 1)
		return ret[:,0:3]

	def speedFilterRuns(self, runs: np.array, minspeed=5.0):
		"""
		Given the runs identified in getRunsOfMinLength, filter for speed and return runs
		that meet the min speed criteria

		The function goes over the runs with a moving window of length equal to self.min_runlength in samples
		and sees if any of those segments meets the speed criteria and splits them out into separate runs if
		true
		
		NB For now this means the same spikes might get included in the autocorrelation procedure later as the 
		moving window will use overlapping periods - can be modified later


		Parameters
		----------
		runs - 3 x nRuns np.array generated from getRunsOfMinLength
		minspeed - float - min running speed in cm/s for an epoch (minimum epoch length defined previously
							in getRunsOfMinLength as minlength, usually 0.4s)

		Returns
		-------
		3 x nRuns np.array - A modified version of the "runs" input variable
		"""
		minlength_in_samples = int(self.pos_sample_rate * self.min_runlength)
		run_list = runs.tolist()
		all_speed = np.array(self.speed)
		for start_idx, end_idx, dir_bin in run_list:
			this_runs_speed = all_speed[start_idx:end_idx]
			this_runs_runs = self._rolling_window(this_runs_speed, minlength_in_samples)
			run_mask = np.all(this_runs_runs > minspeed, 1)
			if np.any(run_mask):
				print("got one")

	def testing(self, cluster: int):
		ts = self.getClusterSpikeTimes(cluster)
		pos_idx = self.getClusterPosIndices(cluster)

		dir_bins = self.getDirectionalBinPerPosition(45)
		cluster_dir_bins = dir_bins[pos_idx.astype(int)]

		from ephysiopy.dacq2py.spikecalcs import SpikeCalcs
		from scipy.signal import periodogram, boxcar, filtfilt
		from collections import OrderedDict
		sub_dict = OrderedDict.fromkeys(('Theta max idx', 'Theta max freq', 'acorr', 'acorr_bins'))
		acorr_dict = OrderedDict.fromkeys(range(1,9),sub_dict)

		acorrs = []
		max_freqs = []
		max_idx = []
		isis = []

		nbins = 501
		
		acorr_range = np.array([-500,500])
		for i in range(1,9):
			this_bin_indices = cluster_dir_bins == i
			this_ts = ts[this_bin_indices] # in seconds still so * 1000 for ms
			y = self.spikeCalcs.xcorr(this_ts*1000, Trange=acorr_range)
			isis.append(y)
			corr, acorr_bins = np.histogram(y[y != 0], bins= 501, range=acorr_range)
			freqs, power = periodogram(corr, fs=200, return_onesided=True)
			# Smooth the power over +/- 1Hz
			b = boxcar(3)
			h = filtfilt(b, 3, power)
			# Square the amplitude first
			sqd_amp = h ** 2
			# Then find the mean power in the +/-1Hz band either side of that
			theta_band_max_idx = np.nonzero(sqd_amp==np.max(sqd_amp[np.logical_and(freqs>6, freqs<11)]))[0][0]
			max_freq = freqs[theta_band_max_idx]
			acorrs.append(corr)
			max_freqs.append(max_freq)
			max_idx.append(theta_band_max_idx)
		return isis, acorrs, max_freqs, max_idx, acorr_bins

	def plotXCorrsByDirection(self, cluster: int):
		acorr_range = np.array([-500,500])
		# plot_range = np.array([-400,400])
		nbins = 501
		isis, acorrs, max_freqs, max_idx, acorr_bins = self.testing(cluster)
		bin_labels = np.arange(0, 360, 45)
		fig, axs = plt.subplots(8)
		max_finding_mask = np.logical_and(acorr_bins[0:-1]>60, acorr_bins[0:-1]<180)
		mask_start_point = np.where(max_finding_mask==True)[0][0]
		from scipy.signal import find_peaks
		pts = []
		for i, a in enumerate(isis):
			axs[i].hist(a[a!=0], bins=nbins, range=acorr_range, color='k', histtype='stepfilled')
			# find the max of the first positive peak
			corr, _ = np.histogram(a[a != 0], bins=nbins, range=acorr_range)
			peaks = find_peaks(corr, height=100, distance=60)

			# for xx in peaks[0]:
			# 	axs[i].plot(xx, 0, 'r*')
			axs[i].set_xlim(acorr_range)
			axs[i].set_ylabel(str(bin_labels[i]))
			axs[i].set_yticklabels('')
			if i < 7:
				axs[i].set_xticklabels('')
			axs[i].spines['right'].set_visible(False)
			axs[i].spines['top'].set_visible(False)
			axs[i].spines['left'].set_visible(False)
		plt.show()
		return pts

	def plotSpeedHisto(self):
		pass



	def intrinsic_freq_autoCorr(self, spkTimes=None, posMask=None, maxFreq=25,
								acBinSize=0.002, acWindow=0.5, plot=True,
								**kwargs):
		"""
		This is taken and adapted from ephysiopy.common.eegcalcs.EEGCalcs

		Parameters
		----------
		spkTimes - np.array of times in seconds of the cells firing
		posMask - boolean array corresponding to the length of spkTimes I guess where True 
					is stuff to keep
		maxFreq - the maximum frequency to do the power spectrum out to
		acBinSize - the bin size of the autocorrelogram in seconds
		acWindow - the range of the autocorr in seconds

		NB Make sure all times are in seconds
		"""
		acBinsPerPos = 1. / self.pos_sample_rate / acBinSize
		acWindowSizeBins = np.round(acWindow / acBinSize)
		binCentres = np.arange(0.5, len(posMask)*acBinsPerPos) * acBinSize
		spkTrHist, _ = np.histogram(spkTimes, bins=binCentres)

		# find start, end and length of each block of trues in posMask
		# from itertools import groupby
		# grouped_runs = [(k,sum(1 for i in g)) for k,g in groupby(posMask)]
		# grouped_runs_array = np.array(grouped_runs)
		# chunkLens = grouped_runs_array[:,1]
		# run_start_indices = np.cumsum(grouped_runs_array[:,1]) - grouped_runs_array[:,1]
		# true_mask = grouped_runs_array[:,0] == 1



		# idxArray = np.array([map(itemgetter(0), itemgetter(0, -1)(list(g))) + [k] for k, g in groupby(enumerate(posMask), itemgetter(1))])

		# split the single histogram into individual chunks
		splitIdx = np.nonzero(np.diff(posMask.astype(int)))[0]+1
		splitMask = np.split(posMask, splitIdx)
		splitSpkHist = np.split(spkTrHist, (splitIdx * acBinsPerPos).astype(int))
		histChunks = []
		for i in range(len(splitSpkHist)):
			if np.all(splitMask[i]):
				if np.sum(splitSpkHist[i]) > 2:
					if len(splitSpkHist[i]) > int(acWindowSizeBins)*2:
						histChunks.append(splitSpkHist[i])
		autoCorrGrid = np.zeros((int(acWindowSizeBins) + 1, len(histChunks)))
		chunkLens = []
		from scipy import signal
		print(f"num chunks = {len(histChunks)}")
		for i in range(len(histChunks)):
			lenThisChunk = len(histChunks[i])
			chunkLens.append(lenThisChunk)
			tmp = np.zeros(lenThisChunk * 2)
			tmp[lenThisChunk//2:lenThisChunk//2+lenThisChunk] = histChunks[i]
			tmp2 = signal.fftconvolve(tmp, histChunks[i][::-1], mode='valid') # the autocorrelation itself
			autoCorrGrid[:, i] = tmp2[lenThisChunk//2:lenThisChunk//2+int(acWindowSizeBins)+1] / acBinsPerPos

		totalLen = np.sum(chunkLens)
		autoCorrSum = np.nansum(autoCorrGrid, 1) / totalLen
		#lags = np.arange(0, acWindowSizeBins) * acBinSize
		meanNormdAc = autoCorrSum[1::] - np.nanmean(autoCorrSum[1::])
		# return meanNormdAc
		out = self.power_spectrum(eeg=meanNormdAc, binWidthSecs=acBinSize,
								  maxFreq=maxFreq, pad2pow=16, **kwargs)
		out.update({'meanNormdAc': meanNormdAc})
		if plot:
			fig = plt.gcf()
			ax = fig.gca()
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			ax.imshow(autoCorrGrid,
					 extent=[maxFreq*0.6, maxFreq,
							 np.max(out['Power'])*0.6, ax.get_ylim()[1]])
			ax.set_ylim(ylim)
			ax.set_xlim(xlim)
		return out

	def power_spectrum(self, eeg, plot=True, binWidthSecs=None,
					   maxFreq=25, pad2pow=None, ymax=None, **kwargs):
		"""
		Method used by eeg_power_spectra and intrinsic_freq_autoCorr
		Signal in must be mean normalised already
		"""
		
		# Get raw power spectrum
		nqLim = 1 / binWidthSecs / 2.
		origLen = len(eeg)
		# if pad2pow is None:
		# 	fftLen = int(np.power(2, self._nextpow2(origLen)))
		# else:
		fftLen = int(np.power(2, pad2pow))
		fftHalfLen = int(fftLen / float(2) + 1)

		fftRes = np.fft.fft(eeg, fftLen)
		# get power density from fft and discard second half of spectrum
		_power = np.power(np.abs(fftRes), 2) / origLen
		power = np.delete(_power, np.s_[fftHalfLen::])
		power[1:-2] = power[1:-2] * 2

		# calculate freqs and crop spectrum to requested range
		freqs = nqLim * np.linspace(0, 1, fftHalfLen)
		freqs = freqs[freqs <= maxFreq].T
		power = power[0:len(freqs)]

		# smooth spectrum using gaussian kernel
		binsPerHz = (fftHalfLen - 1) / nqLim
		kernelLen = np.round(self.smthKernelWidth * binsPerHz)
		kernelSig = self.smthKernelSigma * binsPerHz
		from scipy import signal
		k = signal.gaussian(kernelLen, kernelSig) / (kernelLen/2/2)
		power_sm = signal.fftconvolve(power, k[::-1], mode='same')

		# calculate some metrics
		# find max in theta band
		spectrumMaskBand = np.logical_and(freqs > self.thetaRange[0],
										  freqs < self.thetaRange[1])
		bandMaxPower = np.max(power_sm[spectrumMaskBand])
		maxBinInBand = np.argmax(power_sm[spectrumMaskBand])
		bandFreqs = freqs[spectrumMaskBand]
		freqAtBandMaxPower = bandFreqs[maxBinInBand]
		# self.maxBinInBand = maxBinInBand
		# self.freqAtBandMaxPower = freqAtBandMaxPower
		# self.bandMaxPower = bandMaxPower

		# find power in small window around peak and divide by power in rest
		# of spectrum to get snr
		spectrumMaskPeak = np.logical_and(freqs > freqAtBandMaxPower - self.sn2Width / 2,
										  freqs < freqAtBandMaxPower + self.sn2Width / 2)
		s2n = np.nanmean(power_sm[spectrumMaskPeak]) / np.nanmean(power_sm[~spectrumMaskPeak])
		self.freqs = freqs
		self.power_sm = power_sm
		self.spectrumMaskPeak = spectrumMaskPeak
		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			if ymax is None:
				ymax = np.min([2 * np.max(power), np.max(power_sm)])
				if ymax == 0:
					ymax = 1
			ax.plot(freqs, power, c=[0.9, 0.9, 0.9])
			# ax.hold(True)
			ax.plot(freqs, power_sm, 'k', lw=2)
			ax.axvline(self.thetaRange[0], c='b', ls='--')
			ax.axvline(self.thetaRange[1], c='b', ls='--')
			_, stemlines, _ = ax.stem([freqAtBandMaxPower], [bandMaxPower], linefmt='r')
			# plt.setp(stemlines, 'linewidth', 2)
			ax.fill_between(freqs, 0, power_sm, where=spectrumMaskPeak,
							color='r', alpha=0.25, zorder=25)
			# ax.set_ylim(0, ymax)
			# ax.set_xlim(0, self.xmax)
			ax.set_xlabel('Frequency (Hz)')
			ax.set_ylabel('Power density (W/Hz)')
		out_dict = {'maxFreq': freqAtBandMaxPower, 'Power': power_sm,
					'Freqs': freqs, 's2n': s2n, 'Power_raw': power, 'k': k, 'kernelLen': kernelLen,
					'kernelSig': kernelSig, 'binsPerHz': binsPerHz, 'kernelLen': kernelLen}
		return out_dict
