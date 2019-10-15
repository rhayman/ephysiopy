# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:31:31 2012

@author: robin
"""
import numpy as np
import warnings
from scipy import signal		
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from .utils import blur_image

class SpikeCalcs(object):
	"""
	Mix-in class for use with Tetrode class below.
	
	Extends Tetrodes functionality by adding methods for analysis of spikes/
	spike trains
				
	Note lots of the methods here are native to dacq2py.axonaIO.Tetrode
	
	Note that units are in milliseconds
	"""
	
	
	def getNSpikes(self, cluster):
		if cluster not in self.clusters:
			warnings.warn('Cluster not available. Try again!')
		else:
			return np.count_nonzero(self.cut == cluster)

	def trial_av_firing_rate(self, cluster):
		'''
		returns the trial average firing rate of a cluster in Hz
		'''
		return self.getNSpikes(cluster) / float(self.header['duration'])
	
	def mean_autoCorr(self, cluster, n=40):
		'''
		Returns the autocorrelation function mean from 0 to n ms (default=40)
		Used to help classify units as principal or interneuron
		'''
		if cluster not in self.clusters:
			warnings.warn('Cluster not available. Try again!')
		else:
			bins = 201
			Trange = (-500, 500)
			y = self.xcorr(cluster, Trange=Trange)
			counts, bins = np.histogram(y[y != 0], bins=bins, range=Trange)
			mask = np.logical_and(bins>0, bins<n)
			return np.mean(counts[mask])
		
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

	def xcorr(self, x1, x2=None, Trange=None):
		'''
		Returns the histogram of the ISIs

		Parameters
		---------------
		x1 - 1d np.array list of spike times
		x2 - (optional) 1d np.array of spike times
		Trange - 1x2 np.array for range of times to bin up. Defaults
					to [-500, +500]
		'''
		if x2 is None:
			x2 = x1.copy()
		if Trange is None:
			Trange = np.array([-500, 500])
		y = []
		irange = x1[:, np.newaxis] + Trange[np.newaxis, :]
		dts = np.searchsorted(x2, irange)
		for i, t in enumerate(dts):
			y.extend(x2[t[0]:t[1]] - x1[i])
		y = np.array(y, dtype=float)
		return y

	def smoothSpikePosCount(self, x1, npos, sigma=3.0, shuffle=None):
		'''
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
		'''
		spk_hist = np.bincount(x1, minlength=npos)
		if shuffle is not None:
			spk_hist = np.roll(spk_hist, int(shuffle * 50))
		# smooth the spk_hist (which is a temporal histogram) with a 250ms
		# gaussian as with Kropff et al., 2015
		h = signal.gaussian(13, sigma)
		h = h / float(np.sum(h))
		return signal.filtfilt(h.ravel(), 1, spk_hist)
	
	def getMeanWaveform(self, clusterA):
		'''
		Returns the mean waveform and sem for a given spike train
		
		Parameters
		----------
		clusterA: int
			The cluster to get the mean waveform for
			
		Returns
		-------
		mn_wvs: ndarray (floats) - usually 4x50 for tetrode recordings
			the mean waveforms
		std_wvs: ndarray (floats) - usually 4x50 for tetrode recordings
			the standard deviations of the waveforms
		'''
		if clusterA not in self.clusters:
			warnings.warn('Cluster not available. Try again!')
		x = self.getClustSpks(clusterA)
		return np.mean(x, axis=0), np.std(x, axis=0)

	def thetaBandMaxFreq(self, x1):
		'''
		Calculates the frequency with the max power in the theta band (6-12Hz)
		of a spike trains autocorrelogram. Partly to look for differences
		in theta frequency in different running directions a la Blair (Welday paper)
		'''
		y = self.xcorr(x1)
		corr, _ = np.histogram(y[y != 0], bins=201, range=np.array([-500,500]))
		# Take the fft of the spike train autocorr (from -500 to +500ms)
		from scipy.signal import periodogram
		freqs, power = periodogram(corr, fs=200, return_onesided=True)
		power_masked = np.ma.MaskedArray(power,np.logical_or(freqs<6,freqs>12))
		return freqs[np.argmax(power_masked)]

	def thetaModIdx(self, x1):
		'''
		Calculates a theta modulation index of a spike train based on the cells
		autocorrelogram
		
		Parameters
		----------
		x1: np.array
			The spike time-series
		Returns
		-------
		thetaMod: float
			The difference of the values at the first peak and trough of the
			autocorrelogram
		'''
		y = self.xcorr(x1)
		corr, _ = np.histogram(y[y != 0], bins=201, range=np.array([-500,500]))
		# Take the fft of the spike train autocorr (from -500 to +500ms)
		from scipy.signal import periodogram
		freqs, power = periodogram(corr, fs=200, return_onesided=True)
		# Smooth the power over +/- 1Hz
		b = signal.boxcar(3)
		h = signal.filtfilt(b, 3, power)
		
		# Square the amplitude first
		sqd_amp = h ** 2
		# Then find the mean power in the +/-1Hz band either side of that
		theta_band_max_idx = np.nonzero(sqd_amp==np.max(sqd_amp[np.logical_and(freqs>6, freqs<11)]))[0][0]
		mean_theta_band_power = np.mean(sqd_amp[theta_band_max_idx-1:theta_band_max_idx+1])
		# Find the mean amplitude in the 2-50Hz range
		other_band_idx = np.logical_and(freqs>2, freqs<50)
		mean_other_band_power = np.mean(sqd_amp[other_band_idx])
		# Find the ratio of these two - this is the theta modulation index
		return (mean_theta_band_power - mean_other_band_power) / (mean_theta_band_power + mean_other_band_power)

	def thetaModIdxV2(self, x1):
		'''
		This is a simpler alternative to the thetaModIdx method in that it
		calculates the difference between the normalized temporal autocorrelogram
		at the trough between 50-70ms and the peak between 100-140ms over
		their sum (data is binned into 5ms bins)
		
		Measure used in Cacucci et al., 2004 and Kropff et al 2015
		'''
		y = self.xcorr(x1)
		corr, bins = np.histogram(y[y != 0], bins=201, range=np.array([-500,500]))
		# 'close' the right-hand bin
		bins = bins[0:-1]
		# normalise corr so max is 1.0
		corr = corr/float(np.max(corr))
		thetaAntiPhase = np.min(corr[np.logical_and(bins>50,bins<70)])
		thetaPhase = np.max(corr[np.logical_and(bins>100, bins<140)])
		return (thetaPhase-thetaAntiPhase) / (thetaPhase+thetaAntiPhase)

	def clusterQuality(self, cluster, fet=1):
		'''
		returns the L-ratio and Isolation Distance measures
		calculated on the principal components of the energy in a spike matrix
		'''
		nSpikes, nElectrodes, _ = self.waveforms.shape
		wvs = self.waveforms.copy()
		E = np.sqrt(np.nansum(self.waveforms ** 2, axis=2))
		zeroIdx = np.sum(E, 0) == [0, 0, 0, 0]
		E = E[:, ~zeroIdx]
		wvs = wvs[:, ~zeroIdx, :]
		normdWaves = (wvs.T / E.T).T
		PCA_m = self.getParam(normdWaves, 'PCA', fet=fet)
		# get mahalanobis distance
		idx = self.cut == cluster
		nClustSpikes = np.count_nonzero(idx)
		try:
			d = self._mahal(PCA_m,PCA_m[idx,:])
			# get the indices of the spikes not in the cluster
			M_noise = d[~idx]
			df = np.prod((fet, nElectrodes))
			L = np.sum(1 - stats.chi2.cdf(M_noise, df))
			L_ratio = L / nClustSpikes
			# calculate isolation distance
			if nClustSpikes < nSpikes / 2:
				M_noise.sort()
				isolation_dist = M_noise[nClustSpikes]
			else:
				isolation_dist = np.nan
		except:
			isolation_dist = L_ratio = np.nan
		return L_ratio, isolation_dist

	def _mahal(self, u, v):
		'''
		gets the mahalanobis distance between two vectors u and v
		a blatant copy of the Mathworks fcn as it doesn't require the covariance
		matrix to be calculated which is a pain if there are NaNs in the matrix
		'''
		u_sz = u.shape
		v_sz = v.shape
		if u_sz[1] != v_sz[1]:
			warnings.warn('Input size mismatch: matrices must have same number of columns')
		if v_sz[0] < v_sz[1]:
			warnings.warn('Too few rows: v must have more rows than columns')
		if np.any(np.imag(u)) or np.any(np.imag(v)):
			warnings.warn('No complex inputs are allowed')
		m = np.nanmean(v,axis=0)
		M = np.tile(m, reps=(u_sz[0],1))
		C = v - np.tile(m, reps=(v_sz[0],1))
		Q, R = np.linalg.qr(C)
		ri = np.linalg.solve(R.T, (u-M).T)
		d = np.sum(ri * ri,0).T * (v_sz[0]-1)
		return d

	def plotClusterSpace(self, clusters=None, param='Amp', clusts=None, bins=256, **kwargs):
		'''
		TODO: aspect of plot boxes in ImageGrid not right as scaled by range of
		values now
		'''
		import tintColours as tcols
		import matplotlib.colors as colors
		from itertools import combinations
		from mpl_toolkits.axes_grid1 import ImageGrid
		
		if isinstance(clusters, int):
			clusters = [clusters]

		amps = self.getParam(param=param)
		bad_electrodes = np.setdiff1d(np.array(range(4)),np.array(np.sum(amps,0).nonzero())[0])
		cmap = np.tile(tcols.colours[0],(bins,1))
		cmap[0] = (1,1,1)
		cmap = colors.ListedColormap(cmap)
		cmap._init()
		alpha_vals = np.ones(cmap.N+3)
		alpha_vals[0] = 0
		cmap._lut[:,-1] = alpha_vals
		cmb = combinations(range(4),2)
		if 'fig' in kwargs.keys():
			fig = kwargs['fig']
		else:
			fig = plt.figure(figsize=(8,6))
		if 'rect' in kwargs.keys():
			rect = kwargs['rect']
		else:
			rect = 111
		grid = ImageGrid(fig, rect, nrows_ncols= (2,3), axes_pad=0.1, aspect=False)
		if 'Amp' in param:
			myRange = np.vstack((self.scaling*0, self.scaling*2))
		else:
			myRange = None
		clustCMap0 = np.tile(tcols.colours[0],(bins,1))
		clustCMap0[0] = (1,1,1)
		clustCMap0 = colors.ListedColormap(clustCMap0)
		clustCMap0._init()
		clustCMap0._lut[:,-1] = alpha_vals
		for i, c in enumerate(cmb):
			if c not in bad_electrodes:
				h, ye, xe = np.histogram2d(amps[:,c[0]], amps[:,c[1]], range = myRange[:,c].T, bins=bins)
				x, y = np.meshgrid(xe[0:-1], ye[0:-1])
				grid[i].pcolormesh(x, y, h, cmap=clustCMap0, edgecolors='face')
				if clusters is not None:
					for thisclust in clusters:
						clustidx = self.cut == thisclust
						h, ye, xe = np.histogram2d(amps[clustidx,c[0]],amps[clustidx,c[1]], range=myRange[:,c].T, bins=bins)
						clustCMap = np.tile(tcols.colours[thisclust],(bins,1))
						clustCMap[0] = (1,1,1)
						clustCMap = colors.ListedColormap(clustCMap)
						clustCMap._init()
						clustCMap._lut[:,-1] = alpha_vals
						grid[i].pcolormesh(x, y, h, cmap=clustCMap, edgecolors='face')
			s = str(c[0]+1) + ' v ' + str(c[1]+1)
			grid[i].text(0.05,0.95, s, va='top', ha='left', size='small', color='k', transform=grid[i].transAxes)
			grid[i].set_xlim(xe.min(), xe.max())
			grid[i].set_ylim(ye.min(), ye.max())
		plt.setp([a.get_xticklabels() for a in grid], visible=False)
		plt.setp([a.get_yticklabels() for a in grid], visible=False)
		return fig

	def p2t_time(self, cluster):
		"""
		The peak to trough time of a spike in ms
		
		Parameters
		----------
		cluster: int
			the cluster whose waveforms are to be analysed
			
		Returns
		-------
		p2t: float
			The mean peak-to-trough time for the channel (electrode) that has 
			the strongest (highest amplitude) signal. Units are ms
		"""
		waveforms = self.waveforms[self.cut==cluster, :, :]
		best_chan = np.argmax(np.max(np.mean(waveforms, 0), 1))
		tP = self.getParam(waveforms, param='tP')
		tT = self.getParam(waveforms, param='tT')
		mn_tP = np.mean(tP, 0)
		mn_tT = np.mean(tT, 0)
		p2t = np.abs(mn_tP[best_chan] - mn_tT[best_chan])
		return p2t * 1000
		
	def half_amp_dur(self, cluster):
		"""
		Half amplitude duration of a spike
		
		Parameters
		----------
		A: ndarray
			An nSpikes x nElectrodes x nSamples array
			
		Returns
		-------
		had: float
			The half-amplitude duration for the channel (electrode) that has 
			the strongest (highest amplitude) signal. Units are ms
		"""
		from scipy import interpolate, optimize
		
		waveforms = self.waveforms[self.cut==cluster, :, :]
		best_chan = np.argmax(np.max(np.mean(waveforms, 0), 1))
		mn_wvs = np.mean(waveforms, 0)
		wvs = mn_wvs[best_chan, :]
		half_amp = np.max(wvs) / 2
		half_amp = np.zeros_like(wvs) + half_amp
		t = np.linspace(0, 1/1000., 50)
		# create functions from the data using PiecewisePolynomial
		p1 = interpolate.PiecewisePolynomial(t, wvs[:,np.newaxis])
		p2 = interpolate.PiecewisePolynomial(t, half_amp[:,np.newaxis])
		xs = np.r_[t, t]
		xs.sort()
		x_min = xs.min()
		x_max = xs.max()
		x_mid = xs[:-1] + np.diff(xs) / 2
		roots = set()
		for val in x_mid:
			root, infodict, ier, mesg = optimize.fsolve(lambda x: p1(x)-p2(x), val, full_output=True)
			if ier==1 and x_min < root < x_max:
				roots.add(root[0])
		roots = list(roots)
		if len(roots) > 1:
			r = np.abs(np.diff(roots[0:2]))[0]
		else:
			r = np.nan
		return r

	def getParam(self, waveforms=None, param='Amp', t=200, fet=1):
		'''
		Returns the requested parameter from a spike train as a numpy array
		
		Parameters
		-------------------
		
		waveforms - numpy array		
			Shape of array can be an nSpikes x nSamples
			OR
			a nSpikes x nElectrodes x nSamples
		
		param - str
			Valid values are:
				'Amp' - peak-to-trough amplitude (default)
				'P' - height of peak
				'T' - depth of trough
				'Vt' height at time t
				'tP' - time of peak (in seconds)
				'tT' - time of trough (in seconds)
				'PCA' - first n fet principal components (defaults to 1)
				
		t - int
			The time used for Vt
			
		fet - int
			The number of principal components (used with param 'PCA')
		'''
		from scipy import interpolate
		from sklearn.decomposition import PCA
		
		if waveforms is None:
			waveforms = self.waveforms
			
		if param == 'Amp':
			return np.ptp(waveforms, axis=-1)
		elif param == 'P':
			return np.max(waveforms, axis=-1)
		elif param == 'T':
			return np.min(waveforms, axis=-1)
		elif param == 'Vt':
			times = np.arange(0,1000,20)
			f = interpolate.interp1d(times, range(50), 'nearest')
			if waveforms.ndim == 2:
				return waveforms[:, int(f(t))]
			elif waveforms.ndim == 3:
				return waveforms[:, :, int(f(t))]
		elif param == 'tP':
			idx = np.argmax(waveforms, axis=-1)
			m = interpolate.interp1d([0, waveforms.shape[-1]-1], [0, 1/1000.])
			return m(idx)
		elif param == 'tT':
			idx = np.argmin(waveforms, axis=-1)
			m = interpolate.interp1d([0, waveforms.shape[-1]-1], [0, 1/1000.])
			return m(idx)
		elif param == 'PCA':
			pca = PCA(n_components=fet)
			if waveforms.ndim == 2:
				return pca.fit(waveforms).transform(waveforms).squeeze()
			elif waveforms.ndim == 3:
				out = np.zeros((waveforms.shape[0], waveforms.shape[1] * fet))
				st = np.arange(0, waveforms.shape[1] * fet, fet)
				en = np.arange(fet, fet + (waveforms.shape[1] * fet), fet)
				rng = np.vstack((st, en))
				for i in range(waveforms.shape[1]):
					if ~np.any(np.isnan(waveforms[:,i,:])):
						A = np.squeeze(pca.fit(waveforms[:,i,:].squeeze()).transform(waveforms[:,i,:].squeeze()))
						if A.ndim < 2:
							out[:,rng[0,i]:rng[1,i]] = np.atleast_2d(A).T
						else:
							out[:,rng[0,i]:rng[1,i]] = A
				return out
