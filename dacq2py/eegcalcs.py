# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:31:31 2012

@author: robin
"""
import numpy as np
import os
from scipy import signal
from .axonaIO import EEG as EEGIO
from itertools import groupby
from operator import itemgetter
from .statscalcs import StatsCalcs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from scipy.special._ufuncs import gammainc, gamma
from scipy.optimize import fminbound

class EEGCalcs(EEGIO):
	def __init__(self, fname, eegType='eeg', thetaRange=[7,11], pad2pow=np.nan,
				 smthKernelWidth=2, smthKernelSigma=0.1875, sn2Width=2,
				 maxFreq=125, ymax=None, xmax=25):
		if eegType == 'eeg':
			egf = 0;
		elif eegType == 'egf':
			egf = 1;
		self.fname = os.path.basename(fname)
		self.EEG = EEGIO(fname, egf=egf)
		self.posSampFreq = 50
		self.sampsPerPos = int(self.EEG.sample_rate / self.posSampFreq)
		self.sample_rate = self.EEG.sample_rate
		self.eeg = self.EEG.eeg - np.ma.mean(self.EEG.eeg)
		self.thetaRange = [7,11]
		self.pad2pow = pad2pow
		self.smthKernelWidth = smthKernelWidth
		self.smthKernelSigma = smthKernelSigma
		self.sn2Width = sn2Width
		self.maxFreq = maxFreq
		self.ymax = ymax
		self.xmax = xmax

	def intrinsic_freq_autoCorr(self, spkTimes=None, posMask=None, maxFreq=25,
								acBinSize=0.002, acWindow=0.5, plot=True,
								**kwargs):
		"""
		Be careful that if you've called dacq2py.Tetrode.getSpkTS()
		that they are divided by
		96000 to get into seconds before using here
		"""
		acBinsPerPos = 1. / self.posSampFreq / acBinSize
		acWindowSizeBins = np.round(acWindow / acBinSize)
		binCentres = np.arange(0.5, len(posMask)*acBinsPerPos) * acBinSize
		spkTrHist, _ = np.histogram(spkTimes, bins=binCentres)

		# find start, end and length of each block of trues in posMask
		idxArray = np.array([map(itemgetter(0), itemgetter(0, -1)(list(g))) + [k] for k, g in groupby(enumerate(posMask), itemgetter(1))])
		chunkLens = np.diff(idxArray)[:, 0] + 1

		# split the single histogram into individual chunks
		splitIdx = np.nonzero(np.diff(posMask.astype(int)))[0]+1
		splitMask = np.split(posMask, splitIdx)
		splitSpkHist = np.split(spkTrHist, splitIdx * acBinsPerPos)
		histChunks = []
		for i in range(len(splitSpkHist)):
			if np.all(splitMask[i]):
				if np.sum(splitSpkHist[i]) > 2:
					histChunks.append(splitSpkHist[i])
		autoCorrGrid = np.zeros((acWindowSizeBins + 1, len(histChunks)))
		chunkLens = []
		for i in range(len(histChunks)):
			lenThisChunk = len(histChunks[i])
			chunkLens.append(lenThisChunk)
			tmp = np.zeros(lenThisChunk * 2)
			tmp[lenThisChunk/2:lenThisChunk/2+lenThisChunk] = histChunks[i]
			tmp2 = signal.fftconvolve(tmp, histChunks[i][::-1], mode='valid')
			autoCorrGrid[:, i] = tmp2[lenThisChunk/2:lenThisChunk/2+acWindowSizeBins+1] / acBinsPerPos

		totalLen = np.sum(chunkLens)
		autoCorrSum = np.nansum(autoCorrGrid, 1) / totalLen
		#lags = np.arange(0, acWindowSizeBins) * acBinSize
		meanNormdAc = autoCorrSum[1::] - np.nanmean(autoCorrSum[1::])
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

	def wavelet(self, Y, dt, pad=0, dj=-1, s0=-1, J1=-1, mother=-1, param=-1):
		n1 = len(Y)
	
		if s0 == -1:
			s0 = 2 * dt
		if dj == -1:
			dj = 1. / 4.
		if J1 == -1:
			J1 = np.fix((np.log(n1 * dt / s0) / np.log(2)) / dj)
		if mother == -1:
			mother = 'MORLET'
	
		#....construct time series to analyze, pad if necessary
		x = Y - np.mean(Y)
		if pad == 1:
			base2 = np.fix(np.log(n1) / np.log(2) + 0.4999)  # power of 2 nearest to N
			x = np.concatenate((x, np.zeros(2 ** (base2 + 1) - n1)))
	
		n = len(x)
	
		#....construct wavenumber array used in transform [Eqn(5)]
		kplus = np.arange(1, np.fix(n / 2 + 1))
		kplus = (kplus * 2 * np.pi / (n * dt))
		kminus = (-(kplus[0:-1])[::-1])
		k = np.concatenate(([0.], kplus, kminus))
	
		#....compute FFT of the (padded) time series
		f = np.fft.fft(x)  # [Eqn(3)]
	
		#....construct SCALE array & empty PERIOD & WAVE arrays
		j = np.arange(0,J1+1)
		scale = s0 * 2. ** (j * dj)
		wave = np.zeros(shape=(J1 + 1, n), dtype=complex)  # define the wavelet array
	
		# loop through all scales and compute transform
		for a1 in range(0, int(J1+1)):
			daughter, fourier_factor, coi, dofmin = self.wave_bases(mother, k, scale[a1], param)
			wave[a1, :] = np.fft.ifft(f * daughter)  # wavelet transform[Eqn(4)]
	
		period = fourier_factor * scale  #[Table(1)]
		coi = coi * dt * np.concatenate((np.insert(np.arange((n1 + 1) / 2 - 1), [0], [1E-5]),
										 np.insert(np.flipud(np.arange(0, n1 / 2 - 1)), [-1], [1E-5])))  # COI [Sec.3g]
		wave = wave[:, :n1]  # get rid of padding before returning
	
		return wave, period, scale, coi

	def wave_bases(self, mother, k, scale, param):
		n = len(k)
		kplus = np.array(k > 0., dtype=float)
	
		if mother == 'MORLET':  #-----------------------------------  Morlet
	
			if param == -1:
				param = 6.
	
			k0 = np.copy(param)
			expnt = -(scale * k - k0) ** 2 / 2. * kplus
			norm = np.sqrt(scale * k[1]) * (np.pi ** (-0.25)) * np.sqrt(n)  # total energy=N   [Eqn(7)]
			daughter = norm * np.exp(expnt)
			daughter = daughter * kplus  # Heaviside step function
			fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))  # Scale-->Fourier [Sec.3h]
			coi = fourier_factor / np.sqrt(2)  # Cone-of-influence [Sec.3g]
			dofmin = 2  # Degrees of freedom
		elif mother == 'PAUL':  #--------------------------------  Paul
			if param == -1:
				param = 4.
			m = param
			expnt = -scale * k * kplus
			norm = np.sqrt(scale * k[1]) * (2 ** m / np.sqrt(m*np.prod(np.arange(1, (2 * m))))) * np.sqrt(n)
			daughter = norm * ((scale * k) ** m) * np.exp(expnt) * kplus
			fourier_factor = 4 * np.pi / (2 * m + 1)
			coi = fourier_factor * np.sqrt(2)
			dofmin = 2
		elif mother == 'DOG':  #--------------------------------  DOG
			if param == -1:
				param = 2.
			m = param
			expnt = -(scale * k) ** 2 / 2.0
			norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
			daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
			fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
			coi = fourier_factor / np.sqrt(2)
			dofmin = 1
		else:
			print('Mother must be one of MORLET, PAUL, DOG')
	
		return daughter, fourier_factor, coi, dofmin
		
	def wave_signif(self, Y, dt, scale, sigtest=-1, lag1=-1, siglvl=-1, dof=-1, mother=-1, param=-1):
		n1 = len(np.atleast_1d(Y))
		J1 = len(scale) - 1
		s0 = np.min(scale)
		dj = np.log2(scale[1] / scale[0])
	
		if n1 == 1:
			variance = Y
		else:
			variance = np.std(Y) ** 2
	
		if sigtest == -1:
			sigtest = 0
		if lag1 == -1:
			lag1 = 0.0
		if siglvl == -1:
			siglvl = 0.95
		if mother == -1:
			mother = 'MORLET'
	
		# get the appropriate parameters [see Table(2)]
		if mother == 'MORLET':  #----------------------------------  Morlet
			empir = ([2., -1, -1, -1])
			if param == -1:
				param = 6.
				empir[1:] = ([0.776, 2.32, 0.60])
			k0 = param
			fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))  # Scale-->Fourier [Sec.3h]
		elif mother == 'PAUL':
			empir = ([2, -1, -1, -1])
			if param == -1:
				param = 4
				empir[1:] = ([1.132, 1.17, 1.5])
			m = param
			fourier_factor = (4 * np.pi) / (2 * m + 1)
		elif mother == 'DOG':  #-------------------------------------Paul
			empir = ([1., -1, -1, -1])
			if param == -1:
				param = 2.
				empir[1:] = ([3.541, 1.43, 1.4])
			elif param == 6:  #--------------------------------------DOG
				empir[1:] = ([1.966, 1.37, 0.97])
			m = param
			fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
		else:
			print('Mother must be one of MORLET, PAUL, DOG')
	
		period = scale * fourier_factor
		dofmin = empir[0]  # Degrees of freedom with no smoothing
		Cdelta = empir[1]  # reconstruction factor
		gamma_fac = empir[2]  # time-decorrelation factor
		dj0 = empir[3]  # scale-decorrelation factor
	
		freq = dt / period  # normalized frequency
		fft_theor = (1 - lag1 ** 2) / (1 - 2 * lag1 * np.cos(freq * 2 * np.pi) + lag1 ** 2)  # [Eqn(16)]
		fft_theor = variance * fft_theor  # include time-series variance
		signif = fft_theor
		if len(np.atleast_1d(dof)) == 1:
			if dof == -1:
				dof = dofmin
	
		if sigtest == 0:  # no smoothing, DOF=dofmin [Sec.4]
			dof = dofmin
			chisquare = self.chisquare_inv(siglvl, dof) / dof
			signif = fft_theor * chisquare  # [Eqn(18)]
		elif sigtest == 1:  # time-averaged significance
			if len(np.atleast_1d(dof)) == 1:
				dof = np.zeros(J1) + dof
			dof[dof < 1] = 1
			dof = dofmin * np.sqrt(1 + (dof * dt / gamma_fac / scale) ** 2)  # [Eqn(23)]
			dof[dof < dofmin] = dofmin   # minimum DOF is dofmin
			for a1 in range(0, J1 + 1):
				chisquare = self.chisquare_inv(siglvl, dof[a1]) / dof[a1]
				signif[a1] = fft_theor[a1] * chisquare
			print(chisquare)
		elif sigtest == 2:  # time-averaged significance
			if len(dof) != 2:
				print('ERROR: DOF must be set to [S1,S2], the range of scale-averages')
			if Cdelta == -1:
				print('ERROR: Cdelta & dj0 not defined for ' + mother + ' with param = ' + str(param))
	
			s1 = dof[0]
			s2 = dof[1]
			avg =  np.logical_and(scale >= 2, scale < 8)# scales between S1 & S2
			navg = np.sum(np.array(np.logical_and(scale >= 2, scale < 8), dtype=int))
			if navg == 0:
				print('ERROR: No valid scales between ' + str(s1) + ' and ' + str(s2))
			Savg = 1. / np.sum(1. / scale[avg])  # [Eqn(25)]
			Smid = np.exp((np.log(s1) + np.log(s2)) / 2.)  # power-of-two midpoint
			dof = (dofmin * navg * Savg / Smid) * np.sqrt(1 + (navg * dj / dj0) ** 2)  # [Eqn(28)]
			fft_theor = Savg * np.sum(fft_theor[avg] / scale[avg])  # [Eqn(27)]
			chisquare = chisquare_inv(siglvl, dof) / dof
			signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare  # [Eqn(26)]
		else:
			print('ERROR: sigtest must be either 0, 1, or 2')
	
		return signif

	def chisquare_inv(self, P, V):

		if (1 - P) < 1E-4:
			print('P must be < 0.9999')
	
		if P == 0.95 and V == 2:  # this is a no-brainer
			X = 5.9915
			return X
	
		MINN = 0.01  # hopefully this is small enough
		MAXX = 1  # actually starts at 10 (see while loop below)
		X = 1
		TOLERANCE = 1E-4  # this should be accurate enough
	
		while (X + TOLERANCE) >= MAXX:  # should only need to loop thru once
			MAXX = MAXX * 10.
		# this calculates value for X, NORMALIZED by V
			X = fminbound(self.chisquare_solve, MINN, MAXX, args=(P,V), xtol=TOLERANCE )
			MINN = MAXX
	
		X = X * V  # put back in the goofy V factor
	
		return X  # end of code
		
	def chisquare_solve(self, XGUESS,P,V):

		PGUESS = gammainc(V/2, V*XGUESS/2)  # incomplete Gamma function
	
		PDIFF = np.abs(PGUESS - P)            # error in calculated P
	
		TOL = 1E-4
		if PGUESS >= 1-TOL:  # if P is very close to 1 (i.e. a bad guess)
			PDIFF = XGUESS   # then just assign some big number like XGUESS
	
		return PDIFF
		
	def ifftFilter(self, sig, freqs, fs=250):
		'''
		Calculates the dft of signal and filters out the frequencies in
		freqs from the result and reconstructs the original signal using 
		the inverse fft without those frequencies
		'''
		from scipy import signal
		origLen = len(sig)
		nyq = fs / 2.0
		fftRes = np.fft.fft(sig)
		f = nyq * np.linspace(0, 1, len(fftRes)/2)
		f = np.concatenate([f,f-nyq])		
		
		band = 0.0625
		idx = np.zeros([len(freqs), len(f)]).astype(bool)
		
		for i,freq in enumerate(freqs):
			idx[i,:] = np.logical_and(np.abs(f)<freq+band, np.abs(f)>freq-band)
		
		pollutedIdx = np.sum(idx, 0)
		fftRes[pollutedIdx] = np.mean(fftRes)
		
		reconSig = np.fft.ifft(fftRes)
			
		

	def filterForLaser(self, E=None, width=0.125, dip=15.0, stimFreq=6.66):
		'''
		In some of the optogenetic experiments I ran the frequency of laser
		stimulation was at 6.66Hz - this method attempts to filter those
		frequencies out
		'''
		from scipy.signal import kaiserord, firwin, filtfilt        
		nyq = self.sample_rate / 2.
		width = width / nyq
		dip = dip
		N, beta = kaiserord(dip, width)
		print("N: {0}\nbeta: {1}".format(N, beta))
		upper = np.ceil(nyq/stimFreq)
		c = np.arange(stimFreq, upper*stimFreq, stimFreq)
		dt = np.array([-0.125, 0.125])
		cutoff_hz = dt[:, np.newaxis] + c[np.newaxis, :]
		cutoff_hz = cutoff_hz.ravel()
		cutoff_hz = np.append(cutoff_hz, nyq-1)
		cutoff_hz.sort()
		cutoff_hz_nyq = cutoff_hz / nyq
		taps = firwin(N, cutoff_hz_nyq, window=('kaiser', beta))
		if E is None:
			eeg = self.EEG.eeg
		else:
			eeg = E
		fx = filtfilt(taps, [1.0], eeg)
		return fx
		
	def filterWithButter(self, data, low, high, fs, order=5):
		from scipy.signal import butter, filtfilt
		nyq = fs / 2.
		lowcut = low / nyq
		highcut = high / nyq
		b, a = butter(order, [lowcut, highcut], btype='band')
		y = filtfilt(b, a, data)
		return y
								
	def eeg_instant_freq_power(self, eeg=None, plot=True):
		if eeg is None:
			eeg = self.EEG.eeg
		else:
			eeg = eeg
		filtRange = self.thetaRange
		eegSampFreq = self.EEG.sample_rate
		self.EEG.x1=filtRange[0]
		self.EEG.x2=filtRange[1]
		filteredEEG = self.eegfilter(E=eeg)
		analyticEEG = signal.hilbert(filteredEEG)
		phaseEEGWrpd = np.angle(analyticEEG)
		phaseEEG = np.unwrap(phaseEEGWrpd)
		ampEEG = np.abs(analyticEEG)
		# calculate instantaneous freq
		freqEEG = np.diff(phaseEEG) * eegSampFreq / (2 * np.pi)
		freqEEGposHz = np.nanmean(signal.decimate(freqEEG, self.sampsPerPos))
		ampEEGposHz = np.nanmean(signal.decimate(ampEEG, self.sampsPerPos))
		phaseEEGposHz = np.nanmean(signal.decimate(phaseEEG, self.sampsPerPos))
		out_dict = {'freqEEGposHz': freqEEGposHz, 'ampEEGposHz': ampEEGposHz,
					'phaseEEGposHz': phaseEEGposHz, 'filteredEEG': filteredEEG,
					'phaseEEG': phaseEEG, 'ampEEG': ampEEG,
					'phaseEEGWrpd': phaseEEGWrpd}
		if plot:
			times = np.linspace(0, len(eeg) / eegSampFreq, len(eeg))
			ys = (eeg, filteredEEG, np.real(analyticEEG), ampEEG, -ampEEG)
			fig = plt.figure(figsize=(20,5))
			ax = fig.add_subplot(111)
			ax.set_ylim(np.amin(ys), np.amax(ys))
			ax.set_xlim(np.amin(times), np.amax(times))
			line_segs = LineCollection([list(zip(times,y)) for y in ys],
									   colors=[[0.8,0.8,0.8,1],
												   [0.5,0.5,0.5,1],
													[1,0,0,1], [0,0,1,1],
													[0,0,1,1]],
										   linestyles='solid')
			line_segs.set_array(times)
			ax.add_collection(line_segs)
			ax.set_xlabel('Time(s)')
			ax.set_ylabel('EEG(V)')
			scale = 1
			ax.plot(times, filteredEEG*scale, color=[0.5, 0.5, 0.5], rasterized=True)
			ax.plot(times, np.real(analyticEEG)*scale, color='r', rasterized=True)
			ax.plot(times, ampEEG*scale, color='b', rasterized=True)
			ax.plot(times, -ampEEG*scale, color='b', rasterized=True)
			s = np.nanmedian(ampEEG) / 50. * scale
			phaseInd = np.round(np.mod(phaseEEG, 2*np.pi) / (2 * np.pi) * 100)
			phaseInd[phaseInd==0] = 100
			S = cm.ScalarMappable()
			phaseImg = S.to_rgba(phaseInd)
			phaseImg = phaseImg[:, 0:3]
			ax.imshow(phaseImg, extent=[0, times[-1], -s, s])
			ax.set_aspect(2e8)
			plt.show()
		return out_dict

	def eeg_power_spectra(self, eeg=None, pos2use='all', **kwargs):
		if eeg is None:
			eeg = self.EEG.eeg
		else:
			eeg = eeg
		if isinstance(pos2use, str):
			if 'all' in pos2use:
				eeg = eeg
		elif isinstance(pos2use, np.ndarray):
			# TODO: need to check this doesn't exceed pos len
			eeg = signal.decimate(eeg, self.sampsPerPos)
			eeg = eeg[pos2use]
		out = self.power_spectrum(eeg=eeg, binWidthSecs=1/self.sample_rate,
								  **kwargs)
		return out

	def plv(self, eeg=None, forder=2, thetaband=[4, 8], gammaband=[30, 80],
			plot=True, **kwargs):
		"""
		Computes the phase-amplitude coupling (PAC) of nested oscillations. More
		specifically this is the phase-locking value (PLV) between two nested
		oscillations in EEG data, in this case theta (default between 4-8Hz) 
		and gamma (defaults to 30-80Hz). A PLV of unity indicates perfect phase
		locking (here PAC) and a value of zero indicates no locking (no PAC)
		
		Parameters
		----------
		eeg: numpy array
			the eeg data itself. This is a 1-d array which can be masked or not
		forder: int
			the order of the filter(s) applied to the eeg data
		thetaband/ gammaband: list/ array
			the range of values to bandpass filter for for the theta and gamma
			ranges
		plot: bool (default True)
			whether to plot the resulting binned up polar plot which shows the 
			amplitude of the gamma oscillation found at different phases of the 
			theta oscillation
		Returns
		-------
		plv: float
			the value of the phase-amplitude coupling
		
		"""
		if eeg is None:
			eeg = self.eeg
		eeg = eeg - np.ma.mean(eeg)
		if np.ma.is_masked(eeg):
			eeg = np.ma.compressed(eeg)
		
		lowfreq, highfreq, lowphase, highamp, highamp_f = self._getFreqPhase(eeg, forder, thetaband, gammaband)

		highampphase = np.angle(signal.hilbert(highamp_f))
		phasedf = highampphase - lowphase
		phasedf = np.exp(1j * phasedf)
		phasedf = np.angle(phasedf)
		calcs = StatsCalcs()
		plv = calcs.circ_r(phasedf)
		th = np.linspace(0.0, 2*np.pi, 20, endpoint=False)
		h, xe = np.histogram(phasedf, bins=20)
		h = h / float(len(phasedf))
		
		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(111, polar=True)
			w = np.pi / 10
			ax.bar(th, h, width = w, bottom=0.0)
		return plv, th, h
		
	def modulationindex(self, eeg=None, nbins=20, forder=2,
						thetaband=[4, 8], gammaband=[30, 80],
						plot=True, **kwargs):
		if eeg is None:
			eeg = self.eeg
		eeg = eeg - np.ma.mean(eeg)
		if np.ma.is_masked(eeg):
			eeg = np.ma.compressed(eeg)
		_, _, lowphase, highamp, _= self._getFreqPhase(eeg, forder, thetaband, gammaband)
		inc = 2*np.pi/nbins
		a = np.arange(-np.pi+inc/2, np.pi, inc)
		dt = np.array([-inc/2, inc/2])
		pbins = a[:, np.newaxis] + dt[np.newaxis, :]
		amp = np.zeros((nbins))
		phaselen = np.arange(len(lowphase))
		for i in xrange(nbins):
			pts = np.nonzero((lowphase >= pbins[i,0]) * (lowphase < pbins[i,1]) * phaselen)
			amp[i] = np.mean(highamp[pts])
		amp = amp / np.sum(amp)
		calcs = StatsCalcs()
		mi = calcs.circ_r(pbins[:,1], amp)
		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(111, polar=True)
			w = np.pi / (nbins/2)
			ax.bar(pbins[:,1], amp, width=w)
			ax.set_title("Modulation index={0:.5f}".format(mi))
			fig.canvas.set_window_title(self.fname)
		return mi
		
	def _getFreqPhase(self, eeg, ford, lowband, highband):
		lowband = np.array(lowband, dtype=float)
		highband = np.array(highband, dtype=float)
		b, a = signal.butter(ford, lowband / (self.sample_rate / 2), btype='band')
		e, f = signal.butter(ford, highband / (self.sample_rate / 2), btype='band')
		lowfreq = signal.filtfilt(b, a, eeg, padtype='odd')
		highfreq = signal.filtfilt(e, f, eeg, padtype='odd')
		lowphase = np.angle(signal.hilbert(lowfreq))
		highamp = np.abs(signal.hilbert(highfreq))
		highamp_f = signal.filtfilt(b, a, highamp, padtype='odd')
		return lowfreq, highfreq, lowphase, highamp, highamp_f
		
	def power_spectrum(self, eeg=None, plot=True, binWidthSecs=None,
					   maxFreq=None, pad2pow=None, ymax=None, **kwargs):
		"""
		Method used by eeg_power_spectra and intrinsic_freq_autoCorr
		Signal in must be mean normalised already
		"""
		if eeg is None:
			eeg = self.eeg
		else:
			eeg = eeg
		if maxFreq is None:
			maxFreq = self.maxFreq
		else:
			maxFreq = maxFreq
		# Get raw power spectrum
		nqLim = self.EEG.sample_rate / 2.
		origLen = len(eeg)
		if pad2pow is None:
			fftLen = int(np.power(2, self._nextpow2(origLen)))
		else:
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
		self.maxBinInBand = maxBinInBand
		self.freqAtBandMaxPower = freqAtBandMaxPower
		self.bandMaxPower = bandMaxPower

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
			ax.hold(True)
			ax.plot(freqs, power_sm, 'k', lw=2)
			ax.axvline(self.thetaRange[0], c='b', ls='--')
			ax.axvline(self.thetaRange[1], c='b', ls='--')
			ax.stem([freqAtBandMaxPower], [bandMaxPower], c='r', lw=2)
			ax.fill_between(freqs, 0, power_sm, where=spectrumMaskPeak,
							color='r', alpha=0.25, zorder=25)
			ax.set_ylim(0, ymax)
			ax.set_xlim(0, self.xmax)
			ax.set_xlabel('Frequency (Hz)')
			ax.set_ylabel('Power density (W/Hz)')
		out_dict = {'maxFreq': freqAtBandMaxPower, 'Power': power_sm,
					'Freqs': freqs, 's2n': s2n, 'Power_raw': power, 'k': k, 'kernelLen': kernelLen,
					'kernelSig': kernelSig, 'binsPerHz': binsPerHz, 'kernelLen': kernelLen}
		return out_dict

	def _nextpow2(self, val):
		'''calculates the next power of 2 that will hold val'''
		val = val - 1
		val = (val >> 1) | val
		val = (val >> 2) | val
		val = (val >> 4) | val
		val = (val >> 8) | val
		val = (val >> 16) | val
		val = (val >> 32) | val
		return np.log2(val + 1)
