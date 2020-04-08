"""
LFP-type analysis limmited at the moment to Axona file formats I think
"""
import numpy as np
import os
from scipy import signal, ndimage, stats, optimize
from ephysiopy.dacq2py.axonaIO import EEG as EEGIO
from itertools import groupby
from operator import itemgetter
from ephysiopy.common.statscalcs import StatsCalcs
from ephysiopy.common.utils import bwperim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from scipy.special._ufuncs import gammainc, gamma
from scipy.optimize import fminbound

import skimage
import matplotlib
from ephysiopy.dacq2py import axonaIO
from ephysiopy.dacq2py import dacq2py_util
from sklearn.utils import resample

class EEGCalcs(EEGIO):
	"""
	Has some useful methods in particularly to do with theta-gamma phase coupling
	"""
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
		"""
		Calculates the dft of signal and filters out the frequencies in
		freqs from the result and reconstructs the original signal using 
		the inverse fft without those frequencies
		"""
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
		"""
		In some of the optogenetic experiments I ran the frequency of laser
		stimulation was at 6.66Hz - this method attempts to filter those
		frequencies out
		"""
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
		
		_, _, lowphase, _, highamp_f = self._getFreqPhase(eeg, forder, thetaband, gammaband)

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
		for i in range(nbins):
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
		"""calculates the next power of 2 that will hold val"""
		val = val - 1
		val = (val >> 1) | val
		val = (val >> 2) | val
		val = (val >> 4) | val
		val = (val >> 8) | val
		val = (val >> 16) | val
		val = (val >> 32) | val
		return np.log2(val + 1)

"""
Mostly a total rip-off of code written by Ali Jeewajee for his paper on
2D phase precession in place and grid cells [1]_

.. [1] Jeewajee A, Barry C, Douchamps V, Manson D, Lever C, Burgess N. Theta phase
	precession of grid and place cell firing in open environments. Philos Trans R Soc
	Lond B Biol Sci. 2013 Dec 23;369(1635):20120532. doi: 10.1098/rstb.2012.0532.
"""

class phasePrecession():
	"""Performs phase precession analysis for single unit data
	
	Parameters
	----------
	filename_root : str
		The absolute filename with no suffix
	psr : float, optional
		The sample rate for position data. Default 50.0
	esr : float, optional
		The sample rate for eeg data. Default 250.0
	binsizes : array_like, optional
		The binsizes for the rate maps. Default numpy.array([0.5,0.5])
	smthKernLen : int, optional
		Kernel length for gaussian field smoothing. Default 50
	smthKernSig : int, optional
		Kernel sigma for gaussian field smoothing. Default 5
	fieldThresh : float, optional
		Fractional limit of field peak rate to restrict field size. Default 0.35
	areaThresh : float, optional
		Fractional limit for reducing fields at environment edge. Default numpy.nan
	binsPerCm : int, optional
		The number of bins per cm. Default 2
	allowedminSpkPhase : float, optional
		Defines the start / end of theta cycles. Default numpy.pi
	mnPowPrctThresh : int, optional
		Percentile power below which theta cycles are rejected. Default 0
	allowedThetaLen : array_like, optional
		Bandwidth of theta in bins. Default [20,42]
	spdSmoothWindow : int, optional
		Kernel length for boxcar smoothing of speed. Default 15
	runMinSpd : float, optional
		Minimum allowed running speed in cm/s. Default 2.5
	runMinDuration : int, optional
		Minimum allowed run duration in seconds. Default 2
	runSmoothWindowFrac : float, optional
		Instantaneous firing rate smoothing constant. Default 1/3
	spatialLPCutOff : int, optional
		Spatial low-pass cutoff frequency. Default 3
	ifr_kernelLen : int, optional
		Instantaneous firing rate smoothing kernel length. Default 1
	ifr_kernelSig : float, optional	
		Instantaneous firing rate smoothing kernel sigma. Default 0.5
	binsPerSec : int, optional
		Bins per second for instantaneous firing rate smoothing. Default 50
	"""

	def __init__(self, filename_root, psr=50.0, esr=250.0, binsizes=np.array([0.5, 0.5]),
				 smthKernLen=50, smthKernSig=5, fieldThresh=0.35,
				 areaThresh=np.nan, binsPerCm=2,
				 allowedminSpkPhase=np.pi, mnPowPrctThresh=0,
				 allowedThetaLen=[20,42], spdSmoothWindow=15, runMinSpd=2.5,
				 runMinDuration=2, runSmoothWindowFrac=1/3., spatialLPCutOff=3,
				 ifr_kernLen=1, ifr_kernSig=0.5, binsPerSec=50):
		self.filename_root = filename_root # absolute filename with no suffix
		self.psr = psr # pos sample rate
		self.esr = esr # eeg sample rate
		self.binsizes = binsizes # binsizes for pos maps
		self.smthKernLen = smthKernLen # kernel length for field smoothing (gaussian)
		self.smthKernSig = smthKernSig # kernel sigma for field smoothing (gaussian)
		self.fieldThresh = fieldThresh # fractional limit of field peak rate to restrict field size
		self.areaThresh = areaThresh # fractional limit for reducing fields at environment edge
		self.binsPerCm = binsPerCm # num bins per cm
		self.allowedminSpkPhase = allowedminSpkPhase # defines start/ end of theta cycle
		self.mnPowPrctThresh = mnPowPrctThresh # percentile power below which theta cycles are rejected
		self.allowedThetaLen = allowedThetaLen # bandwidth of theta in bins
		self.spdSmoothWindow = spdSmoothWindow # kernel length for smoothing speed (boxcar)
		self.runMinSpd = runMinSpd # min allowed running speed
		self.runMinDuration = runMinDuration # min allowed run duration
		self.runSmoothWindowFrac = runSmoothWindowFrac # ifr smoothing constant
		self.spatialLPCutOff = spatialLPCutOff # spatial low-pass cutoff freq
		self.ifr_kernLen = ifr_kernLen # instantaneous firing rate (ifr) smoothing kernel length
		self.ifr_kernSig = ifr_kernSig # ifr kernel sigma
		self.binsPerSec = binsPerSec # bins per second for ifr smoothing

		# get the EEG data
		T = dacq2py_util.Trial(filename_root, cm=True)
		T.ppm = 430
		self.Trial = T
		# add some eeg and pos info - this obv won't change within a trial
		self.EEG = None
		self.eeg = T.EEG.eeg
		T.EEG.thetaAmpPhase()
		self.filteredEEG = T.EEG.eegfilter()
		self.phase = T.EEG.EEGphase
		self.phaseAdj = None
		self.xy = T.POS.xy
		self.xydir = T.POS.dir
		self.spd = T.POS.speed
		# add a bit of spiking info - this will change within a trial given the tetrode/ cluster changing
		self.tetrode = None
		self.cluster = None
		self.spikeTS = None

		# create a dict of regressors and a dict to hold the stats values
		self.regressors = {'spk_numWithinRun': {'values': None, 'pha': None, 'slope': None, 'intercept': None, 'cor': None, 'p': None,
					   'cor_boot': None, 'p_shuffled': None, 'ci': None, 'reg': None}, # updated in getSpikeProps
						  'pos_exptdRate_cum': {'values': None, 'pha': None, 'slope': None, 'intercept': None, 'cor': None, 'p': None,
					   'cor_boot': None, 'p_shuffled': None, 'ci': None, 'reg': None}, # updated in getPosProps
						  'pos_instFR': {'values': None, 'pha': None, 'slope': None, 'intercept': None, 'cor': None, 'p': None,
					   'cor_boot': None, 'p_shuffled': None, 'ci': None, 'reg': None}, # updated in getPosProps
						  'pos_timeInRun': {'values': None, 'pha': None, 'slope': None, 'intercept': None, 'cor': None, 'p': None,
					   'cor_boot': None, 'p_shuffled': None, 'ci': None, 'reg': None}, # updated in getPosProps
						  'pos_d_cum': {'values': None, 'pha': None, 'slope': None, 'intercept': None, 'cor': None, 'p': None,
					   'cor_boot': None, 'p_shuffled': None, 'ci': None, 'reg': None}, # updated in getPosProps
						  'pos_d_meanDir': {'values': None, 'pha': None, 'slope': None, 'intercept': None, 'cor': None, 'p': None,
					   'cor_boot': None, 'p_shuffled': None, 'ci': None, 'reg': None}, # updated in getPosProps
						  'pos_d_currentdir': {'values': None, 'pha': None, 'slope': None, 'intercept': None, 'cor': None, 'p': None,
					   'cor_boot': None, 'p_shuffled': None, 'ci': None, 'reg': None}, # updated in getPosProps
						  'spk_thetaBatchLabelInRun': {'values': None, 'pha': None, 'slope': None, 'intercept': None, 'cor': None, 'p': None,
					   'cor_boot': None, 'p_shuffled': None, 'ci': None, 'reg': None}} # updated in getSpikeProps}

		self.k = 1000
		self.alpha = 0.05
		self.hyp = 0
		self.conf = True

	def performRegression(self, tetrode, cluster, laserEvents=None, **kwargs):
		"""Wrapper function for doing the actual regression which has multiple stages.
		
		Specifically here we parition fields into sub-fields, get a bunch of information
		about the position, spiking and theta data and then do the actual regresssion.
		
		Parameters
		----------
		tetrode, cluster : int
			The tetrode, cluster to examine
		laserEvents : array_like, optional
			The on times for laser events if present, by default None

		See Also
		--------
		ephysiopy.common.eegcalcs.phasePrecession.partitionFields()
		ephysiopy.common.eegcalcs.phasePrecession.getPosProps()
		ephysiopy.common.eegcalcs.phasePrecession.getThetaProps()
		ephysiopy.common.eegcalcs.phasePrecession.getSpikeProps()
		ephysiopy.common.eegcalcs.phasePrecession._ppRegress()
		"""
		if 'binsizes' in kwargs.keys():
			self.binsizes = kwargs.pop('binsizes')
		peaksXY, peaksRate, labels, rmap = self.partitionFields(tetrode, cluster, plot=True)
		posD, runD = self.getPosProps(tetrode, cluster, labels, peaksXY, laserEvents=laserEvents, plot=True)
		self.getThetaProps(tetrode, cluster)
		spkD = self.getSpikeProps(tetrode, cluster, posD['runLabel'], runD['meanDir'],
								  runD['runDurationInPosBins'])
		self._ppRegress(spkD, plot=True)

	def partitionFields(self, tetrode, cluster, ftype='g', plot=False, **kwargs):
		"""
		Partitions fileds.

		Partitions spikes into fields by finding the watersheds around the
		peaks of a super-smoothed ratemap

		Parameters
		----------
		tetrode, cluster : int
			The tetrode / cluster to examine
		ftype : str
			'p' or 'g' denoting place or grid cells - not implemented yet
		plot : boolean
			Whether to produce a debugging plot or not

		Returns
		-------
		peaksXY : array_like
			The xy coordinates of the peak rates in each field
		peaksRate : array_like
			The peak rates in peaksXY
		labels : numpy.ndarray
			An array of the labels corresponding to each field (indices start at 1)
		rmap : numpy.ndarray
			The ratemap of the tetrode / cluster
		"""

		if np.logical_or(tetrode != self.tetrode, cluster != self.cluster):
			spikeTS = self.Trial.TETRODE[tetrode].getClustTS(cluster) / 96000.
			self.spikeTS = spikeTS
			self.tetrode = tetrode
			self.cluster = cluster
		else:
			spikeTS = self.spikeTS
			self.tetrode = tetrode
			self.cluster = cluster
		xy = self.xy
		rmap, _ = self.Trial._getMap(tetrode, cluster, smooth_sz=self.smthKernSig, gaussian=True)
		rmap[np.isnan(rmap)] = 0
		# start image processing:
		# get some markers
		# NB I've tried a variety of techniques to optimise this part and the
		# best seems to be the local adaptive thresholding technique which)
		# smooths locally with a gaussian - see the skimage docs for more
		markers = skimage.filters.threshold_adaptive(rmap, 3, 'gaussian', mode='mirror', param=self.smthKernSig)
		# label these markers so each blob has a unique id
		labels = ndimage.label(markers)[0]
		# labels is now a labelled int array from 0 to however many fields have
		# been detected
		# get the number of spikes in each field - NB this is done against a
		# flattened array so we need to figure out which count corresponds to
		# which particular field id using np.unique
		fieldId, _ = np.unique(labels, return_index=True)
		fieldId = fieldId[1::]
		# TODO: come back to this as may need to know field id ordering
		peakCoords = np.array(ndimage.measurements.maximum_position(rmap, labels=labels, index=fieldId)).astype(int)
		peaksInBinUnits = peakCoords - 0.5
		peaksXY = peaksInBinUnits * self.binsizes
		peaksXY = peaksXY + np.min(xy, 1)
		# find the peak rate at each of the centre of the detected fields to
		# subsequently threshold the field at some fraction of the peak value
		peakRates = rmap[peakCoords[:, 0], peakCoords[:, 1]]
		fieldThresh = peakRates * self.fieldThresh
		rmFieldMask = np.zeros_like(rmap)
		for fid in fieldId:
			f = labels[peakCoords[fid-1, 0], peakCoords[fid-1, 1]]
			rmFieldMask[labels==f] = rmap[labels==f] > fieldThresh[f-1]
		labels[~rmFieldMask.astype(bool)] = 0
		peakBinInds = np.ceil(peaksInBinUnits)
		# re-order some vars to get into same format as fieldLabels
		peakLabels = labels[peakCoords[:, 0], peakCoords[:, 1]]
		peaksXY = peaksXY[peakLabels-1, :]
		peaksRate = peakRates[peakLabels-1]
		peakBinInds = peakBinInds[peakLabels-1, :]
		peaksInBinUnits = peaksInBinUnits[peakLabels-1, :]
		peaksXY = peakCoords - np.min(xy, 1)

		if ~np.isnan(self.areaThresh):
			#TODO: this needs fixing so sensible values are used and that the
			# modified bool array is propagated to the relevant arrays ie makes
			# sense to have a function that applies a bool array to whatever
			# arrays are used as output and call it in a couple of places
			# areaInBins = self.areaThresh * self.binsPerCm
			lb = ndimage.label(markers)[0]
			rp = skimage.measure.regionprops(lb)
			for reg in rp:
				print(reg.filled_area)
			markers = skimage.morphology.remove_small_objects(lb, min_size=4000, connectivity=4, in_place=True)
		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(211)
			ax.imshow(rmap, interpolation='nearest', origin='lower')
			ax.set_title('Smoothed ratemap + peaks')
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			ax.set_aspect('equal')
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()
			ax.plot(peakCoords[:, 1], peakCoords[:, 0], 'ko')
			ax.set_ylim(ylim)
			ax.set_xlim(xlim)

			ax = fig.add_subplot(212)
			ax.imshow(labels, interpolation='nearest', origin='lower')
			ax.set_title('Labelled restricted fields')
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			ax.set_aspect('equal')

		return peaksXY, peaksRate, labels, rmap

	def getPosProps(self, tetrode, cluster, labels, peaksXY,
					laserEvents=None, plot=False, **kwargs):
		"""
		Uses the output of partitionFields and returns vectors the same
		length as pos.

		Parameters
		----------
		tetrode, cluster : int
			The tetrode / cluster to examine
		peaksXY : array_like
			The x-y coords of the peaks in the ratemap
		laserEvents : array_like
			The position indices of on events (laser on)

		Returns
		-------
		pos_dict, run_dict : dict
			Contains a whole bunch of information for the whole trial (pos_dict) and
			also on a run-by-run basis (run_dict). See the end of this function for all
			the key / value pairs.
		"""

		if self.spikeTS is None:
			spikeTS = self.Trial.TETRODE[tetrode].getClustTS(cluster) / 96000.
			self.spikeTS = spikeTS
		else:
			spikeTS = self.spikeTS
		xy = self.xy
		xydir = self.xydir
		spd = self.spd

		if xydir.ndim == 2:
			xydir = np.squeeze(xydir)
		if spd.ndim == 2:
			spd = np.squeeze(spd)
		spkPosInd = np.ceil(spikeTS * self.psr).astype(int) - 1
		spkPosInd[spkPosInd > len(xy.T)] = len(xy.T) - 1
		nPos = xy.shape[1]
		xy_old = xy.copy()
		xydir = np.squeeze(xydir)
		xydir_old = xydir.copy()
		
		xy_nrmd = xy - np.min(xy, 1)[:, np.newaxis]
		rmap, (xe, ye) = self.Trial._getMap(tetrode, cluster, smooth_sz=self.smthKernSig, gaussian=True)
		xe = xe - np.min(xy, 1)[1]
		ye = ye - np.min(xy, 1)[0]
		
		rmap[np.isnan(rmap)] = 0
		xBins = np.digitize(xy_nrmd[0], ye[:-1])
		yBins = np.digitize(xy_nrmd[1], xe[:-1])
		fieldLabel = labels[yBins-1, xBins-1]

		fieldPerimMask = bwperim(labels)
		fieldPerimYBins, fieldPerimXBins = np.nonzero(fieldPerimMask)
		fieldPerimX = ye[fieldPerimXBins]
		fieldPerimY = xe[fieldPerimYBins]
		fieldPerimXY = np.vstack((fieldPerimX, fieldPerimY))
		peaksXYBins = np.array(ndimage.measurements.maximum_position(rmap, labels=labels, index=np.unique(labels)[1::])).astype(int)
		peakY = xe[peaksXYBins[:, 0]]
		peakX = ye[peaksXYBins[:, 1]]
		peaksXY = np.vstack((peakX, peakY)).T

		posRUnsmthd = np.zeros((nPos)) * np.nan
		posAngleFromPeak = np.zeros_like(posRUnsmthd) * np.nan
		perimAngleFromPeak = np.zeros((fieldPerimXY.shape[1])) * np.nan
		for i, peak in enumerate(peaksXY):
			i = i + 1
			# grab each fields perim coords and the pos samples within it
			thisFieldPerim = fieldPerimXY[:, labels[fieldPerimMask]==i]
			if thisFieldPerim.any():
				this_xy = xy_nrmd[:, fieldLabel == i]
				# calculate angle from the field peak for each point on the perim
				# and each pos sample that lies within the field
				thisPerimAngle = np.arctan2(thisFieldPerim[1, :]-peak[1], thisFieldPerim[0, :]-peak[0])
				thisPosAngle = np.arctan2(this_xy[1, :]-peak[1], this_xy[0, :]-peak[0])
				posAngleFromPeak[fieldLabel==i] = thisPosAngle
				perimAngleFromPeak[labels[fieldPerimMask]==i] = thisPerimAngle
				#for each pos sample calculate which point on the perim is most
				# colinear with the field centre - see _circ_abs for more
				thisAngleDf = self._circ_abs(thisPerimAngle[:, np.newaxis] - thisPosAngle[np.newaxis, :])
				thisPerimInd = np.argmin(thisAngleDf, 0)
				# calculate the distance to the peak from pos and the min perim
				# point and calculate the ratio (r - see OUtputs for method)
				tmp = this_xy.T - peak.T
				thisDistFromPos2Peak = np.hypot(tmp[:, 0], tmp[:, 1])
				tmp = thisFieldPerim[:, thisPerimInd].T - peak.T
				thisDistFromPerim2Peak = np.hypot(tmp[:, 0], tmp[:, 1])
				posRUnsmthd[fieldLabel==i] = thisDistFromPos2Peak / thisDistFromPerim2Peak
		# the skimage find_boundaries method combined with the labelled mask
		# strive to make some of the values in thisDistFromPos2Peak larger than
		# those in thisDistFromPerim2Peak which means that some of the vals in
		# posRUnsmthd larger than 1 which means the values in xy_new later are
		# wrong - so lets cap any value > 1 to 1. The same cap is applied later
		# to rho when calculating the angular values. Print out a warning
		# message letting the user know how many values have been capped
		print('\n\n{:.2%} posRUnsmthd values have been capped to 1\n\n'.format(np.sum(posRUnsmthd>=1)/ posRUnsmthd.size))
		posRUnsmthd[posRUnsmthd>=1] = 1
		# label non-zero contiguous runs with a unique id
		runLabel = self._labelContigNonZeroRuns(fieldLabel)
		isRun = runLabel > 0
		runStartIdx = self._getLabelStarts(runLabel)
		runEndIdx = self._getLabelEnds(runLabel)
		# find runs that are too short, have low speed or too few spikes
		runsSansSpikes = np.ones(len(runStartIdx), dtype=bool)
		spkRunLabels = runLabel[spkPosInd] - 1
		runsSansSpikes[spkRunLabels[spkRunLabels>0]] = False
		k = signal.boxcar(self.spdSmoothWindow) / float(self.spdSmoothWindow)
		spdSmthd = signal.convolve(np.squeeze(spd), k, mode='same')
		runDurationInPosBins = runEndIdx - runStartIdx + 1
		runsMinSpeed = []
		runId = np.unique(runLabel)[1::]
		for run in runId:
			runsMinSpeed.append(np.min(spdSmthd[runLabel==run]))
		runsMinSpeed = np.array(runsMinSpeed)
		badRuns = np.logical_or(np.logical_or(runsMinSpeed < self.runMinSpd,
											  runDurationInPosBins < self.runMinDuration),
											  runsSansSpikes)
		badRuns = np.squeeze(badRuns)
		runLabel = self._applyFilter2Labels(~badRuns, runLabel)
		runStartIdx = runStartIdx[~badRuns]
		runEndIdx = runEndIdx[~badRuns]# + 1
		runsMinSpeed = runsMinSpeed[~badRuns]
		runDurationInPosBins = runDurationInPosBins[~badRuns]
		isRun = runLabel > 0

		# calculate mean and std direction for each run
		runComplexMnDir = np.squeeze(np.zeros_like(runStartIdx,dtype=np.complex))
		np.add.at(runComplexMnDir, runLabel[isRun]-1, np.exp(1j * (xydir[isRun] * (np.pi/180))))
		meanDir = np.angle(runComplexMnDir) #circ mean
		tortuosity = 1 - np.abs(runComplexMnDir) / runDurationInPosBins

		# caculate angular distance between the runs main direction and the
		# pos's direction to the peak centre
		posPhiUnSmthd = np.ones_like(fieldLabel) * np.nan
		posPhiUnSmthd[isRun] = posAngleFromPeak[isRun] - meanDir[runLabel[isRun]-1]

		#smooth r and phi in cartesian space
		# convert to cartesian coords first
		posXUnSmthd, posYUnSmthd = self._pol2cart(posRUnsmthd, posPhiUnSmthd)
		posXYUnSmthd = np.vstack((posXUnSmthd, posYUnSmthd))

		# filter each run with filter of appropriate length
		filtLen = np.squeeze(np.floor((runEndIdx - runStartIdx + 1) * self.runSmoothWindowFrac))
		xy_new = np.zeros_like(xy_old) * np.nan
		for i in range(len(runStartIdx)):
			if filtLen[i] > 2:
				filt = signal.firwin(filtLen[i] - 1, cutoff=self.spatialLPCutOff / self.psr*2,
									 window='blackman')
				xy_new[:, runStartIdx[i]:runEndIdx[i]] = signal.filtfilt(filt, [1], posXYUnSmthd[:, runStartIdx[i]:runEndIdx[i]], axis=1)

		r, phi = self._cart2pol(xy_new[0], xy_new[1])
		r[r>1] = 1

		# calculate the direction of the smoothed data
		xydir_new = np.arctan2(np.diff(xy_new[1]), np.diff(xy_new[0]))
		xydir_new = np.append(xydir_new, xydir_new[-1])
		xydir_new[runEndIdx] = xydir_new[runEndIdx-1]

		# project the distance value onto the current direction
		d_currentdir = r * np.cos(xydir_new - phi)

		# calculate the cumulative distance travelled on each run
		dr = np.sqrt(np.diff(np.power(r, 2) ,1))
		d_cumulative = self._labelledCumSum(np.insert(dr, 0, 0), runLabel)

		# calculate cumulative sum of the expected normalised firing rate
		exptdRate_cumulative = self._labelledCumSum(1-r, runLabel)

		# direction projected onto the run mean direction is just the x coord
		d_meandir = xy_new[0]

		# smooth binned spikes to get an instantaneous firing rate
		# set up the smoothing kernel
		kernLenInBins = np.round(self.ifr_kernLen * self.binsPerSec)
		kernSig = self.ifr_kernSig * self.binsPerSec
		k = signal.gaussian(kernLenInBins, kernSig)
		# get a count of spikes to smooth over
		spkCount = np.bincount(spkPosInd, minlength=nPos)
		# apply the smoothing kernel
		instFiringRate = signal.convolve(spkCount, k, mode='same')
		instFiringRate[~isRun] = np.nan

		# find time spent within run
		time = np.ones(nPos)
		time = self._labelledCumSum(time, runLabel)
		timeInRun = time / self.psr

		fieldNum = fieldLabel[runStartIdx]
		mnSpd = np.squeeze(np.zeros_like(fieldNum, dtype=np.float))
		np.add.at(mnSpd, runLabel[isRun]-1, spd[isRun])
		nPts = np.bincount(runLabel[isRun]-1, minlength=len(mnSpd))
		mnSpd = mnSpd / nPts
		centralPeripheral = np.squeeze(np.zeros_like(fieldNum, dtype=np.float))
		np.add.at(centralPeripheral, runLabel[isRun]-1, xy_new[1, isRun])
		centralPeripheral = centralPeripheral / nPts
		if plot:
			fig = plt.figure()
			ax = fig.add_subplot(221)
			ax.plot(xy_new[0], xy_new[1])
			ax.set_title('Unit circle x-y')
			ax.set_aspect('equal')
			ax.set_xlim([-1, 1])
			ax.set_ylim([-1, 1])
			ax = fig.add_subplot(222)
			ax.plot(fieldPerimX, fieldPerimY,'k.')
			ax.set_title('Field perim and laser on events')
			ax.plot(xy_nrmd[0, fieldLabel>0],xy_nrmd[1, fieldLabel>0], 'y.')
			if laserEvents is not None:
				validOns = np.setdiff1d(laserEvents, np.nonzero(~np.isnan(r))[0])
				ax.plot(xy_nrmd[0, validOns], xy_nrmd[1, validOns], 'rx')
			ax.set_aspect('equal')
			angleCMInd = np.round(perimAngleFromPeak / np.pi * 180) + 180
			angleCMInd[angleCMInd == 0] = 360
			im = np.zeros_like(fieldPerimMask, dtype=np.float)
			im[fieldPerimMask] = angleCMInd
			imM = np.ma.MaskedArray(im, mask=~fieldPerimMask, copy=True)
			#############################################
			# create custom colormap
			cmap = plt.cm.jet_r
			cmaplist = [cmap(i) for i in range(cmap.N)]
			cmaplist[0] = (1, 1, 1, 1)
			cmap = cmap.from_list('Runvals cmap', cmaplist, cmap.N)
			bounds = np.linspace(0, 1.0, 100)
			norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
			# add the runs through the fields
			runVals = np.zeros_like(im, dtype=np.float)
			runVals[yBins[isRun]-1, xBins[isRun]-1] = r[isRun]
			runVals = runVals
			ax = fig.add_subplot(223)
			imm = ax.imshow(runVals, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
			plt.colorbar(imm, orientation='horizontal')
			ax.hold(True)
			ax.set_aspect('equal')
			# add a custom colorbar for colors in runVals

			# create a custom colormap for the plot
			cmap = plt.cm.hsv
			cmaplist = [cmap(i) for i in range(cmap.N)]
			cmaplist[0] = (1, 1, 1, 1)
			cmap = cmap.from_list('Perim cmap', cmaplist, cmap.N)
			bounds = np.linspace(0, 360, 361)
			norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

			imm = ax.imshow(imM, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
			plt.colorbar(imm)
			ax.set_title('Runs by distance and angle')
			ax.plot(peaksXYBins[:, 1], peaksXYBins[:, 0], 'ko')
			ax.set_xlim(0, im.shape[1])
			ax.set_ylim(0, im.shape[0])
			#############################################
			ax = fig.add_subplot(224)
			ax.imshow(rmap, origin='lower', interpolation='nearest')
			ax.set_aspect('equal')
			ax.set_title('Smoothed ratemap')

		# update the regressor dict from __init__ with relevant values
		self.regressors['pos_exptdRate_cum']['values'] = exptdRate_cumulative
		self.regressors['pos_instFR']['values'] = instFiringRate
		self.regressors['pos_timeInRun']['values'] = timeInRun
		self.regressors['pos_d_cum']['values'] = d_cumulative
		self.regressors['pos_d_meanDir']['values'] = d_meandir
		self.regressors['pos_d_currentdir']['values'] = d_currentdir
		posKeys = ('xy', 'xydir', 'r', 'phi', 'xy_old','xydir_old', 'fieldLabel', 'runLabel',
				   'd_currentdir', 'd_cumulative', 'exptdRate_cumulative', 'd_meandir',
				   'instFiringRate', 'timeInRun', 'fieldPerimMask', 'perimAngleFromPeak','posAngleFromPeak')
		runsKeys = ('runStartIdx','runEndIdx','runDurationInPosBins',
					'runsMinSpeed','meanDir','tortuosity', 'mnSpd', 'centralPeripheral')
		posDict = dict.fromkeys(posKeys, np.nan)
		for thiskey in posDict.keys():
			posDict[thiskey] = locals()[thiskey]# neat trick: locals is a dict that holds all locally scoped variables
		runsDict = dict.fromkeys(runsKeys, np.nan)
		for thiskey in runsDict.keys():
			runsDict[thiskey] = locals()[thiskey]
		return posDict, runsDict


	def getThetaProps(self, tetrode, cluster, **kwargs):
		if self.spikeTS is None:
			spikeTS = self.Trial.TETRODE[tetrode].getClustTS(cluster) / 96000.
			self.spikeTS = spikeTS
		else:
			spikeTS = self.spikeTS
		phase = self.phase
		filteredEEG = self.filteredEEG
		oldAmplt = filteredEEG.copy()
		# get indices of spikes into eeg
		spkEEGIdx = np.ceil(spikeTS * self.esr).astype(int) - 1
		spkEEGIdx[spkEEGIdx > len(phase)] = len(phase) - 1
		spkCount = np.bincount(spkEEGIdx, minlength=len(phase))
		spkPhase = phase[spkEEGIdx]
		minSpikingPhase = self._getPhaseOfMinSpiking(spkPhase)
		phaseAdj = self._fixAngle(phase - minSpikingPhase * (np.pi / 180)
								  + self.allowedminSpkPhase)
		isNegFreq = np.diff(np.unwrap(phaseAdj)) < 0
		isNegFreq = np.append(isNegFreq, isNegFreq[-1])
		# get start of theta cycles as points where diff > pi
		phaseDf = np.diff(phaseAdj)
		cycleStarts = phaseDf[1::] < -np.pi
		cycleStarts = np.append(cycleStarts, True)
		cycleStarts = np.insert(cycleStarts, 0, True)
		cycleStarts[isNegFreq] = False
		cycleLabel = np.cumsum(cycleStarts)

		# caculate power and find low power cycles
		power = np.power(filteredEEG, 2)
		cycleTotValidPow = np.bincount(cycleLabel[~isNegFreq],
									   weights=power[~isNegFreq])
		cycleValidBinCount = np.bincount(cycleLabel[~isNegFreq])
		cycleValidMnPow = cycleTotValidPow / cycleValidBinCount
		powRejectThresh = np.percentile(cycleValidMnPow, self.mnPowPrctThresh)
		cycleHasBadPow = cycleValidMnPow < powRejectThresh

		# find cycles too long or too short
		cycleTotBinCount = np.bincount(cycleLabel)
		cycleHasBadLen = np.logical_or(cycleTotBinCount > self.allowedThetaLen[1],
									   cycleTotBinCount < self.allowedThetaLen[0])

		# remove data calculated as 'bad'
		isBadCycle = np.logical_or(cycleHasBadLen, cycleHasBadPow)
		isInBadCycle = isBadCycle[cycleLabel]
		isBad = np.logical_or(isInBadCycle, isNegFreq)
		phaseAdj[isBad] = np.nan
		self.phaseAdj = phaseAdj
		ampAdj = filteredEEG.copy()
		ampAdj[isBad] = np.nan
		cycleLabel[isBad] = 0
		self.cycleLabel = cycleLabel
		out = {'phase': phaseAdj, 'amp': ampAdj, 'cycleLabel': cycleLabel,
			   'oldPhase': phase.copy(), 'oldAmplt': oldAmplt, 'spkCount': spkCount}
		return out

	def getSpikeProps(self, tetrode, cluster, runLabel, meanDir,
					  durationInPosBins):

		if self.spikeTS is None:
			spikeTS = self.Trial.TETRODE[tetrode].getClustTS(cluster) / 96000.
			self.spikeTS = spikeTS
		else:
			spikeTS = self.spikeTS
		xy = self.xy
		phase = self.phaseAdj
		cycleLabel = self.cycleLabel
		spkEEGIdx = np.ceil(spikeTS * self.esr) - 1
		spkEEGIdx[spkEEGIdx > len(phase)] = len(phase) - 1
		spkEEGIdx = spkEEGIdx.astype(int)
		spkPosIdx = np.ceil(spikeTS * self.psr) - 1
		spkPosIdx[spkPosIdx > xy.shape[1]] = xy.shape[1] - 1
		spkRunLabel = runLabel[spkPosIdx.astype(int)]
		thetaCycleLabel = cycleLabel[spkEEGIdx.astype(int)]

		# build mask true for spikes in 1st half of cycle
		firstInTheta = thetaCycleLabel[:-1] != thetaCycleLabel[1::]
		firstInTheta = np.insert(firstInTheta, 0, True)
		lastInTheta = firstInTheta[1::]
		# calculate two kinds of numbering for spikes in a run
		numWithinRun = self._labelledCumSum(np.ones_like(spkPosIdx), spkRunLabel)
		thetaBatchLabelInRun = self._labelledCumSum(firstInTheta.astype(float), spkRunLabel)

		spkCount = np.bincount(spkRunLabel[spkRunLabel > 0], minlength=len(meanDir))
		rateInPosBins = spkCount[1::] / durationInPosBins.astype(float)
		# update the regressor dict from __init__ with relevant values
		self.regressors['spk_numWithinRun']['values'] = numWithinRun
		self.regressors['spk_thetaBatchLabelInRun']['values'] = thetaBatchLabelInRun
		spkKeys = ('spikeTS', 'spkPosIdx', 'spkEEGIdx', 'spkRunLabel', 'thetaCycleLabel',
				   'firstInTheta', 'lastInTheta', 'numWithinRun', 'thetaBatchLabelInRun',
				   'spkCount', 'rateInPosBins')
		spkDict = dict.fromkeys(spkKeys, np.nan)
		for thiskey in spkDict.keys():
			spkDict[thiskey] = locals()[thiskey]
		return spkDict

	def _ppRegress(self, spkDict, whichSpk='first', plot=False, **kwargs):

		phase = self.phaseAdj
		newSpkRunLabel = spkDict['spkRunLabel'].copy()
		# TODO: need code to deal with splitting the data based on a group of
		# variables
		spkUsed = newSpkRunLabel > 0
		if 'first' in whichSpk:
			spkUsed[~spkDict['firstInTheta']] = False
		elif 'last' in whichSpk:
			spkUsed[~spkDict['lastInTheta']] = False
		spkPosIdxUsed = spkDict['spkPosIdx'].astype(int)
		# copy self.regressors and update with spk/ pos of interest
		regressors = self.regressors.copy()
		for k in regressors.keys():
			if k.startswith('spk_'):
				regressors[k]['values'] = regressors[k]['values'][spkUsed]
			elif k.startswith('pos_'):
				regressors[k]['values'] = regressors[k]['values'][spkPosIdxUsed[spkUsed]]
		phase = phase[spkDict['spkEEGIdx'][spkUsed]]
		phase = phase.astype(np.double)
		if 'mean' in whichSpk:
			goodPhase = ~np.isnan(phase)
			cycleLabels = spkDict['thetaCycleLabel'][spkUsed]
			sz = np.max(cycleLabels)
			cycleComplexPhase = np.squeeze(np.zeros(sz, dtype=np.complex))
			np.add.at(cycleComplexPhase, cycleLabels[goodPhase]-1, np.exp(1j * phase[goodPhase]))
			phase = np.angle(cycleComplexPhase)
			spkCountPerCycle = np.bincount(cycleLabels[goodPhase], minlength=sz)
			for k in regressors.keys():
				regressors[k]['values'] = np.bincount(cycleLabels[goodPhase],
													  weights=regressors[k]['values'][goodPhase], minlength=sz) / spkCountPerCycle

		goodPhase = ~np.isnan(phase)
		for k in regressors.keys():
			goodRegressor = ~np.isnan(regressors[k]['values'])
			reg = regressors[k]['values'][np.logical_and(goodRegressor, goodPhase)]
			pha = phase[np.logical_and(goodRegressor, goodPhase)]
			regressors[k]['slope'], regressors[k]['intercept'] = self._circRegress(reg, pha)
			regressors[k]['pha'] = pha
			mnx = np.mean(reg)
			reg = reg - mnx
			mxx = np.max(np.abs(reg)) + np.spacing(1)
			reg = reg / mxx
			# problem regressors = instFR, pos_d_cum

			theta = np.mod(np.abs(regressors[k]['slope']) * reg, 2*np.pi)
			rho, p, rho_boot, p_shuff, ci = self._circCircCorrTLinear(theta, pha, self.k, self.alpha, self.hyp, self.conf)
			regressors[k]['reg'] = reg
			regressors[k]['cor'] = rho
			regressors[k]['p'] = p
			regressors[k]['cor_boot'] = rho_boot
			regressors[k]['p_shuffled'] = p_shuff
			regressors[k]['ci'] = ci

		if plot:
			fig = plt.figure()

			ax = fig.add_subplot(2, 1, 1)
			ax.plot(regressors['pos_d_currentdir']['values'], phase, 'k.')
			ax.plot(regressors['pos_d_currentdir']['values'], phase + 2*np.pi, 'k.')
			slope = regressors['pos_d_currentdir']['slope']
			intercept = regressors['pos_d_currentdir']['intercept']
			mm = (0, -2*np.pi, 2*np.pi, 4*np.pi)
			for m in mm:
				ax.plot((-1, 1), (-slope + intercept + m, slope + intercept + m), 'r', lw=3)
			ax.set_xlim(-1, 1)
			ax.set_ylim(-np.pi, 3*np.pi)
			ax.set_title('pos_d_currentdir')
			ax.set_ylabel('Phase')

			ax = fig.add_subplot(2, 1, 2)
			ax.plot(regressors['pos_d_meanDir']['values'], phase, 'k.')
			ax.plot(regressors['pos_d_meanDir']['values'], phase + 2*np.pi, 'k.')
			slope = regressors['pos_d_meanDir']['slope']
			intercept = regressors['pos_d_meanDir']['intercept']
			mm = (0, -2*np.pi, 2*np.pi, 4*np.pi)
			for m in mm:
				ax.plot((-1, 1), (-slope + intercept + m, slope + intercept + m), 'r', lw=3)
			ax.set_xlim(-1, 1)
			ax.set_ylim(-np.pi, 3*np.pi)
			ax.set_title('pos_d_meanDir')
			ax.set_ylabel('Phase')
			ax.set_xlabel('Normalised position')


		self.reg_phase = phase

		return regressors

	def plotPPRegression(self, regressorDict, regressor2plot='pos_d_cum', ax=None):
		if self.EEG is None:
			self.EEG = axonaIO.EEG(self.filename_root)
		if self.tetrode is None:
			raise AttributeError("I need a tetrode!")
		if self.cluster is None:
			raise AttributeError("I need a cluster!")
		idx = self.Trial.TETRODE[self.tetrode].getClustIdx(self.cluster)
		t = self.Trial._getClusterPhaseVals(self.tetrode, self.cluster)
		x = self._getClusterXPos(self.tetrode, self.cluster)
		label, xe, _ = self.Trial._getFieldLims(self.tetrode, self.cluster)
		xInField = xe[label.nonzero()[1]]
		mask = np.logical_and(x > np.min(xInField), x < np.max(xInField))
		x = x[mask]
		t = t[mask]
		# keep x between -1 and +1
		mnx = np.mean(x)
		xn = x - mnx
		mxx = np.max(np.abs(xn))
		x = xn / mxx
		# keep tn between 0 and 2pi
		t = np.remainder(t, 2 * np.pi)
		slope, intercept = self._circRegress(x, t)
		rho, p, rho_boot, p_shuff, ci = self._circCircCorrTLinear(x, t)
		plt.figure()
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		else:
			ax = ax
		ax.plot(x, t, '.', color='k')
		ax.plot(x, t+2*np.pi, '.', color='k')
		mm = (0, -2*np.pi, 2*np.pi, 4*np.pi)
		for m in mm:
			ax.plot((-1, 1), (-slope + intercept + m, slope + intercept + m), 'r', lw=3)
		ax.set_xlim((-1, 1))
		ax.set_ylim((-np.pi, 3*np.pi))
		return {'slope': slope, 'intercept': intercept, 'rho': rho, 'p': p,
				'rho_boot': rho_boot, 'p_shuff': p_shuff, 'ci': ci}

	def thetaMod(self, eeg, spikeTS=None, pos2use=None, **kwargs):
		"""
		Calculates theta modulation properties of cells and EEG
		"""
		pass

	def _getClusterXPos(self, tetrode, cluster):
		"""
		Returns the x pos of a given cluster taking account of any masking
		"""
		idx = self.Trial.TETRODE[tetrode].getClustIdx(cluster)
		x = self.Trial.POS.xy[0, idx]
		return x

	def _labelledCumSum(self, X, L):
		X = np.ravel(X)
		L = np.ravel(L)
		if len(X) != len(L):
			print('The two inputs need to be of the same length')
			return
		X[np.isnan(X)] = 0
		S = np.cumsum(X)

		mask = L.astype(bool)
		LL = L[:-1] != L[1::]
		LL = np.insert(LL, 0, True)
		isStart = np.logical_and(mask, LL)
		startInds = np.nonzero(isStart)[0]
		if startInds[0] == 0:
			S_starts = S[startInds[1::]-1]
			S_starts = np.insert(S_starts, 0, 0)
		else:
			S_starts = S[startInds-1]

		L_safe = np.cumsum(isStart)
		S[mask] = S[mask] - S_starts[L_safe[mask]-1]
		S[L==0] = np.nan
		return S


	def _cart2pol(self, x, y):
		r = np.hypot(x, y)
		th = np.arctan2(y, x)
		return r, th

	def _pol2cart(self, r, theta):
		x = r * np.cos(theta)
		y = r * np.sin(theta)
		return x, y

	def _applyFilter2Labels(self, M, x):
		"""
		M is a logical mask specifying which label numbers to keep
		x is an array of positive integer labels

		This method sets the undesired labels to 0 and renumbers the remaining
		labels 1 to n when n is the number of trues in M
		"""
		newVals = M * np.cumsum(M)
		x[x>0] = newVals[x[x>0]-1]
		return x

	def _getLabelStarts(self, x):
		x = np.ravel(x)
		xx = np.ones(len(x) + 1)
		xx[1::] = x
		xx = xx[:-1] != xx[1::]
		xx[0] = True
		return np.nonzero(np.logical_and(x, xx))[0]

	def _getLabelEnds(self, x):
		x = np.ravel(x)
		xx = np.ones(len(x) + 1)
		xx[:-1] = x
		xx = xx[:-1] != xx[1::]
		xx[-1] = True
		return np.nonzero(np.logical_and(x, xx))[0]

	def _circ_abs(self, x):
		return np.abs(np.mod(x+np.pi, 2*np.pi) - np.pi)

	def _labelContigNonZeroRuns(self, x):
		x = np.ravel(x)
		xx = np.ones(len(x) + 1)
		xx[1::] = x
		xx = xx[:-1] != xx[1::]
		xx[0] = True
		L = np.cumsum(np.logical_and(x, xx))
		L[np.logical_not(x)] = 0
		return L

	def _toBinUnits2(self, nd_data, binsizes, **kwargs):
		if ~(np.min(nd_data, 1)==0).all():
			nd_data = nd_data - np.min(nd_data, 1)[:, np.newaxis]
		nBins = np.max(np.ceil(np.max(nd_data, 1) / binsizes)).astype(int)
		nd_binned, ye, xe = np.histogram2d(nd_data[0], nd_data[1], bins=(nBins, nBins))
		return nd_data, nd_binned, ye, xe


	def _toBinUnits(self, nd_data, nd_data_ranges, binsizes, **kwargs):
		"""
		data should be in cms (nd_data) so raw data should be divided by ppm
		and multiplied by 100 to get into cm
		"""
		ndims, npoints = np.shape(nd_data)
		if np.logical_or(np.logical_or(nd_data_ranges.shape[0] > 2,
									   nd_data_ranges.shape[0] < 1),
									   nd_data_ranges.shape[0] != nd_data.ndim):
			print('Ranges array must have 1 or 2 rows & the same number of columns as the nd_data array.')
			return
		if np.logical_or(binsizes.ndim != 1, binsizes.size != nd_data.ndim):
			print('Binsizes array must have 1 row & the same number of columns as the nd_data array.')
			return
		# if nd_data_ranges has 2 rows the first is the minima and should be
		# subtracted from the data
		if nd_data_ranges.shape[0] == 2:
			nd_data = (nd_data.T - nd_data_ranges[0, :]).T
			maxBinUnits = np.diff(nd_data_ranges, axis=0) / binsizes
		else:
			maxBinUnits = nd_data_ranges / binsizes
		# remove low points...
		binUnits = nd_data / binsizes[:, np.newaxis]
		badLowPts = binUnits < np.spacing(1)
		binUnits[badLowPts] = np.spacing(1)
		# ... and high points
		badHighPts = binUnits > maxBinUnits.T
		binUnits[0, badHighPts[0, :]] = maxBinUnits[0][0]
		binUnits[1, badHighPts[1, :]] = maxBinUnits[0][1]

		if (np.sum(badLowPts) + np.sum(badHighPts)) / (npoints * ndims) > 0.1:
			print('More than a tenth of data points outside range')
			return

		return binUnits

	def _getPhaseOfMinSpiking(self, spkPhase):
		kernelLen = 180
		kernelSig = kernelLen / 4

		k = signal.gaussian(kernelLen, kernelSig)
		bins = np.arange(-179.5, 180, 1)
		phaseDist, _ = np.histogram(spkPhase / np.pi * 180, bins=bins)
		phaseDist = ndimage.filters.convolve(phaseDist, k)
		phaseMin = bins[int(np.ceil(np.nanmean(np.nonzero(phaseDist==np.min(phaseDist))[0])))]
		return phaseMin

	def _fixAngle(self, a):
		"""
		Ensure angles lie between -pi and pi
		a must be in radians
		"""
		b = np.mod(a + np.pi, 2 * np.pi) - np.pi
		return b

	@staticmethod
	def _ccc(t, p):
		"""
		Calculates correlation between two random circular variables
		"""
		n = len(t)
		A = np.sum(np.cos(t) * np.cos(p), dtype=np.float)
		B = np.sum(np.sin(t) * np.sin(p), dtype=np.float)
		C = np.sum(np.cos(t) * np.sin(p), dtype=np.float)
		D = np.sum(np.sin(t) * np.cos(p), dtype=np.float)
		E = np.sum(np.cos(2 * t), dtype=np.float)
		F = np.sum(np.sin(2 * t), dtype=np.float)
		G = np.sum(np.cos(2 * p), dtype=np.float)
		H = np.sum(np.sin(2 * p), dtype=np.float)
		rho = 4 * (A*B - C*D) / np.sqrt((n**2 - E**2 - F**2)
										* (n**2 - G**2 - H**2))
		return rho

	@staticmethod
	def _ccc_jack(t, p):
		"""
		Function used to calculate jackknife estimates of correlation
		"""
		n = len(t) - 1
		A = np.cos(t) * np.cos(p)
		A = np.sum(A, dtype=np.float) - A
		B = np.sin(t) * np.sin(p)
		B = np.sum(B, dtype=np.float) - B
		C = np.cos(t) * np.sin(p)
		C = np.sum(C, dtype=np.float) - C
		D = np.sin(t) * np.cos(p)
		D = np.sum(D, dtype=np.float) - D
		E = np.cos(2 * t)
		E = np.sum(E, dtype=np.float) - E
		F = np.sin(2 * t)
		F = np.sum(F, dtype=np.float) - F
		G = np.cos(2 * p)
		G = np.sum(G, dtype=np.float) - G
		H = np.sin(2 * p)
		H = np.sum(H, dtype=np.float) - H
		rho = 4 * (A*B - C*D) / np.sqrt((n**2 - E**2 - F**2)
										* (n**2 - G**2 - H**2))
		return rho

	def _circCircCorrTLinear(self, theta, phi, k=1000, alpha=0.05, hyp=0, conf=True):
		"""
		====
		circCircCorrTLinear
		====

		Definition: circCircCorrTLinear(theta, phi, k = 1000, alpha = 0.05,
		hyp = 0, conf = 1)

		----

		An almost direct copy from AJs Matlab fcn to perform correlation between
		2 circular random variables

		Returns the correlation value (rho), p-value, bootstrapped correlation
		values, shuffled p values and correlation values

		Parameters
		----------
		theta, phi: array_like
			   mx1 array containing circular data (in radians) whose correlation
			   is to be measured
		k: int, optional (default = 1000)
			   number of permutations to use to calculate p-value from
			   randomisation and bootstrap estimation of confidence intervals.
			   Leave empty to calculate p-value analytically (NB confidence
			   intervals will not be calculated)
		alpha: float, optional (default = 0.05)
			   hypothesis test level e.g. 0.05, 0.01 etc
		hyp:   int, optional (default = 0)
			   hypothesis to test; -1/ 0 / 1 (negatively correlated / correlated
			   in either direction / positively correlated)
		conf:  bool, optional (default = True)
			   True or False to calculate confidence intervals via jackknife or
			   bootstrap


		References
		---------
		$6.3.3 Fisher (1993), Statistical Analysis of Circular Data,
				   Cambridge University Press, ISBN: 0 521 56890 0
		"""
		theta = theta.ravel()
		phi = phi.ravel()

		if not len(theta) == len(phi):
			raise ValueError()
			print('theta and phi not same length - try again!')

		# estimate correlation
		rho = self._ccc(theta, phi)
		n = len(theta)

		# derive p-values
		if k:
			p_shuff = self._shuffledPVal(theta, phi, rho, k, hyp)
			p = np.nan
		else:
			p = self._analyticPVal(theta, phi, rho, n, alpha, hyp)
			p_shuff = np.nan

		#estimtate ci's for correlation
		if n >= conf:
			#obtain jackknife estimates of rho and its ci's
			rho_jack = self._ccc_jack(theta, phi)
			rho_jack = n * rho - (n - 1) * rho_jack
			rho_boot = np.mean(rho_jack)
			rho_jack_std = np.std(rho_jack)
			ci = (rho_boot - (1 / np.sqrt(n)) * rho_jack_std * stats.norm.ppf(alpha/2, (0,1))[0],
				 rho_boot + (1 / np.sqrt(n)) * rho_jack_std * stats.norm.ppf(alpha/2, (0,1))[0])
		elif conf and k and n < 25 and n > 4:
			# set up the bootstrapping parameters
			idx = resample(theta, n_samples=k)
			rho_boot = []
			for i in idx:
				rho_boot.append(self._ccc(theta[i], phi[i]))
			rho_boot = np.mean(rho_boot)
			import scipy.stats as stats
			ci = stats.t.interval(alpha=alpha, n_samples=k, loc=np.mean(theta), scale=stas.sem(theta))
		else:
			rho_boot = np.nan
			ci = np.nan

		return rho, p, rho_boot, p_shuff, ci

	@staticmethod
	def _shuffledPVal(theta, phi, rho, k, hyp):
		"""
		Calculates shuffled p-values for correlation
		"""
		n = len(theta)
		idx = np.zeros((n, k))
		for i in range(k):
			idx[:, i] = np.random.permutation(np.arange(n, dtype=np.int))

		thetaPerms = theta[idx.astype(int)]

		A = np.dot(np.cos(phi), np.cos(thetaPerms))
		B = np.dot(np.sin(phi), np.sin(thetaPerms))
		C = np.dot(np.sin(phi), np.cos(thetaPerms))
		D = np.dot(np.cos(phi), np.sin(thetaPerms))
		E = np.sum(np.cos(2 * theta), dtype=np.float)
		F = np.sum(np.sin(2 * theta), dtype=np.float)
		G = np.sum(np.cos(2 * phi), dtype=np.float)
		H = np.sum(np.sin(2 * phi), dtype=np.float)

		rho_sim = 4 * (A*B - C*D) / np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))

		if hyp == 1:
			p_shuff = np.sum(rho_sim >= rho, dtype=np.float) / float(k)
		elif hyp == -1:
			p_shuff = np.sum(rho_sim <= rho, dtype=np.float) / float(k)
		elif hyp == 0:
			p_shuff = np.sum(np.fabs(rho_sim) > np.fabs(rho), dtype=np.float) / float(k)
		else:
			p_shuff = np.nan

		return p_shuff


	def _circRegress(self, x, t):
		"""
		Function to find approximation to circular-linear regression for phase
		precession.
		x - n-by-1 list of in-field positions (linear variable)
		t - n-by-1 list of phases, in degrees (converted to radians internally)
		neither can contain NaNs, must be paired (of equal length).
		"""
		# transform the linear co-variate to the range -1 to 1
		mnx = np.mean(x, dtype=np.float)
		xn = x - mnx
		mxx = np.max(np.fabs(xn))
		xn = xn / mxx
		# keep tn between 0 and 2pi
		tn = np.remainder(t, 2 * np.pi)
		# constrain max slope to give at most 720 degrees of phase precession
		# over the field
		max_slope = (2 * np.pi) / (np.max(xn) - np.min(xn))
		# perform slope optimisation and find intercept
		cost = lambda m, x, t: -np.abs(np.sum(np.exp(1j*(t-m*x)))) / len(t-m*x)
		slope = optimize.fminbound(cost, -1*max_slope, max_slope, args=(xn, tn))
		intercept = np.arctan2(np.sum(np.sin(tn - slope*xn)), np.sum(np.cos(tn - slope*xn)))
		intercept = intercept + (-slope*(mnx / mxx))
		slope = slope / mxx
		return slope, intercept
