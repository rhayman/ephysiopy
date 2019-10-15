# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:31:31 2012

@author: robin
"""
import numpy as np
import skimage
from scipy import ndimage, signal, stats, optimize
import matplotlib.pylab as plt
import mahotas
import matplotlib
from . import axonaIO
from . import dacq2py_util
from sklearn.utils import resample

class phasePrecession():
	"""
	NB : MAKE SURE TO SEND 'CM' TO THE POS CONSTRUCTOR SO POS XY VALS ARE
	RETURNED IN CM NOT PIXELS
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
		if 'binsizes' in kwargs.keys():
			self.binsizes = kwargs.pop('binsizes')
		peaksXY, peaksRate, labels, rmap = self.partitionFields(tetrode, cluster, plot=True)
		posD, runD = self.getPosProps(tetrode, cluster, labels, peaksXY, laserEvents=laserEvents, plot=True)
		thetaProps = self.getThetaProps(tetrode, cluster)
		spkD = self.getSpikeProps(tetrode, cluster, posD['runLabel'], runD['meanDir'],
								  runD['runDurationInPosBins'])
		regressD = self._ppRegress(spkD, plot=True)

	def partitionFields(self, tetrode, cluster, ftype='g', plot=False, **kwargs):
		'''
		Partitions fileds.
		Partitions spikes into fields by finding the watersheds around the
		peaks of a super-smoothed ratemap
		Parameters
			psr - the pos sampling rate (defaults to 50.0Hz)
			dacq2py.Tetrode.getClustTS() will have to be divided by 96000)
			ftype - 'p' or 'g' denoting place or grid cells - not implemented
			binsizes - the binsize in cm
			smthKernLen - the size of the kernel to smooth the data with (in cm)
			smthKernSig - the sigma of the kernel if using 'gaussian'
			fieldThresh - the fraction to threshold the field (float >0 <=1)
			areaThresh - size in cm of the small fields to remove
			binsPerCm - the number of bins per cm
			plot - boolean. Whether to produce a debugging plot or not

			Valid keyword arguments include:
				TODO
		Outputs:
			peaksXY - the xy coordinates of the peak rates in each field
			peaksRate - the peak rates in peaksXY
			labels - an array of the labels corresponding to each field (starts
			at 1)

		'''
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
		fieldId, firstIdx = np.unique(labels, return_index=True)
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
		'''
		Uses the output of partitionFields and returns vectors the same
		length as pos.
		Parameters:
			xy, xydir, spd - obvious spatial variables as numpy arrays
			rmap - the smootheed, binned ratemap
			spikeTS - the spike timestamps in seconds
			labels - a labelled ratemap (integer id for each field; 0 = background)
			peaksXY - x-y coords of the peaks in the ratemap
			laserEvents - position indices of on events (laser on)

		Output:
			fieldLabel - the field label for each pos sample (0 outside)
			runLabel - the run label for each pos (0 outside)
			r - distance from pos to centre / distance from perim to centre
			phi - the difference between the pos's angle to the peak and the
					runs mean direction (nan outisde)
			NB: both r and phi are smoothed in cartesian space
			xy - r and phi in cartesian coords
			xyDir - direction of travel according to xy coords
			d_meanDir - r projected onto the runs mean direction
			d_currentDir - r projected on the runs current direction
			xy_old - the original xy values
			dir_old - the original dir values

		'''
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

		fieldPerimMask = mahotas.bwperim(labels)
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

		'''
		Calculates theta modulation properties of cells and EEG
		'''
		pass

	def _getClusterXPos(self, tetrode, cluster):
		'''
		Returns the x pos of a given cluster taking account of any masking
		'''
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
		'''
		M is a logical mask specifying which label numbers to keep
		x is an array of positive integer labels

		This method sets the undesired labels to 0 and renumbers the remaining
		labels 1 to n when n is the number of trues in M
		'''
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
		'''
		data should be in cms (nd_data) so raw data should be divided by ppm
		and multiplied by 100 to get into cm
		'''
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
		'''
		Calculates correlation between two random circular variables
		'''
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
		'''
		Function used to calculate jackknife estimates of correlation
		'''
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
		'''
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
		'''
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
		'''
		Calculates shuffled p-values for correlation
		'''
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
		'''
		Function to find approximation to circular-linear regression for phase
		precession.
		x - n-by-1 list of in-field positions (linear variable)
		t - n-by-1 list of phases, in degrees (converted to radians internally)
		neither can contain NaNs, must be paired (of equal length).
		'''
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