# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:33:39 2012

@author: robin
"""
import numpy as np
import scipy, scipy.io, scipy.signal
import skimage, skimage.morphology, skimage.measure, skimage.feature, skimage.segmentation
import matplotlib.pyplot as plt
import warnings
import matplotlib.cm as cm
from .utils import polar, rect
import mahotas
import collections

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in greater")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

class SAC(object):
	def __init__(self):
		pass
	def autoCorr2D(self, A, nodwell, tol=1e-10):
		'''
		Performs a spatial autocorrelation on A
		Inputs:
			A - an n-dimensional array of ratemaps
				A can be either 2 or 3 dimensional. In the former case it is 
				simply the binned up ratemap where the two dimensions correspond
				to x and y. In the latter case the first two dimensions are x
				and y and the third is 'stack' of ratemaps
			nodwell - a boolean array corresponding the bins in the ratemap that
				weren't visited. Usually this is generated as:
					nodwell = ~np.isfinite(A) NB use this form even if A is 3D
			tol - values below this are set to zero to deal with v small values
				thrown up by the fft
		'''

		if np.ndim(A) == 2:
			m,n = np.shape(A)
			o = 1
			x = np.reshape(A, (m,n,o))
			nodwell = np.reshape(nodwell, (m,n,o))
		elif np.ndim(A) == 3:
			m,n,o = np.shape(A)
			x = A.copy()
		
		x[nodwell] = 0
		# [Step 1] Obtain FFTs of x, the sum of squares and bins visited
		Fx = np.fft.fft(np.fft.fft(x,2*m-1,axis=0),2*n-1,axis=1)
		FsumOfSquares_x = np.fft.fft(np.fft.fft(np.power(x,2),2*m-1,axis=0),2*n-1,axis=1)
		Fn = np.fft.fft(np.fft.fft(np.invert(nodwell).astype(int),2*m-1,axis=0),2*n-1,axis=1)
		# [Step 2] Multiply the relevant transforms and invert to obtain the
		# equivalent convolutions
		rawCorr = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fx * np.conj(Fx),axis=1),axis=0)),axes=(0,1))
		sums_x = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(np.conj(Fx) * Fn,axis=1),axis=0)),axes=(0,1))
		sumOfSquares_x = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn * np.conj(FsumOfSquares_x),axis=1),axis=0)),axes=(0,1))
		N = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn * np.conj(Fn),axis=1),axis=0)),axes=(0,1))
		# [Step 3] Account for rounding errors.
		rawCorr[np.abs(rawCorr) < tol] = 0
		sums_x[np.abs(sums_x) < tol] = 0
		sumOfSquares_x[np.abs(sumOfSquares_x) < tol] = 0
		N = np.round(N)
		N[N<=1] = np.nan
		# [Step 4] Compute correlation matrix
		mapStd = np.sqrt((sumOfSquares_x * N) - sums_x**2)
		mapCovar = (rawCorr * N) - sums_x * sums_x[::-1,:,:][:,::-1,:][:,:,:]

		return np.squeeze(mapCovar / mapStd / mapStd[::-1,:,:][:,::-1,:][:,:,:])

	def crossCorr2D(self, A, B, A_nodwell, B_nodwell, tol=1e-10):
		if np.ndim(A) != np.ndim(B):
			raise ValueError('Both arrays must have the same dimensionality')
		if np.ndim(A) == 2:
			ma, na = np.shape(A)
			mb, nb = np.shape(B)
			oa = ob = 1
		elif np.ndim(A) == 3:
			[ma,na,oa] = np.shape(A)
			[mb,nb,ob] = np.shape(B)
		A = np.reshape(A, (ma, na, oa))
		B = np.reshape(B, (mb, nb, ob))
#		import pdb
#		pdb.set_trace()
		A_nodwell = np.reshape(A_nodwell, (ma, na, oa))
		B_nodwell = np.reshape(B_nodwell, (mb, nb, ob))
		A[A_nodwell] = 0
		B[B_nodwell] = 0
		# [Step 1] Obtain FFTs of x, the sum of squares and bins visited
		Fa = np.fft.fft(np.fft.fft(A,2*mb-1,axis=0),2*nb-1,axis=1)
		FsumOfSquares_a = np.fft.fft(np.fft.fft(np.power(A,2),2*mb-1,axis=0),2*nb-1,axis=1)
		Fn_a = np.fft.fft(np.fft.fft(np.invert(A_nodwell).astype(int),2*mb-1,axis=0),2*nb-1,axis=1)

		Fb = np.fft.fft(np.fft.fft(B,2*ma-1,axis=0),2*na-1,axis=1)
		FsumOfSquares_b = np.fft.fft(np.fft.fft(np.power(B,2),2*ma-1,axis=0),2*na-1,axis=1)
		Fn_b = np.fft.fft(np.fft.fft(np.invert(B_nodwell).astype(int),2*ma-1,axis=0),2*na-1,axis=1)
		# [Step 2] Multiply the relevant transforms and invert to obtain the
		# equivalent convolutions
		rawCorr = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fa * np.conj(Fb),axis=1),axis=0)))
		sums_a = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fa * np.conj(Fn_b),axis=1),axis=0)))
		sums_b = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn_a * np.conj(Fb),axis=1),axis=0)))
		sumOfSquares_a = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(FsumOfSquares_a * np.conj(Fn_b),axis=1),axis=0)))
		sumOfSquares_b = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn_a * np.conj(FsumOfSquares_b),axis=1),axis=0)))
		N = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn_a * np.conj(Fn_b),axis=1),axis=0)))
		# [Step 3] Account for rounding errors.
		rawCorr[np.abs(rawCorr) < tol] = 0
		sums_a[np.abs(sums_a) < tol] = 0
		sums_b[np.abs(sums_b) < tol] = 0
		sumOfSquares_a[np.abs(sumOfSquares_a) < tol] = 0
		sumOfSquares_b[np.abs(sumOfSquares_b) < tol] = 0
		N = np.round(N)
		N[N<=1] = np.nan
		# [Step 4] Compute correlation matrix
		mapStd_a = np.sqrt((sumOfSquares_a * N) - sums_a**2)
		mapStd_b = np.sqrt((sumOfSquares_b * N) - sums_b**2)
		mapCovar = (rawCorr * N) - sums_a * sums_b

		return np.squeeze(mapCovar / (mapStd_a * mapStd_b))

	def t_win_SAC(self, xy, spkIdx, ppm = 365, winSize=10, pos_sample_rate=50, nbins=71, boxcar=5, Pthresh=100, downsampfreq=50, plot=False):
		'''
		[Stage 0] Get some numbers
		'''
		xy = xy / ppm * 100
		n_samps = xy.shape[1]
		n_spks = len(spkIdx)
		winSizeBins = np.min([winSize * pos_sample_rate, n_samps])
		downsample = np.ceil(pos_sample_rate / downsampfreq) # factor by which positions are downsampled.
		Pthresh = Pthresh / downsample # take account of downsampling

		'''
		[Stage 1] Calculate number of spikes in the window for each spikeInd (ignoring spike itself)
		'''
		#1a. Loop preparation
		nSpikesInWin = np.zeros(n_spks, dtype=np.int)

		#1b. Keep looping until we have dealt with all spikes
		for i, s in enumerate(spkIdx):
			t = np.searchsorted(spkIdx, (s, s + winSizeBins))
			nSpikesInWin[i] = len(spkIdx[t[0]:t[1]]) - 1 # i.e. ignore ith spike

		'''
		[Stage 2] Prepare for main loop
		'''
		#2a. Work out offset inidices to be used when storing spike data
		off_spike = np.cumsum([nSpikesInWin])
		off_spike = np.pad(off_spike,(1,0),'constant',constant_values=(0))

		#2b. Work out number of downsampled pos bins in window and offset indicies for storing data
		nPosInWindow = np.minimum(winSizeBins, n_samps - spkIdx)
		nDownsampInWin = np.floor((nPosInWindow-1)/downsample)+1

		off_dwell = np.cumsum(nDownsampInWin.astype(int))
		off_dwell = np.pad(off_dwell,(1,0),'constant',constant_values=(0))
		

		#2c. Pre-allocate dwell and spike arrays, singles for speed
		dwell = np.zeros((2, off_dwell[-1]),dtype=np.single) * np.nan
		spike = np.zeros((2, off_spike[-1]), dtype=np.single) * np.nan

		filled_pvals = 0
		filled_svals = 0

		for i in range(n_spks):
			# calculate dwell displacements
			winInd_dwell = np.arange(spkIdx[i] + 1, np.minimum(spkIdx[i]+winSizeBins, n_samps), downsample, dtype=np.int)
			WL = len(winInd_dwell)
			dwell[:, filled_pvals:filled_pvals + WL] = np.rot90(np.array(np.rot90(xy[:, winInd_dwell]) - xy[:,spkIdx[i]]))
			filled_pvals = filled_pvals + WL
			# calculate spike displacements
			winInd_spks = i + (spkIdx[i+1:n_spks] < spkIdx[i]+winSizeBins).nonzero()[0]
			WL = len(winInd_spks)
			spike[:, filled_svals:filled_svals+WL] = np.rot90(np.array(np.rot90(xy[:, spkIdx[winInd_spks]]) - xy[:,spkIdx[i]]))
			filled_svals = filled_svals + WL

		dwell = np.delete(dwell, np.isnan(dwell).nonzero()[1], axis=1)
		spike = np.delete(spike, np.isnan(spike).nonzero()[1], axis=1)

		dwell = np.hstack((dwell, -dwell))
		spike = np.hstack((spike, -spike))

		dwell_min = np.min(dwell, axis=1)
		dwell_max = np.max(dwell, axis=1)

		binsize = (dwell_max[1] - dwell_min[1]) / nbins

		dwell = np.round((dwell - np.ones_like(dwell) * dwell_min[:,np.newaxis]) / binsize)
		spike = np.round((spike - np.ones_like(spike) * dwell_min[:,np.newaxis]) / binsize)

		binsize = np.max(dwell, axis=1)
		binedges = np.array(((-0.5,-0.5),binsize+0.5)).T
		Hp = np.histogram2d(dwell[0,:], dwell[1,:], range=binedges, bins=binsize)[0]
		Hs = np.histogram2d(spike[0,:], spike[1,:], range=binedges, bins=binsize)[0]

#        # reverse y,x order
		Hp = np.swapaxes(Hp, 1, 0)
		Hs = np.swapaxes(Hs, 1, 0)

		fHp = self.__blur_image__(Hp, boxcar)
		fHs = self.__blur_image__(Hs, boxcar)

		H = fHs / fHp
		H[Hp < Pthresh] = np.nan

		if plot:
			plt.figure()
			plt.imshow(H.T, interpolation='nearest')
			plt.show()
		return H

	def getMeasures(self, A, maxima='centroid', field_extent_method=2, allProps=True, **kwargs):
		'''
		A clone of the Matlab version of this code in the m-file called
		autoCorrProps.m written by Caswell Barry and Daniel Manson and
		others
		Attempt to see how close a python instantiation replicates a
		Matlab one...
		'''
		A_tmp = A.copy()
		A_tmp[~np.isfinite(A)] = -1
		A_tmp[A_tmp <= 0] = -1
		A_sz = np.array(np.shape(A))
		# [STAGE 1] find peaks & identify 7 closest to centre
		if 'min_distance' in kwargs.keys():
			min_distance = kwargs.pop('min_distance')
		else:
			min_distance = np.ceil(np.min(A_sz / 2) / 8.).astype(int)
		peaksMask = skimage.feature.peak_local_max(A_tmp, indices=False, min_distance=min_distance,exclude_border=False)
		peaksLabel = skimage.measure.label(peaksMask, 8)
		if maxima == 'centroid':
			S = skimage.measure.regionprops(peaksLabel)
			xyCoordPeaks = np.fliplr(np.array([(x['Centroid'][1],x['Centroid'][0]) for x in S]))
		elif maxima == 'single':
			xyCoordPeaks = np.fliplr(np.rot90(np.array(np.nonzero(peaksLabel))))# flipped so xy instead of yx
		# Convert to a new reference frame which has the origin at the centre of the autocorr
		centralPoint = np.ceil(A_sz/2).astype(int)
		xyCoordPeaksCentral = xyCoordPeaks - centralPoint
		# calculate distance of peaks from centre and find 7 closest
		# NB one is central peak - dealt with later
		peaksDistToCentre = np.hypot(xyCoordPeaksCentral[:,1],xyCoordPeaksCentral[:,0])
		orderOfClose = np.argsort(peaksDistToCentre)
		#Get id and coordinates of closest peaks1
		# NB closest peak at index 0 will be centre
		closestPeaks = orderOfClose[0:np.min((7,len(orderOfClose)))]
		closestPeaksCoord = xyCoordPeaks[closestPeaks,:]
		closestPeaksCoord = np.floor(closestPeaksCoord).astype(np.int)
		# [Stage 2] Expand peak pixels into the surrounding half-height region
		if field_extent_method == 1:
			peakLabel = np.zeros((A.shape[0], A.shape[1], len(closestPeaks)))
			perimeterLabel = np.zeros_like(peakLabel)
			for i in range(len(closestPeaks)):
				peakLabel[:,:,i], perimeterLabel[:,:,i] = self.__findPeakExtent__(A, closestPeaks[i], closestPeaksCoord[i])
			fieldsLabel = np.max(peakLabel,2)
			fieldsMask = fieldsLabel > 0
		elif field_extent_method == 2:
			# 2a find the inverse drainage bin for each peak
			fieldsLabel = skimage.morphology.watershed(image=-A_tmp, markers=peaksLabel)
#            fieldsLabel = skimage.segmentation.random_walker(-A, peaksLabel)
			# 2b. Work out what threshold to use in each drainage-basin
			nZones = np.max(fieldsLabel.ravel())
			fieldIDs = fieldsLabel[closestPeaksCoord[:,0],closestPeaksCoord[:,1]]
			thresholds = np.ones((nZones,1)) * np.inf
			# set thresholds for each sub-field at half-maximum
			thresholds[fieldIDs - 1, 0] = A[closestPeaksCoord[:,0],closestPeaksCoord[:,1]] / 2
			fieldsMask = np.zeros((A.shape[0],A.shape[1],nZones))
			for field in fieldIDs:
				sub = fieldsLabel == field
				fieldsMask[:,:, field-1] = np.logical_and(sub, A>thresholds[field-1])
				# TODO: the above step can fragment a sub-field in poorly formed SACs
				# need to deal with this...perhaps by only retaining the largest
				# sub-sub-field
				labelled_sub_field = skimage.measure.label(fieldsMask[:,:, field-1],8)
				sub_props = skimage.measure.regionprops(labelled_sub_field)
				if len(sub_props) > 1:
					distFromCentre = []
					for s in range(len(sub_props)):
						centroid = sub_props[s]['Centroid']
						distFromCentre.append(np.hypot(centroid[0]-A_sz[1],centroid[1]-A_sz[0]))
					idx = np.argmin(distFromCentre)
					tmp = np.zeros_like(A)
					tmp[sub_props[idx]['Coordinates'][:,0],sub_props[idx]['Coordinates'][:,1]] = 1
					fieldsMask[:,:, field-1] = tmp.astype(bool)
			fieldsMask = np.max(fieldsMask,2).astype(bool)
			fieldsLabel[~fieldsMask] = 0
		fieldPerim = mahotas.bwperim(fieldsMask)
		fieldsLabel = fieldsLabel.astype(int)
		# [Stage 3] Calculate a couple of metrics based on the closest peaks
		#Find the (mean) autoCorr value at the closest peak pixels
		nPixelsInLabel = np.bincount(fieldsLabel.ravel())
		sumRInLabel = np.bincount(fieldsLabel.ravel(), weights=A.ravel())
		meanRInLabel = sumRInLabel[closestPeaks+1] / nPixelsInLabel[closestPeaks+1]
		# get scale of grid
		closestPeakDistFromCentre = peaksDistToCentre[closestPeaks[1:]]
		scale = np.median(closestPeakDistFromCentre.ravel())
		# get orientation
		try:
			orientation = self.getorientation(xyCoordPeaksCentral, closestPeaks)
		except:
			orientation = np.nan
		# calculate gridness
		# THIS STEP MASKS THE MIDDLE AND OUTER PARTS OF THE SAC
		# 
		# crop to the central region of the image and remove central peak
		x = np.linspace(-centralPoint[0], centralPoint[0], A_sz[0])
		y = np.linspace(-centralPoint[1], centralPoint[1], A_sz[1])
		xx, yy = np.meshgrid(x, y, indexing = 'ij')
		dist2Centre = np.hypot(xx,yy)
		maxDistFromCentre = np.nan
		if len(closestPeaks) >= 7:
			maxDistFromCentre = np.max(dist2Centre[fieldsMask])
		if np.logical_or(np.isnan(maxDistFromCentre), maxDistFromCentre > np.min(np.floor(A_sz/2))):
			maxDistFromCentre = np.min(np.floor(A_sz/2))
		gridnessMaskAll = dist2Centre <= maxDistFromCentre
		centreMask = fieldsLabel == fieldsLabel[centralPoint[0],centralPoint[1]]
		gridnessMask = np.logical_and(gridnessMaskAll, ~centreMask)
		W = np.ceil(maxDistFromCentre).astype(int)
		autoCorrMiddle = A.copy()
		autoCorrMiddle[~gridnessMask] = np.nan
		autoCorrMiddle = autoCorrMiddle[-W + centralPoint[0]:W + centralPoint[0],-W+centralPoint[1]:W+centralPoint[1]]
		# crop the edges of the middle if there are rows/ columns of nans
		if np.any(np.all(np.isnan(autoCorrMiddle), 1)):
			autoCorrMiddle = np.delete(autoCorrMiddle, np.nonzero((np.all(np.isnan(autoCorrMiddle), 1)))[0][0], 0)
		if np.any(np.all(np.isnan(autoCorrMiddle), 0)):
			autoCorrMiddle = np.delete(autoCorrMiddle, np.nonzero((np.all(np.isnan(autoCorrMiddle), 0)))[0][0], 1)
		if 'step' in kwargs.keys():
			step = kwargs.pop('step')
		else:
			step = 30
		gridness, rotationCorrVals, rotationArr = self.getgridness(autoCorrMiddle, step=step)
		# attempt to fit an ellipse to the closest peaks
		if allProps:
			try:
				a = self.__fit_ellipse__(closestPeaksCoord[1:,0],closestPeaksCoord[1:,1])
				im_centre = self.__ellipse_center__(a)
				ellipse_axes = self.__ellipse_axis_length__(a)
				ellipse_angle = self.__ellipse_angle_of_rotation__(a)
	#            ang =  ang + np.pi
				ellipseXY = self.__getellipseXY__(ellipse_axes[0], ellipse_axes[1], ellipse_angle, im_centre)
				# get the minimum containing circle based on the minor axis of the ellipse
				circleXY = self.__getcircleXY__(im_centre, np.min(ellipse_axes))
			except:
				im_centre = centralPoint
				ellipse_angle = np.nan
				ellipse_axes = (np.nan, np.nan)
				ellipseXY = centralPoint
				circleXY = centralPoint
		else:
			ellipseXY = None
			circleXY = None
			ellipse_axes = None
			ellipse_angle = None
			im_centre = None
		# collect all the following keywords into a dict for output
		dictKeys = ('gridness','scale', 'orientation', 'gridnessMaskAll','gridnessMask', 'fieldsMask',
		'fieldsLabel', 'fieldPerim', 'meanRInLabel', 'closestPeaksCoord', 'xyCoordPeaksCentral', 'closestPeaks',
		'ellipseXY', 'circleXY', 'ellipse_axes', 'ellipse_angle', 'im_centre', 'autoCorrMiddle','rotationArr','rotationCorrVals')
		outDict = dict.fromkeys(dictKeys,np.nan)
		for thiskey in outDict.keys():
			outDict[thiskey] = locals()[thiskey]# neat trick: locals is a dict that holds all locally scoped variables
		return outDict

	def getextrema(self, rotationArray):
		'''
		Uses peak_local_max to find the extrema in the rotational correlation
		plot used to calculate gridness
		NB: requires a rotation array that spans values from 0 to 180 degrees
		'''
		maxima = skimage.feature.peak_local_max(rotationArray)
		minima = skimage.feature.peak_local_max(-rotationArray)
		return maxima, minima

	def getorientation(self, peakCoords, closestPeakIdx):
		if len(closestPeakIdx) == 1:
			return np.nan
		else:
			closestPeaksCoordCentral = peakCoords[closestPeakIdx[1::]]
			theta = polar(closestPeaksCoordCentral[:,1], -closestPeaksCoordCentral[:,0], deg=1)[1]
			return np.sort(theta.compress(theta>0))[0]

	def getgridness(self, image, step=30):
		#TODO: add options in here for whether the full range of correlations are wanted
		# or whether a reduced set is wanted (i.e. at the 30-tuples)
		rotationalCorrVals = collections.OrderedDict.fromkeys(np.arange(0,181,step),np.nan)
		rotationArr = np.zeros(len(rotationalCorrVals)) * np.nan
		# autoCorrMiddle needs to be rescaled or the image rotation falls down
		# as values are cropped to lie between 0 and 1.0
		in_range = (np.nanmin(image), np.nanmax(image))
		out_range = (0, 1)
		autoCorrMiddleRescaled = skimage.exposure.rescale_intensity(image, in_range, out_range)
		origNanIdx = np.isnan(autoCorrMiddleRescaled.ravel())
		for idx, angle in enumerate(rotationalCorrVals.keys()):
			rotatedA = skimage.transform.rotate(autoCorrMiddleRescaled, angle=angle, cval=np.nan, order=3)
			# ignore nans
			rotatedNanIdx = np.isnan(rotatedA.ravel())
			allNans = np.logical_or(origNanIdx, rotatedNanIdx)
			# get the correlation between the original and rotated images and assign
			rotationalCorrVals[angle] = scipy.stats.pearsonr(autoCorrMiddleRescaled.ravel()[~allNans], rotatedA.ravel()[~allNans])[0]
			rotationArr[idx] = rotationalCorrVals[angle]
		gridscore = np.min((rotationalCorrVals[60],rotationalCorrVals[120])) - np.max((rotationalCorrVals[150],rotationalCorrVals[30],rotationalCorrVals[90]))
		return gridscore, rotationalCorrVals, rotationArr

	def show(self, A, inDict, ax=None, **kwargs):
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		Am = A.copy()
		Am[~inDict['gridnessMaskAll']] = np.nan
		Am = np.ma.masked_invalid(np.atleast_2d(Am))
		ret = ax.imshow(A, cmap=cm.gray_r, interpolation='nearest')
		cmap = plt.cm.jet
		cmap.set_bad('w', 0)
		ax.pcolormesh(Am, cmap=cmap, edgecolors='face')
		# horizontal green line at 3 o'clock
		ax.plot((inDict['closestPeaksCoord'][0,1],np.max(inDict['closestPeaksCoord'][:,1])),
				  (inDict['closestPeaksCoord'][0,0],inDict['closestPeaksCoord'][0,0]),'-g', **kwargs)
		mag = inDict['scale'] * 0.5
		th = np.linspace(0, inDict['orientation'], 50)
		[x, y] = rect(mag, th, deg=1)
		# angle subtended by orientation
		ax.plot(x + (inDict['gridnessMask'].shape[1] / 2), (inDict['gridnessMask'].shape[0] / 2) - y, 'r', **kwargs)
		# plot lines from centre to peaks above middle
		for p in inDict['closestPeaksCoord']:
			if p[0] <= inDict['gridnessMask'].shape[0] / 2:
				ax.plot((inDict['gridnessMask'].shape[1] / 2,p[1]),(inDict['gridnessMask'].shape[0] / 2,p[0]),'k', **kwargs)
		all_ax = ax.axes
		x_ax = all_ax.get_xaxis()
		x_ax.set_tick_params(which='both', bottom=False, labelbottom=False,
							 top=False)
		y_ax = all_ax.get_yaxis()
		y_ax.set_tick_params(which='both', left=False, labelleft=False,
							 right=False)
		all_ax.set_aspect('equal')
		all_ax.set_xlim((0.5, inDict['gridnessMask'].shape[1]-1.5))
		all_ax.set_ylim((inDict['gridnessMask'].shape[0]-.5, -.5))
		plt.setp(ax.get_xticklabels(), visible=False)
		plt.setp(ax.get_yticklabels(), visible=False)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		if "show_gridscore" in kwargs.keys():
			ax.annotate('{:.2f}'.format(inDict['gridness']), (0.9,0.15), \
				xycoords='figure fraction', textcoords='figure fraction', color='k', size=30, weight='bold', ha='center', va='center')
		return ret

	def deformSAC(self, A, circleXY, ellipseXY):
		A[np.isnan(A)] = 0
		if circleXY.shape[0] == 2:
			circleXY = circleXY.T
		if ellipseXY.shape[0] == 2:
			ellipseXY = ellipseXY.T
		tform = skimage.transform.AffineTransform()
		tform.estimate(ellipseXY, circleXY)
		'''
		the transformation algorithms used here crop values < 0 to 0. Need to
		rescale the SAC values before doing the deformation and then rescale
		again so the values assume the same range as in the unadulterated SAC
		'''
		SACmin = np.nanmin(A.flatten())#should be 1
		SACmax = np.nanmax(A.flatten())
		AA = A + 1
		deformedSAC = skimage.transform.warp(AA / np.nanmax(AA.flatten()), inverse_map=tform.inverse, cval=0)
		return skimage.exposure.rescale_intensity(deformedSAC, out_range=(SACmin,SACmax))

	def __getcircleXY__(self, centre, radius):
		'''
		function XY = getcircleXY(centre, radius):
		given the origin (1x2 array) and radius this returns 100 x and y points
		for plotting of a circle
		'''
		npts = 100
		t = np.linspace(0+(np.pi/4), (2*np.pi)+(np.pi/4), npts)
		r = np.repeat(radius, npts)
		x = r * np.cos(t) + centre[1]
		y = r * np.sin(t) + centre[0]
		return np.array((x,y))

	def __getellipseXY__(self, a, b, ang, im_centre):
		'''
		function XY = getellipseXY(a, b, ang, im_centre):
		angles are in radians
		given the lengths of the major and minor axes of an ellipse (a and b), the angle of
		rotation and the origin (1x2 array) this returns 100 x and y points for
		plotting of an ellipse
		'''
		pts = 100
		cos_a, sin_a = np.cos(ang), np.sin(ang)
		theta = np.linspace(0, 2*np.pi, pts)
		X = a*np.cos(theta)*cos_a - sin_a*b*np.sin(theta) + im_centre[1]
		Y = a*np.cos(theta)*sin_a + cos_a*b*np.sin(theta) + im_centre[0]
		return np.array((X,Y))

	def __fit_ellipse__(self, x, y):
		x = x[:,np.newaxis]
		y = y[:,np.newaxis]
		D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
		S = np.dot(D.T,D)
		C = np.zeros([6,6])
		C[0,2] = C[2,0] = 2; C[1,1] = -1
		E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
		n = np.argmax(np.abs(E))
		a = V[:,n]
		return a

	def __ellipse_center__(self, a):
		b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
		num = b*b-a*c
		x0=(c*d-b*f)/num
		y0=(a*f-b*d)/num
		return np.array([x0,y0])

	def __ellipse_angle_of_rotation__(self, a):
		b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
		return 0.5*np.arctan(2*b/(a-c))

	def __ellipse_axis_length__(self, a):
		b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
		_up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
		down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
		down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
		res1=np.sqrt(_up/np.abs(down1))
		res2=np.sqrt(_up/np.abs(down2))
		return np.array([res1, res2])

	def __blur_image__(self, im, n, ny=None, ftype='box'):
		""" blurs the image by convolving with a filter ('gauss' or 'box') of
			size n. The optional keyword argument ny allows for a different
			size in the y direction.
		"""
		#g = gauss_kern(n, sizey=ny)quit
		# check for dimensionality of image to be blurred and form correct filter
		if ftype == 'box':
			if np.ndim(im) == 1:
				g = scipy.signal.boxcar(n) / float(n)
			elif np.ndim(im) == 2:
				g = scipy.signal.boxcar([n, n]) / float(n)
		elif ftype == 'gauss':
			g = self.gauss_kern(n, sizey=ny)
			if np.ndim(im) == 1:
				g = g[n, :]
		improc = scipy.signal.convolve(im, g, mode='same')
		return improc

	def __findPeakExtent__(self, A, peakID, peakCoord):
		'''
		Finds extent of field that belongs to each peak - defined as area
		in half-height and also perimieter.
		NB - peakCoord must by m,n pair in normal matrix coords
		'''
		peakLabel = np.zeros((A.shape[0], A.shape[1]))
		perimeterLabel = np.zeros_like(peakLabel)

		# define threshold to use - currently this is half-height
		halfHeight = A[peakCoord[1], peakCoord[0]] * .5
		aboveHalfHeightLabel = scipy.ndimage.label(A > halfHeight, structure=np.ones((3,3)))[0]
		peakIDTmp = aboveHalfHeightLabel[peakCoord[1], peakCoord[0]]
		peakLabel[aboveHalfHeightLabel == peakIDTmp] = peakID
		perimeterLabel[mahotas.bwperim(aboveHalfHeightLabel==peakIDTmp)] = peakID
		return peakLabel, perimeterLabel