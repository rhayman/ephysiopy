# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:31:31 2012

@author: robin
"""
import numpy as np
from scipy import signal, spatial, misc, ndimage
import skimage as skimage
import matplotlib.pyplot as plt
import warnings

class FieldCalcs:
	'''
	A series of methods are defined here that allow quantification of ratemaps
	Methods:
		- _blur_image(im, n, ny=None, ftype='boxcar')
		- limit_to_one(rmap, prc=50, min_dist=5)
		- global_threshold
		- local_threshold
		- get_field_props
		- coherence(smoothed_rate, unsmoothed rate)
		- kldiv_dir(polarplot)
		- kldiv(X, pvect1,pvect2,variant=None)
		- skaggsInfo(ratemap, dwelltimes)
		- xPearson(ratemap1,ratemap2=None,mode='full')
		- linearStackAverage(ratemap)
	'''
	def _blur_image(self, im, n, ny=None, ftype='boxcar'):
		""" blurs the image by convolving with a filter ('gaussian' or
			'boxcar') of
			size n. The optional keyword argument ny allows for a different
			size in the y direction.
		"""
		n = int(n)
		if not ny:
			ny = n
		else:
			ny = int(ny)
		#  keep track of nans
		nan_idx = np.isnan(im)
		im[nan_idx] = 0
		if ftype == 'boxcar':
			if np.ndim(im) == 1:
				g = signal.boxcar(n) / float(n)
			elif np.ndim(im) == 2:
				g = signal.boxcar([n, ny]) / float(n)
		elif ftype == 'gaussian':
			x, y = np.mgrid[-n:n+1, -ny:ny+1]
			g = np.exp(-(x**2/float(n) + y**2/float(ny)))
			g = g / g.sum()
			if np.ndim(im) == 1:
				g = g[n, :]
		improc = signal.convolve(im, g, mode='same')
		improc[nan_idx] = np.nan
		return improc    
	
	def limit_to_one(self, A, prc=50, min_dist=5):
		"""
		Processes a multi-peaked ratemap (ie grid cell) and returns a matrix
		where the multi-peaked ratemap consist of a single peaked field that is
		a) not connected to the border and b) close to the middle of the ratemap
		"""
		Ac = A.copy()
		Ac[np.isnan(A)] = 0
		# smooth Ac more to remove local irregularities
		n = ny = 5
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		g = g / g.sum()
		Ac = signal.convolve(Ac, g, mode='same')
		peak_mask = skimage.feature.peak_local_max(Ac, min_distance=min_dist,
												   exclude_border=False,
												   indices=False)
		peak_labels = skimage.measure.label(peak_mask, 8)
		field_labels = skimage.morphology.watershed(image=-Ac,
													markers=peak_labels)
		nFields = np.max(field_labels)
		sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
		labelled_sub_field_mask = np.zeros_like(sub_field_mask)
		sub_field_props = skimage.measure.regionprops(field_labels,
													  intensity_image=Ac)
		sub_field_centroids = []
		sub_field_size = []

		for sub_field in sub_field_props:
			tmp = np.zeros(Ac.shape).astype(bool)
			tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
			tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
			sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
			labelled_sub_field_mask[sub_field.label-1, np.logical_and(tmp2, tmp)] = sub_field.label
			sub_field_centroids.append(sub_field.centroid)
			sub_field_size.append(sub_field.area)  # in bins
		sub_field_mask = np.sum(sub_field_mask, 0)
		middle = np.round(np.array(A.shape) / 2)
		normd_dists = sub_field_centroids - middle
		field_dists_from_middle = np.hypot(normd_dists[:, 0], normd_dists[:, 1])
		central_field_idx = np.argmin(field_dists_from_middle)
		central_field = np.squeeze(labelled_sub_field_mask[central_field_idx, :, :])
		# collapse the labelled mask down to an 2d array
		labelled_sub_field_mask = np.sum(labelled_sub_field_mask, 0)
		# clear the border
		cleared_mask = skimage.segmentation.clear_border(central_field)
		# check we've still got stuff in the matrix or fail
		if ~np.any(cleared_mask):
			print('No fields were detected away from edges so nothing returned')
			return None, None, None
		else:
			central_field_props = sub_field_props[central_field_idx]
			return central_field_props, central_field, central_field_idx

	def global_threshold(self, A, prc=50, min_dist=5):
		'''
		Globally thresholds a ratemap and counts number of fields found
		'''
		Ac = A.copy()
		Ac[np.isnan(A)] = 0
		n = ny = 5
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		g = g / g.sum()
		Ac = signal.convolve(Ac, g, mode='same')
		maxRate = np.nanmax(np.ravel(Ac))
		Ac[Ac < maxRate*(prc/float(100))] = 0
		peak_mask = skimage.feature.peak_local_max(Ac, min_distance=min_dist,
												   exclude_border=False,
												   indices=False)
		peak_labels = skimage.measure.label(peak_mask, 8)
		field_labels = skimage.morphology.watershed(image=-Ac,
													markers=peak_labels)
		nFields = np.max(field_labels)
		return nFields

	def local_threshold(self, A, prc=50, min_dist=5):
		"""
		Locally thresholds a ratemap to take only the surrounding prc amount
		around any local peak
		"""
		Ac = A.copy()
		nanidx = np.isnan(Ac)
		Ac[nanidx] = 0
		# smooth Ac more to remove local irregularities
		n = ny = 5
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		g = g / g.sum()
		Ac = signal.convolve(Ac, g, mode='same')
		peak_mask = skimage.feature.peak_local_max(Ac, min_distance=min_dist,
												   exclude_border=False,
												   indices=False)
		peak_labels = skimage.measure.label(peak_mask, 8)
		field_labels = skimage.morphology.watershed(image=-Ac,
													markers=peak_labels)
		nFields = np.max(field_labels)
		sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
		sub_field_props = skimage.measure.regionprops(field_labels,
													  intensity_image=Ac)
		sub_field_centroids = []
		sub_field_size = []

		for sub_field in sub_field_props:
			tmp = np.zeros(Ac.shape).astype(bool)
			tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
			tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
			sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
			sub_field_centroids.append(sub_field.centroid)
			sub_field_size.append(sub_field.area)  # in bins
		sub_field_mask = np.sum(sub_field_mask, 0)
		A_out = np.zeros_like(A)
		A_out[sub_field_mask.astype(bool)] = A[sub_field_mask.astype(bool)]
		A_out[nanidx] = np.nan
		return A_out

	def getBorderScore(self, A, B=None, shape='square', fieldThresh=0.3, smthKernSig=3,
					circumPrc=0.2, binSize=3.0, minArea=200, debug=False):
		'''
		Calculates a border score totally dis-similar to that calculated in Solstad et al
		(2008)

		Parameters
		----------
		A : np.array
			Should be the ratemap
		B : np.array
			This should be a boolean mask where True (1)
			is equivalent to the presence of a border and False (0)
			is equivalent to 'open space'. Naievely this will be the 
			edges of the ratemap but could be used to take account of 
			boundary insertions/ creations to check tuning to multiple
			environmental boundaries. Default None: when the mask is 
			None then a mask is created that has 1's at the edges of the
			ratemap i.e. it is assumed that occupancy = environmental 
			shape
		shape : str
			description of environment shape. Currently
			only 'square' or 'circle' accepted. Used to calculate the
			proportion of the environmental boundaries to examine for
			firing
		fieldThresh : float
			Between 0 and 1 this is the percentage 
			amount of the maximum firing rate
			to remove from the ratemap (i.e. to remove noise)
		smthKernSig : float
			the sigma value used in smoothing the ratemap
			(again!) with a gaussian kernel
		circumPrc : float
			The percentage amount of the circumference
			of the environment that the field needs to be to count
			as long enough to make it through
		binSize : float
			bin size in cm
		minArea : float
			min area for a field to be considered
		debug : bool
			If True then some plots and text will be output

		Returns
		-------
			float : the border score

		Notes
		-----
		If the cell is a border cell (BVC) then we know that it should
		fire at a fixed distance from a given boundary (possibly more
		than one). In essence this algorithm estimates the amount of 
		variance in this distance i.e. if the cell is a border cell this
		number should be small. This is achieved by first doing a bunch of 
		morphological operations to isolate individual fields in the
		ratemap (similar to the code used in phasePrecession.py - see 
		the partitionFields method therein). These partitioned fields are then
		thinned out (using skimage's skeletonize) to a single pixel
		wide field which will lie more or less in the middle of the
		(highly smoothed) sub-field. It is the variance in distance from the
		nearest boundary along this pseudo-iso-line that is the boundary measure

		Other things to note are that the pixel-wide field has to have some minimum
		length. In the case of a circular environment this is set to 
		20% of the circumference; in the case of a square environment markers
		this is at least half the length of the longest side

		'''

		dwell = np.isfinite(A)
		# need to know borders of the environment so we can see if a field
		# touches the edges, and the perimeter length of the environment
		# deal with square or circles differently
		borderMask = np.zeros_like(A)
		A_rows, A_cols = np.shape(A)
		if 'circle' in shape:
			radius = np.max(np.array(np.shape(A))) / 2.0
			dist_mask = skimage.morphology.disk(radius)
			if np.shape(dist_mask) > np.shape(A):
				dist_mask = dist_mask[1:A_rows+1, 1:A_cols+1]
			tmp = np.zeros([A_rows + 2, A_cols + 2])
			tmp[1:-1, 1:-1] = dist_mask
			dists = ndimage.morphology.distance_transform_bf(tmp)
			dists = dists[1:-1, 1:-1]
			perimeter =  2.0 * radius * np.pi
			borderMask = np.logical_xor(dists <= 0, dists < 2)
			# open up the border mask a little
			borderMask = skimage.morphology.binary_dilation(borderMask, skimage.morphology.disk(1))
		elif 'square' in shape:
			perimeter = np.sum(np.array(np.shape(A)*2))
			borderMask[0:3, :] = 1
			borderMask[-3:, :] = 1
			borderMask[:, 0:3] = 1
			borderMask[:, -3:] = 1
			tmp = np.zeros([A_rows + 2, A_cols + 2])
			dist_mask = np.ones_like(A)
			tmp[1:-1, 1:-1] = dist_mask
			dists = ndimage.morphology.distance_transform_bf(tmp)
			# remove edges to make same shape as input ratemap
			dists = dists[1:-1, 1:-1]
		A[np.isnan(A)] = 0
		# get some morphological info about the fields in the ratemap
		# start image processing:
		# get some markers
		# NB I've tried a variety of techniques to optimise this part and the
		# best seems to be the local adaptive thresholding technique which)
		# smooths locally with a gaussian - see the skimage docs for more
		idx = A >= np.nanmax(np.ravel(A)) * fieldThresh		
		A_thresh = np.zeros_like(A)
		A_thresh[idx] = A[idx]

		# label these markers so each blob has a unique id
		labels, nFields = ndimage.label(A_thresh)
		# remove small objects
		min_size = int(minArea / binSize) - 1
		if debug:
			plt.figure()
			plt.imshow(A_thresh)
			ax = plt.gca()
			ax.set_title('Before removing small objects')
		skimage.morphology.remove_small_objects(labels, min_size=min_size, connectivity=2, in_place=True)
		labels = skimage.segmentation.relabel_sequential(labels)[0]
		nFields = np.max(labels)
		if nFields == 0:
			return np.nan
		# Iterate over the labelled parts of the array labels calculating
		# how much of the total circumference of the environment edge it
		# covers

		fieldAngularCoverage = np.zeros([1, nFields]) * np.nan
		fractionOfPixelsOnBorder = np.zeros([1, nFields]) * np.nan
		fieldsToKeep = np.zeros_like(A)
		for i in range(1, nFields+1):
			fieldMask = np.logical_and(labels==i, borderMask)

			# check the angle subtended by the fieldMask
			if np.sum(fieldMask.astype(int)) > 0:
				s = skimage.measure.regionprops(fieldMask.astype(int), intensity_image=A_thresh)[0]
				x = s.coords[:,0] - (A_cols / 2.0)
				y = s.coords[:,1] - (A_rows / 2.0)
				subtended_angle = np.rad2deg(np.ptp(np.arctan2(x,y)))
				if subtended_angle > (360 * circumPrc):
					pixelsOnBorder = np.count_nonzero(fieldMask) / float(np.count_nonzero(labels==i))
					fractionOfPixelsOnBorder[:,i-1] = pixelsOnBorder
					if pixelsOnBorder > 0.5:
						fieldAngularCoverage[0, i-1] = subtended_angle

				fieldsToKeep = np.logical_or(fieldsToKeep, labels==i)
		if debug:
			fig, ax = plt.subplots(4,1,figsize=(3,9))
			ax1 = ax[0]
			ax2 = ax[1]
			ax3 = ax[2]
			ax4 = ax[3]
			ax1.imshow(A)
			ax2.imshow(labels)
			ax3.imshow(A_thresh)
			ax4.imshow(fieldsToKeep)
			plt.show()
			for i,f in enumerate(fieldAngularCoverage.ravel()):
				print("angle subtended by field {0} = {1:.2f}".format(i+1, f))
			for i,f in enumerate(fractionOfPixelsOnBorder.ravel()):
				print("% pixels on border for field {0} = {1:.2f}".format(i+1, f))
		fieldAngularCoverage = (fieldAngularCoverage / 360.)
		if np.sum(fieldsToKeep) == 0:
			return np.nan
		rateInField = A[fieldsToKeep]
		# normalize firing rate in the field to sum to 1
		rateInField = rateInField / np.nansum(rateInField)
		dist2WallInField = dists[fieldsToKeep]
		Dm = np.dot(dist2WallInField, rateInField)
		if 'circle' in shape:
			Dm = Dm / radius
		elif 'square' in shape:
			Dm = Dm / (np.max(np.shape(A)) / 2.0)
		borderScore = (fractionOfPixelsOnBorder-Dm) / (fractionOfPixelsOnBorder+Dm)
		return np.max(borderScore)

	def get_field_props(self, A, min_dist=5, neighbours=2, prc=50,
						plot=False, ax=None, tri=False, verbose=True, **kwargs):
		"""
		Returns a dictionary of properties of the field(s) in a ratemap A
		Parameters
		----------
		A : numpy.array
			a ratemap (but could be any image)
		min_dist : float
			the separation (in bins) between fields for measures
			such as field distance to make sense. Used to
			partition the image into separate fields in the call to
			skimage.feature.peak_local_max
		neighbours : int
			the number of fields to consider as neighbours to
			any given field. Defaults to 2
		prc : float
			percent of fields to consider
		ax : matplotlib.Axes
			user supplied axis. If None a new figure window is created
		tri : bool
			whether to do Delaunay triangulation between fields
			and add to plot
		verbose : bool
			dumps the properties to the console
		plot : bool
			whether to plot some output - currently consists of the
			ratemap A, the fields of which are outline in a black
			contour. Default False

		Returns
		-------
		result : dict
			The properties of the field(s) in the input ratemap A
		"""

		from scipy.spatial import Delaunay
		from skimage.measure import find_contours
		from sklearn.neighbors import NearestNeighbors
		import gridcell
		import matplotlib.cm as cm
		nan_idx = np.isnan(A)
		Ac = A.copy()
		Ac[np.isnan(A)] = 0
		# smooth Ac more to remove local irregularities
		n = ny = 5
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		g = g / g.sum()
		Ac = signal.convolve(Ac, g, mode='same')
		if 'clear_border' in kwargs.keys():
			clear_border = True
		else:
			clear_border = False
		peak_idx = skimage.feature.peak_local_max(Ac, min_distance=min_dist,
												  exclude_border=clear_border,
												  indices=True)
		if neighbours > len(peak_idx):
			print('neighbours value of {0} > the {1} peaks found'.format(neighbours, len(peak_idx)))
			print('Reducing neighbours to number of peaks found')
			neighbours = len(peak_idx)
		peak_mask = skimage.feature.peak_local_max(Ac, min_distance=min_dist, exclude_border=clear_border,
												   indices=False)
		peak_labels = skimage.measure.label(peak_mask, 8)
		field_labels = skimage.morphology.watershed(image=-Ac,
													markers=peak_labels)
		nFields = np.max(field_labels)
		sub_field_mask = np.zeros((nFields, Ac.shape[0], Ac.shape[1]))
		sub_field_props = skimage.measure.regionprops(field_labels,
													  intensity_image=Ac)
		sub_field_centroids = []
		sub_field_size = []

		for sub_field in sub_field_props:
			tmp = np.zeros(Ac.shape).astype(bool)
			tmp[sub_field.coords[:, 0], sub_field.coords[:, 1]] = True
			tmp2 = Ac > sub_field.max_intensity * (prc/float(100))
			sub_field_mask[sub_field.label - 1, :, :] = np.logical_and(tmp2, tmp)
			sub_field_centroids.append(sub_field.centroid)
			sub_field_size.append(sub_field.area)  # in bins
		sub_field_mask = np.sum(sub_field_mask, 0)
		contours = skimage.measure.find_contours(sub_field_mask, 0.5)
		# find the nearest neighbors to the peaks of each sub-field


		nbrs = NearestNeighbors(n_neighbors=neighbours, algorithm='ball_tree').fit(peak_idx)
		distances, indices = nbrs.kneighbors(peak_idx)
		mean_field_distance = np.mean(distances[:, 1:neighbours])


		nValid_bins = np.sum(~nan_idx)
		# calculate the amount of out of field firing
		A_non_field = np.zeros_like(A) * np.nan
		A_non_field[~sub_field_mask.astype(bool)] = A[~sub_field_mask.astype(bool)]
		A_non_field[nan_idx] = np.nan
		out_of_field_firing_prc = (np.count_nonzero(A_non_field > 0) / float(nValid_bins)) * 100
		Ac[np.isnan(A)] = np.nan
		"""
		get some stats about the field ellipticity
		"""
		central_field_props, central_field, central_field_idx = self.limit_to_one(A, prc=50)
		if central_field is None:
			ellipse_ratio = np.nan
		else:
			contour_coords = find_contours(central_field, 0.5)
			G = gridcell.SAC()
			a = G.__fit_ellipse__(contour_coords[0][:,0], contour_coords[0][:,1])
			ellipse_axes = G.__ellipse_axis_length__(a)
			ellipse_ratio = np.min(ellipse_axes) / np.max(ellipse_axes)
		''' using the peak_idx values calculate the angles of the triangles that
		make up a delaunay tesselation of the space if the calc_angles arg is
		in kwargs
		'''
		if 'calc_angs' in kwargs.keys():
			try:
				angs = self.calc_angs(peak_idx)
			except:
				angs = np.nan
		else:
			angs = None

		if plot:
			if ax is None:
				fig = plt.figure()
				ax = fig.add_subplot(111)
			else:
				ax = ax
			Am = np.ma.MaskedArray(Ac, mask=nan_idx, copy=True)
			ax.pcolormesh(Am, cmap=cm.jet, edgecolors='face')
			for c in contours:
				ax.plot(c[:, 1], c[:, 0], 'k')
			# do the delaunay magic
			if tri:
				tri = Delaunay(peak_idx)
				ax.triplot(peak_idx[:,1], peak_idx[:,0], tri.simplices.copy(), color='w', marker='o')
			ax.set_xlim(0, Ac.shape[1] - 0.5)
			ax.set_ylim(0, Ac.shape[0] - 0.5)
			ax.set_xticklabels('')
			ax.set_yticklabels('')
			ax.invert_yaxis()
		props = {'Ac' : Ac,
				 'Peak_rate': np.nanmax(A),
				 'Mean_rate': np.nanmean(A),
				 'Field_size': np.mean(sub_field_size),
				 'Pct_bins_with_firing': (np.sum(sub_field_mask) / nValid_bins) * 100,
				 'Out_of_field_firing_prc': out_of_field_firing_prc,
				 'Dist_between_fields': mean_field_distance,
				 'Num_fields': float(nFields),
				 'Sub_field_mask': sub_field_mask,
				 'Smoothed_map': Ac,
				 'field_labels': field_labels,
				 'Peak_idx': peak_idx,
				 'angles': angs,
				 'contours': contours,
				 'ellipse_ratio': ellipse_ratio}

		if verbose:
			print('\nPercentage of bins with firing: {:.2%}'.format(np.sum(sub_field_mask) / nValid_bins))
			print('Percentage out of field firing: {:.2%}'.format(np.count_nonzero(A_non_field > 0) / float(nValid_bins)))
			print('Peak firing rate: {:.3} Hz'.format(np.nanmax(A)))
			print('Mean firing rate: {:.3} Hz'.format(np.nanmean(A)))
			print('Number of fields: {0}'.format(nFields))
			print('Mean field size: {:.5} cm'.format(np.mean(sub_field_size)))  # 3 is binsize)
			print('Mean inter-peak distance between fields: {:.4} cm'.format(mean_field_distance))
		return props

	def calc_angs(self, points):
		"""
		Calculates the angles for all triangles in a delaunay tesselation of
		the peak points in the ratemap
		"""

		# calculate the lengths of the sides of the triangles
		sideLen = np.hypot(points[:, 0], points[:, 1])
		tri = spatial.Delaunay(points)
		indices, indptr = tri.vertex_neighbor_vertices
		nTris = tri.nsimplex
		outAngs = []
		for k in range(nTris):
			idx = indptr[indices[k]:indices[k+1]]
			a = sideLen[k]
			b = sideLen[idx[0]]
			c = sideLen[idx[1]]
			angA = self._getAng(a, b, c)
			angB = self._getAng(b, c, a)
			angC = self._getAng(c, a, b)
			outAngs.append((angA, angB, angC))
		return np.array(outAngs).T

	def _getAng(self, a, b, c):
		'''
		Given lengths a,b,c of the sides of a triangle this returns the angles
		in degress of all 3 angles
		'''
		return np.degrees(np.arccos((c**2 - b**2 - a**2)/(-2.0 * a * b)))

	def corr_maps(self, map1, map2, maptype='normal'):
		'''
		correlates two ratemaps together ignoring areas that have zero sampling
		'''
		if map1.shape > map2.shape:
			map2 = misc.imresize(map2, map1.shape, interp='nearest', mode='F')
		elif map1.shape < map2.shape:
			map1 = misc.imresize(map1, map2.shape, interp='nearest', mode='F')
		map1 = map1.flatten()
		map2 = map2.flatten()
		if maptype is 'normal':
			valid_map1 = np.logical_or((map1 > 0), ~np.isnan(map1))
			valid_map2 = np.logical_or((map2 > 0), ~np.isnan(map2))
		elif maptype is 'grid':
			valid_map1 = ~np.isnan(map1)
			valid_map2 = ~np.isnan(map2)
		valid = np.logical_and(valid_map1, valid_map2)
		r = np.corrcoef(map1[valid], map2[valid])
		if r.any():
			return r[1][0]
		else:
			return np.nan

	def coherence(self, smthd_rate, unsmthd_rate):
		'''calculates coherence of receptive field via correlation of smoothed
		and unsmoothed ratemaps
		'''
		smthd = smthd_rate.ravel()
		unsmthd = unsmthd_rate.ravel()
		si = ~np.isnan(smthd)
		ui = ~np.isnan(unsmthd)
		idx = ~(~si | ~ui)
		coherence = np.corrcoef(unsmthd[idx], smthd[idx])
		return coherence[1,0]

	def kldiv_dir(self, polarPlot):
		"""
		Returns a kl divergence for directional firing: measure of directionality.
		Calculates kl diveregence between a smoothed ratemap (probably should be smoothed
		otherwise information theoretic measures don't 'care' about position of bins relative to
		one another) and a pure circular distribution. The larger the divergence the more
		tendancy the cell has to fire when the animal faces a specific direction.

		Parameters
		----------
		polarPlot: 1D-array
			The binned and smoothed directional ratemap

		Returns
		-------
		klDivergence: float
			The divergence from circular of the 1D-array from a uniform circular
			distribution
		"""

		__inc = 0.00001
		polarPlot = np.atleast_2d(polarPlot)
		polarPlot[np.isnan(polarPlot)] = __inc
		polarPlot[polarPlot == 0] = __inc
		normdPolar = polarPlot / float(np.nansum(polarPlot))
		nDirBins = polarPlot.shape[1]
		compCirc = np.ones_like(polarPlot) / float(nDirBins)
		kldivergence = self.kldiv(np.arange(0,nDirBins), normdPolar, compCirc)
		return kldivergence

	def kldiv(self, X, pvect1, pvect2, variant=None):
		'''
		Calculates the Kullback-Leibler or Jensen-Shannon divergence between two distributions.

		kldiv(X,P1,P2) returns the Kullback-Leibler divergence between two
		distributions specified over the M variable values in vector X.  P1 is a
		length-M vector of probabilities representing distribution 1, and P2 is a
		length-M vector of probabilities representing distribution 2.  Thus, the
		probability of value X(i) is P1(i) for distribution 1 and P2(i) for
		distribution 2.  The Kullback-Leibler divergence is given by:

		.. math:: KL(P1(x),P2(x)) = sum_[P1(x).log(P1(x)/P2(x))]

		If X contains duplicate values, there will be an warning message, and these
		values will be treated as distinct values.  (I.e., the actual values do
		not enter into the computation, but the probabilities for the two
		duplicate values will be considered as probabilities corresponding to
		two unique values.)  The elements of probability vectors P1 and P2 must
		each sum to 1 +/- .00001.

		kldiv(X,P1,P2,'sym') returns a symmetric variant of the Kullback-Leibler
		divergence, given by [KL(P1,P2)+KL(P2,P1)]/2 [1]_

		kldiv(X,P1,P2,'js') returns the Jensen-Shannon divergence, given by
		[KL(P1,Q)+KL(P2,Q)]/2, where Q = (P1+P2)/2.  See the Wikipedia article
		for "Kullback–Leibler divergence".  This is equal to 1/2 the so-called
		"Jeffrey divergence." [2]_

		References
		----------
		.. [1] Johnson, D.H. and S. Sinanovic. "Symmetrizing the Kullback-Leibler
		distance." IEEE Transactions on Information Theory (Submitted).
		.. [2] Rubner, Y., Tomasi, C., and Guibas, L. J., 2000. "The Earth Mover's
		distance as a metric for image retrieval." International Journal of
		Computer Vision, 40(2): 99-121.

		See Also
		--------
		Cover, T.M. and J.A. Thomas. "Elements of Information Theory," Wiley, 1991.

		https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

		Notes
		-----
		This function is taken from one on the Mathworks file exchange
		'''

		if not np.equal(np.unique(X), np.sort(X)).all():
			warnings.warn('X contains duplicate values. Treated as distinct values.',
						  UserWarning)
		if not np.equal(np.shape(X), np.shape(pvect1)).all() or not np.equal(np.shape(X), np.shape(pvect2)).all():
			warnings.warn('All inputs must have the same dimension.', UserWarning)
		if (np.abs(np.sum(pvect1) - 1) > 0.00001) or (np.abs(np.sum(pvect2) - 1) > 0.00001):
			warnings.warn('Probabilities don''t sum to 1.', UserWarning)
		if variant:
			if variant == 'js':
				logqvect = np.log2((pvect2 + pvect1) / 2)
				KL = 0.5 * (np.nansum(pvect1 * (np.log2(pvect1) - logqvect)) + np.sum(pvect2 * (np.log2(pvect2) - logqvect)))
				return KL
			elif variant == 'sym':
				KL1 = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
				KL2 = np.nansum(pvect2 * (np.log2(pvect2) - np.log2(pvect1)))
				KL = (KL1 + KL2) / 2
				return KL
			else:
				warnings.warn('Last argument not recognised', UserWarning)
		KL = np.nansum(pvect1 * (np.log2(pvect1) - np.log2(pvect2)))
		return KL

	def skaggsInfo(self, ratemap, dwelltimes):
		'''
		Calculates Skaggs information measure

		Parameters
		----------
		ratemap : numpy.ndarray
		dwelltimes: numpy.ndarray
			Must be same size as ratemap

		Returns
		-------
		bits_per_spike : float
			Skaggs information score

		Notes
		-----
		Returns Skaggs et al's estimate of spatial information in bits per spike:
			NB THIS DATA SHOULD UNDERGO ADAPTIVE BINNING - See adaptiveBin in binning class above
		I = sum_x p(x) r(x) log(r(x)/r)
		divided by mean rate over bins to get bits per spike
		Inputs:
			array of firing rates and dwell times per bin.
		Outputs:
			bits per spike
		binning could be over any single spatial variable (e.g. location, direction, speed).
		'''

		dwelltimes = dwelltimes / 50 # assumed sample rate of 50Hz
		if np.shape(ratemap) > 1:
			ratemap = np.reshape(ratemap,(np.prod(np.shape(ratemap)),1))
			dwelltimes = np.reshape(dwelltimes,(np.prod(np.shape(dwelltimes)),1))
		duration = np.nansum(dwelltimes)
		meanrate = np.nansum(ratemap * dwelltimes) / duration
		if meanrate <= 0.0:
			bits_per_spike = np.nan
			return bits_per_spike
		p_x = dwelltimes / duration
		p_r = ratemap / meanrate
		dum = p_x * ratemap
		ind = np.nonzero(dum)[0]
		bits_per_spike = np.nansum(dum[ind] * np.log2(p_r[ind]))
		bits_per_spike = bits_per_spike / meanrate
		return bits_per_spike