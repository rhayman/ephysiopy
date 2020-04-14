"""
Calculation of the various metrics for quantifying the behaviour of grid cells
and some graphical output etc
"""

import numpy as np
import scipy, scipy.io, scipy.signal
import skimage, skimage.morphology, skimage.measure, skimage.feature, skimage.segmentation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ephysiopy.common.utils import rect, bwperim
from ephysiopy.common.binning import RateMap
from ephysiopy.common.ephys_generic import FieldCalcs
import collections

class SAC(object):
	"""
	Spatial AutoCorrelation (SAC) class
	"""
	def __init__(self):
		pass
	def autoCorr2D(self, A, nodwell, tol=1e-10):
		"""
		Performs a spatial autocorrelation on the array A

		Parameters
		----------
		A : array_like
			Either 2 or 3D. In the former it is simply the binned up ratemap 
			where the two dimensions correspond to x and y. 
			If 3D then the first two dimensions are x
			and y and the third (last dimension) is 'stack' of ratemaps
		nodwell : array_like
			A boolean array corresponding the bins in the ratemap that
			weren't visited. See Notes below.
		tol : float, optional
			Values below this are set to zero to deal with v small values
			thrown up by the fft. Default 1e-10

		Returns
		-------

		sac : array_like
			The spatial autocorrelation in the relevant dimensionality

		Notes
		-----
		In order to maintain backward compatibility I've kept this method here as a
		wrapper into ephysiopy.common.binning.RateMap.autoCorr2D()

		See Also
		--------
		ephysiopy.common.binning.RateMap.autoCorr2D()
		
		"""
		R = RateMap()
		return R.autoCorr2D(A, nodwell, tol)

		
	def crossCorr2D(self, A, B, A_nodwell, B_nodwell, tol=1e-10):
		"""
		Performs a spatial crosscorrelation between the arrays A and B

		Parameters
		----------
		A, B : array_like
			Either 2 or 3D. In the former it is simply the binned up ratemap 
			where the two dimensions correspond to x and y. 
			If 3D then the first two dimensions are x
			and y and the third (last dimension) is 'stack' of ratemaps
		nodwell_A, nodwell_B : array_like
			A boolean array corresponding the bins in the ratemap that
			weren't visited. See Notes below.
		tol : float, optional
			Values below this are set to zero to deal with v small values
			thrown up by the fft. Default 1e-10

		Returns
		-------

		sac : array_like
			The spatial crosscorrelation in the relevant dimensionality

		Notes
		-----
		In order to maintain backward compatibility I've kept this method here as a
		wrapper into ephysiopy.common.binning.RateMap.autoCorr2D()
		
		"""
		R = RateMap()
		return R.crossCorr2D(A, B, A_nodwell, B_nodwell, tol)

	def t_win_SAC(self, xy, spkIdx, ppm = 365, winSize=10, pos_sample_rate=50, nbins=71, boxcar=5, Pthresh=100, downsampfreq=50, plot=False):
		"""
		Temporal windowed spatial autocorrelation. For rationale see Notes below

		Parameters
		----------
		xy : array_like
			The position data
		spkIdx : array_like
			The indices in xy where the cell fired
		ppm : int, optional
			The camera pixels per metre. Default 365
		winSize : int, optional
			The window size for the temporal search
		pos_sample_rate : int, optional
			The rate at which position was sampled. Default 50
		nbins : int, optional
			The number of bins for creating the resulting ratemap. Default 71
		boxcar : int, optional
			The size of the smoothing kernel to smooth ratemaps. Default 5
		Pthresh : int, optional
			The cut=off for values in the ratemap; values < Pthresh become nans.
			Default 100
		downsampfreq : int, optional
			How much to downsample. Default 50
		plot : bool, optional
			Whether to show a plot of the result. Default False

		Returns
		-------
		H : array_like
			The temporal windowed SAC

		Notes
		-----
		In order to maintain backward compatibility I've kept this method here as a
		wrapper into ephysiopy.common.binning.RateMap.crossCorr2D()

		"""
		R = RateMap()
		return R.t_win_SAC(xy, spkIdx, ppm, winSize, pos_sample_rate, nbins, boxcar, Pthresh, downsampfreq, plot)
		
	def getMeasures(self, A, maxima='centroid', field_extent_method=2, allProps=True, **kwargs):
		"""
		Extracts various measures from a spatial autocorrelogram

		Parameters
		----------
		A : array_like
			The spatial autocorrelogram (SAC)
		maxima : str, optional
			The method used to detect the peaks in the SAC. 
			Legal values are 'single' and 'centroid'. Default 'centroid'
		field_extent_method : int, optional
			The method used to delimit the regions of interest in the SAC
			Legal values:
			* 1 - uses the half height of the ROI peak to limit field extent
			* 2 - uses a watershed method to limit field extent
			Default 2
		allProps : bool, optional
			Whether to return a dictionary that contains the attempt to fit an
			ellipse around the edges of the central size peaks. See below
			Default True
		
		Returns
		-------
		props : dict
			A dictionary containing measures of the SAC. The keys include things like:
			* gridness score
			* scale
			* orientation
			* the coordinates of the peaks (nominally 6) closest to the centre of the SAC
			* a binary mask that defines the extent of the 6 central fields around the centre
			* values of the rotation procedure used to calculate the gridness score
			* ellipse axes and angle (if allProps is True and the procedure worked)

		Notes
		-----
		In order to maintain backward comaptibility this is a wrapper for ephysiopy.common.ephys_generic.FieldCalcs.getGridFieldMeasures()

		See Also
		--------
		ephysiopy.common.ephys_generic.FieldCalcs.getGridFieldMeasures()

		"""
		F = FieldCalcs()
		return F.getGridFieldMeasures(A, maxima, field_extent_method, allProps, **kwargs)

	def getorientation(self, peakCoords, closestPeakIdx):
		"""
		Calculates the angle of the peaks working counter-clockwise from 3 o'clock

		Parameters
		----------
		peakCoords : array_like
			The peak coordinates as pairs of xy
		closestPeakIdx : array_like
			A 1D array of the indices in peakCoords of the peaks closest to the centre
			of the SAC
		
		Returns
		-------
		peak_orientations : array_like
			An array of the angles of the peaks in the SAC working counter-clockwise
			from a line extending from the middle of the SAC to 3 o'clock. The array
			is sorted from closest peak to the centre to the most distant
		"""
		if len(closestPeakIdx) == 1:
			return np.nan
		else:
			from .utils import polar
			closestPeaksCoordCentral = peakCoords[closestPeakIdx[1::]]
			theta = polar(closestPeaksCoordCentral[:,1], -closestPeaksCoordCentral[:,0], deg=1)[1]
			return np.sort(theta.compress(theta>0))[0]

	def getgridness(self, image, step=30):
		"""
		Calculates the gridness score in a grid cell spatial autocorrelogram (SAC).

		Briefly, the data in `image` is rotated in `step` amounts and each rotated array
		is correlated with the original. The maximum of the values at 30, 90 and 150 degrees
		is the subtracted from the minimum of the values at 60, 120 and 180 degrees to give the
		grid score.

		Parameters
		----------
		image : array_like
			The spatial autocorrelogram
		step : int, optional
			The amount to rotate the SAC by in each step of the rotational correlation
			procedure

		Returns
		-------
		gridmeasures : 3-tuple
			The gridscore, the correlation values at each `step` and the rotational array

		Notes
		-----
		The correlation performed is a Pearsons R. Some rescaling of the values in `image` is
		performed following rotation.

		See Also
		--------
		skimage.transform.rotate : for how the rotation of `image` is done
		skimage.exposure.rescale_intensity : for the resscaling following rotation

		"""
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
		"""
		Displays the result of performing a spatial autocorrelation (SAC) on a grid cell.

		Uses the dictionary containing measures of the grid cell SAC to make a pretty picture

		Parameters
		----------
		A : array_like
			The spatial autocorrelogram
		inDict : dict
			The dictionary calculated in getmeasures
		ax : matplotlib.axes._subplots.AxesSubplot, optional
			If given the plot will get drawn in these axes. Default None

		Returns
		-------
		ret : matplotlib.image.AxesImage
			The axes in which the SAC is shown

		See Also
		--------
		ephysiopy.common.binning.RateMap.autoCorr2D()
		ephysiopy.common.ephys_generic.FieldCalcs.getMeaures()
		"""
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
		"""
		Deforms a SAC that is non-circular to be more circular

		Basically a blatant attempt to improve grid scores, possibly introduced in
		a paper by Matt Nolan...

		Parameters
		----------
		A : array_like
			The SAC
		circleXY : array_like
			The xy coordinates defining a circle. See Notes
		ellipseXY : array_like
			The xy coordinates defining an ellipse

		Returns
		-------
		deformed_sac : array_like
			The SAC deformed to be more circular

		See Also
		--------
		skimage.transform.AffineTransform : for calculation of the affine transform
		skimage.transform.warp : for performance of the image warping
		skimage.exposure.rescale_intensity : for rescaling following deformation
		"""
		A[np.isnan(A)] = 0
		if circleXY.shape[0] == 2:
			circleXY = circleXY.T
		if ellipseXY.shape[0] == 2:
			ellipseXY = ellipseXY.T
		tform = skimage.transform.AffineTransform()
		tform.estimate(ellipseXY, circleXY)
		"""
		the transformation algorithms used here crop values < 0 to 0. Need to
		rescale the SAC values before doing the deformation and then rescale
		again so the values assume the same range as in the unadulterated SAC
		"""
		SACmin = np.nanmin(A.flatten())#should be 1
		SACmax = np.nanmax(A.flatten())
		AA = A + 1
		deformedSAC = skimage.transform.warp(AA / np.nanmax(AA.flatten()), inverse_map=tform.inverse, cval=0)
		return skimage.exposure.rescale_intensity(deformedSAC, out_range=(SACmin,SACmax))

	def __getcircleXY__(self, centre, radius):
		"""
		Calculates xy coordinate pairs that define a circle

		Parameters
		----------
		centre : array_like
			The xy coordinate of the centre of the circle
		radius : int
			The radius of the circle

		Returns
		-------
		circ : array_like
			100 xy coordinate pairs that describe the circle
		"""
		npts = 100
		t = np.linspace(0+(np.pi/4), (2*np.pi)+(np.pi/4), npts)
		r = np.repeat(radius, npts)
		x = r * np.cos(t) + centre[1]
		y = r * np.sin(t) + centre[0]
		return np.array((x,y))

	def __getellipseXY__(self, a, b, ang, im_centre):
		"""
		Calculates xy coordinate pairs that define an ellipse

		Parameters
		----------
		a, b : float
			The major and minor axes of the ellipse respectively
		ang : float
			The angle of orientation of the ellipse
		im_centre : array_like
			The xy coordinate of the centre of the ellipse

		Returns
		-------
		ellipse : array_like
			100 xy coordinate pairs that describe the ellipse
		"""
		pts = 100
		cos_a, sin_a = np.cos(ang), np.sin(ang)
		theta = np.linspace(0, 2*np.pi, pts)
		X = a*np.cos(theta)*cos_a - sin_a*b*np.sin(theta) + im_centre[1]
		Y = a*np.cos(theta)*sin_a + cos_a*b*np.sin(theta) + im_centre[0]
		return np.array((X,Y))

	def __fit_ellipse__(self, x, y):
		"""
		Does a best fits of an ellipse to the x/y coordinates provided

		Parameters
		----------
		x, y : array_like
			The x and y coordinates

		Returns
		-------
		a : array_like
			The xy coordinate pairs that fit
		"""
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
		"""
		Finds the centre of an ellipse

		Parameters
		----------
		a : array_like
			The values that describe the ellipse; major, minor axes etc

		Returns
		-------
		xy_centre : array_like
			The xy coordinates of the centre of the ellipse
		"""
		b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
		num = b*b-a*c
		x0=(c*d-b*f)/num
		y0=(a*f-b*d)/num
		return np.array([x0,y0])

	def __ellipse_angle_of_rotation__(self, a):
		"""
		Finds the angle of rotation of an ellipse

		Parameters
		----------
		a : array_like
			The values that describe the ellipse; major, minor axes etc

		Returns
		-------
		angle : array_like
			The angle of rotation of the ellipse
		"""
		b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
		return 0.5*np.arctan(2*b/(a-c))

	def __ellipse_axis_length__(self, a):
		"""
		Finds the axis length of an ellipse

		Parameters
		----------
		a : array_like
			The values that describe the ellipse; major, minor axes etc

		Returns
		-------
		axes_length : array_like
			The length of the major and minor axes (I think)
		"""
		b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
		_up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
		down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
		down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
		res1=np.sqrt(_up/np.abs(down1))
		res2=np.sqrt(_up/np.abs(down2))
		return np.array([res1, res2])

	def __findPeakExtent__(self, A, peakID, peakCoord):
		"""
		Finds extent of field that belongs to each peak.
		
		The extent is defined as the area that falls under the half-height.

		Parameters
		----------
		A : array_like
			The SAC
		peakID : array_like
			I think this is a list of the peak identities i.e. [1, 2, 3 etc]
		peakCoord : array_like
			xy coordinates into A that contain the full peaks

		Returns
		-------
		out : 2-tuple
			Consisting of the labelled peaks and their labelled perimeters
		"""
		peakLabel = np.zeros((A.shape[0], A.shape[1]))
		perimeterLabel = np.zeros_like(peakLabel)

		# define threshold to use - currently this is half-height
		halfHeight = A[peakCoord[1], peakCoord[0]] * .5
		aboveHalfHeightLabel = scipy.ndimage.label(A > halfHeight, structure=np.ones((3,3)))[0]
		peakIDTmp = aboveHalfHeightLabel[peakCoord[1], peakCoord[0]]
		peakLabel[aboveHalfHeightLabel == peakIDTmp] = peakID
		perimeterLabel[bwperim(aboveHalfHeightLabel==peakIDTmp)] = peakID
		return peakLabel, perimeterLabel
