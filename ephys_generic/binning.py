import numpy as np
from scipy import signal
from astropy import convolution # deals with nans unlike other convs

class RateMap(object):
	"""
	Bins up positional data (xy, head direction etc) and produces rate maps
	of the relevant kind. This is a generic class meant to be independent of
	any particular recording format

	Parameters
	----------
	xy : array_like, optional
		The xy data, usually given as a 2 x n sample numpy array
	hdir : array_like, optional
		The head direction data, usualy a 1 x n sample numpy array
	speed : array_like, optional
		Similar to hdir
	pos_weights : array_like, optional
		A 1D numpy array n samples long which is used to weight a particular
		position sample when binning data. For example, if there were 5 positions
		recorded and a cell spiked once in position 2 and 5 times in position 3 and
		nothing anywhere else then pos_weights looks like: [0 0 1 5 0]
		In the case of binning up position this will be an array of mostly just 1's
		unless there are some positions you want excluded for some reason
	ppm : int, optional
		Pixels per metre. Specifies how many camera pixels per metre so this,
		in combination with cmsPerBin, will determine how many bins there are
		in the rate map
	xyInCms : bool, optional, default False
		Whether the positional data is in cms
	cmsPerBin : int, optional, default 3
		How many cms on a side each bin is in a rate map OR the number of degrees
		per bin in the case of directional binning
	smooth_sz : int, optional, default = 5
		The width of the smoothing kernel for smoothing rate maps

	Notes
	----
	There are several instance variables you can set, see below
	
	"""
	def __init__(self, xy=None, hdir=None, speed=None, pos_weights=None, ppm=430, xyInCms=False, cmsPerBin=3, smooth_sz=5):
		self.xy = xy
		self.dir = hdir
		self.speed = speed
		self.__pos_weights = pos_weights
		self.__ppm = ppm #pixels per metre
		self.__cmsPerBin = cmsPerBin
		self.__inCms = xyInCms
		self.__binsize__ = None # has setter and getter - see below
		self.__smooth_sz = smooth_sz
		self.__smoothingType = 'gaussian' # 'boxcar' or 'gaussian'
		self.whenToSmooth = 'before' # or 'after'

	@property
	def inCms(self):
		# Whether the units are in cms or not
		return self.__inCms

	@inCms.setter
	def inCms(self, value):
		self.__inCms = value

	@property
	def ppm(self):
		# Get the current pixels per metre (ppm)
		return self.__ppm

	@ppm.setter
	def ppm(self, value):
		self.__ppm = value
		self.__binsize__ = self.__calcBinSize(self.cmsPerBin)

	@property
	def binsize(self):
		# Returns binsize calculated in __calcBinSize and based on cmsPerBin
		if self.__binsize__ is None:
			try:
				self.__binsize__ = self.__calcBinSize(self.cmsPerBin)
			except AttributeError:
				self.__binsize__ = None
		return self.__binsize__

	@binsize.setter
	def binsize(self, value):
		self.__binsize__ = value

	@property
	def pos_weights(self):
		"""
		The 'weights' used as an argument to np.histogram* for binning up position
		Mostly this is just an array of 1's equal to the length of the pos
		data, but usefully can be adjusted when masking data in the trial
		by
		"""
		return self.__pos_weights

	@pos_weights.setter
	def pos_weights(self, value):
		self.__pos_weights = value

	@property
	def cmsPerBin(self):
		# The number of cms per bin of the binned up map
		return self.__cmsPerBin

	@cmsPerBin.setter
	def cmsPerBin(self, value):
		self.__cmsPerBin = value
		self.__binsize__ = self.__calcBinSize(self.cmsPerBin)

	@property
	def smooth_sz(self):
		# The size of the smoothing window applied to the binned data (1D or 2D)
		return self.__smooth_sz

	@smooth_sz.setter
	def smooth_sz(self, value):
		self.__smooth_sz = value

	@property
	def smoothingType(self):
		# The type of smoothing to do - legal values are 'boxcar' or 'gaussian'
		return self.__smoothingType

	@smoothingType.setter
	def smoothingType(self, value):
		self.__smoothingType = value

	@property
	def pixelsPerBin(self):
		# Calculates the number of camera pixels per bin of the binned data
		if getattr(self, 'inCms'):
			return getattr(self, 'cmsPerBin')
		else:
			return (getattr(self, 'ppm') / 100.) * getattr(self, 'cmsPerBin')

	def __calcBinSize(self, cmsPerBin=3):
		"""
		Aims to get the right number of bins for x and y dims given the ppm
		in the set header and the x and y extent

		Parameters
		----------
		cmsPerBin : int, optional, default = 3
			The number of cms per bin OR degrees in the case of directional binning
		"""
		x_lims = (np.min(self.xy[0]), np.max(self.xy[0]))
		y_lims = (np.min(self.xy[1]), np.max(self.xy[1]))
		ppb = getattr(self, 'pixelsPerBin')
		self.binsize = np.array((np.ceil(np.ptp(y_lims) / ppb)-1,
								 np.ceil(np.ptp(x_lims) / ppb)-1), dtype=np.int)
		return self.binsize

	def getMap(self, spkWeights, varType='xy', mapType='rate', smoothing=True):
		"""
		Bins up the variable type varType and returns a tuple of (rmap, binnedPositionDir) or
		(rmap, binnedPostionX, binnedPositionY)

		Parameters
		----------
		spkWeights : array_like
			Shape equal to number of positions samples captured and consists of
			position weights. For example, if there were 5 positions
			recorded and a cell spiked once in position 2 and 5 times in position 3 and
			nothing anywhere else then pos_weights looks like: [0 0 1 5 0]
		varType : str, optional, default 'xy'
			The variable to bin up. Legal values are: 'xy', 'dir', and 'speed'
		mapType : str, optional, default 'rate'
			If 'rate' then the binned up spikes are divided by varType. Otherwise return
			binned up position. Options are 'rate' or 'pos'
		smoothing : bool, optional, default True
			Whether to smooth the data or not

		Returns
		-------
		binned_data, binned_pos : tuple
			This is either a 2-tuple or a 3-tuple depening on whether binned pos
			(mapType is 'pos') or binned spikes (mapType is 'rate') is asked for,
			respectively

		"""
		sample = getattr(self, varType)
		assert(sample is not None) # might happen if head direction not supplied for example

		if 'xy' in varType:
			self.binsize = self.__calcBinSize(self.cmsPerBin)
		elif 'dir' in varType:
			self.binsize = np.arange(0, 360+self.cmsPerBin, self.cmsPerBin)
		elif 'speed' in varType:
			self.binsize = np.arange(0, 50, 1)

		binned_pos = self.__binData(sample, self.binsize, self.pos_weights)

		if binned_pos.ndim == 1: # directional binning
			binned_pos_edges = binned_pos[1]
			binned_pos = binned_pos[0]
		elif binned_pos.ndim == 2:
			binned_pos_edges = (binned_pos[1])
			binned_pos = binned_pos[0]
		elif len(binned_pos) == 3:
			binned_pos_edges = binned_pos[1:]
			binned_pos = binned_pos[0]
		nanIdx = binned_pos == 0

		if 'pos' in mapType: #return just binned up position
			if smoothing:
				if 'dir' in varType:
					binned_pos = self.__circPadSmooth(binned_pos, n=self.smooth_sz)
				else:
					binned_pos = self.blurImage(binned_pos, self.smooth_sz, ftype=self.smoothingType)
			return binned_pos, binned_pos_edges
		binned_spk = self.__binData(sample, self.binsize, spkWeights)[0]
		# binned_spk is returned as a tuple of the binned data and the bin
		# edges
		if 'after' in self.whenToSmooth:
			rmap = binned_spk[0] / binned_pos
			if 'dir' in varType:
				rmap = self.__circPadSmooth(rmap, self.smooth_sz)
			else:
				rmap = self.blurImage(rmap, self.smooth_sz, ftype=self.smoothingType)
		else: # default case
			if not smoothing:
				if len(binned_pos_edges) == 1: #directional map
					return binned_spk / binned_pos, binned_pos_edges
				elif len(binned_pos_edges) == 2:
					if isinstance(binned_spk, np.object): #__binData returns np.object when binning multiple clusters
						nClusters = spkWeights.shape[0]
						multi_binned_spks = np.zeros([self.binsize[0], self.binsize[1], nClusters])
						for i in range(nClusters):
							multi_binned_spks[:, :, i] = binned_spk[i]
						return multi_binned_spks / binned_pos[:, :, np.newaxis], binned_pos_edges[0], binned_pos_edges[1]
					else:
						return binned_spk / binned_pos, binned_pos_edges[0], binned_pos_edges[1]
			if 'dir' in varType:
				binned_pos = self.__circPadSmooth(binned_pos, self.smooth_sz)
				binned_spk = self.__circPadSmooth(binned_spk, self.smooth_sz)
				if spkWeights.ndim == 1:
					rmap = binned_spk / binned_pos
				elif spkWeights.ndim == 2:
					rmap = np.zeros([spkWeights.shape[0], binned_pos.shape[0]])
					for i in range(spkWeights.shape[0]):
						rmap[i, :] = binned_spk[i] / binned_pos
			else:
				if isinstance(binned_spk.dtype, np.object): #__binData returns np.object when binning multiple clusters
					binned_pos = self.blurImage(binned_pos, self.smooth_sz, ftype=self.smoothingType)
					if binned_spk.ndim == 2:
						pass
					elif (binned_spk.ndim == 3 or binned_spk.ndim == 1):
						binned_spk_tmp = np.zeros([binned_spk.shape[0], binned_spk[0].shape[0], binned_spk[0].shape[1]])
						for i in range(binned_spk.shape[0]):
							binned_spk_tmp[i, :, :] = binned_spk[i]
						binned_spk = binned_spk_tmp
					binned_spk = self.blurImage(binned_spk, self.smooth_sz, ftype=self.smoothingType)
					rmap = binned_spk / binned_pos
					if rmap.ndim <= 2:
						rmap[nanIdx] = np.nan
					elif rmap.ndim == 3:
						rmap[:,nanIdx] = np.nan

		return rmap, binned_pos_edges

	def blurImage(self, im, n, ny=None, ftype='boxcar'):
		"""
		Smooths a 2D image by convolving with a filter

		Parameters
		----------
		im : array_like
			The array to smooth
		n, ny : int
			The size of the smoothing kernel
		ftype : str
			The type of smoothing kernel. Either 'boxcar' or 'gaussian'

		Returns
		-------
		res: array_like
			The smoothed vector with shape the same as im
		"""
		n = int(n)
		if not ny:
			ny = n
		else:
			ny = int(ny)
		#  keep track of nans
		nan_idx = np.isnan(im)
		im[nan_idx] = 0
		g = signal.boxcar(n) / float(n)
		if ftype == 'boxcar':
			if im.ndim == 1:
				g = signal.boxcar(n) / float(n)
			elif im.ndim == 2:
				g = signal.boxcar(n) / float(n)
				g = np.tile(g, (1, ny, 1))
			elif im.ndim == 3: # mutlidimensional binning
				g = signal.boxcar([n, ny]) / float(n)
				g = g[None, :, :]
		elif ftype == 'gaussian':
			x, y = np.mgrid[-n:n+1, -ny:ny+1]
			g = np.exp(-(x**2/float(n) + y**2/float(ny)))
			g = g / g.sum()
			if np.ndim(im) == 1:
				g = g[n, :]
		improc = signal.convolve(im, g, mode='same')
		improc[nan_idx] = np.nan
		return improc

	def __binData(self, var, bin_edges, weights):
		"""
		Bins data taking account of possible multi-dimensionality

		Parameters
		----------
		var : array_like
			The variable to bin
		bin_edges : array_like
			The edges of the data - see numpys histogramdd for more
		weights : array_like
			The weights attributed to the samples in var
		
		Returns
		-------
		ndhist : 2-tuple
			Think this always returns a two-tuple of the binned variable and
			the bin edges - need to check to be sure...		

		Notes
		-----
		This breaks compatability with numpys histogramdd
		In the 2d histogram case below I swap the axes around so that x and y
		are binned in the 'normal' format i.e. so x appears horizontally and y
		vertically. 
		Multi-binning issue is dealt with awkwardly through checking
		the dimensionality of the weights array - 'normally' this would be 1 dim
		but when multiple clusters are being binned it will be 2 dim. In that case
		np.apply_along_axis functionality is applied. The spike weights in
		that case might be created like so:

		>>> spk_W = np.zeros(shape=[len(trial.nClusters), trial.npos])
		>>> for i, cluster in enumerate(trial.clusters):
		>>>		x1 = trial.getClusterIdx(cluster)
		>>>		spk_W[i, :] = np.bincount(x1, minlength=trial.npos)

		This can then be fed into this fcn something like so:

		>>> rng = np.array((np.ma.min(trial.POS.xy, 1).data, np.ma.max(rial.POS.xy, 1).data))
		>>> h = __binData(var=trial.POS.xy, bin_edges=np.array([64, 64]), weights=spk_W, rng=rng)

		Returned will be a tuple containing the binned up data and the bin edges for x and y (obv this will be the same for all
		entries of h)
		"""
		if weights is None:
			weights = np.ones_like(var)
		dims = weights.ndim
		orig_dims = weights.ndim
		if (dims == 1 and var.ndim == 1):
			var = var[np.newaxis, :]
			bin_edges = bin_edges[np.newaxis, :]
		elif (dims > 1 and var.ndim == 1):
			var = var[np.newaxis, :]
			bin_edges = bin_edges[np.newaxis, :]
		else:
			var = np.flipud(var)
		ndhist = np.apply_along_axis(lambda x: np.histogramdd(var.T, weights=x, bins=bin_edges), 0, weights.T)
		if ndhist.ndim == 1:
			if var.ndim == 2: # 1-dimenstional spike weights and xy
				return ndhist
		if ndhist.ndim == 2:
			# a single map has been asked for, pos, single map or dir
			return ndhist[0], ndhist[-1][0]
		elif ndhist.ndim == 1:
			if orig_dims == 1: # directional binning
				return ndhist
			# multi-dimensional binning
			result = np.zeros((len(ndhist[0]), ndhist[0][0].shape[0], ndhist[0][0].shape[1]))
			for i in range(len(ndhist)):
				result[i,:,:] = ndhist[0][i]
			return result, ndhist[::-1]


	def __circPadSmooth(self, var, n=3, ny=None):
		"""
		Smooths a vector by convolving with a gaussian
		Mirror reflects the start and end of the vector to
		deal with edge effects

		Parameters
		----------
		var : array_like
			The vector to smooth
		n, ny : int
			Size of the smoothing (sigma in gaussian)

		Returns
		-------
		res : array_like
			The smoothed vector with shape the same as var
		"""

		tn = len(var)
		t2 = int(np.floor(tn / 2))
		var = np.concatenate((var[t2:tn], var, var[0:t2]))
		if ny is None:
			ny = n
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		if np.ndim(var) == 1:
			g = g[n, :]
		g = g / g.sum()
		improc = signal.convolve(var, g, mode='same')
		improc = improc[tn-t2:tn-t2+tn]
		return improc

	def __circularStructure(self, radius):
		"""
		Generates a circular binary structure for use with morphological
		operations such as ndimage.binary_dilation etc

		This is only used in this implementation for adaptively binning
		ratemaps for use with information theoretic measures (Skaggs etc)

		Parameters
		----------
		radius : int
			the size of the circular structure

		Returns
		-------
		res : array_like
			Binary structure with shape [(radius*2) + 1,(radius*2) + 1]

		See Also
		--------
		RateMap.__adpativeMap
		"""
		crad = np.ceil(radius-0.5).astype(np.int)
		x, y = np.mgrid[-crad:crad+1, -crad:crad+1].astype(float)
		maxxy = np.maximum(abs(x), abs(y))
		minxy = np.minimum(abs(x), abs(y))

		m1 = ((radius ** 2 < (maxxy+0.5)**2 + (minxy-0.5)**2) * (minxy-0.5) +
			  (radius**2 >= (maxxy+0.5)**2 + (minxy-0.5)**2) *
			  np.real(np.sqrt(np.asarray(radius**2 - (maxxy + 0.5)**2,
										 dtype=complex))))
		m2 = ((radius**2 > (maxxy-0.5)**2 + (minxy+0.5)**2) * (minxy+0.5) +
			  (radius**2 <= (maxxy-0.5)**2 + (minxy+0.5)**2) *
			  np.real(np.sqrt(np.asarray(radius**2 - (maxxy - 0.5)**2,
										 dtype=complex))))

		sgrid = ((radius**2*(0.5*(np.arcsin(m2/radius) - np.arcsin(m1/radius)) +
			  0.25*(np.sin(2*np.arcsin(m2/radius)) - np.sin(2*np.arcsin(m1/radius)))) -
			 (maxxy-0.5)*(m2-m1) + (m1-minxy+0.5)) *
			 ((((radius**2 < (maxxy+0.5)**2 + (minxy+0.5)**2) &
			 (radius**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)) |
			 ((minxy == 0) & (maxxy-0.5 < radius) & (maxxy+0.5 >= radius)))) )

		sgrid = sgrid + ((maxxy+0.5)**2 + (minxy+0.5)**2 < radius**2)
		sgrid[crad,crad] = np.minimum(np.pi*radius**2,np.pi/2)
		if ((crad>0) and (radius > crad-0.5) and (radius**2 < (crad-0.5)**2+0.25)):
			m1  = np.sqrt(radius**2 - (crad - 0.5)**2)
			m1n = m1/radius
			sg0 = 2*(radius**2*(0.5*np.arcsin(m1n) + 0.25*np.sin(2*np.arcsin(m1n)))-m1*(crad-0.5))
			sgrid[2*crad,crad]   = sg0
			sgrid[crad,2*crad]   = sg0
			sgrid[crad,0]        = sg0
			sgrid[0,crad]        = sg0
			sgrid[2*crad-1,crad] = sgrid[2*crad-1,crad] - sg0
			sgrid[crad,2*crad-1] = sgrid[crad,2*crad-1] - sg0
			sgrid[crad,1]        = sgrid[crad,1]        - sg0
			sgrid[1,crad]        = sgrid[1,crad]        - sg0

		sgrid[crad,crad] = np.minimum(sgrid[crad,crad],1)
		kernel = sgrid/sgrid.sum()
		return kernel

	def __adaptiveMap(self, pos_binned, spk_binned, alpha=200):
		"""
		Produces a ratemap that has been adaptively binned according to the
		algorithm described in Skaggs et al., 1996) [1]_.

		Parameters
		----------
		pos_binned : array_like
			The binned positional data. For example that returned from getMap
			above with mapType as 'pos'
		spk_binned : array_like
			The binned spikes
		alpha : int, optional, default = 200
			A scaling parameter determing the amount of occupancy to aim at
			in each bin

		Returns
		-------
		Returns adaptively binned spike and pos maps. Use to generate Skaggs
		information measure

		Notes
		-----
		Positions with high rates mean proportionately less error than those
		with low rates, so this tries to even the playing field a bit. This is the
		kind of binning that should be used for calculations of spatial info
		as with the skaggsInfo method in the fieldcalcs class (see below)
		alpha is a scaling parameter that might need tweaking for different
		data sets.
		From the paper:
			The data [are] first binned
			into a 64 X 64 grid of spatial locations, and then the firing rate
			at each point in this grid was calculated by expanding a circle
			around the point until the following criterion was met:
				Nspks > alpha / (Nocc^2 * r^2)
			where Nspks is the number of spikes emitted in a circle of radius
			r (in bins), Nocc is the number of occupancy samples, alpha is the
			scaling parameter
			The firing rate in the given bin is then calculated as:
				sample_rate * (Nspks / Nocc)
		
		References
		----------
		.. [1] W. E. Skaggs, B. L. McNaughton, K. M. Gothard & E. J. Markus
			"An Information-Theoretic Approach to Deciphering the Hippocampal Code"
			Neural Information Processing Systems, 1993.
				
		"""
		#  assign output arrays
		smthdPos = np.zeros_like(pos_binned)
		smthdSpk = np.zeros_like(spk_binned)
		smthdRate = np.zeros_like(pos_binned)
		idx = pos_binned == 0
		pos_binned[idx] = np.nan
		spk_binned[idx] = np.nan
		visited = np.zeros_like(pos_binned)
		visited[pos_binned > 0] = 1
		# array to check which bins have made it
		binCheck = np.isnan(pos_binned)
		maxR = np.max(pos_binned.shape)
		r = 1
		while np.any(~binCheck):
			if r > (maxR / 2):
				smthdPos[~binCheck] = np.nan
				smthdSpk[~binCheck] = np.nan
				break
			# create the filter kernel
			h = self.__circularStructure(r)
			h[h >= np.max(h) / 3.0] = 1
			h[h != 1] = 0
			# filter the arrays using astropys convolution
			filtPos = convolution.convolve(pos_binned, h, boundary='None')
			filtSpk = convolution.convolve(spk_binned, h, boundary='None')
			filtVisited = convolution.convolve(visited, h, boundary='None')
			# get the bins which made it through this iteration
			trueBins = alpha / (np.sqrt(filtSpk) * filtPos) <= r
			trueBins = np.logical_and(trueBins, ~binCheck)
			# insert values where true
			smthdPos[trueBins] = filtPos[trueBins] / filtVisited[trueBins]
			smthdSpk[trueBins] = filtSpk[trueBins] / filtVisited[trueBins]
			binCheck[trueBins] = True
			r += 1
		smthdRate = smthdSpk / smthdPos
		smthdRate[idx] = np.nan
		smthdSpk[idx] = np.nan
		smthdPos[idx] = np.nan
		return smthdRate, smthdSpk, smthdPos