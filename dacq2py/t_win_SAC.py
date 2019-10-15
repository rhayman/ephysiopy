# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:57:17 2014

@author: Robin
"""

def t_win_SAC(self, xy, spkIdx, ppm = 365, winSize=10, pos_sample_rate=50, nbins=71, boxcar=5, Pthresh=100, downsampfreq=50, plot=False):
	'''
	[Stage 0] Get some numbers
	'''
	xy = xy / ppm * 100
	n_samps = pos.shape[1]
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
		plt.imshow(H, interpolation='nearest')
		plt.show()
	return H