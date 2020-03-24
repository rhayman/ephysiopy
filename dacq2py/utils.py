# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:00:57 2012

@author: robin
"""
import numpy as np
import scipy.interpolate
import scipy.ndimage
import math
from scipy.signal import boxcar
from scipy.signal import convolve

def blur_image(im, n, ny=None, ftype='boxcar'):
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
			g = boxcar(n) / float(n)
		elif np.ndim(im) == 2:
			g = boxcar([n, ny]) / float(n)
	elif ftype == 'gaussian':
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		if np.ndim(im) == 1:
			g = g[n, :]
		g = g / g.sum()
	improc = convolve(im, g, mode='same')
	improc[nan_idx] = np.nan
	return improc 

def count_to(self, n):
	"""By example:

		#    0  1  2  3  4  5  6  7  8
		n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
		res = [0, 1, 2, 0, 1, 0, 1, 0]

	That is it is equivalent to something like this :

		hstack((arange(n_i) for n_i in n))

	This version seems quite a bit faster, at least for some
	possible inputs, and at any rate it encapsulates a task
	in a function.
	"""
	if n.ndim != 1:
		raise Exception("n is supposed to be 1d array.")

	n_mask = n.astype(bool)
	n_cumsum = np.cumsum(n)
	ret = np.ones(n_cumsum[-1]+1,dtype=int)
	ret[n_cumsum[n_mask]] -= n[n_mask]
	ret[0] -= 1
	return np.cumsum(ret)[:-1]

def repeat_ind(self, n):
	"""By example:

		#    0  1  2  3  4  5  6  7  8
		n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
		res = [2, 2, 2, 5, 5, 7, 7, 8]

	That is the input specifies how many times to repeat the given index.

	It is equivalent to something like this :

		hstack((zeros(n_i,dtype=int)+i for i, n_i in enumerate(n)))

	But this version seems to be faster, and probably scales better, at
	any rate it encapsulates a task in a function.
	"""
	if n.ndim != 1:
		raise Exception("n is supposed to be 1d array.")

	n_mask = n.astype(bool)
	n_inds = np.nonzero(n_mask)[0]
	n_inds[1:] = n_inds[1:]-n_inds[:-1] # take diff and leave 0th value in place
	n_cumsum = np.empty(len(n)+1,dtype=int)
	n_cumsum[0] = 0
	n_cumsum[1:] = np.cumsum(n)
	ret = np.zeros(n_cumsum[-1],dtype=int)
	ret[n_cumsum[n_mask]] = n_inds # note that n_mask is 1 element shorter than n_cumsum
	return np.cumsum(ret)

def rect(r, w, deg=0):
	"""
	Convert from polar (r,w) to rectangular (x,y)
	x = r cos(w)
	y = r sin(w)
	"""
	# radian if deg=0; degree if deg=1 
	if deg: w = np.pi * w / 180.0 
	return r * np.cos(w), r * np.sin(w)
	
def polar(x, y, deg=0): 
	""" Convert from rectangular (x,y) to polar (r,w) 
	r = sqrt(x^2 + y^2) 
	w = arctan(y/x) = [-\pi,\pi] = [-180,180]
	"""
	# radian if deg=0; degree if deg=1 
	if deg:
		return np.hypot(x, y), 180.0 * np.arctan2(y, x) / np.pi
	else:
		return np.hypot(x, y), np.arctan2(y, x)

def spiral(self, X, Y):
	'''
	Given an array of shape X x Y this returns the coordinates needed to step
	out from the centre of the array to the edge in a spiral fashion:
		see http://stackoverflow.com/questions/398299/looping-in-a-spiral?rq=1
		for original code and question/ solution(s)
	'''
	x = 0
	y = 0
	dx = 0
	dy = -1
	x_out = []
	y_out = []
	for i in range(max(X, Y)**2):
		x_out.append(x)
		y_out.append(y)
		if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
			dx, dy = -dy, dx
		x, y = x+dx, y+dy
		
	return np.array(x_out), np.array(y_out)
	
class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'

	def disable(self):
		self.HEADER = ''
		self.OKBLUE = ''
		self.OKGREEN = ''
		self.WARNING = ''
		self.FAIL = ''
		self.ENDC = ''