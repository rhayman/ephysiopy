# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:20:36 2012

@author: robin
"""
import numpy as np
from scipy import ndimage

def normalize(points):
	""" Normalize a collection of points in
	homogeneous coordinates so that last row = 1. """
	for row in points:
		row /= points[-1]
	return points

def make_homog(points):
	""" Convert a set of points (dim*n array) to
	homogeneous coordinates. """
	return np.vstack((points,np.ones((1,points.shape[1]))))

def H_from_points(fp,tp):
	""" Find homography H, such that fp is mapped to tp
	using the linear DLT method. Points are conditioned
	automatically. """
	if fp.shape != tp.shape:
		raise RuntimeError("number of points do not match")
	# condition points (important for numerical reasons)
	# --from points--
	m = np.mean(fp[:2], axis=1)
	maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
	C1 = np.diag([1/maxstd, 1/maxstd, 1])
	C1[0][2] = -m[0]/maxstd
	C1[1][2] = -m[1]/maxstd
	fp = np.dot(C1,fp)
	# --to points--
	m = np.mean(tp[:2], axis=1)
	maxstd = max(np.std(tp[:2], axis=1)) + 1e-9
	C2 = np.diag([1/maxstd, 1/maxstd, 1])
	C2[0][2] = -m[0]/maxstd
	C2[1][2] = -m[1]/maxstd
	tp = np.dot(C2,tp)
	# create matrix for linear method, 2 rows for each correspondence pair
	nbr_correspondences = fp.shape[1]
	A = np.zeros((2*nbr_correspondences,9))
	for i in range(nbr_correspondences):
		A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
			  tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
		A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
			  tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]
	U,S,V = np.linalg.svd(A)
	H = V[8].reshape((3,3))
	# decondition
	H = np.dot(np.linalg.inv(C2),np.dot(H,C1))
	# normalize and return
	return H / H[2,2]

def Haffine_from_points(pts8):
	""" find H, affine transformation, such that
		tp is affine transf of fp"""
	assert len(pts8) == 8

	tp = pts8[0:4];fp = pts8[4:8]

	#condition points
	#-from points-
	m = np.mean(fp[:2], axis=1)
	maxstd = max(np.std(fp[:2], axis=1))
	C1 = np.diag([1/maxstd, 1/maxstd, 1])
	C1[0][2] = -m[0]/maxstd
	C1[1][2] = -m[1]/maxstd
	fp_cond = np.dot(C1,fp)

	#-to points-
	m = np.mean(tp[:2], axis=1)
	C2 = C1.copy() #must use same scaling for both point sets
	C2[0][2] = -m[0]/maxstd
	C2[1][2] = -m[1]/maxstd
	tp_cond = np.dot(C2,tp)

	#conditioned points have mean zero, so translation is zero
	A = np.concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
	U,S,V = np.linalg.svd(A.T)

	#create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
	tmp = V[:2].T
	B = tmp[:2]
	C = tmp[2:4]

	tmp2 = np.concatenate((np.dot(C,np.linalg.pinv(B)),np.zeros((2,1))), axis=1)
	H = np.vstack((tmp2,[0,0,1]))

	#decondition
	H = np.dot(np.linalg.inv(C2),np.dot(H,C1))

	return H / H[2][2]

def image_in_image(im1,im2,tp):
	""" put im1 in im2 with an affine transformation
		such that corners are as close to tp as possible.
		tp are homogeneous and counter-clockwise from top left."""

	#points to warp from
	m,n = im1.shape[:2]
	fp = np.array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

	#compute affine transform and apply
	H = Haffine_from_points(tp,fp)
	im1_t = ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])
	alpha = (im1_t > 0)

	return (1-alpha)*im2 + alpha*im1_t

def get_transform_data(pts8, backward=True ):
	'''This method returns a perspective transform 8-tuple (a,b,c,d,e,f,g,h).

	Use to transform an image:
	X = (a x + b y + c)/(g x + h y + 1)
	Y = (d x + e y + f)/(g x + h y + 1)

	Image.transform: Use 4 source coordinates, followed by 4 corresponding
		destination coordinates. Use backward=True (the default)

	To calculate the destination coordinate of a single pixel, either reverse
		the pts (4 dest, followed by 4 source, backward=True) or use the same
		pts but set backward to False.

	@arg pts8: four source and four corresponding destination coordinates
	@kwarg backward: True to return coefficients for calculating an originating
		position. False to return coefficients for calculating a destination
		coordinate. (Image.transform calculates originating position.)
	'''
	assert len(pts8) == 8, 'Requires a tuple of eight coordinate tuples (x,y)'

	b0,b1,b2,b3,a0,a1,a2,a3 = pts8 if backward else pts8[::-1]

	# CALCULATE THE COEFFICIENTS
	A = np.array([[a0[0], a0[1], 1,     0,     0, 0, -a0[0]*b0[0], -a0[1]*b0[0]],
			   [    0,     0, 0, a0[0], a0[1], 1, -a0[0]*b0[1], -a0[1]*b0[1]],
			   [a1[0], a1[1], 1,     0,     0, 0, -a1[0]*b1[0], -a1[1]*b1[0]],
			   [    0,     0, 0, a1[0], a1[1], 1, -a1[0]*b1[1], -a1[1]*b1[1]],
			   [a2[0], a2[1], 1,     0,     0, 0, -a2[0]*b2[0], -a2[1]*b2[0]],
			   [    0,     0, 0, a2[0], a2[1], 1, -a2[0]*b2[1], -a2[1]*b2[1]],
			   [a3[0], a3[1], 1,     0,     0, 0, -a3[0]*b3[0], -a3[1]*b3[0]],
			   [    0,     0, 0, a3[0], a3[1], 1, -a3[0]*b3[1], -a3[1]*b3[1]]] )

	B = np.array([b0[0], b0[1], b1[0], b1[1], b2[0], b2[1], b3[0], b3[1]])

	C = np.linalg.solve(A, B)
	return C

def transform_pt(pt , coeffs ):
	T = coeffs
	x = (T[0]*pt[0] + T[1]*pt[1] + T[2])/(T[6]*pt[0] + T[7]*pt[1] + 1)
	y = (T[3]*pt[0] + T[4]*pt[1] + T[5])/(T[6]*pt[0] + T[7]*pt[1] + 1)
	return x,y