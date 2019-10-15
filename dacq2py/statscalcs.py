"""
Created on Mon Jun 18 18:31:31 2012

@author: robin
"""
import numpy as np

class StatsCalcs():
	def __init__(self):
		pass
	def circ_r(self, alpha, w=None, d=0, axis=0):
		"""
		Computes the mean resultant vector length for circular data
		
		Parameters
		----------
		alpha: array or list
			sample of angles in radians
		w: array or list
			counts in the case of binned data. Must be same length as alpha
		d: array or list
			spacing of bin centres for binned data; if supplied, correction
			factor is used to correct for bias in estimation of r, in radians
		axis: int
			the dimension along which to compute, Default is 0

		Returns
		-------
		r: float
			the mean resultant vector length
			
		References
		----------
		Statistical analysis of circular data, N.I.Fisher
		Topics in circular statistics, S.R.Jamalamadaka et al.
		Biostatistical Analysis, J.H.Zar
		"""
		if w is None:
			w = np.ones_like(alpha, dtype=float)
		else:
			assert(len(alpha) == len(w))
		#TODO: error check for size constancy
		r = np.sum(w * np.exp(1j * alpha))
		r = np.abs(r) / np.sum(w)
		if d != 0:
			c = d/2./np.sin(d/2.)
			r = c * r
		return r
		
	def mean_resultant_vector(self, angles):
		'''
		Calculate the mean resultant length and direction for angles
		
		Parameters
		----------
		angles: np.array
			sample of angles in radians
		
		Returns
		-------
		r: float
			the mean resultant vector length
		th: float
			the mean resultant vector direction
		'''
		S = np.sum(np.sin(angles)) * (1/float(len(angles)))
		C = np.sum(np.cos(angles)) * (1/float(len(angles)))
		r = np.hypot(S, C)
		th = np.arctan(S / C)
		if (C < 0):
			th = np.pi + th
		return r, th
		
	def V_test(self, angles, test_direction):
		"""
		Taken from '100 Statistical Tests' G.J.Kanji, 2006 Sage Publications
		Test is also known as the modified Rayleigh test

		angles is a vector of angular values in degrees
		test_direction is a single angular value in degrees
		The Watson U2 tests whether the observed angles have a tendency to
		cluster around a given angle indicating a lack of randomness in the
		distribution
		Limitations:
			For grouped data the length of the mean vector must be adjusted,
			and for axial data all angles must be doubled.
		"""
		n = len(angles)
		x_hat = np.sum(np.cos(np.radians(angles))) / float(n)
		y_hat = np.sum(np.sin(np.radians(angles))) / float(n)
		r = np.sqrt(x_hat**2 + y_hat**2)
		theta_hat = np.degrees(np.arctan(y_hat / x_hat))
		if theta_hat < 0:
			theta_hat = theta_hat + 360
		v_squiggle = r * np.cos(np.radians(theta_hat) - np.radians(test_direction))
		V = np.sqrt(2 * n) * v_squiggle
		return V

	def duplicates_as_complex(self, x, already_sorted=False):
		""" Example:
			x = [9.9    9.9     12.3    15.2    15.2    15.2    ]
			ret=[9.9+0j 9.9+1j  12.3+0j 15.2+0j 15.2+1j 15.2+2j ]
		"""
		if not already_sorted:
			x = np.sort(x)
		is_start = np.empty(len(x),dtype=bool)
		is_start[0], is_start[1:] = True, x[:-1] != x[1:]
		labels = np.cumsum(is_start)-1
		sub_idx = np.arange(len(x)) - is_start.nonzero()[0][labels]
		return x + 1j*sub_idx

	def watsonsU2(self, a, b):
		'''
		Taken from '100 Statistical Tests' G.J.Kanji, 2006 Sage Publications
		To test whether two samples from circular observations differ
		significantly from each other with regard to mean direction or angular
		variance.
		Limitation
		Both samples must come from a continuous distribution. In the case of
		grouping the class interval should not exceed 5.
		'''
		a = np.sort(np.ravel(a))
		b = np.sort(np.ravel(b))
		n_a = len(a)
		n_b = len(b)
		N = float(n_a + n_b)
		a_complex, b_complex = self.duplicates_as_complex(a, True), self.duplicates_as_complex(b, True)
		a_and_b = np.union1d(a_complex,b_complex)

		# get index for a
		a_ind = np.zeros(len(a_and_b),dtype=int)
		a_ind[np.searchsorted(a_and_b,a_complex)] = 1
		a_ind = np.cumsum(a_ind)

		# same for b
		b_ind = np.zeros(len(a_and_b),dtype=int)
		b_ind[np.searchsorted(a_and_b,b_complex)] = 1
		b_ind = np.cumsum(b_ind)

		d_k = (a_ind / float(n_a)) - (b_ind / float(n_b))

		d_k_sq = d_k ** 2

		U2 = ((n_a*n_b) / N**2) * (np.sum(d_k_sq) - ((np.sum(d_k)**2) / N))
		return U2


	def watsonsU2n(self, angles):
		"""
		Taken from '100 Statistical Tests' G.J.Kanji, 2006 Sage Publications

		To test whether the given distribution fits a random sample of angular values
		Limitations:
			This test is suitable for both unimodal and the multimodal cases.
			It can be used as a test for randomness.
		"""
		angles = np.sort(angles)
		n = len(angles)
		Vi = angles / float(360)
		sum_Vi = np.sum(Vi)
		sum_sq_Vi = np.sum(Vi**2)
		Ci = (2 * np.arange(1, n+1)) - 1
		sum_Ci_Vi_ov_n = np.sum(Ci * Vi / n)
		V_bar = (1 / float(n)) * sum_Vi
		U2n = sum_sq_Vi - sum_Ci_Vi_ov_n + (n * (1/float(3) - (V_bar - 0.5)**2))
		test_vals = {'0.1': 0.152, '0.05': 0.187, '0.025': 0.221, '0.01': 0.267, '0.005': 0.302}
		for key, val in test_vals.iteritems():
			if U2n > val:
				print('The Watsons U2 statistic is {0} which is greater than\n the critical value of {1} at p={2}'.format(U2n, val, key))
			else:
				print('The Watsons U2 statistic is not significant at p={0}'.format(key))
				return

	def watsonWilliams(self, a, b):
		n = len(a)
		m = len(b)
		N = n + m
		#v_1 = 1 # needed to do p-value lookup in table of critical values of F distribution
		#v_2 = N - 2 # needed to do p-value lookup in table of critical values of F distribution
		C_1 = np.sum(np.cos(np.radians(a)))
		S_1 = np.sum(np.sin(np.radians(a)))
		C_2 = np.sum(np.cos(np.radians(b)))
		S_2 = np.sum(np.sin(np.radians(b)))
		C = C_1 + C_2
		S = S_1 + S_2
		R_1 = np.hypot(C_1, S_1)
		R_2 = np.hypot(C_2, S_2)
		R = np.hypot(C, S)
		R_hat = (R_1 + R_2) / float(N)
		import os
		fid = os.path.join(os.path.getcwd(), 'mle_von_mises_vals.txt')
		with open(fid, 'r') as f:
			mle_von_mises = np.loadtxt(f)
		mle_von_mises = np.sort(mle_von_mises, 0)
		k_hat = mle_von_mises[(np.abs(mle_von_mises[:,0]-R_hat)).argmin(), 1]
		g = 1 - (3 / 8 * k_hat)
		F = g * (N-2) * ((R_1 + R_2 - R) / (N - (R_1 + R_2)))
		return F