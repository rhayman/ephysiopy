ephysiopy.common.statscalcs
===========================

.. py:module:: ephysiopy.common.statscalcs


Classes
-------

.. autoapisummary::

   ephysiopy.common.statscalcs.CircStatsResults
   ephysiopy.common.statscalcs.RegressionResults


Functions
---------

.. autoapisummary::

   ephysiopy.common.statscalcs.V_test
   ephysiopy.common.statscalcs.box_cox_normalize
   ephysiopy.common.statscalcs.ccc
   ephysiopy.common.statscalcs.ccc_jack
   ephysiopy.common.statscalcs.circCircCorrTLinear
   ephysiopy.common.statscalcs.circRegress
   ephysiopy.common.statscalcs.circ_r
   ephysiopy.common.statscalcs.duplicates_as_complex
   ephysiopy.common.statscalcs.mean_resultant_vector
   ephysiopy.common.statscalcs.rayleigh_test
   ephysiopy.common.statscalcs.shuffledPVal
   ephysiopy.common.statscalcs.watsonWilliams
   ephysiopy.common.statscalcs.watsonsU2
   ephysiopy.common.statscalcs.watsonsU2n
   ephysiopy.common.statscalcs.z_normalize


Module Contents
---------------

.. py:class:: CircStatsResults

   
   Dataclass to hold results from circular statistics
















   ..
       !! processed by numpydoc !!

   .. py:method:: __post_init__()


   .. py:method:: __repr__()


   .. py:attribute:: ci
      :type:  float


   .. py:attribute:: intercept


   .. py:attribute:: p
      :type:  float


   .. py:attribute:: p_shuffled
      :type:  float


   .. py:attribute:: rho
      :type:  float


   .. py:attribute:: rho_boot
      :type:  float


   .. py:attribute:: slope


.. py:class:: RegressionResults

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: phase
      :type:  numpy.ndarray


   .. py:attribute:: regressor
      :type:  numpy.ndarray


   .. py:attribute:: stats
      :type:  CircStatsResults


.. py:function:: V_test(angles, test_direction)

   
   The Watson U2 tests whether the observed angles have a tendency to
   cluster around a given angle indicating a lack of randomness in the
   distribution. Also known as the modified Rayleigh test.

   :param angles: Vector of angular values in degrees.
   :type angles: array_like
   :param test_direction: A single angular value in degrees.
   :type test_direction: int

   .. rubric:: Notes

   For grouped data the length of the mean vector must be adjusted,
   and for axial data all angles must be doubled.















   ..
       !! processed by numpydoc !!

.. py:function:: box_cox_normalize(scores, lam)

   
   Box-Cox normalize an array of scores.

   :param scores: The scores to normalize.
   :type scores: np.ndarray
   :param lam: The lambda parameter for Box-Cox transformation.
   :type lam: float

   :returns: The normalized scores.
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: ccc(t, p)

   
   Calculates correlation between two random circular variables

   :param t: The first variable
   :type t: np.ndarray
   :param p: The second variable
   :type p: np.ndarray

   :returns: The correlation between the two variables
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: ccc_jack(t, p)

   
   Function used to calculate jackknife estimates of correlation
   between two circular random variables

   :param t: The first variable
   :type t: np.ndarray
   :param p: The second variable
   :type p: np.ndarray

   :returns: The jackknife estimates of the correlation between the two variables
   :rtype: np.ndarray















   ..
       !! processed by numpydoc !!

.. py:function:: circCircCorrTLinear(theta, phi, regressor=1000, alpha=0.05, hyp=0, conf=True)

   
   An almost direct copy from AJs Matlab fcn to perform correlation
   between 2 circular random variables.

   Returns the correlation value (rho), p-value, bootstrapped correlation
   values, shuffled p values and correlation values.

   :param theta: The two circular variables to correlate (in radians)
   :type theta: np.ndarray
   :param phi: The two circular variables to correlate (in radians)
   :type phi: np.ndarray
   :param regressor: number of permutations to use to calculate p-value from randomisation
                     and bootstrap estimation of confidence intervals.
                     Leave empty to calculate p-value analytically (NB confidence
                     intervals will not be calculated).
   :type regressor: int, default=1000
   :param alpha: hypothesis test level e.g. 0.05, 0.01 etc.
   :type alpha: float, default=0.05
   :param hyp: hypothesis to test; -1/ 0 / 1 (-ve correlated / correlated in either
               direction / positively correlated).
   :type hyp: int, default=0
   :param conf: True or False to calculate confidence intervals via
                jackknife or bootstrap.
   :type conf: bool, default=True

   .. rubric:: References

   Fisher (1993), Statistical Analysis of Circular Data,
       Cambridge University Press, ISBN: 0 521 56890 0















   ..
       !! processed by numpydoc !!

.. py:function:: circRegress(x, t)

   
   Finds approximation to circular-linear regression for phase precession.

   :param x: The linear variable and the phase variable (in radians)
   :type x: np.ndarray
   :param t: The linear variable and the phase variable (in radians)
   :type t: np.ndarray

   .. rubric:: Notes

   Neither x nor t can contain NaNs, must be paired (of equal length).















   ..
       !! processed by numpydoc !!

.. py:function:: circ_r(alpha, w=None, d=0, axis=0)

   
   Computes the mean resultant vector length for circular data.

   :param alpha: Sample of angles in radians.
   :type alpha: array or list
   :param w: Counts in the case of binned data.
             Must be same length as alpha.
   :type w: array or list
   :param d: Spacing of bin centres for binned data; if
             supplied, correction factor is used to correct for bias in
             estimation of r, in radians.
   :type d: array or list, optional
   :param axis: The dimension along which to compute.
                Default is 0.
   :type axis: int, optional

   :returns: The mean resultant vector length.
   :rtype: r (float)















   ..
       !! processed by numpydoc !!

.. py:function:: duplicates_as_complex(x, already_sorted=False)

   
   Finds duplicates in x

   :param x: The list to find duplicates in.
   :type x: array_like
   :param already_sorted: Whether x is already sorted.
                          Default False.
   :type already_sorted: bool, optional

   :returns:

             A complex array where the complex part is the count of
                 the number of duplicates of the real value.
   :rtype: x (array_like)

   .. rubric:: Examples

   >>>     x = [9.9, 9.9, 12.3, 15.2, 15.2, 15.2]
   >>> ret = duplicates_as_complex(x)
   >>>     print(ret)
   [9.9+0j, 9.9+1j,  12.3+0j, 15.2+0j, 15.2+1j, 15.2+2j]















   ..
       !! processed by numpydoc !!

.. py:function:: mean_resultant_vector(angles)

   
   Calculate the mean resultant length and direction for angles.

   :param angles: Sample of angles in radians.
   :type angles: np.array

   :returns: The mean resultant vector length.
             th (float): The mean resultant vector direction.
   :rtype: r (float)

   Notes:
   Taken from Directional Statistics by Mardia & Jupp, 2000















   ..
       !! processed by numpydoc !!

.. py:function:: rayleigh_test(angles)

   
   Perform the Rayleigh test for uniformity of circular data.

   :param angles: Vector of angular values in radians.
   :type angles: array_like

   :returns: The Rayleigh test statistic.
             p_value (float): The p-value for the test.
   :rtype: Z (float)















   ..
       !! processed by numpydoc !!

.. py:function:: shuffledPVal(theta, phi, rho, regressor, hyp)

   
   Calculates shuffled p-values for correlation

   :param theta: The two circular variables to correlate (in radians)
   :type theta: np.ndarray
   :param phi: The two circular variables to correlate (in radians)
   :type phi: np.ndarray

   :returns: The shuffled p-value for the correlation between the two variables
   :rtype: float















   ..
       !! processed by numpydoc !!

.. py:function:: watsonWilliams(a, b)

   
   The Watson-Williams F test tests whether a set of mean directions are
   equal given that the concentrations are unknown, but equal, given that
   the groups each follow a von Mises distribution.

   :param a: The directional samples
   :type a: array_like
   :param b: The directional samples
   :type b: array_like

   :returns: The F-statistic
   :rtype: F_stat (float)















   ..
       !! processed by numpydoc !!

.. py:function:: watsonsU2(a, b)

   
   Tests whether two samples from circular observations differ significantly
   from each other with regard to mean direction or angular variance.

   :param a: The two samples to be tested
   :type a: array_like
   :param b: The two samples to be tested
   :type b: array_like

   :returns: The test statistic
   :rtype: U2 (float)

   .. rubric:: Notes

   Both samples must come from a continuous distribution. In the case of
   grouping the class interval should not exceed 5.
   Taken from '100 Statistical Tests' G.J.Kanji, 2006 Sage Publications















   ..
       !! processed by numpydoc !!

.. py:function:: watsonsU2n(angles)

   
   Tests whether the given distribution fits a random sample of angular
   values.

   :param angles: The angular samples.
   :type angles: array_like

   :returns: The test statistic.
   :rtype: U2n (float)

   .. rubric:: Notes

   This test is suitable for both unimodal and the multimodal cases.
   It can be used as a test for randomness.
   Taken from '100 Statistical Tests' G.J.Kanji, 2006 Sage Publications.















   ..
       !! processed by numpydoc !!

.. py:function:: z_normalize(scores)

   
   Z-normalize an array of scores.
















   ..
       !! processed by numpydoc !!

