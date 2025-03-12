ephysiopy.common.statscalcs
===========================

.. py:module:: ephysiopy.common.statscalcs


Functions
---------

.. autoapisummary::

   ephysiopy.common.statscalcs.V_test
   ephysiopy.common.statscalcs.circ_r
   ephysiopy.common.statscalcs.duplicates_as_complex
   ephysiopy.common.statscalcs.mean_resultant_vector
   ephysiopy.common.statscalcs.watsonWilliams
   ephysiopy.common.statscalcs.watsonsU2
   ephysiopy.common.statscalcs.watsonsU2n


Module Contents
---------------

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

