"""
A lot of the functionality here has been more generically implemented
in the ephys_generic.ephys_generic.SpikeCalcsGeneric class
"""
import numpy as np
import warnings
from scipy import signal		
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from ephysiopy.common.utils import blur_image

"""
NB AS OF 19/10/20 I MOVED MOST OF THE METHODS OF THE CLASSSPIKECALCS
INTO EPHYSIOPY.COMMON.EPHYS_GENERIC.SPIKECALCSGENERIC

THE ONES REMAINING BELOW INVOLVE COMBINING SPIKES AND POSITION
INFORMATION SO SHOULD BE IN SOME OTHER PLACE...
"""

class SpikeCalcs(object):
	"""
	Mix-in class for use with Tetrode class below.
	
	Extends Tetrodes functionality by adding methods for analysis of spikes/
	spike trains
				
	Note lots of the methods here are native to dacq2py.axonaIO.Tetrode
	
	Note that units are in milliseconds
	"""
	
	


	
	