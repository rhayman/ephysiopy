"""

.. include:: README.md

"""
__version__ = '1.5.50'
import mahotas # this is to get around a weird, possibly python3.6 related issue
kk_path = '/home/robin/klustakwik/KlustaKwik'
# used to bandpass filter the continuous raw openephys data when converting to
# Axona eeg/ egf format
lfp_lowcut = 1
lfp_highcut = 500
fs = 30000 # sampling rate of continuous data in openephys