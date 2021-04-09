import os
try:
    kk_path = os.path.join(os.environ.get('HOME'), 'klustakwik', 'Klustakwik')
except Exception:
    kk_path = None
# used to bandpass filter the continuous raw openephys data when converting to
# Axona eeg/ egf format
lfp_lowcut = 1
lfp_highcut = 500
fs = 30000  # sampling rate of continuous data in openephys
