import pytest
from ephysiopy.dacq2py.axona_headers import PosHeader
from ephysiopy.dacq2py.axona_headers import EEGHeader
from ephysiopy.dacq2py.axona_headers import EGFHeader
from ephysiopy.dacq2py.axona_headers import TetrodeHeader
from ephysiopy.dacq2py.axona_headers import SetHeader


def test_pos_header():
    P = PosHeader()
    P.print()


def test_eeg_header():
    E = EEGHeader()
    E.n_samples = 1000
    E.n_samples


def test_egf_header():
    E = EGFHeader()
    E.n_samples = 1000
    E.n_samples
    E.print()


def test_tetrode_header():
    T = TetrodeHeader()
    T.print()


def test_set_header():
    S = SetHeader()
    S.print()
