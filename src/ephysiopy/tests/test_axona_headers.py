from ephysiopy.axona.file_headers import (
    EEGHeader,
    EGFHeader,
    PosHeader,
    SetHeader,
    TetrodeHeader,
)


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
