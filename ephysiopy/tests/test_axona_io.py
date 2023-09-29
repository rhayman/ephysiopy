import os
import numpy as np
import pytest
from ephysiopy.axona.axonaIO import IO
from ephysiopy.axona.file_headers import PosHeader
from pathlib import Path, PurePath


def test_io(path_to_axona_data):
    fname_root = os.path.splitext(path_to_axona_data)[0]
    io = IO(fname_root)
    data = io.getData(fname_root + '.eeg')
    assert isinstance(data, np.ndarray)
    data = io.getData(fname_root + '.1')
    assert isinstance(data, np.ndarray)
    with pytest.raises(IOError):
        io.getData(fname_root + '.blurt')


def test_get_cut(path_to_axona_data):
    fname_root = PurePath(os.path.splitext(path_to_axona_data)[0])
    io = IO(fname_root)
    data = io.getCut(1)
    assert isinstance(data, list)
    data = io.getCluCut(1)
    assert isinstance(data, np.ndarray)
    # a file that doesn't exist
    nothing = io.getCut(100)
    assert nothing is None


def test_save_data(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.pos"
    pos_header = PosHeader()
    io = IO()
    io.setHeader(str(p), pos_header)
    data_to_save = np.random.randint(0, 10, (8, 100))
    io.setData(str(p), data_to_save)


def test_read_data(path_to_axona_data):
    from ephysiopy.axona.axonaIO import Pos
    fname_root = Path(os.path.splitext(path_to_axona_data)[0])
    pos_data = Pos(fname_root)
    pos_data.ppm
    pos_data.ppm = 300
    Pos(fname_root, cm=True)


def test_tetrode_io(path_to_axona_data):
    from ephysiopy.axona.axonaIO import Tetrode
    fname_root = Path(os.path.splitext(path_to_axona_data)[0])
    # tetrode 1 has a cut and clu file, tetrode 2
    # only has a clu file and tetrode 3 has neither
    tets = [1, 2]
    for t in tets:
        tetrode = Tetrode(fname_root, t)
        spk_ts = tetrode.getSpkTS()
        assert isinstance(spk_ts, np.ndarray)
        clust_ts = tetrode.getClustTS(1)
        assert isinstance(clust_ts, np.ndarray)
        pos_samples = tetrode.getPosSamples()
        assert isinstance(pos_samples, np.ndarray)
        clust_spk_ts = tetrode.getClustSpks(1)
        assert isinstance(clust_spk_ts, np.ndarray)
        clust_idx = tetrode.getClustIdx(1)
        assert isinstance(clust_idx, np.ndarray)
        unique_clusts = tetrode.getUniqueClusters()
        assert isinstance(unique_clusts, np.ndarray)
    tetrode.getClustTS()
    tetrode.cut = None
    tetrode.getClustTS(1)
    tetrode.getClustTS(2)
    tetrode.cut = None
    tetrode.getClustSpks(1)
    tetrode.cut = None
    tetrode.pos_samples = None
    tetrode.getClustIdx(1)
    tetrode.cut = None
    tetrode.getUniqueClusters()


def test_eeg_io(path_to_axona_data):
    from ephysiopy.axona.axonaIO import EEG
    fname_root = os.path.splitext(path_to_axona_data)[0]
    EEG(fname_root)
    EEG(fname_root, egf=1)


def test_stim_io(path_to_axona_data):
    from ephysiopy.axona.axonaIO import Stim
    fname_root = os.path.splitext(path_to_axona_data)[0]
    stim = Stim(fname_root)
    stim.update(foo=2)
    with pytest.raises(KeyError):
        stim['blurt']
