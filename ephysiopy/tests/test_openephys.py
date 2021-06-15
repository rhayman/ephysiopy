from ephysiopy.openephys2py.OESettings import Settings


def test_settings(path_to_OE_settings):
    S = Settings(path_to_OE_settings)
    S.parse()


def test_oebin(path_to_OE_spikeSorter_settings):
    S = Settings(path_to_OE_spikeSorter_settings)
    S.parse()