import pathlib

import numpy as np

import ecofuture.preproc


def test_filename_parsing():

    example = "ga_ls_landcover_class_cyear_2_1-0-0_au_x9y-24_1993-01-01_level4"

    parsed = ecofuture.preproc.parse_filename(filename=pathlib.Path(example))

    assert parsed.year == 1993
    assert parsed.position.x == 9.0
    assert parsed.position.y == -24.0
    assert parsed.measurement == "level4"


def test_remapping():

    # pick a few random values from the level4 labels
    dea_data_eg = np.array([[27, 36], [103, 85]], dtype=np.uint8)

    # remap
    remapped_data = ecofuture.preproc.remap_data(data=dea_data_eg)

    assert remapped_data.shape == dea_data_eg.shape

    expected_remapped_data = np.array([[5, 12], [20, 16]], dtype=np.uint8)

    assert np.all(remapped_data == expected_remapped_data)
