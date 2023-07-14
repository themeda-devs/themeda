import pathlib

import pytest

import ecofuture.preproc.chiplets
import ecofuture.preproc.climate.dload
import ecofuture.preproc.climate.chiplets
import ecofuture.preproc.climate.preproc


TESTDATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "test_data"

DEA_MEASUREMENT = "level4"
DEA_CHIP_DIR = TESTDATA_PATH

REGION_FILE = TESTDATA_PATH / "savanna_region.geojson"

MEASUREMENTS = ["rain", "tmax"]


@pytest.fixture(scope="session")
def chip_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("chips")


@pytest.fixture(scope="session")
def chiplet_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("chiplets")


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("raw")


@pytest.fixture(scope="session", autouse=True)
def download(raw_dir):

    for measurement in MEASUREMENTS:
        ecofuture.preproc.climate.dload.download(
            output_dir=raw_dir,
            measurement=measurement,
            start_year=1988,
            end_year=1990,
            verbose=False,
        )


def test_chip_formation(raw_dir, chip_dir):

    for measurement in MEASUREMENTS:
        ecofuture.preproc.climate.chiplets.save_chips(
            raw_climate_dir=raw_dir,
            measurement=measurement,
            dea_chip_dir=DEA_CHIP_DIR,
            climate_chip_dir=chip_dir,
            overwrite=True,
        )


def test_saving(chip_dir, chiplet_dir):

    for measurement in MEASUREMENTS:
        ecofuture.preproc.chiplets.save_chiplets(
            chip_dir=chip_dir,
            chiplet_dir=chiplet_dir,
            region_file=REGION_FILE,
            measurement=measurement,
            show_progress=False,
            overwrite=True,
        )
