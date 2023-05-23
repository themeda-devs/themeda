import pathlib
import random
import collections

import numpy as np

import pytest

import ecofuture.preproc


TESTDATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "test_data"


@pytest.fixture
def chip_dir():
    return TESTDATA_PATH


@pytest.fixture
def region_file():
    return TESTDATA_PATH / "savanna_region.geojson"


@pytest.fixture(scope="session")
def chiplet_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("chiplets")


def test_saving(chip_dir, chiplet_dir, region_file):
    ecofuture.preproc.save_chiplets(
        chip_dir=chip_dir,
        chiplet_dir=chiplet_dir,
        region_file=region_file,
    )


def test_loading(chiplet_dir):

    chiplets = ecofuture.preproc.load_chiplets(chiplet_dir=chiplet_dir)

    for _ in chiplets:
        pass


def test_loading_year(chiplet_dir):

    test_year = 1989

    chiplets = ecofuture.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        years=[test_year],
    )

    for chiplet in chiplets:
        assert chiplet.year == test_year


def test_loading_subsets(chiplet_dir):

    subset_nums = [2, 4]

    chiplets = ecofuture.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        subset_nums=subset_nums,
    )

    for chiplet in chiplets:
        assert chiplet.subset_num in subset_nums


def test_loading_subset_instance_nums(chiplet_dir):

    subset_instance_nums = [20, 10, 56, 64]

    # just do it for a single year/subset so can test the ordering more easily
    chiplets = ecofuture.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        years=[1989],
        subset_nums=[3],
        subset_instance_nums=subset_instance_nums,
    )

    for (expected_subset_instance_num, chiplet) in zip(
        subset_instance_nums, chiplets
    ):
        assert chiplet.subset_instance_num == expected_subset_instance_num


def test_subset_nums_over_years(chiplet_dir):

    # check that each position is associated with the same subset info
    # across years

    pos_items = collections.defaultdict(list)

    chiplets = ecofuture.preproc.load_chiplets(chiplet_dir=chiplet_dir)

    for chiplet in chiplets:
        pos_items[chiplet.position].append(chiplet)

    for pos_item in pos_items.values():

        assert len(pos_item) > 1

        # there is to be only one unique subset number and subset instance
        # number across the years
        for attr in ["subset_num", "subset_instance_num"]:
            assert len(
                set(getattr(curr_pos_item, attr) for curr_pos_item in pos_item)
            ) == 1

    return pos_items


def test_against_manual(chip_dir, chiplet_dir, region_file):

    # need to test that the chosen TIFF file is within the region
    region = ecofuture.preproc.get_region(region_file=region_file)

    # pick a TIFF file
    avail_chip_paths = list(chip_dir.glob("*.tif*"))
    random.shuffle(avail_chip_paths)

    for chip_path in avail_chip_paths:

        chip = ecofuture.preproc.read_chip(filename=chip_path)
        chip_bbox = ecofuture.preproc.get_bbox(data=chip)
        all_chip_within_region = ecofuture.preproc.is_bbox_within_region(
            bbox=chip_bbox, region=region
        )

        if all_chip_within_region:
            break

    chip_meta = ecofuture.preproc.parse_chip_filename(filename=chip_path)
    chip = ecofuture.preproc.read_chip(filename=chip_path, load_data=True)

    chiplets = [
        chiplet
        for chiplet in ecofuture.preproc.load_chiplets(
            chiplet_dir=chiplet_dir,
            years=[chip_meta.year],
            measurement=chip_meta.measurement,
        )
    ]

    # sniff the size of each chiplet
    (chiplet_size, _) = chiplets[0].data.shape

    chip_size = chip.sizes["x"]

    n_chiplets_per_dim = int(chip_size / chiplet_size)

    n_rand = 10

    # just randomly pick some offsets to manually confirm
    for iteration in range(n_rand):

        i_x_offset = random.randint(0, n_chiplets_per_dim - 1) * chiplet_size
        i_y_offset = random.randint(0, n_chiplets_per_dim - 1) * chiplet_size

        # manually extract the subset from the chip
        chip_subset = chip.isel(
            x=range(i_x_offset, i_x_offset + chiplet_size),
            y=range(i_y_offset, i_y_offset + chiplet_size),
        )

        chip_pos = ecofuture.preproc.Position(
            x=chip_subset.x.mean().item(),
            y=chip_subset.y.mean().item(),
        )

        # remap
        chip_data = ecofuture.preproc.remap_data(data=chip_subset.data)

        # find the matching chiplet
        (matching_chiplet,) = [
            chiplet
            for chiplet in chiplets
            if chiplet.position == chip_pos
        ]

        # confirm that the data are the same
        assert np.all(chip_data == matching_chiplet.data)


def test_chip_filename_parsing():

    example = "ga_ls_landcover_class_cyear_2_1-0-0_au_x9y-24_1993-01-01_level4"

    parsed = ecofuture.preproc.parse_chip_filename(filename=pathlib.Path(example))

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
