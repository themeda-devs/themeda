import pathlib
import random
import collections

import numpy as np

import pytest

import themeda.preproc.roi
import themeda.preproc.chips
import themeda.preproc.chiplets
import themeda.preproc.dea.preproc


TESTDATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "test_data"

MEASUREMENT = "level4"


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
    themeda.preproc.dea.preproc.save_chiplets(
        chip_dir=chip_dir,
        chiplet_dir=chiplet_dir,
        region_file=region_file,
    )


def test_loading(chiplet_dir):

    chiplets = themeda.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        measurement=MEASUREMENT,
    )

    for _ in chiplets:
        pass


def test_loading_year(chiplet_dir):

    test_year = 1989

    chiplets = themeda.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        years=[test_year],
        measurement=MEASUREMENT,
    )

    for chiplet in chiplets:
        assert chiplet.year == test_year

    chiplets = themeda.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        exclude_years=[test_year],
        measurement=MEASUREMENT,
    )

    for chiplet in chiplets:
        assert chiplet.year != test_year


def test_loading_subsets(chiplet_dir):

    subset_nums = [2, 4]

    chiplets = themeda.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        subset_nums=subset_nums,
        measurement=MEASUREMENT,
    )

    for chiplet in chiplets:
        assert chiplet.subset_num in subset_nums

    chiplets = themeda.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        exclude_subset_nums=subset_nums,
        measurement=MEASUREMENT,
    )

    for chiplet in chiplets:
        assert chiplet.subset_num not in subset_nums


def test_loading_subset_instance_nums(chiplet_dir):

    subset_instance_nums = [20, 10, 56, 64]

    # just do it for a single year/subset so can test the ordering more easily
    chiplets = themeda.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        years=[1989],
        subset_nums=[3],
        subset_instance_nums=subset_instance_nums,
        measurement=MEASUREMENT,
    )

    for (expected_subset_instance_num, chiplet) in zip(subset_instance_nums, chiplets):
        assert chiplet.subset_instance_num == expected_subset_instance_num


def test_subset_nums_over_years(chiplet_dir):

    # check that each position is associated with the same subset info
    # across years

    pos_items = collections.defaultdict(list)

    chiplets = themeda.preproc.load_chiplets(
        chiplet_dir=chiplet_dir,
        measurement=MEASUREMENT,
    )

    for chiplet in chiplets:
        pos_items[chiplet.position].append(chiplet)

    for pos_item in pos_items.values():

        assert len(pos_item) > 1

        # there is to be only one unique subset number and subset instance
        # number across the years
        for attr in ["subset_num", "subset_instance_num"]:
            assert (
                len(set(getattr(curr_pos_item, attr) for curr_pos_item in pos_item))
                == 1
            )


def get_random_chips(chip_dir, region_file):

    # need to test that the chosen TIFF file is within the region
    region = themeda.preproc.roi.get_region(region_file=region_file)

    # pick a TIFF file
    avail_chip_paths = themeda.preproc.chips.get_chip_filenames(
        chip_dir=chip_dir,
        measurement="level4",
    )
    random.shuffle(avail_chip_paths)

    for chip_path in avail_chip_paths:

        chip = themeda.preproc.chips.read_chip(filename=chip_path)
        chip_bbox = themeda.preproc.roi.get_bbox(data=chip)
        all_chip_within_region = themeda.preproc.roi.is_bbox_within_region(
            bbox=chip_bbox, region=region
        )

        if all_chip_within_region:
            yield chip_path


def test_years_against_manual(
    chip_dir,
    chiplet_dir,
    region_file,
    chiplet_spatial_size_pix=160,
):

    # get the details of a random chip within the region
    base_chip_path = next(get_random_chips(chip_dir=chip_dir, region_file=region_file))
    base_chip_meta = themeda.preproc.chips.parse_chip_filename(
        filename=base_chip_path
    )

    year_chip_meta = []

    for chip_path in themeda.preproc.chips.get_chip_filenames(
        chip_dir=chip_dir, measurement="level4"
    ):

        chip_meta = themeda.preproc.chips.parse_chip_filename(
            filename=chip_path
        )

        if chip_meta.grid_ref == base_chip_meta.grid_ref:
            year_chip_meta.append(chip_meta)

    expected_chiplet_data = {}

    chiplet_pos = None

    for chip_meta in year_chip_meta:

        chip = themeda.preproc.chips.read_chip(
            filename=chip_meta.filename, load_data=True
        )

        # extract the top corner as the chiplet region to use
        chip_chiplet = chip.isel(
            x=range(chiplet_spatial_size_pix),
            y=range(chiplet_spatial_size_pix),
        )

        # work out the position of the chiplet
        chip_chiplet_pos = themeda.preproc.chiplets.Position(
            x=chip_chiplet.x.mean().item(),
            y=chip_chiplet.y.mean().item(),
        )

        if chiplet_pos is None:
            chiplet_pos = chip_chiplet_pos
        else:
            assert chiplet_pos == chip_chiplet_pos

        # store the data
        expected_chiplet_data[
            chip_meta.year
        ] = themeda.preproc.dea.preproc.remap_data(data=chip_chiplet.values)

    # now to check against the processed chiplet data
    for (year, year_expected_chiplet_data) in expected_chiplet_data.items():

        for chiplet in themeda.preproc.load_chiplets(
            chiplet_dir=chiplet_dir,
            years=[year],
            measurement="level4",
        ):

            if chiplet.position != chiplet_pos:
                continue

            # matching chiplet position
            assert np.all(chiplet.data == year_expected_chiplet_data)

            break

        else:
            raise ValueError("Unexpected error")


def test_against_manual(chip_dir, chiplet_dir, region_file):

    chip_path = next(get_random_chips(chip_dir=chip_dir, region_file=region_file))

    chip_meta = themeda.preproc.chips.parse_chip_filename(filename=chip_path)
    chip = themeda.preproc.chips.read_chip(filename=chip_path, load_data=True)

    chiplets = [
        chiplet
        for chiplet in themeda.preproc.load_chiplets(
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

        chip_pos = themeda.preproc.chiplets.Position(
            x=chip_subset.x.mean().item(),
            y=chip_subset.y.mean().item(),
        )

        # remap
        chip_data = themeda.preproc.dea.preproc.remap_data(data=chip_subset.data)

        # find the matching chiplet
        (matching_chiplet,) = [
            chiplet for chiplet in chiplets if chiplet.position == chip_pos
        ]

        # confirm that the data are the same
        assert np.all(chip_data == matching_chiplet.data)


def test_chip_filename_parsing():

    example = "ga_ls_landcover_class_cyear_2_1-0-0_au_x9y-24_1993-01-01_level4"

    parsed = themeda.preproc.chips.parse_chip_filename(
        filename=pathlib.Path(example)
    )

    assert parsed.year == 1993
    assert parsed.grid_ref.x == 9
    assert parsed.grid_ref.y == -24
    assert parsed.measurement == "level4"


def test_remapping():

    # pick a few random values from the level4 labels
    dea_data_eg = np.array([[27, 36], [103, 85]], dtype=np.uint8)

    # remap
    remapped_data = themeda.preproc.dea.preproc.remap_data(data=dea_data_eg)

    assert remapped_data.shape == dea_data_eg.shape

    expected_remapped_data = np.array([[5, 12], [20, 16]], dtype=np.uint8)

    assert np.all(remapped_data == expected_remapped_data)
