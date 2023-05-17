"""
Pre-processing the DEA database files into an analysis-ready format.

From the DEA database, we have a series of GeoTIFF files that each encode the data
for a given 'measurement' over a particular spatial region for a particular year. The
data for a particular spatial region for a particular year is called a 'chip'.

However, this represention is not ideal for the purposes of analysis. The chips are
quite large, with each being 4000x4000 pixels in size. Some of them are also not within
the defined savanna region. The GeoTIFF format also includes metadata that is ancillary
to the current analysis purposes. The data labelling conventions are also not desirable
for our analyses. Finally, we would like to randomly assign spatial regions to subsets
for the purposes of cross-validation.

The code in this file aims to resolve the above limitations by pre-processing the set
of GeoTIFF files and creating a new representation more amenable to our analyses.

The basic workflow is:
    1. Find the set of GeoTIFF files of interest.
    2. Create a representation based on 'chiplets', where a chiplet is a small spatial
    region (default is 160x160 pixels) that is within the defined savanna region and
    contains the data for a particular year.
    3. Identify each chiplet with the same spatial location as belonging to a random
    subset of all chiplets, based on a default of 5 subsets.
    4. Remap the labelled data within each chiplet.
    5. Write each chiplet to a separate `npz` file.

Note that the execution is pretty slow, particularly due to the savanna region bounds
checking. There is scope for parallelisation, but it also only needs to be run once-ish.
It also occupies a fair bit of RAM.

The pre-processed data can then be loaded using the `load_chiplets` function, which
returns a generator that yields chiplet data and metadata - optionally filtered by year
and subset.
"""

import pathlib
import dataclasses
import collections
import argparse
import typing

import numpy as np
import numpy.typing as npt

import xarray as xr
import rioxarray

import geojson
import pyproj
import shapely


@dataclasses.dataclass(frozen=True)
class Position:
    x: float
    y: float


@dataclasses.dataclass
class Chip:
    filename: pathlib.Path
    year: int
    position: Position
    measurement: str


@dataclasses.dataclass
class Chiplet(Chip):
    data: xr.DataArray
    subset_num: int | None = None
    subset_instance_num: int | None = None


@dataclasses.dataclass
class ChipletFilenameInfo:
    measurement: str
    year: int
    subset_num: int
    subset_instance_num: int


@dataclasses.dataclass
class ChipletFile(ChipletFilenameInfo):
    data: npt.NDArray[np.uint8]
    position: Position


def load_chiplets(
    chiplet_dir: pathlib.Path,
    years: collections.abc.Container[int] | None = None,
    subset_nums: collections.abc.Container[int] | None = None,
    sorted_filenames: bool = False,
) -> typing.Iterator[ChipletFile]:
    """
    Loads pre-processed chiplets from disk.

    Parameters
    ----------
    chiplet_dir:
        Directory containing the chiplet files.
    years:
        A collection of years to include; default is to include all years.
    subset_nums:
        A collection of subsets to include; default is to include all subsets.
    sorted_filenames:
        Whether to sort the matching filenames prior to yielding. If `False` (the
        default), they will be in arbitrary order. However, setting `True` will load
        all the filenames into memory.

    Returns
    -------
    A generator that yields chiplet data and metadata.
    """

    potential_filenames: typing.Generator[pathlib.Path, None, None] | list[
        pathlib.Path
    ] = chiplet_dir.glob("*.npz")

    if sorted_filenames:
        # this makes it into a list, which may require lots of memory
        potential_filenames = sorted(potential_filenames)

    for potential_filename in potential_filenames:

        filename_info = parse_chiplet_filename(filename=potential_filename)

        if (years is not None and filename_info.year not in years) or (
            subset_nums is not None and filename_info.subset_num not in subset_nums
        ):
            continue

        info = np.load(file=potential_filename)

        chiplet = ChipletFile(
            measurement=info["measurement"].item(),
            year=info["year"].item(),
            subset_num=info["subset_num"].item(),
            subset_instance_num=info["subset_instance_num"].item(),
            data=info["data"],
            position=Position(
                x=info["position"][0],
                y=info["position"][1],
            ),
        )

        yield chiplet


def parse_chiplet_filename(filename: pathlib.Path) -> ChipletFilenameInfo:
    "Converts a chiplet filename into a structured representation"

    items = filename.stem.split("_")

    (study, form, measurement, year, subset, subset_num, subset_instance_num) = items

    if (
        not filename.stem.startswith("ecofuture_chiplet")
        or study != "ecofuture"
        or form != "chiplet"
        or subset != "subset"
    ):
        raise ValueError("Unknown chiplet file")

    chiplet_filename_info = ChipletFilenameInfo(
        measurement=measurement,
        year=int(year),
        subset_num=int(subset_num),
        subset_instance_num=int(subset_instance_num),
    )

    return chiplet_filename_info


def save_chiplets(
    chip_dir: pathlib.Path,
    chiplet_dir: pathlib.Path,
    region_file: pathlib.Path,
    chiplet_spatial_size_pix: int = 160,
    remap: bool = True,
    n_subsets: int = 5,
    subset_seed: int | None = 254204982,
    overwrite: bool = False,
) -> None:
    """
    Converts and saves a set of DEA chips as chiplets.

    Parameters
    ----------
    chip_dir:
        Directory containing the DEA chip files (i.e., .tif files).
    chiplet_dir:
        Directory to write the chiplet files.
    region_file:
        Path to a GeoJSON file that specifies the region of interest.
    chiplet_spatial_size_pix:
        Size of the sides of each chiplet. Must divide evenly into the chip size.
    remap:
        Whether to convert the DEA Level 4 labels into a collaborator-specified
        remapping.
    n_subsets:
        Number of subsets to break the chiplet locations into.
    subset_seed:
        Random number generator seed for the chiplet subset allocator.
    overwrite:
        Whether to overwrite already-existing chiplet files.
    """

    product = "ga_ls_landcover_class_cyear_2"
    measurement = "level4"

    # get the GeoTIFF filenames matching the provided criteria
    filenames = get_chip_filenames(
        chip_dir=chip_dir,
        product=product,
        measurement=measurement,
    )

    # get the info on each chip
    chips = parse_filenames(filenames=filenames)

    # get the polygon capturing the spatial region-of-interest
    region = get_region(region_file=region_file)

    # get the info on each chiplet
    chiplets = form_chiplets(
        chips=chips,
        chiplet_spatial_size_pix=chiplet_spatial_size_pix,
        region=region,
    )

    # give each chiplet a subset (operates in-place)
    assign_subset_labels(
        chiplets=chiplets,
        n_subsets=n_subsets,
        subset_seed=subset_seed,
    )

    # save the chiplets to disk
    render_chiplets(
        chiplets=chiplets,
        chiplet_dir=chiplet_dir,
        remap=remap,
        overwrite=overwrite,
    )


def render_chiplets(
    chiplets: dict[Position, dict[int, Chiplet]],
    chiplet_dir: pathlib.Path,
    remap: bool = True,
    overwrite: bool = False,
) -> None:
    "Formats and saves chiplets to disk."

    if remap:
        remap_lut = get_remapping_lut()

    for year_chiplets in chiplets.values():
        for chiplet in year_chiplets.values():
            filename = get_chiplet_filename(chiplet=chiplet)

            save_path = chiplet_dir / filename

            if save_path.exists() and not overwrite:
                print(f"File {filename} exists in {chiplet_dir}; skipping")

            else:
                # make a copy of the data so we don't retain a reference
                # to the full data
                chiplet_data = chiplet.data.copy().values

                if remap:
                    chiplet_data = remap_data(
                        data=chiplet_data,
                        lut=remap_lut,
                    )

                # finally, save
                # probably excessive metadata, but best to save rather than
                # not save and miss it later
                np.savez(
                    file=save_path,
                    data=chiplet_data,
                    position=np.array([chiplet.position.x, chiplet.position.y]),
                    subset_num=np.array([chiplet.subset_num]),
                    subset_instance_num=np.array([chiplet.subset_instance_num]),
                    raw_filename=np.array([chiplet.filename], dtype=str),
                    year=np.array([chiplet.year]),
                    measurement=np.array([chiplet.measurement], dtype=str),
                )


def assign_subset_labels(
    chiplets: dict[Position, dict[int, Chiplet]],
    n_subsets: int,
    subset_seed: int | None = None,
) -> None:
    """Gives each chiplet with a unique spatial location a randomly-assigned subset
    number"""
    # count the instance within a given subset
    # all years for the same spatial location have the same subset instance number
    subset_instance_counter: collections.Counter[int] = collections.Counter()

    rand = np.random.default_rng(seed=subset_seed)

    for year_chiplets in chiplets.values():
        subset_num = rand.integers(low=1, high=n_subsets + 1)

        subset_instance_counter[subset_num] += 1

        subset_instance_num = subset_instance_counter[subset_num]

        for chiplet in year_chiplets.values():
            if chiplet.subset_num is not None:
                pass
                # raise ValueError("Subset number unexpectedly assigned")

            chiplet.subset_num = subset_num
            chiplet.subset_instance_num = subset_instance_num


def get_chiplet_filename(chiplet: Chiplet) -> str:
    "Formats a chiplet filename from its metadata"
    filename = (
        f"ecofuture_chiplet_{chiplet.measurement}_{chiplet.year}_subset_"
        + f"{chiplet.subset_num}_{chiplet.subset_instance_num:08d}.npz"
    )

    return filename


def form_chiplets(
    chips: dict[Position, dict[int, Chip]],
    chiplet_spatial_size_pix: int,
    region: shapely.geometry.polygon.Polygon,
) -> dict[Position, dict[int, Chiplet]]:
    """
    Converts a set of chips into chiplets that lie within the region of interest.
    """

    chiplets: dict[Position, dict[int, Chiplet]] = {}

    # `pos_chips` will contain the entries for each year for a given position
    for pos_chips in chips.values():
        # the region test is quite slow, so assume that the geometry is the
        # same across years and just assess it once, using the first entry
        # as representative
        (rep_chip, *_) = pos_chips.values()

        # just load the metadata and not the full data
        rep_chip_data = read_chip(
            filename=rep_chip.filename,
            load_data=False,
        )

        # given the slowness of the region test, we can avoid doing it for each
        # chiplet if we know that the bounding box of the chip is within the region
        all_rep_chip_within_region = is_bbox_within_region(
            bbox=get_bbox(data=rep_chip), region=region
        )

        if all_rep_chip_within_region:
            region_arg = None
        else:
            region_arg = region

        rep_chip_splits = split_chip(
            data=rep_chip_data,
            chiplet_spatial_size_pix=chiplet_spatial_size_pix,
            region=region_arg,
        )

        if all_rep_chip_within_region:
            validity = [True for _ in rep_chip_splits]
        else:
            # get the validity (whether the chiplet is in the region) of each chiplet
            validity = [rep_chip_split.in_region for rep_chip_split in rep_chip_splits]

        # now iterate over the yearly chips at this position
        for chip in pos_chips.values():
            chip_data = read_chip(
                filename=chip.filename,
                load_data=False,
            )

            potential_chiplets_data = split_chip(
                data=chip_data,
                chiplet_spatial_size_pix=chiplet_spatial_size_pix,
            )

            for chiplet_validity, potential_chiplet_data in zip(
                validity,
                potential_chiplets_data,
            ):
                # if we aren't dealing with a valid chiplet, move on
                if not chiplet_validity:
                    continue

                chiplet_position = Position(
                    x=potential_chiplet_data.x_centre,
                    y=potential_chiplet_data.y_centre,
                )

                chiplet = Chiplet(
                    filename=chip.filename,
                    year=chip.year,
                    position=chiplet_position,
                    data=potential_chiplet_data,
                    measurement=chip.measurement,
                )

                if chiplet_position not in chiplets:
                    chiplets[chiplet_position] = {}

                chiplets[chiplet_position][chiplet.year] = chiplet

    return chiplets


def split_chip(
    data: xr.DataArray,
    chiplet_spatial_size_pix: int,
    region: shapely.geometry.polygon.Polygon | None = None,
) -> list[xr.DataArray]:
    "Spatially tiles a chip into a set of chiplets."

    chip_subsets = []

    for i_x_left in range(0, data.sizes["x"], chiplet_spatial_size_pix):
        for i_y_left in range(0, data.sizes["y"], chiplet_spatial_size_pix):
            chip_subset = data.isel(
                x=range(i_x_left, i_x_left + chiplet_spatial_size_pix),
                y=range(i_y_left, i_y_left + chiplet_spatial_size_pix),
            )

            chip_subset.attrs["x_centre"] = chip_subset.x.mean().item()
            chip_subset.attrs["y_centre"] = chip_subset.y.mean().item()

            if region is not None:
                bbox = get_bbox(data=chip_subset)
                chip_subset.attrs["in_region"] = is_bbox_within_region(
                    bbox=bbox, region=region
                )

            chip_subsets.append(chip_subset)

    return chip_subsets


def is_bbox_within_region(
    bbox: shapely.geometry.polygon.Polygon,
    region: shapely.geometry.polygon.Polygon,
) -> bool:
    is_within: bool = bbox.within(other=region)
    return is_within


def get_bbox(data: xr.DataArray) -> shapely.geometry.polygon.Polygon:
    "Gets the bounding-box of a chip or chiplet's data"

    bbox = shapely.Polygon(
        shell=[
            [data.x[0], data.y[0]],
            [data.x[-1], data.y[0]],
            [data.x[-1], data.y[-1]],
            [data.x[0], data.y[-1]],
        ],
    )

    return bbox


def read_chip(filename: pathlib.Path, load_data: bool = False) -> xr.DataArray:
    "Reads a GeoTIFF file from disk"

    data = rioxarray.open_rasterio(filename=filename)

    if not isinstance(data, xr.DataArray):
        raise ValueError("Unexpected data format")

    # has a redundant `band` dimension
    data = data.squeeze(dim="band", drop=True)

    if load_data:
        data.load()

    return data


def get_region(
    region_file: pathlib.Path,
    src_crs: int = 4326,
    dst_crs: int = 3577,
) -> shapely.geometry.polygon.Polygon:
    """
    Loads a spatial region-of-interest from a GeoJSON file, optionally converts the
    projection, and formats the region as a `shapely` Polygon.
    """

    region = geojson.loads(region_file.read_text())

    (feature,) = region["features"]

    (coords,) = feature["geometry"]["coordinates"]

    if src_crs != dst_crs:
        transformer = pyproj.Transformer.from_crs(
            crs_from=src_crs,
            crs_to=dst_crs,
            always_xy=True,
        )

        # splat from [(x1,y1), (x2, y2), ...] to ([x1, x2, ...], [y1, y2, ...])
        (x, y) = list(zip(*coords))

        # convert projection
        (x, y) = transformer.transform(xx=x, yy=y, errcheck=True)

        # unsplat
        coords = list(zip(x, y))

    polygon = shapely.Polygon(shell=coords)

    # might speed up calls that use the geometry
    shapely.prepare(geometry=polygon)

    return polygon


def parse_filenames(filenames: list[pathlib.Path]) -> dict[Position, dict[int, Chip]]:
    "Parses the metadata in a set of chip filenames"

    chips: dict[Position, dict[int, Chip]] = {}

    for filename in filenames:
        chip = parse_filename(filename=filename)

        if chip.position not in chips:
            chips[chip.position] = {}

        chips[chip.position][chip.year] = chip

    return chips


def parse_filename(filename: pathlib.Path) -> Chip:
    "Parses the metadata in a chip filename"

    # example: ga_ls_landcover_class_cyear_2_1-0-0_au_x9y-24_1993-01-01_level4

    (*_, xy, date, measurement) = filename.stem.split("_")

    if not xy.startswith("x"):
        raise ValueError("Unexpected filename format")

    x = float(xy[1 : xy.index("y")])
    y = float(xy[xy.index("y") + 1 :])

    if not date.count("-") == 2:
        raise ValueError("Unexpected filename format")

    (year, *_) = date.split("-")

    chip = Chip(
        filename=filename,
        year=int(year),
        position=Position(x=x, y=y),
        measurement=measurement,
    )

    return chip


def get_chip_filenames(
    chip_dir: pathlib.Path,
    product: str = "ga_ls_landcover_class_cyear_2",
    year: str | int | None = None,
    measurement: str | None = None,
) -> list[pathlib.Path]:
    """
    Gets a list of DEA chip filenames, with optional filters for year and
    measurement.
    """

    glob = f"{product}*"

    if year is not None:
        glob += f"{year}-01-01*"

    if measurement is not None:
        glob += measurement

    glob += ".tif"

    filenames = sorted(chip_dir.glob(glob))

    if len(filenames) == 0:
        raise ValueError("No matching filenames found")

    return filenames


def remap_data(
    data: npt.NDArray[np.uint8],
    lut: npt.NDArray[np.uint8] | None = None,
) -> npt.NDArray[np.uint8]:
    """
    Applies a remapping to a data array.
    """

    if lut is None:
        lut = get_remapping_lut()

    remapped_data: npt.NDArray[np.uint8] = lut[data]

    if np.any(remapped_data > 104):
        raise ValueError("Data remapping error")

    return remapped_data


def get_remapping_lut() -> npt.NDArray[np.uint8]:
    "Gets an array that acts as a LUT for remapping."

    # mapping from the input to output values
    # from Rob's `transforms.py`
    lut_dict = {
        0: 0,
        14: 1,
        15: 2,
        16: 3,
        17: 4,
        18: 4,
        27: 5,
        28: 6,
        29: 7,
        30: 8,
        31: 8,
        32: 9,
        33: 10,
        34: 11,
        35: 12,
        36: 12,
        63: 13,
        64: 13,
        65: 13,
        66: 14,
        67: 14,
        68: 14,
        69: 15,
        70: 15,
        71: 15,
        72: 15,
        73: 15,
        74: 15,
        75: 15,
        76: 15,
        77: 15,
        78: 16,
        79: 16,
        80: 16,
        81: 16,
        82: 16,
        83: 16,
        84: 16,
        85: 16,
        86: 16,
        87: 16,
        88: 16,
        89: 16,
        90: 16,
        91: 16,
        92: 16,
        93: 17,
        94: 18,
        95: 12,
        96: 12,
        97: 18,
        98: 19,
        99: 19,
        100: 19,
        101: 19,
        102: 19,
        103: 20,
        104: 20,
    }

    sentinel_val = 111

    lut = np.ones(104 + 1) * sentinel_val

    for src_val, dst_val in lut_dict.items():
        lut[src_val] = dst_val

    lut_dt: npt.NDArray[np.uint8] = lut.astype(np.uint8)

    return lut_dt


def run() -> None:

    # using `argparse` rather than `typer` because other project dependencies
    # require a very old version of `typer` that is missing functionality

    parser = argparse.ArgumentParser(
        description=(
            "Convert the data from a DEA dataset query into a set of 'chiplets' - "
            + "files that contain the image from a given year and asset over a "
            + "small spatial window"
        ),
    )

    parser.add_argument(
        "-chip_dir",
        required=True,
        type=pathlib.Path,
        help="Directory containing the result from `dea_data_loader download`",
    )

    parser.add_argument(
        "-chiplet_dir",
        required=True,
        type=pathlib.Path,
        help="Directory to write the chiplet files",
    )

    parser.add_argument(
        "-region_file",
        required=True,
        type=pathlib.Path,
        help="Path to the GeoJSON file that specifies the region-of-interest",
    )

    parser.add_argument(
        "-no_remapping",
        required=False,
        default=False,
        action="store_true",
        help=(
            "The labels for each file are remapped to our convention by default; use"
            + "this flag to disable"
        ),
    )

    parser.add_argument(
        "-n_spatial_subsets",
        required=False,
        type=int,
        default=5,
        help=(
            "Number of spatial subsets to use; each chiplet is randomly assigned a "
            + "subset from 1 to `n_spatial_subsets`"
        ),
    )

    parser.add_argument(
        "-random_seed",
        required=False,
        default=False,
        type=int,
        help="Seed for the random assignment of chiplets to subsets",
    )

    parser.add_argument(
        "-chiplet_spatial_size_pix",
        required=False,
        type=int,
        default=160,
        help=(
            "Spatial extent of each side of each square chiplet. Must divide evenly "
            + "with the chip size"
        ),
    )

    parser.add_argument(
        "-overwrite",
        required=False,
        default=False,
        action="store_true",
        help="Whether it is OK to overwrite if the chiplet file already exists",
    )

    args = parser.parse_args()

    save_chiplets(
        chip_dir=args.chip_dir,
        chiplet_dir=args.chiplet_dir,
        region_file=args.region_file,
        chiplet_spatial_size_pix=args.chiplet_spatial_size_pix,
        remap=not args.no_remapping,
        n_subsets=args.n_spatial_subsets,
        subset_seed=args.random_seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run()
