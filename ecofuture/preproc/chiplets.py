import pathlib
import dataclasses
import collections
import typing
import zipfile

import numpy as np
import numpy.typing as npt

import xarray as xr

import shapely

import tqdm

from .chips import (
    Chip,
    GridRef,
    read_chip,
    split_chip,
    get_chip_filenames,
    parse_chip_filenames,
)
from .roi import get_bbox, is_bbox_within_region, get_region


@dataclasses.dataclass(frozen=True)
class Position:
    x: float
    y: float


@dataclasses.dataclass
class Chiplet:
    year: int
    measurement: str
    filename: typing.Optional[pathlib.Path]
    position: Position
    data: xr.DataArray
    subset_num: typing.Optional[int] = None
    subset_instance_num: typing.Optional[int] = None


@dataclasses.dataclass
class ChipletFilenameInfo:
    filename: pathlib.Path
    measurement: str
    year: int
    subset_num: int
    subset_instance_num: int


@dataclasses.dataclass
class ChipletFile(ChipletFilenameInfo):
    data: npt.NDArray[np.uint8]
    position: Position


@dataclasses.dataclass
class ChipletFilename:
    container_path: pathlib.Path
    filename: pathlib.Path


def load_chiplets(
    chiplet_dir: pathlib.Path,
    measurement: str,
    years: typing.Optional[typing.Iterable[int]] = None,
    exclude_years: typing.Iterable[int] = (2019, 2020, 2021, 2022, 2023),
    subset_nums: typing.Optional[typing.Iterable[int]] = None,
    exclude_subset_nums: typing.Iterable[int] = tuple(),
    subset_instance_nums: typing.Optional[list[int]] = None,
    just_filenames: bool = False,
) -> typing.Union[typing.Iterator[ChipletFile], typing.Iterator[ChipletFilenameInfo]]:
    """
    Loads pre-processed chiplets from disk.

    Parameters
    ----------
    chiplet_dir:
        Directory containing the chiplet files.
    measurement:
        The measurement name recorded in the files.
    years:
        A collection of years to include; default is to include all years.
    exclude_years:
        A collection of years to exclude; default is to exclude 2019 and beyond.
    subset_nums:
        A collection of subsets to include; default is to include all subsets.
    exclude_subset_nums:
        A collection of subsets to exclude; default is not to exclude any.
    subset_instance_nums:
        A collection of subset instance numbers to include; default is to include all
        subset instance numbers. Note that these are not allowed to vary across subset
        numbers.
    just_filenames:
        If `True`, only return the matching filename info rather than the data.

    Returns
    -------
    A generator that yields chiplet data and metadata or filenames.
    """

    # if the years or subset numbers haven't been specified, then parse the
    # available zip files for the years and subset numbers that are present
    if years is None or subset_nums is None:
        (avail_years, avail_subset_nums) = get_zip_info(
            chiplet_dir=chiplet_dir, measurement=measurement
        )

        if years is None:
            years = avail_years

        if subset_nums is None:
            subset_nums = avail_subset_nums

    # apply exclusions
    years = [year for year in years if year not in exclude_years]
    subset_nums = [
        subset_num
        for subset_num in subset_nums
        if subset_num not in exclude_subset_nums
    ]

    for year in years:
        for subset_num in subset_nums:

            zip_path = get_zip_path(
                chiplet_dir=chiplet_dir,
                measurement=measurement,
                year=year,
                subset_num=subset_num,
            )

            with zipfile.ZipFile(zip_path, mode="r") as zip_handle:

                chiplets_in_zip = [
                    parse_chiplet_filename(filename=pathlib.Path(filename))
                    for filename in zip_handle.namelist()
                ]

                if subset_instance_nums is None:
                    curr_subset_instance_nums = list(range(1, len(chiplets_in_zip) + 1))
                else:
                    curr_subset_instance_nums = subset_instance_nums

                for subset_instance_num in curr_subset_instance_nums:

                    (curr_chiplet_info,) = [
                        chiplet_info
                        for chiplet_info in chiplets_in_zip
                        if chiplet_info.subset_instance_num == subset_instance_num
                    ]

                    if just_filenames:
                        yield curr_chiplet_info

                    else:
                        with zip_handle.open(
                            str(curr_chiplet_info.filename), mode="r"
                        ) as file_handle:

                            info = np.load(file=file_handle)

                            chiplet = ChipletFile(
                                filename=curr_chiplet_info.filename,
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


def get_zip_path(
    chiplet_dir: pathlib.Path,
    measurement: str,
    year: int,
    subset_num: int,
) -> pathlib.Path:
    "Get the path on disk to a particular zip file"

    zip_path = (
        chiplet_dir / f"ecofuture_chiplet_{measurement}_{year}_subset_{subset_num}.zip"
    )

    return zip_path


def get_zip_info(
    chiplet_dir: pathlib.Path,
    measurement: str,
) -> tuple[list[int], list[int]]:
    """
    Glob a directory for zip files and return the years and subset numbers that
    are contained.
    """

    available_zip_paths = sorted(chiplet_dir.glob("ecofuture_chiplet*.zip"))

    years = []
    subset_nums = []

    for potential_zip_path in available_zip_paths:

        (
            front,
            chiplet_str,
            meas,
            year,
            subset_str,
            subset_num,
        ) = potential_zip_path.stem.split("_")

        if meas != measurement:
            continue

        if front != "ecofuture" or chiplet_str != "chiplet" or subset_str != "subset":
            raise ValueError("Unexpected zip file name")

        years.append(int(year))
        subset_nums.append(int(subset_num))

    years = sorted(set(years))
    subset_nums = sorted(set(subset_nums))

    return (years, subset_nums)


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
        filename=filename,
        measurement=measurement,
        year=int(year),
        subset_num=int(subset_num),
        subset_instance_num=int(subset_instance_num),
    )

    return chiplet_filename_info


def assign_subset_labels(
    chiplets: dict[Position, dict[int, Chiplet]],
    n_subsets: int,
    subset_seed: typing.Optional[int] = None,
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
            chiplet.subset_num = subset_num
            chiplet.subset_instance_num = subset_instance_num

    # now to do the shuffling of instance numbers within each instance
    shuffle_lut = get_shuffled_instance_num_lut(
        subset_counts=dict(subset_instance_counter),
        rand=rand,
    )

    for year_chiplets in chiplets.values():
        for chiplet in year_chiplets.values():

            if chiplet.subset_num is None or chiplet.subset_instance_num is None:
                raise ValueError("Unexpected null")

            lut_key = (chiplet.subset_num, chiplet.subset_instance_num)
            new_instance_num = shuffle_lut[lut_key]
            chiplet.subset_instance_num = new_instance_num


def get_shuffled_instance_num_lut(
    subset_counts: dict[int, int],
    rand: np.random._generator.Generator,
) -> dict[tuple[int, int], int]:

    subset_nums = sorted(subset_counts.keys())
    n_subsets = len(subset_nums)

    lut = {}

    for subset_num in range(1, n_subsets + 1):

        # how many positions are part of this subset
        n_subset_instances = subset_counts[subset_num]

        # this will (implicitly) map from original instance number
        # to shuffled instance number
        instance_num_lut = np.arange(1, n_subset_instances + 1)
        rand.shuffle(instance_num_lut)

        for (orig_instance_num, new_instance_num) in enumerate(instance_num_lut, 1):

            key = (subset_num, orig_instance_num)

            lut[key] = new_instance_num

    return lut


def get_chiplet_filename(chiplet: Chiplet) -> ChipletFilename:
    "Formats a chiplet filename from its metadata"

    # the outer zip file
    container_path = pathlib.Path(
        f"ecofuture_chiplet_{chiplet.measurement}_{chiplet.year}_subset_"
        + f"{chiplet.subset_num}.zip"
    )

    # the path *within* the zip file
    filename = pathlib.Path(
        f"ecofuture_chiplet_{chiplet.measurement}_{chiplet.year}_subset_"
        + f"{chiplet.subset_num}_{chiplet.subset_instance_num:08d}.npz"
    )

    return ChipletFilename(container_path=container_path, filename=filename)


def form_chiplets(
    chips: dict[GridRef, dict[int, Chip]],
    chiplet_spatial_size_pix: int,
    region: shapely.geometry.polygon.Polygon,
    show_progress: bool = True,
) -> dict[Position, dict[int, Chiplet]]:
    """
    Converts a set of chips into chiplets that lie within the region of interest.
    """

    if show_progress:
        progress_bar = tqdm.tqdm(iterable=None, total=len(chips))

    chiplets: dict[Position, dict[int, Chiplet]] = {}

    # `pos_chips` will contain the entries for each year for a given grid ref
    for pos_chips in chips.values():
        # the region test is quite slow, so assume that the geometry is the
        # same across years and just assess it once, using the first entry
        # as representative
        (rep_chip, *_) = pos_chips.values()

        if rep_chip.data is None:

            if rep_chip.filename is None:
                raise ValueError("Expected to have a filename at this point")

            # just load the metadata and not the full data
            rep_chip_data = read_chip(
                filename=rep_chip.filename,
                load_data=False,
            )
        else:
            rep_chip_data = rep_chip.data

        # given the slowness of the region test, we can avoid doing it for each
        # chiplet if we know that the bounding box of the chip is within the region
        all_rep_chip_within_region = is_bbox_within_region(
            bbox=get_bbox(data=rep_chip_data),
            region=region,
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

        # now iterate over the yearly chips at this grid ref
        for chip in pos_chips.values():

            if chip.data is None:

                if chip.filename is None:
                    raise ValueError("Expected to have a filename")

                chip_data = read_chip(
                    filename=chip.filename,
                    load_data=False,
                )
            else:
                chip_data = chip.data

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

                # to save memory, drop some un-used elements from the chip
                # data structure
                potential_chiplet_data = potential_chiplet_data.drop_vars(
                    names=["x", "y", "spatial_ref"],
                )
                potential_chiplet_data.attrs = {}

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

        if show_progress:
            progress_bar.update()

    if show_progress:
        progress_bar.close()

    return chiplets


def save_chiplets(
    chip_dir: typing.Optional[pathlib.Path],
    chiplet_dir: pathlib.Path,
    region_file: pathlib.Path,
    measurement: str,
    chips: typing.Optional[dict[GridRef, dict[int, Chip]]] = None,
    chiplet_spatial_size_pix: int = 160,
    remapper: typing.Optional[
        typing.Callable[[npt.NDArray[typing.Any]], npt.NDArray[typing.Any]]
    ] = None,
    n_subsets: int = 5,
    subset_seed: typing.Optional[int] = 254204982,
    overwrite: bool = False,
    show_progress: bool = True,
    as_float16: bool = False,
) -> None:
    """
    Converts and saves a set of chips as chiplets.

    Parameters
    ----------
    chip_dir:
        Directory containing the chip files (i.e., .tif files).
    chiplet_dir:
        Directory to write the chiplet files.
    region_file:
        Path to a GeoJSON file that specifies the region of interest.
    chiplet_spatial_size_pix:
        Size of the sides of each chiplet. Must divide evenly into the chip size.
    remapper:
        Function that takes in a set of chiplet data and returns remapped data
        that replaces the data in the chiplet.
    n_subsets:
        Number of subsets to break the chiplet locations into.
    subset_seed:
        Random number generator seed for the chiplet subset allocator.
    overwrite:
        Whether to overwrite already-existing chiplet files.
    show_progress:
        Whether to show a progress bar during the saving stage.
    """

    if show_progress:
        print("Preparing...")

    if chips is None:

        if chip_dir is None:
            raise ValueError("Either `chip_dir` or `chips` needs to be provided")

        # get the GeoTIFF filenames matching the provided criteria
        filenames = get_chip_filenames(
            chip_dir=chip_dir,
            measurement=measurement,
        )

        # get the info on each chip
        chips = parse_chip_filenames(filenames=filenames)

    else:
        if chip_dir is not None:
            raise ValueError(
                "Either `chip_dir` or `chips` needs to be provided, but not both"
            )

    # get the polygon capturing the spatial region-of-interest
    region = get_region(region_file=region_file)

    if show_progress:
        print("Forming chiplets...")

    # get the info on each chiplet
    chiplets = form_chiplets(
        chips=chips,
        chiplet_spatial_size_pix=chiplet_spatial_size_pix,
        region=region,
        show_progress=show_progress,
    )

    if show_progress:
        print("Assigning subset labels...")

    # give each chiplet a subset (operates in-place)
    assign_subset_labels(
        chiplets=chiplets,
        n_subsets=n_subsets,
        subset_seed=subset_seed,
    )

    if show_progress:
        print("Rendering chiplets...")

    # save the chiplets to disk
    render_chiplets(
        chiplets=chiplets,
        chiplet_dir=chiplet_dir,
        remapper=remapper,
        overwrite=overwrite,
        show_progress=show_progress,
        as_float16=as_float16,
    )


def render_chiplets(
    chiplets: dict[Position, dict[int, Chiplet]],
    chiplet_dir: pathlib.Path,
    remapper: typing.Optional[
        typing.Callable[[npt.NDArray[typing.Any]], npt.NDArray[typing.Any]]
    ] = None,
    overwrite: bool = False,
    show_progress: bool = True,
    as_float16: bool = False,
) -> None:
    "Formats and saves chiplets to disk."

    # rearrange chiplets as {year: {subset_num : {subset_instance_num: chiplet}}}
    # otherwise writing the zips takes *forever*
    chiplets_rearr: dict[int, dict[int, dict[int, Chiplet]]] = {}

    # keep a track of the total number of chiplets
    n_total_chiplets = 0

    for pos_chiplets in chiplets.values():

        for (year, chiplet) in pos_chiplets.items():

            if year not in chiplets_rearr:
                year_dict: dict[int, dict[int, Chiplet]] = {}
                chiplets_rearr[year] = year_dict

            subset_num = chiplet.subset_num

            if subset_num is None:
                raise ValueError("Unexpected subset number")

            if subset_num not in chiplets_rearr[year]:
                year_subset_dict: dict[int, Chiplet] = {}
                chiplets_rearr[year][subset_num] = year_subset_dict

            subset_instance_num = chiplet.subset_instance_num

            if subset_instance_num is None:
                raise ValueError("Unexpected subset instance number")

            chiplets_rearr[year][subset_num][subset_instance_num] = chiplet

            n_total_chiplets += 1

    if show_progress:
        progress_bar = tqdm.tqdm(iterable=None, total=n_total_chiplets)

    for (year, year_chiplets) in chiplets_rearr.items():
        for (subset_num, year_subset_chiplets) in year_chiplets.items():

            # sniff the first chiplet for the container zip path
            rep_filename = get_chiplet_filename(chiplet=year_subset_chiplets[1])
            container_path = chiplet_dir / rep_filename.container_path

            if container_path.exists() and not overwrite:
                print(f"Zip path {container_path} exists; skipping")
                break

            # no/little compression
            with zipfile.ZipFile(
                container_path,
                mode="w",
                compression=zipfile.ZIP_BZIP2,
                compresslevel=1,
            ) as zip_file:

                for chiplet in year_subset_chiplets.values():

                    filename = get_chiplet_filename(chiplet=chiplet)

                    if (chiplet_dir / filename.container_path) != container_path:
                        raise ValueError("Unexpected container path")

                    save_path = str(filename.filename)

                    if save_path in zip_file.namelist() and not overwrite:
                        print(
                            f"File {filename.filename} exists in "
                            + "{filename.container_path}; skipping"
                        )

                    else:
                        # make a copy of the data so we don't retain a reference
                        # to the full data
                        chiplet_data = chiplet.data.copy().values

                        if remapper is not None:
                            chiplet_data = remapper(chiplet_data)

                        if as_float16:
                            chiplet_data = chiplet_data.astype(np.float16)

                        with zip_file.open(save_path, mode="w") as zip_file_handle:

                            # finally, save
                            # probably excessive metadata, but best to save rather than
                            # not save and miss it later
                            np.savez(
                                file=zip_file_handle,
                                data=chiplet_data,
                                position=np.array(
                                    [chiplet.position.x, chiplet.position.y]
                                ),
                                subset_num=np.array([chiplet.subset_num]),
                                subset_instance_num=np.array(
                                    [chiplet.subset_instance_num]
                                ),
                                raw_filename=np.array([chiplet.filename], dtype=str),
                                year=np.array([chiplet.year]),
                                measurement=np.array([chiplet.measurement], dtype=str),
                            )

                    if show_progress:
                        progress_bar.update()

    if show_progress:
        progress_bar.close()
