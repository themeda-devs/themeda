"""
Convert the raw ANU climate data into chiplets based on the DEA layout, optionally
saving intermediate chip files.
"""

import pathlib
import collections
import typing
import warnings

import numpy as np

import xarray as xr
import rioxarray
import rasterio

import themeda.preproc.chips
import themeda.preproc.chiplets
import themeda.preproc.dea.preproc


def form_chips(
    raw_climate_dir: typing.Union[pathlib.Path, str],
    dea_chip_dir: typing.Union[pathlib.Path, str],
    measurement: str,
    climate_chip_dir: typing.Optional[pathlib.Path] = None,
    years: typing.Optional[list[int]] = None,
) -> typing.Iterator[themeda.preproc.chips.SpatialChipCollectionType]:
    """
    Using the DEA chips as a template, this converts the raw climate data
    into corresponding chips. It returns an iterator that yields a year's
    worth of chips at a time.
    """

    if measurement not in ["rain", "tmax"]:
        raise ValueError("Unknown measurement")

    raw_climate_dir = pathlib.Path(raw_climate_dir)
    dea_chip_dir = pathlib.Path(dea_chip_dir)

    if years is None:
        years = get_raw_years(data_dir=raw_climate_dir, measurement=measurement)

    # {grid_ref: {year: info}}
    dea_chip_meta_info = themeda.preproc.chips.parse_chip_filenames(
        filenames=themeda.preproc.chips.get_chip_filenames(
            chip_dir=dea_chip_dir,
            measurement="level4",
        ),
    )

    for year in years:

        # each year just has a single climate dataset (after being summarised over
        # months), covering the whole country
        climate_data = load_raw(
            data_dir=raw_climate_dir,
            measurement=measurement,
            year=year,
            summarise=True,
        )

        chips: themeda.preproc.chips.SpatialChipCollectionType = {}

        for (dea_chip_pos, dea_chip_yearly_meta) in dea_chip_meta_info.items():

            # just take the first DEA chip, since it is assumed that everything
            # of relevance is constant across years
            (dea_chip_meta, *_) = list(dea_chip_yearly_meta.values())

            # load the DEA chip - no need for the actual data
            dea_chip = themeda.preproc.chips.read_chip(
                filename=dea_chip_meta.filename,
                load_data=False,
            )

            # reproject the climate data (which covers the whole country) to match
            # the spatial region and resolution of the DEA chip
            climate_chip = climate_data.rio.reproject_match(
                match_data_array=dea_chip,
                resampling=rasterio.enums.Resampling.nearest,
            )

            # we changed nodata to nan when loading, so explain that change
            climate_chip.rio.set_nodata(
                input_nodata=np.nan,
                inplace=True,
            )

            # figure out the output filename for the new climate chip
            climate_chip_meta = gen_chip_metadata(
                base_chip=dea_chip_meta,
                year=year,
                measurement=measurement,
                chip_dir=climate_chip_dir,
                data=climate_chip,
            )

            if climate_chip_meta.grid_ref not in chips:
                chips[climate_chip_meta.grid_ref] = {}
            else:
                if year in chips[climate_chip_meta.grid_ref]:
                    raise ValueError("Unexpected value")

            chips[climate_chip_meta.grid_ref][year] = climate_chip_meta

        yield chips


def save_chips(
    raw_climate_dir: typing.Union[pathlib.Path, str],
    dea_chip_dir: typing.Union[pathlib.Path, str],
    climate_chip_dir: typing.Union[pathlib.Path, str],
    measurement: str,
    overwrite: bool = False,
    year: typing.Optional[int] = None,
) -> None:
    """
    Saves the (intermediate) chips to disk.
    """

    if year is None:
        years = None
    else:
        years = [year]

    chips = form_chips(
        raw_climate_dir=raw_climate_dir,
        dea_chip_dir=dea_chip_dir,
        climate_chip_dir=pathlib.Path(climate_chip_dir),
        measurement=measurement,
        years=years,
    )

    for year_chips in chips:
        for year_chip in year_chips.values():
            for chip in year_chip.values():

                if chip.filename is None:
                    raise ValueError("Expected a filename")

                if chip.filename.exists() and not overwrite:
                    print(f"Output path {chip.filename} exists; skipping")
                else:
                    chip.data.rio.to_raster(raster_path=chip.filename)


def gen_chip_metadata(
    base_chip: themeda.preproc.chips.Chip,
    year: int,
    measurement: str,
    chip_dir: typing.Optional[typing.Union[pathlib.Path, str]],
    data: typing.Optional[xr.DataArray] = None,
) -> themeda.preproc.chips.Chip:

    if chip_dir is None:
        filename = None
    else:
        chip_dir = pathlib.Path(chip_dir)
        filename = chip_dir / (
            "ANUClimate_"
            + f"x{base_chip.grid_ref.x:d}y{base_chip.grid_ref.y:d}_"
            + f"{year}_{measurement}.tif"
        )

    chip = themeda.preproc.chips.Chip(
        year=year,
        grid_ref=base_chip.grid_ref,
        measurement=measurement,
        filename=filename,
        data=data,
    )

    return chip


def load_raw(
    data_dir: typing.Union[pathlib.Path, str],
    measurement: str,
    year: int,
    summarise: bool = True,
) -> xr.DataArray:

    data_dir = pathlib.Path(data_dir)

    data = []

    for month_num in range(1, 12 + 1):

        filename = data_dir / get_raw_filename(
            measurement=measurement,
            year=year,
            month_num=month_num,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            handle = rioxarray.open_rasterio(filename=filename, masked=True)

        # the climate data is stored in a separate variable, so is a Dataset
        # rather than a DataArray
        if isinstance(handle, xr.Dataset):
            # convert to a DataArray
            handle = handle[measurement]

        try:
            data.append(handle.load())
        finally:
            handle.close()

    data = xr.concat(objs=data, dim="time")

    if summarise:

        # the crs gets lost following the summary function, so store
        crs = data.rio.crs

        if measurement == "rain":
            summ_func = data.sum
        elif measurement == "tmax":
            summ_func = data.mean
        else:
            raise ValueError("Unexpected measurement type")

        data = summ_func(dim="time", skipna=False, keep_attrs=True)

        # restore the crs
        data.rio.write_crs(crs, inplace=True)

    return data


def get_raw_filename(
    measurement: str,
    year: int,
    month_num: int,
) -> str:
    return f"ANUClimate_v2-0_{measurement}_monthly_{year}{month_num:02d}.nc"


def get_raw_years(
    data_dir: typing.Union[pathlib.Path, str],
    measurement: str,
) -> list[int]:

    data_dir = pathlib.Path(data_dir)

    filenames = data_dir.glob(f"ANUClimate_v2-0_{measurement}_monthly_*.nc")

    year_counter = collections.defaultdict(list)

    for filename in filenames:

        (*_, year_month) = filename.stem.split("_")

        (year, month_num) = (int(year_month[:4]), int(year_month[4:]))

        year_counter[year].append(month_num)

    valid_years = sorted(
        [
            year
            for (year, month_nums) in year_counter.items()
            if len(set(month_nums)) == 12
        ]
    )

    return valid_years
