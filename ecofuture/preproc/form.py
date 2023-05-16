import pathlib
import typing
import datetime

import numpy as np

import xarray as xr
import rioxarray

import geojson

import dea_data_loader


def form_chiplets(
    query_name: str,
    asset_name: str,
    chip_data_dir: pathlib.Path,
    chiplet_data_dir: pathlib.Path,
    bounds_path: pathlib.Path,
    query_db_path: pathlib.Path | None = None,
    relabel: bool = True,
    n_spatial_folds: int = 5,
    chiplet_spatial_size_pix: int = 160,
    overwrite: bool = False,
    random_seed: int | None = 287411685,
    dataset: xr.Dataset | None = None,
):

    if query_db_path is None:
        query_db_path = dea_data_loader.get_default_query_db_path(
            data_dir=chip_data_dir
        )

    if dataset is None:
        dataset = dea_data_loader.form_dataset_from_query(
            name=query_name,
            data_dir=chip_data_dir,
        )

    # make sure that the time dimension is in order
    dataset = dataset.sortby(variables="time")
    # and only has the asset we're interested in
    dataset = dataset.sel(band=asset_name)

    # load the GeoJSON file that is used to provide the precise boundaries
    bounds = geojson.loads(bounds_path.read_text())

    # check that the provided chiplet size divides evenly
    check_chiplet_sizes(
        dataset=dataset,
        chiplet_spatial_size_pix=chiplet_spatial_size_pix,
    )

    test = []

    rand = np.random.default_rng(seed=random_seed)

    for chip_num, chip_data in dataset.groupby(group="chip"):

        # split into subsets
        chip_subsets = split_into_chip_subsets(
            data=chip_data,
            chiplet_spatial_size_pix=chiplet_spatial_size_pix,
        )

        for chip_subset in chip_subsets:
            test.append(
                [
                    [np.mean(chip_subset.x.values), np.mean(chip_subset.y.values)],
                    check_within_bounds(
                        data=chip_subset.isel(time=0).load(),
                        bounds=bounds,
                    )
                ]
            )

        for chip_datetime, chip_year_data in chip_data.groupby(group="time"):

            # now we have just a 2D chip array in `.data`

            for (i_chip_subset, chip_subset) in enumerate(chip_subsets):

                pass


    return test

    """
    # this contains each combination of year and spatial location (chip)
    """


def check_within_bounds(
    data: xr.DataArray,
    bounds: geojson.feature.FeatureCollection,
    bounds_crs: int =4326,
) -> bool:

    (feature,) = bounds["features"]

    try:
        clipped = data.rio.clip(
            geometries=[feature["geometry"]],
            crs=bounds_crs,
        )
    # if entire data is outside of the bounds, then an error is raised
    except rioxarray.raster_array.NoDataInBounds:
        within_bounds = False
    # otherwise, need to see if the size changed
    else:
        within_bounds = data.equals(clipped)

    return within_bounds


def split_into_chip_subsets(data: xr.Dataset, chiplet_spatial_size_pix: int) -> xr.DataArray:
    """
    Splits a chip into a bunch of smaller tiles, each represented as a xr DataArray
    """

    chip_subsets = []

    for i_x_left in range(0, data.sizes["x_offset"], chiplet_spatial_size_pix):
        for i_y_left in range(0, data.sizes["y_offset"], chiplet_spatial_size_pix):

            chip_subset = data.data.isel(
                x_offset=range(i_x_left, i_x_left + chiplet_spatial_size_pix),
                y_offset=range(i_y_left, i_y_left + chiplet_spatial_size_pix),
            )

            # convert back into absolute, rather than chip-centre-relative, coordinates
            for direction in ["x", "y"]:
                chip_subset[f"{direction}_offset"] = (
                    chip_subset[f"{direction}_offset"]
                    + data.centre_pos.sel(direction=direction).values.item()
                )

            chip_subset = chip_subset.rename({"x_offset": "x", "y_offset": "y"})

            chip_subsets.append(chip_subset)

    return chip_subsets


def get_year_from_datetime64(date_val: np.datetime64) -> int:
    # this seems more complicated than it should be
    return date_val.astype("datetime64[Y]").astype(datetime.datetime).year


def check_chiplet_sizes(
    dataset: xr.Dataset,
    chiplet_spatial_size_pix: int,
) -> None:

    for spatial_dim in ["x", "y"]:

        dim_size = dataset.sizes[f"{spatial_dim}_offset"]

        n_chiplets_per_dim = dim_size / chiplet_spatial_size_pix

        if not n_chiplets_per_dim.is_integer():
            raise ValueError(
                f"Provided chiplet size ({chiplet_spatial_size_pix}) does not "
                + "divide evenly into the chip size ({dim_size})"
            )


def sniff_query_result(
    query_result: dict[str, typing.Any],
    asset_name: str,
    chiplet_spatial_size_pix: int,
):
    """
    Go through the items returned in the query result and sniff for a few
    parameters that encompass the dataset.
    """

    years = set()
    chiplets_per_chip = None
