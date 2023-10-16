import pathlib
import dataclasses
import typing
import warnings

import xarray as xr
import rioxarray

import shapely

from .roi import get_bbox, is_bbox_within_region


@dataclasses.dataclass(frozen=True)
class GridRef:
    x: int
    y: int


# note that chips have a 'grid reference' rather than a 'position'
# see https://docs.dea.ga.gov.au/reference/collection_3_summary_grid.html
@dataclasses.dataclass
class Chip:
    year: int
    measurement: str
    grid_ref: GridRef
    filename: typing.Optional[pathlib.Path] = None
    data: typing.Optional[xr.DataArray] = None


# e.g., {1988: Chip(...), 1989: Chip(...), ...}
YearChipCollectionType = dict[int, Chip]
# e.g., {GridRef(x=..., y=...): {1988: Chip(...), ...}}
SpatialChipCollectionType = dict[GridRef, YearChipCollectionType]


def read_chip(filename: pathlib.Path, load_data: bool = False) -> xr.DataArray:
    "Reads a chip file from disk"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        data = rioxarray.open_rasterio(filename=filename)

    if not isinstance(data, xr.DataArray):
        raise ValueError("Unexpected data format")

    if "band" in data.dims:
        # has a redundant `band` dimension
        data = data.squeeze(dim="band", drop=True)

    if load_data:
        data.load()

    return data


def split_chip(
    data: xr.DataArray,
    chiplet_spatial_size_pix: int,
    region: typing.Optional[shapely.geometry.polygon.Polygon] = None,
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
                in_region = is_bbox_within_region(bbox=bbox, region=region)
                chip_subset.attrs["in_region"] = in_region

            chip_subsets.append(chip_subset)

    return chip_subsets


def parse_chip_filenames(
    filenames: list[pathlib.Path],
) -> SpatialChipCollectionType:
    "Parses the metadata in a set of chip filenames"

    chips: SpatialChipCollectionType = {}

    for filename in filenames:
        chip = parse_chip_filename(filename=filename)

        if chip.grid_ref not in chips:
            chips[chip.grid_ref] = {}

        chips[chip.grid_ref][chip.year] = chip

    return chips


def parse_chip_filename(filename: pathlib.Path) -> Chip:
    "Parses the metadata in a chip filename"

    # example: ga_ls_landcover_class_cyear_2_1-0-0_au_x9y-24_1993-01-01_level4

    (*_, xy, date, measurement) = filename.stem.split("_")

    if not xy.startswith("x"):
        raise ValueError("Unexpected filename format")

    x = float(xy[1 : xy.index("y")])
    y = float(xy[xy.index("y") + 1 :])

    if any([not x.is_integer() or not y.is_integer()]):
        raise ValueError("Grid reference is unexpectedly float")

    x = int(x)
    y = int(y)

    (year, *_) = date.split("-")

    chip = Chip(
        filename=filename,
        year=int(year),
        grid_ref=GridRef(x=x, y=y),
        measurement=measurement,
    )

    return chip


def get_chip_filenames(
    chip_dir: pathlib.Path,
    year: typing.Optional[typing.Union[str, int]] = None,
    measurement: typing.Optional[str] = None,
) -> list[pathlib.Path]:

    glob = "*"

    if year is not None:
        glob += f"{year}*"

    if measurement is not None:
        glob += measurement

    glob += ".tif"

    filenames = sorted(chip_dir.glob(glob))

    return filenames
