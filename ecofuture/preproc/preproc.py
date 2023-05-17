import pathlib
import dataclasses

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


@dataclasses.dataclass(frozen=True)
class Chip:
    filename: pathlib.Path
    year: int
    position: Position


@dataclasses.dataclass(frozen=True)
class Chiplet(Chip):
    data: xr.DataArray


def save_chiplets(
    chip_dir: pathlib.Path,
    chiplet_dir: pathlib.Path,
    region_file: pathlib.Path,
    chiplet_spatial_size_pix: int = 160,
    overwrite: bool = False,
) -> None:

    product = "ga_ls_landcover_class_cyear_2"
    measurement = "level4"

    filenames = get_filenames(
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


def render_chiplets(
    chiplets: dict[Position, dict[int, Chiplet]],
    chiplet_dir: pathlib.Path,
    fold_seed: int | None = None,
) -> None:
    pass


def form_chiplets(
    chips: dict[Position, dict[int, Chip]],
    chiplet_spatial_size_pix: int,
    region: shapely.geometry.polygon.Polygon,
) -> dict[Position, dict[int, Chiplet]]:

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

        rep_chip_splits = split_chip(
            data=rep_chip_data,
            chiplet_spatial_size_pix=chiplet_spatial_size_pix,
            region=region,
        )

        # get the validity (whether the chiplet is in the region) of each chiplet
        validity = [
            rep_chip_split.in_region
            for rep_chip_split in rep_chip_splits
        ]

        n_valid_chiplets = sum(validity)

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

            for (chiplet_validity, potential_chiplet_data) in zip(
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
                chip_subset.attrs["in_region"] = bbox.within(other=region)

            chip_subsets.append(chip_subset)

    return chip_subsets


def get_bbox(data: xr.DataArray) -> shapely.geometry.polygon.Polygon:

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

    chips: dict[Position, dict[int, Chip]] = {}

    for filename in filenames:

        chip = parse_filename(filename=filename)

        if chip.position not in chips:
            chips[chip.position] = {}

        chips[chip.position][chip.year] = chip

    return chips


def parse_filename(filename: pathlib.Path) -> Chip:

    # example: ga_ls_landcover_class_cyear_2_1-0-0_au_x9y-24_1993-01-01_level4

    (*_, xy, date, measurement) = filename.name.split("_")

    if not xy.startswith("x"):
        raise ValueError("Unexpected filename format")

    x = float(xy[1:xy.index("y")])
    y = float(xy[xy.index("y") + 1:])

    if not date.count("-") == 2:
        raise ValueError("Unexpected filename format")

    (year, *_) = date.split("-")

    chip = Chip(
        filename=filename,
        year=int(year),
        position=Position(x=x, y=y),
    )

    return chip


def get_filenames(
    chip_dir: pathlib.Path,
    product: str = "ga_ls_landcover_class_cyear_2",
    year: str | int | None = None,
    measurement: str | None = None,
) -> list[pathlib.Path]:

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

    if lut is None:
        lut = get_remapping_lut()

    remapped_data: npt.NDArray[np.uint8] = lut[data]

    if np.any(remapped_data > 104):
        raise ValueError("Data remapping error")

    return remapped_data



def get_remapping_lut() -> npt.NDArray[np.uint8]:

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

    for (src_val, dst_val) in lut_dict.items():
        lut[src_val] = dst_val

    lut_dt: npt.NDArray[np.uint8] = lut.astype(np.uint8)

    return lut_dt


