import pathlib
import typing

import numpy as np

import shapely

import tqdm

import ecofuture.preproc.dload
import ecofuture.preproc.roi
import ecofuture.preproc.chips

import ecofuture.preproc.dea.meta


def get_valid_grid_refs(
    region_file: typing.Union[pathlib.Path, str],
    metadata_dir: typing.Union[pathlib.Path, str],
) -> list[ecofuture.preproc.chips.GridRef]:

    region = ecofuture.preproc.roi.get_region(region_file=pathlib.Path(region_file))

    metadata_dir = pathlib.Path(metadata_dir)

    metadata_paths = sorted(metadata_dir.glob("*metadata.yaml"))

    grid_refs = []

    for metadata_path in metadata_paths:

        metadata = ecofuture.preproc.dea.meta.get_metadata(
            filename=metadata_path.name,
            metadata_dir=metadata_dir,
        )

        chip_region = get_chip_region_from_metadata(metadata=metadata)

        intersecting = is_chip_intersecting_region(
            chip_region=chip_region,
            region=region,
        )

        if intersecting:
            chip_metadata = ecofuture.preproc.chips.parse_chip_filename(
                filename=pathlib.Path(metadata["measurements"]["level4"]["path"])
            )
            grid_refs.append(chip_metadata.grid_ref)

    return grid_refs


def get_chip_region_from_metadata(
    metadata: dict[str, typing.Any]
) -> shapely.geometry.polygon.Polygon:

    if metadata["crs"] != "epsg:3577":
        raise ValueError("Unexpected CRS")

    coords = np.array(metadata["geometry"]["coordinates"][0])

    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError("Unexpected coordinates")

    chip_region = shapely.Polygon(shell=coords)

    if len(chip_region.interiors) != 0:
        raise ValueError("Unexpected region shape")

    return chip_region


def is_chip_intersecting_region(
    chip_region: shapely.geometry.polygon.Polygon,
    region: shapely.geometry.polygon.Polygon,
) -> bool:
    # "Returns True if A and B share any portion of space"
    intersects: bool = chip_region.intersects(other=region)
    return intersects


def download(
    output_dir: typing.Union[pathlib.Path, str],
    region_file: typing.Union[pathlib.Path, str],
    metadata_dir: typing.Union[pathlib.Path, str],
    skip_existing: bool = True,
    start_year: int = 1988,
    end_year: int = 2020,
    verbose: bool = True,
) -> None:
    """
    Download raw files from the ANU Climate database.
    """

    output_dir = pathlib.Path(output_dir)

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"The output path {output_dir} exists but is not a directory")

    if not output_dir.exists():
        output_dir.mkdir()

    # get the chip grid refs that are valid given the region of interest
    grid_refs = get_valid_grid_refs(region_file=region_file, metadata_dir=metadata_dir)

    measurement = "level4"

    protocol = "https://"

    if verbose:
        progress_bar = tqdm.tqdm(
            iterable=None, total=len(grid_refs) * (end_year - start_year + 1)
        )

    for year in range(start_year, end_year + 1):

        for grid_ref in grid_refs:

            remote_path = get_remote_path(
                grid_ref=grid_ref,
                year=year,
                measurement=measurement,
            )

            output_path = output_dir / remote_path.name

            if not (output_path.exists() and skip_existing):

                url = protocol + str(remote_path)

                ecofuture.preproc.dload.download_file(
                    url=url,
                    output_path=output_path,
                    overwrite=True,
                )

            if verbose:
                progress_bar.update()

    if verbose:
        progress_bar.close()


def get_remote_path(
    grid_ref: ecofuture.preproc.chips.GridRef,
    year: int,
    measurement: str = "level4",
) -> pathlib.Path:

    filename = (
        "ga_ls_landcover_class_cyear_2_1-0-0_au_"
        + f"x{grid_ref.x:d}y{grid_ref.y:d}_{year}-01-01_{measurement}.tif"
    )

    host = pathlib.Path("data.dea.ga.gov.au")

    remote_path = (
        host
        / "derivative"
        / "ga_ls_landcover_class_cyear_2"
        / "1-0-0"
        / str(year)
        / f"x_{grid_ref.x:d}"
        / f"y_{grid_ref.y:d}"
        / filename
    )

    return remote_path
