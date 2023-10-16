import pathlib

import numpy as np
import numpy.typing as npt

import xarray as xr

import shapely
import pyproj


def is_bbox_within_region(
    bbox: shapely.geometry.polygon.Polygon,
    region: shapely.geometry.polygon.Polygon,
) -> bool:
    # "Returns True if geometry A is completely inside geometry B"
    is_within: bool = bbox.within(other=region)
    return is_within


def get_bbox(data: xr.DataArray) -> shapely.geometry.polygon.Polygon:
    "Gets the bounding-box of a chip or chiplet's data"

    bbox = shapely.geometry.box(
        minx=data.x.min().item(),
        miny=data.y.min().item(),
        maxx=data.x.max().item(),
        maxy=data.y.max().item(),
    )

    return bbox


def get_region(
    region_file: pathlib.Path,
    src_crs: int = 4326,
    dst_crs: int = 3577,
) -> shapely.geometry.polygon.Polygon:
    """
    Loads a spatial region-of-interest from a GeoJSON file, optionally converts the
    projection, and formats the region as a `shapely` Polygon.
    """

    geoms = shapely.from_geojson(geometry=region_file.read_text())

    try:
        (region,) = geoms.geoms
    except ValueError:
        raise ValueError("Unexpected number of components in the region file")

    if src_crs != dst_crs:
        transformer = pyproj.Transformer.from_crs(
            crs_from=src_crs,
            crs_to=dst_crs,
            always_xy=True,
        )

        def shapely_transform(points: npt.NDArray[float]) -> npt.NDArray[float]:
            return np.column_stack(
                transformer.transform(
                    xx=points[:, 0],
                    yy=points[:, 1],
                    errcheck=True,
                )
            )

        region = shapely.transform(
            geometry=region,
            transformation=shapely_transform,
        )

    # might speed up calls that use the geometry
    shapely.prepare(geometry=region)

    return region
