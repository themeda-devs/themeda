"""
Pre-processing the ANU climate files into an analysis-ready format.
"""

import pathlib
import typing
import functools

import ecofuture.preproc.climate.dload
import ecofuture.preproc.climate.chiplets


def run(
    raw_data_dir: pathlib.Path,
    dea_chip_dir: pathlib.Path,
    chiplet_dir: pathlib.Path,
    region_file: pathlib.Path,
    measurement: str,
    chip_dir: typing.Optional[pathlib.Path] = None,
    overwrite: bool = False,
    show_progress: bool = True,
    download_step: str = "normal",
    year: typing.Optional[int] = None,
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
    overwrite:
        Whether to overwrite already-existing chiplet files.
    """

    if download_step in ["only", "normal"]:

        # if we've been given a specific year, then use that to fix the
        # relevant parameters for the downloader
        # (the defaults are set in the function arguments, so can't just
        # pass `year` as `None`)
        if year is not None:
            dload_func = functools.partial(
                ecofuture.preproc.climate.dload.download,
                start_year=year,
                end_year=year,
            )
        else:
            dload_func = ecofuture.preproc.climate.dload.download

        # download the raw climate data
        dload_func(
            output_dir=raw_data_dir,
            measurement=measurement,
            verbose=show_progress,
        )

    if download_step != "only":

        if chip_dir is not None:

            # convert raw data to chips
            ecofuture.preproc.climate.chips.save_chips(
                raw_climate_dir=raw_data_dir,
                measurement=measurement,
                dea_chip_dir=dea_chip_dir,
                climate_chip_dir=chip_dir,
                overwrite=overwrite,
            )

            ecofuture.preproc.chiplets.save_chiplets(
                chip_dir=chip_dir,
                chiplet_dir=chiplet_dir,
                region_file=region_file,
                measurement=measurement,
                show_progress=show_progress,
                overwrite=overwrite,
            )

        else:

            chips = ecofuture.preproc.climate.chiplets.form_chips(
                raw_climate_dir=raw_data_dir,
                dea_chip_dir=dea_chip_dir,
                measurement=measurement,
                climate_chip_dir=None,
                years=[year],
            )

            for year_chips in chips:

                ecofuture.preproc.chiplets.save_chiplets(
                    chips=year_chips,
                    chip_dir=None,
                    chiplet_dir=chiplet_dir,
                    region_file=region_file,
                    measurement=measurement,
                    show_progress=show_progress,
                    overwrite=overwrite,
                )
