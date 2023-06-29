"""
Downloading the raw data from the ANU Climate Database.
"""

import pathlib
import typing

import ecofuture.preproc.dload


def download(
    output_dir: typing.Union[pathlib.Path, str],
    measurement: str,
    skip_existing: bool = True,
    start_year: int = 1900,
    end_year: int = 2024,
    verbose: bool = True,
) -> None:

    if measurement not in ["rain", "tmax"]:
        raise ValueError(f"Unknown measurement: {measurement}")

    output_dir = pathlib.Path(output_dir)

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"The output path {output_dir} exists but is not a directory")

    if not output_dir.exists():
        output_dir.mkdir()

    year = start_year

    protocol = "https://"

    n_years_downloaded = 0

    while n_years_downloaded == 0 or year <= end_year:

        for month_num in range(1, 12 + 1):

            remote_path = get_remote_path(
                measurement=measurement,
                year=year,
                month_num=month_num,
            )

            output_path = output_dir / remote_path.name

            if output_path.exists() and skip_existing:
                if verbose:
                    print(f"Output path {remote_path.name} exists; skipping")
                continue

            url = protocol + str(remote_path)

            try:
                ecofuture.preproc.dload.download_file(
                    url=url, output_path=output_path, overwrite=True
                )
            except ValueError:
                if verbose:
                    print(f"Output path {remote_path.name} not found; skipping year")
                    break

            if verbose:
                print(f"Saved {remote_path.name}")

            n_years_downloaded += 1

        year += 1


def get_remote_path(
    measurement: str,
    year: int,
    month_num: int,
) -> pathlib.Path:

    remote_path = pathlib.Path(
        "dapds00.nci.org.au/"
        + "thredds/fileServer/gh70/ANUClimate/v2-0/stable/month/"
        + f"{measurement}/{year}/ANUClimate_v2-0_{measurement}_"
        + f"monthly_{year}{month_num:02d}.nc"
    )

    return remote_path
