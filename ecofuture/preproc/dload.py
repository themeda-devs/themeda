import pathlib
import typing

import requests


def download_file(
    url: str,
    output_path: typing.Union[pathlib.Path, str],
    overwrite: bool = False,
) -> None:
    """
    Downloads a file from a URL, raising a `ValueError` if there is a problem.
    """

    output_path = pathlib.Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file {output_path} already exists")

    response = requests.get(url=url, stream=True)

    if not response.ok:
        raise ValueError(
            f"Error downloading {url}; response code was {response.status_code}"
        )

    with open(output_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            handle.write(chunk)
