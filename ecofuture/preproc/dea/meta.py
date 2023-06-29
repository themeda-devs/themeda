"""
Downloads the metadata from the DEA database.
"""

import pathlib
import typing

import boto3
import botocore
import mypy_boto3_s3.client

import yaml


BUCKET = "dea-public-data"
PRODUCT = "ga_ls_landcover_class_cyear_2"
REFYEAR = 1988


def get_metadata(
    filename: str,
    metadata_dir: typing.Union[pathlib.Path, str],
) -> dict[str, typing.Any]:
    """
    Returns the metadata associated with the spatial location of a given filename for a
    file in the DEA database.
    """

    if not filename.startswith(PRODUCT):
        raise ValueError("Unexpected filename")

    metadata_dir = pathlib.Path(metadata_dir)

    # just look at the part of the filename after the known product beginning
    filename_tail = filename.removeprefix(PRODUCT)

    # the end of the filename can vary in its `_` separators, so get the known starting
    # points - the first four
    filename_tail_components = filename_tail.split("_")[:4]

    # rebuild the filename head
    filename_head = PRODUCT + "_".join(filename_tail_components)

    metadata_filename = f"{filename_head}_{REFYEAR}-01-01.odc-metadata.yaml"

    metadata_path = metadata_dir / metadata_filename

    if not metadata_path.exists():
        raise FileNotFoundError(f"Required metadata was not found at {metadata_path}")

    with open(metadata_path, "r") as handle:
        metadata: dict[str, typing.Any] = yaml.safe_load(handle)

    return metadata


def download(
    output_dir: typing.Union[pathlib.Path, str],
    skip_existing: bool = True,
) -> None:
    """
    Downloads the metadata for all the files in the DEA database for this product.
    """

    output_dir = pathlib.Path(output_dir)

    client = init_client()

    paginator = client.get_paginator("list_objects")

    pages = paginator.paginate(
        Bucket=BUCKET,
        Prefix=f"derivative/{PRODUCT}/1-0-0/{REFYEAR}/",
    )

    for page in pages:
        for item in page["Contents"]:

            path = pathlib.Path(item["Key"])

            if path.name.endswith("metadata.yaml"):

                local_path = output_dir / path.name

                if not (skip_existing and local_path.exists()):

                    client.download_file(
                        Bucket=BUCKET,
                        Key=item["Key"],
                        Filename=str(local_path),
                    )

    client.close()


def init_client() -> mypy_boto3_s3.client.S3Client:
    config = botocore.client.Config(signature_version=botocore.UNSIGNED)

    client = boto3.client("s3", config=config)

    return client
