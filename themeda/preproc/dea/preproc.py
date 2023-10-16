"""
Acquiring and pre-processing the DEA database files into an analysis-ready format.
"""

import pathlib
import typing
import functools

import numpy as np
import numpy.typing as npt

import themeda.preproc.roi
import themeda.preproc.chips
import themeda.preproc.chiplets

import themeda.preproc.dea.meta
import themeda.preproc.dea.dload


def run(
    metadata_dir: pathlib.Path,
    chip_dir: pathlib.Path,
    chiplet_dir: pathlib.Path,
    region_file: pathlib.Path,
    measurement: str = "level4",
    overwrite: bool = False,
    show_progress: bool = True,
    download_step: str = "normal",
) -> None:

    if download_step in ["only", "normal"]:

        # download the raw metadata
        themeda.preproc.dea.meta.download(
            output_dir=metadata_dir,
            skip_existing=not overwrite,
        )

        # download the raw data
        themeda.preproc.dea.dload.download(
            output_dir=chip_dir,
            metadata_dir=metadata_dir,
            region_file=region_file,
            skip_existing=not overwrite,
            verbose=show_progress,
        )

    if download_step != "only":

        save_chiplets(
            chip_dir=chip_dir,
            chiplet_dir=chiplet_dir,
            region_file=region_file,
            overwrite=overwrite,
            show_progress=show_progress,
        )


def save_chiplets(
    chip_dir: pathlib.Path,
    chiplet_dir: pathlib.Path,
    region_file: pathlib.Path,
    overwrite: bool = False,
    show_progress: bool = True,
) -> None:

    remap_lut = get_remapping_lut()

    remapper = functools.partial(remap_data, lut=remap_lut)

    themeda.preproc.chiplets.save_chiplets(
        chip_dir=chip_dir,
        chiplet_dir=chiplet_dir,
        region_file=region_file,
        measurement="level4",
        remapper=remapper,
        show_progress=show_progress,
    )


def remap_data(
    data: npt.NDArray[np.uint8],
    lut: typing.Optional[npt.NDArray[np.uint8]] = None,
) -> npt.NDArray[np.uint8]:
    "Applies a remapping to a data array."

    if lut is None:
        lut = get_remapping_lut()

    remapped_data: npt.NDArray[np.uint8] = lut[data]

    if np.any(remapped_data > 104):
        raise ValueError("Data remapping error")

    return remapped_data


def get_remapping_lut() -> npt.NDArray[np.uint8]:
    "Gets an array that acts as a LUT for remapping."

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

    for src_val, dst_val in lut_dict.items():
        lut[src_val] = dst_val

    lut_dt: npt.NDArray[np.uint8] = lut.astype(np.uint8)

    return lut_dt
