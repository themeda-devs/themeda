"""
Pre-processing launcher with arguments set for running on `spartan`.
"""


import argparse
import pathlib

import ecofuture.preproc.dea.preproc
import ecofuture.preproc.climate.preproc


def run() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-measurement",
        required=True,
        choices=["level4", "rain", "tmax"],
    )

    parser.add_argument(
        "-download_step",
        choices=["skip", "only", "normal"],
        default="normal",
    )

    parser.add_argument(
        "-dont_save_chips",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-year",
        type=int,
        required=False,
    )

    args = parser.parse_args()

    if args.dont_save_chips and args.measurement not in ["rain", "tmax"]:
        raise ValueError("Can only not save chips for the ANU climate data")

    project_dir = pathlib.Path("/data/gpfs/projects/punim1932")

    if not project_dir.exists():
        raise ValueError(
            "The project directory does not exist; this needs to be run on spartan"
        )

    base_data_dir = project_dir / "Data"

    chip_dir = base_data_dir / "chips" / args.measurement
    chiplet_dir = base_data_dir / "chiplets" / args.measurement
    region_file = base_data_dir / "NAust_mask_IBRA_WGS1984.geojson"

    raw_data_dir = base_data_dir / "raw" / args.measurement

    if args.measurement in ["rain", "tmax"]:

        dea_chip_dir = base_data_dir / "chips" / "level4"

        if args.dont_save_chips:
            chip_dir = None

        ecofuture.preproc.climate.preproc.run(
            raw_data_dir=raw_data_dir,
            chip_dir=chip_dir,
            dea_chip_dir=dea_chip_dir,
            chiplet_dir=chiplet_dir,
            region_file=region_file,
            measurement=args.measurement,
            download_step=args.download_step,
            year=args.year,
        )

    elif args.measurement == "level4":

        ecofuture.preproc.dea.preproc.run(
            metadata_dir=raw_data_dir,
            chip_dir=chip_dir,
            chiplet_dir=chiplet_dir,
            region_file=region_file,
            download_step=args.download_step,
        )


if __name__ == "__main__":
    run()
