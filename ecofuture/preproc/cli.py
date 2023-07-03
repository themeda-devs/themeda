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
        "-chip_dir",
        type=pathlib.Path,
    )

    parser.add_argument(
        "-chiplet_dir",
        type=pathlib.Path,
        required=True,
    )

    parser.add_argument(
        "-raw_dir",
        type=pathlib.Path,
    )

    parser.add_argument(
        "-region_file",
        type=pathlib.Path,
        required=True,
    )

    parser.add_argument(
        "-dea_chip_dir",
        type=pathlib.Path,
        required=False,
    )

    parser.add_argument(
        "-overwrite",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-download_step",
        choices=["skip", "only", "normal"],
        default="normal",
    )

    parser.add_argument(
        "-hide_progress",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-year",
        type=int,
        required=False,
    )

    args = parser.parse_args()

    if args.measurement == "level4":
        ecofuture.preproc.dea.preproc.run(
            metadata_dir=args.raw_dir,
            chip_dir=args.chip_dir,
            chiplet_dir=args.chiplet_dir,
            region_file=args.region_file,
            overwrite=args.overwrite,
            show_progress=not args.hide_progress,
            download_step=args.download_step,
        )

    elif args.measurement in ["rain", "tmax"]:

        if args.dea_chip_dir is None:
            raise ValueError(
                "For this measurement, the `dea_chip_dir` argument "
                + "needs to be provided"
            )

        ecofuture.preproc.climate.preproc.run(
            raw_data_dir=args.raw_dir,
            chip_dir=args.chip_dir,
            dea_chip_dir=args.dea_chip_dir,
            chiplet_dir=args.chiplet_dir,
            region_file=args.region_file,
            measurement=args.measurement,
            overwrite=args.overwrite,
            show_progress=not args.hide_progress,
            download_step=args.download_step,
            year=args.year,
        )


if __name__ == "__main__":
    run()
