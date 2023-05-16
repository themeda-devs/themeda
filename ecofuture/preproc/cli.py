import pathlib
import argparse

import ecofuture.preproc.form


def run():

    # using `argparse` rather than `typer` because other project dependencies
    # require a very old version of `typer` that is missing functionality

    parser = argparse.ArgumentParser(
        description=(
            "Convert the data from a DEA dataset query into a set of 'chiplets' - "
            + "files that contain the image from a given year and asset over a "
            + "small spatial window"
        ),
    )

    parser.add_argument(
        "-query_name",
        required=True,
        help=(
            "Name of the previously-run query to `dea_data_loader query`"
        ),
    )

    parser.add_argument(
        "-chip_data_dir",
        required=True,
        type=pathlib.Path,
        help="Directory containing the result from `dea_data_loader download`",
    )

    parser.add_argument(
        "-chiplet_data_dir",
        required=True,
        type=pathlib.Path,
        help="Directory to write the chiplet files",
    )

    parser.add_argument(
        "-asset_name",
        required=False,
        help=(
            "Name of the asset to extract from within the query. Not required "
            + "if the query was only run with one asset"
        ),
    )

    parser.add_argument(
        "-query_db_path",
        required=False,
        type=pathlib.Path,
        default=None,
        help="Path to the query database provided to `dea_data_loader`",
    )

    parser.add_argument(
        "-no_relabelling",
        required=False,
        default=False,
        action="store_true",
        help=(
            "The labels for each file are remapped to our convention by default; use" +
            + "this flag to disable",
        ),
    )

    parser.add_argument(
        "-n_spatial_folds",
        required=False,
        type=int,
        default=5,
        help=(
            "Number of spatial folds to use; each chiplet is randomly assigned a "
            + "fold from 1 to `n_spatial_folds`"
        ),
    )

    parser.add_argument(
        "-random_seed",
        required=False,
        default=False,
        type=int,
        help="Seed for the random assignment of chiplets to folds",
    )

    parser.add_argument(
        "-chiplet_spatial_size_pix",
        required=False,
        type=int,
        default=160,
        help=(
            "Spatial extent of each side of each square chiplet. Must divide evenly "
            + "with the chip size"
        ),
    )

    parser.add_argument(
        "-overwrite",
        required=False,
        default=False,
        action="store_true",
        help="Whether it is OK to overwrite if the chiplet file already exists",
    )

    args = parser.parse_args()

    ecofuture.preproc.form_chiplets(
        query_name=args.query_name,
        asset_name=args.asset_name,
        chip_data_dir=args.chip_data_dir,
        chiplet_data_dir=args.chiplet_data_dir,
        query_db_path=args.query_db_path,
        relabel=not args.no_relabelling,
        n_spatial_folds=args.n_spatial_folds,
        chiplet_spatial_size_pix=args.chiplet_spatial_size_pix,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
    ):


if __name__ == "__main__":
    run()
