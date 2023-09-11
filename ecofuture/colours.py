import importlib.resources
from typing import Dict
import csv


def get_land_cover_colours() -> Dict[str,str]:
    result = {}
    csv_path = importlib.resources.files("ecofuture_preproc.resources.relabel").joinpath("LCNS_codes_colours.csv")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(f=handle)
        for index, row in enumerate(reader):
            label = row["LCNS_label"]
            colour = row["LCNS_HexCol"]
            if not label:
                break
            assert int(row['LCNS_n']) == index

            result[ label ] = colour
    return result

LEVEL4_COLOURS = get_land_cover_colours()