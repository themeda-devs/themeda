import importlib.resources
from typing import Dict, List
import csv


def get_land_cover_column(column) -> List[str]:
    result = []
    csv_path = importlib.resources.files("themeda_preproc.resources.relabel").joinpath("LCNS_codes_colours.csv")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(f=handle)
        for index, row in enumerate(reader):
            if not row['LCNS_n']:
                break
            if index != int(row['LCNS_n']):
                raise ValueError(f"Error reading land cover CSV")

            result.append(row[column])

    return result


def get_land_cover_colours() -> Dict[str,str]:
    labels = get_land_cover_column("LCNS_label")
    colours = get_land_cover_column("LCNS_HexCol")
    return {label: colour for label, colour in zip(labels, colours)}


LEVEL4_COLOURS = get_land_cover_colours()