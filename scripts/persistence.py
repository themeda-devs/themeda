from themeda.transforms import ChipletBlock
from themeda.dataloaders import get_chiplets_list
from themeda.apps import get_dates, Interval
from pathlib import Path
import typer
from rich.progress import track

def main(
    chiplet_dir: Path, 
    max_chiplets:int=0,
    start:str=typer.Option("1988-01-01", help="The start date."),
    end:str=typer.Option("2018-01-01", help="The end date."),
    interval:Interval=typer.Option(Interval.YEARLY.value, help="The time interval to use."),
    max_years:int=0,
):
    chiplets = get_chiplets_list(chiplet_dir, max_chiplets)
    dates = get_dates(start=start, end=end, interval=interval)        

    block = ChipletBlock(base_dir=chiplet_dir, dates=dates, max_years=max_years)

    persistence_total = 0.0
    with open("persistence.csv", "w") as f:
        print("subset", "id", "x", "y", "persistence_mean", sep=",", file=f)
        for chiplet in track(chiplets, "Chiplets:"):
            position = block.get_position(chiplet)
            t = block.tuple_to_tensor(chiplet)
            persistent = t[1:] == t[:-1]
            persistence_mean = persistent.float().mean().item()
            persistence_total += persistence_mean
            print(chiplet.subset, chiplet.id, position[0], position[1], persistence_mean, sep=",", file=f, flush=True)

    print("persistence:", persistence_total/len(chiplets))

if __name__ == "__main__":
    typer.run(main)
