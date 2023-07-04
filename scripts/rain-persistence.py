from ecofuture.transforms import ChipletBlock
from ecofuture.dataloaders import get_chiplets_list
from ecofuture.apps import get_dates, Interval
from pathlib import Path
import typer
from rich.progress import track

import torch.nn.functional as F

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

    block = ChipletBlock(base_dir=chiplet_dir, dates=dates, max_years=max_years, pad=False)

    persistence_total = 0.0
    total = 0.0
    pixel_count = 0

    with open("rain-persistence.csv", "w") as f:
        print("subset", "id", "x", "y", "persistence_mean", sep=",", file=f)
        for chiplet in track(chiplets, "Chiplets:"):
            position = block.get_position(chiplet)
            breakpoint()
            t = block(chiplet)

            # After you've found the mean and standard deviation
            # We can normalise the value
            # t = (t - mean)/std

            total += t.mean()

            time_t_plus_1 = t[1:]
            time_t = t[:-1]
            persistent = F.smooth_l1_loss(time_t, time_t_plus_1)

            persistence_mean = persistent.float().mean().item()
            persistence_total += persistence_mean
            print(chiplet.subset, chiplet.id, position[0], position[1], persistence_mean, sep=",", file=f, flush=True)

    chiplet_count = len(chiplets)
    mean = total/chiplet_count
    print("mean", mean)
    #print("std", std)

    print("persistence:", persistence_total/len(chiplets))

if __name__ == "__main__":
    typer.run(main)
