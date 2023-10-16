import os
import csv
from pathlib import Path
def read_last_n_lines(filename, n):
    with open(filename, 'rb') as f:
        lines = []
        f.seek(0, os.SEEK_END)
        position = f.tell()
        while len(lines) < n:
            try:
                f.seek(position)
                if f.read(1) == b'\n':
                    line = f.readline().decode().strip()
                    if line:
                        lines.append(line)
            except OSError:
                f.seek(0)
                line = f.readline().decode().strip()
                if line:
                    lines.append(line)
                break
            position -= 1
    return lines[::-1]

def write_to_csv(list_outs,out_csv_name,header):
    rows = []
    for out in list_outs:
        elements_to_write = out[2:]
        values = [element.split()[-1] for element in elements_to_write]
        rows.append(values)
    with open(out_csv_name, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)

last_n_lines = 8
results_dir = '/data/gpfs/projects/punim1932/themeda/'
directory = Path(results_dir)
slurm_jobs_name = 'slurm-49066091_'
out_csv_name = 'validation-persistence.csv'
header = ["subset", "loss", "categorical_accuracy", "kl_divergence_proportions", "generalized_dice","smooth_l1_rain","smooth_l1_tmax"]
files = [read_last_n_lines(file,last_n_lines) for file in directory.glob(f'{slurm_jobs_name}*.out')]
write_to_csv(files,out_csv_name,header)


