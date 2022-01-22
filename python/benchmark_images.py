import sys
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from time import time

import pandas as pd

import matplotlib.pyplot as plt

from mmap_ninja.ragged import RaggedMmap

path = Path(sys.argv[1])
data_path = Path(sys.argv[2])


# https://stackoverflow.com/a/1392549/6636290
def get_dir_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


# Modified https://stackoverflow.com/a/31631711/6636290
def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    GB = float(KB ** 3)  # 1,073,741,824
    return '{0:.6f} GB'.format(B / GB)


def main():
    df_data = defaultdict(list)

    start_t = time()
    in_memory = np.load(data_path / 'in_memory.npy', allow_pickle=True)
    df_data['in_memory'].append(time() - start_t)
    start_t = time()
    for i in range(len(in_memory)):
        img = in_memory[i]
    df_data['in_memory'].append(time() - start_t)
    df_data['in_memory'].append(humanbytes(sum([arr.nbytes for arr in in_memory])))
    df_data['in_memory'].append(humanbytes(os.path.getsize(data_path / 'in_memory.npy')))

    start_t = time()
    training_images = RaggedMmap(data_path / 'training_images')
    df_data['ragged_mmap'].append(time() - start_t)
    start_t = time()
    for i in range(len(training_images)):
        img = training_images[i]
    df_data['ragged_mmap'].append(time() - start_t)
    biggest_image_nbytes = in_memory[training_images.flattened_shapes.argmax()].nbytes
    df_data['ragged_mmap'].append(humanbytes(biggest_image_nbytes))
    df_data['ragged_mmap'].append(humanbytes(get_dir_size(data_path / 'training_images')))

    df_data['imread_from_disk'].append(0.)
    start_t = time()
    for file_path in path.iterdir():
        img = plt.imread(file_path)
    df_data['imread_from_disk'].append(time() - start_t)
    df_data['imread_from_disk'].append(humanbytes(biggest_image_nbytes))
    df_data['imread_from_disk'].append(humanbytes(get_dir_size(path)))

    benchmark = pd.DataFrame(df_data).T
    benchmark.columns = ['Initial load (s)', 'Time for iteration (s)', 'Memory usage (GB)', 'Disk usage (GB)']

    print()

    print(benchmark.to_markdown(floatfmt=".6f"))


if __name__ == '__main__':
    main()
