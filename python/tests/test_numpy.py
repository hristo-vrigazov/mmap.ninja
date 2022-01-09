import numpy as np

from mmap_ninja.numpy import NumpyMmap


def test_numpy(tmp_path):
    arr = np.arange(10)
    memmap = NumpyMmap.from_ndarray(arr, tmp_path / 'numpy_mmap').memmap

    for i, el in enumerate(arr):
        assert el == memmap[i]
