import numpy as np

from mmap_ninja import numpy


def test_numpy(tmp_path):
    arr = np.arange(10)
    memmap = numpy.from_ndarray(arr, tmp_path / 'numpy_mmap')

    for i, el in enumerate(arr):
        assert el == memmap[i]


def test_numpy_from_empty(tmp_path):
    arr = np.arange(10)
    memmap = numpy.empty(tmp_path / 'numpy_mmap', dtype=np.int64, shape=(10,), order='C')
    memmap[:] = arr
