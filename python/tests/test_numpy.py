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


def test_numpy_extend(tmp_path):
    arr = np.arange(3)
    memmap = numpy.from_ndarray(arr, tmp_path / 'growable')
    numpy.extend_dir(tmp_path / 'growable', np.arange(11, 13))
    memmap = numpy.open_existing(tmp_path / 'growable')
    assert memmap[0] == 0
    assert memmap[1] == 1
    assert memmap[2] == 2
    assert memmap[3] == 11
    assert memmap[4] == 12
