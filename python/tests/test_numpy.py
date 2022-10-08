import numpy as np

from mmap_ninja import numpy as np_ninja


def test_numpy(tmp_path):
    arr = np.arange(10)
    memmap = np_ninja.from_ndarray(tmp_path / 'numpy_mmap', arr)

    for i, el in enumerate(arr):
        assert el == memmap[i]


def test_numpy_from_empty(tmp_path):
    arr = np.arange(10)
    memmap = np_ninja.empty(tmp_path / 'numpy_mmap', dtype=np.int64, shape=(10,), order='C')
    memmap[:] = arr


def test_numpy_extend(tmp_path):
    arr = np.arange(3)
    memmap = np_ninja.from_ndarray(tmp_path / 'growable', arr)
    np_ninja.extend_dir(tmp_path / 'growable', np.arange(11, 13))
    memmap = np_ninja.open_existing(tmp_path / 'growable')
    assert memmap[0] == 0
    assert memmap[1] == 1
    assert memmap[2] == 2
    assert memmap[3] == 11
    assert memmap[4] == 12


def test_numpy_extend_alternative_api(tmp_path):
    arr = np.arange(3)
    memmap = np_ninja.from_ndarray(tmp_path / 'growable', arr)
    np_ninja.extend(memmap, np.arange(11, 13))
    memmap = np_ninja.open_existing(tmp_path / 'growable')
    assert memmap[0] == 0
    assert memmap[1] == 1
    assert memmap[2] == 2
    assert memmap[3] == 11
    assert memmap[4] == 12


def simple_gen():
    for i in range(30):
        yield i


def test_numpy_from_generator(tmp_path):
    memmap = np_ninja.from_generator(simple_gen(), tmp_path / 'generator', n=30, batch_size=4, verbose=True)
    for i in range(30):
        assert i == memmap[i]

