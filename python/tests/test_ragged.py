import numpy as np

from mmap_ninja.ragged import RaggedMmap


def test_base_case(tmp_path):
    simple = list(range(4))
    mmap = RaggedMmap.from_lists(tmp_path / 'simple', simple)
    for i in range(4):
        assert i == mmap[i]
    assert len(mmap) == 4


def test_open_existing_case(tmp_path):
    simple = list(range(4))
    mmap = RaggedMmap.from_lists(tmp_path / 'simple', simple)
    mmap = RaggedMmap.open_existing(tmp_path / 'simple')
    for i in range(4):
        assert i == mmap[i]


def test_np_case(tmp_path):
    simple = [
        np.array([11, 13, -1, 17]),
        np.array([2, 3, 4, 19]),
        np.array([90, 12])
    ]
    mmap = RaggedMmap.from_lists(tmp_path / 'simple', simple)
    for i in range(3):
        assert np.allclose(simple[i], mmap[i])


def test_set_np_case(tmp_path):
    simple = [
        np.array([11, 13, -1, 17]),
        np.array([2, 3, 4, 19]),
        np.array([90, 12])
    ]
    mmap = RaggedMmap.from_lists(tmp_path / 'simple', simple)
    changed = np.array([-1, -1, -1, -2])
    mmap[1] = changed
    for i in range(3):
        if i == 1:
            assert np.allclose(changed, mmap[i])
        else:
            assert np.allclose(simple[i], mmap[i])


def test_set_multiple_case(tmp_path):
    simple = [
        np.array([11, 13, -1, 17]),
        np.array([2, 3, 4, 19]),
        np.array([90, 12])
    ]
    mmap = RaggedMmap.from_lists(tmp_path / 'simple', simple)
    changed = np.array([-1, -1, -1, -2])
    mmap[:2] = [changed, changed]
    for i in range(3):
        if i < 2:
            assert np.allclose(changed, mmap[i])
        else:
            assert np.allclose(simple[i], mmap[i])


def test_get_multiple_case(tmp_path):
    simple = [
        np.array([11, 13, -1, 17]),
        np.array([2, 3, 4, 19]),
        np.array([90, 12])
    ]
    mmap = RaggedMmap.from_lists(tmp_path / 'simple', simple)
    result = mmap[:2]
    for i in range(2):
        assert np.allclose(result[i], simple[i])


def test_wrapper(tmp_path):
    simple = [
        np.array([11, 13, -1, 17]),
        np.array([2, 3, 4, 19]),
        np.array([90, 12])
    ]
    mmap = RaggedMmap.from_lists(tmp_path / 'simple', simple, wrapper_fn=lambda x: np.array(x, dtype=np.int8))
    assert mmap[0].dtype == np.int8
