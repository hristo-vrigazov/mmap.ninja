import numpy as np
import pytest

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
    mmap = RaggedMmap(tmp_path / 'simple')
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


def test_extend(tmp_path):
    simple = [
        np.array([11, 13, -1, 17]),
        np.array([2, 3, 4, 19]),
        np.array([90, 12])
    ]
    mmap = RaggedMmap.from_lists(tmp_path / 'base', simple, wrapper_fn=lambda x: np.array(x, dtype=np.int16))
    extended = [
        np.array([123, -1]),
        np.array([-1, 0, 123, 92, 12])
    ]
    mmap.extend(extended)
    assert np.allclose(mmap[3], extended[0])
    assert np.allclose(mmap[4], extended[1])
    new_arr = np.array([-1, -2, 628])
    mmap.append(new_arr)
    assert np.allclose(mmap[5], new_arr)


def generate_arrs(n):
    for i in range(n):
        yield np.ones(12) * i


@pytest.mark.parametrize("n", [30, 3])
def test_from_generator(tmp_path, n):
    memmap = RaggedMmap.from_generator(tmp_path / 'strings_memmap', generate_arrs(n), 4, verbose=True)
    for i in range(n):
        assert np.allclose(np.ones(12) * i, memmap[i])


def test_nd_case(tmp_path):
    simple = [
        np.array([
            [11, 13],
            [-1, 17]
        ]),
        np.array([
            [2, 3],
            [4, 19]
        ]),
        np.array([[90], [12]])
    ]
    mmap = RaggedMmap.from_lists(tmp_path / 'base', simple, wrapper_fn=lambda x: np.array(x, dtype=np.int16))
    assert np.allclose(mmap[-1], simple[-1])


def test_different_number_of_axes(tmp_path):
    simple = [
        np.array([
            [11, 13],
            [-1, 17]
        ]),
        np.array([2, 3]),
        np.array([[90], [12]])
    ]
    mmap = RaggedMmap.from_lists(tmp_path / 'base', simple, wrapper_fn=lambda x: np.array(x, dtype=np.int16))
    assert np.allclose(mmap[-1], simple[-1])


def test_different_number_of_axes_gen(tmp_path):
    simple = [
        np.array([
            [11, 13],
            [-1, 17]
        ]),
        np.array([2, 3]),
        np.array([[90], [12]])
    ]
    mmap = RaggedMmap.from_generator(tmp_path / 'base', simple, batch_size=1)
    assert np.allclose(mmap[-1], simple[-1])

