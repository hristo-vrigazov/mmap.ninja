import os
import stat

import numpy as np
import pytest

from mmap_ninja import generic
from mmap_ninja.ragged import RaggedMmap
from joblib import delayed, Parallel


def test_base_case(tmp_path):
    simple = list(range(4))
    mmap = RaggedMmap.from_lists(tmp_path / "simple", simple)
    for i in range(4):
        assert i == mmap[i]
    assert len(mmap) == 4

    print(generic.open_existing(tmp_path / "simple"))


def test_open_existing_case(tmp_path):
    simple = list(range(4))
    mmap = RaggedMmap.from_lists(tmp_path / "simple", simple)
    mmap = RaggedMmap(tmp_path / "simple")
    for i in range(4):
        assert i == mmap[i]


def test_np_case(tmp_path):
    simple = [np.array([11, 13, -1, 17]), np.array([2, 3, 4, 19]), np.array([90, 12])]
    mmap = RaggedMmap.from_lists(tmp_path / "simple", simple)
    for i in range(3):
        assert np.allclose(simple[i], mmap[i])


def test_set_np_case(tmp_path):
    simple = [np.array([11, 13, -1, 17]), np.array([2, 3, 4, 19]), np.array([90, 12])]
    mmap = RaggedMmap.from_lists(tmp_path / "simple", simple)
    changed = np.array([-1, -1, -1, -2])
    mmap[1] = changed
    for i in range(3):
        if i == 1:
            assert np.allclose(changed, mmap[i])
        else:
            assert np.allclose(simple[i], mmap[i])


def test_set_multiple_case(tmp_path):
    simple = [np.array([11, 13, -1, 17]), np.array([2, 3, 4, 19]), np.array([90, 12])]
    mmap = RaggedMmap.from_lists(tmp_path / "simple", simple)
    changed = np.array([-1, -1, -1, -2])
    mmap[:2] = [changed, changed]
    for i in range(3):
        if i < 2:
            assert np.allclose(changed, mmap[i])
        else:
            assert np.allclose(simple[i], mmap[i])


def test_get_multiple_case(tmp_path):
    simple = [np.array([11, 13, -1, 17]), np.array([2, 3, 4, 19]), np.array([90, 12])]
    mmap = RaggedMmap.from_lists(tmp_path / "simple", simple)
    result = mmap[:2]
    for i in range(2):
        assert np.allclose(result[i], simple[i])


def test_wrapper(tmp_path):
    simple = [np.array([11, 13, -1, 17]), np.array([2, 3, 4, 19]), np.array([90, 12])]
    mmap = RaggedMmap.from_lists(tmp_path / "simple", simple, wrapper_fn=lambda x: np.array(x, dtype=np.int8))
    assert mmap[0].dtype == np.int8


def test_extend(tmp_path):
    simple = [np.array([11, 13, -1, 17]), np.array([2, 3, 4, 19]), np.array([90, 12])]
    mmap = RaggedMmap.from_lists(tmp_path / "base", simple, wrapper_fn=lambda x: np.array(x, dtype=np.int16))
    extended = [np.array([123, -1]), np.array([-1, 0, 123, 92, 12])]
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
    memmap = RaggedMmap.from_generator(tmp_path / "strings_memmap", generate_arrs(n), 4, verbose=True)
    for i in range(n):
        assert np.allclose(np.ones(12) * i, memmap[i])


@pytest.fixture
def indexable_obj(request):
    length, has_length = request.param

    class _Indexable:
        def __init__(self, _length, _has_length):
            self.length = _length
            self.has_length = _has_length

        def __len__(self):
            if not self.has_length:
                raise TypeError
            return self.length

        def __getitem__(self, item):
            if 0 <= item < self.length:
                return np.ones(12) * item
            raise IndexError(item)

    return _Indexable(length, has_length)


@pytest.mark.parametrize("n, indexable_obj", [(30, (30, True)), (3, (3, False))], indirect=["indexable_obj"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_from_indexable_obj(tmp_path, n, indexable_obj, n_jobs):
    memmap = RaggedMmap.from_indexable(tmp_path / "strings_memmap", indexable_obj, 4, n_jobs=n_jobs, verbose=True)
    for i in range(n):
        assert np.allclose(np.ones(12) * i, memmap[i])


@pytest.fixture
def indexable_func(request):

    total = request.param

    def func(item):
        if 0 <= item < total:
            return np.ones(12) * item
        raise IndexError(item)

    return func


@pytest.mark.parametrize("n, indexable_func", [(30, 30), (3, 3)], indirect=["indexable_func"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_from_indexable_func(tmp_path, n, indexable_func, n_jobs):
    memmap = RaggedMmap.from_indexable(tmp_path / "strings_memmap", indexable_func, 4, n_jobs=n_jobs, verbose=True)
    for i in range(n):
        assert np.allclose(np.ones(12) * i, memmap[i])


def test_nd_case(tmp_path):
    simple = [
        np.array([[11, 13], [-1, 17]]),
        np.array([[2, 3], [4, 19]]),
        np.array([[90], [12]]),
    ]
    mmap = RaggedMmap.from_lists(tmp_path / "base", simple, wrapper_fn=lambda x: np.array(x, dtype=np.int16))
    assert np.allclose(mmap[-1], simple[-1])


def test_different_number_of_axes(tmp_path):
    simple = [np.array([[11, 13], [-1, 17]]), np.array([2, 3]), np.array([[90], [12]])]
    mmap = RaggedMmap.from_lists(tmp_path / "base", simple, wrapper_fn=lambda x: np.array(x, dtype=np.int16))
    assert np.allclose(mmap[-1], simple[-1])


@pytest.fixture
def np_array_with_different_number_of_axes():
    return [np.array([[11, 13], [-1, 17]]), np.array([2, 3]), np.array([[90], [12]])]


def test_different_number_of_axes_gen(tmp_path, np_array_with_different_number_of_axes):
    mmap = RaggedMmap.from_generator(tmp_path / "base", np_array_with_different_number_of_axes, batch_size=1)
    assert np.allclose(mmap[-1], np_array_with_different_number_of_axes[-1])


def test_if_batch_size_exceeds_n_samples_crash(tmp_path, np_array_with_different_number_of_axes):
    mmap = RaggedMmap.from_generator(tmp_path / "base", np_array_with_different_number_of_axes, batch_size=100)
    assert np.allclose(mmap[-1], np_array_with_different_number_of_axes[-1])


def read_something(arr, i):
    return arr[i]


def test_parallel_read(tmp_path, np_array_with_different_number_of_axes):
    n_workers = 32
    RaggedMmap.from_generator(tmp_path / "base", np_array_with_different_number_of_axes, batch_size=100)
    ragged = RaggedMmap(tmp_path / "base")
    delayed_funcs = [delayed(read_something)(ragged, np.random.randint(len(ragged))) for _ in range(n_workers)]
    p = Parallel()
    r = p(delayed_funcs)
    assert len(r) == n_workers


def test_empty_init(tmp_path):
    ragged = RaggedMmap(tmp_path / "samples")
    assert len(ragged) == 0
    ragged.append(np.array([[1.0, 2.0, 3], [4.0, 5.0, 6.0]]))
    assert len(ragged) == 1


def test_ragged_index_error(tmp_path):
    ragged = RaggedMmap(tmp_path / "samples")
    with pytest.raises(IndexError):
        ragged[0]


def test_read_only(tmp_path, np_array_with_different_number_of_axes):
    out_path = tmp_path / "read_only_ragged_memmap"
    RaggedMmap.from_generator(out_path, np_array_with_different_number_of_axes, batch_size=100)
    # Make all memmap files read-only for the user
    for p in out_path.glob(r"**/*"):
        if not p.is_file():
            continue
        os.chmod(p, stat.S_IRUSR)
    # Open in read-only mode
    ragged = RaggedMmap(out_path, mode="r")
    for i, arr in enumerate(np_array_with_different_number_of_axes):
        assert np.allclose(arr, ragged[i])


def test_empty_element(tmp_path):
    arrays = [[1, 2, 3], [4, 5], [], [6, 7]]
    res = RaggedMmap.from_lists(tmp_path / "int", arrays)
    for i in range(len(arrays)):
        _ = res[i]


def test_wrapper_fn_without_copy(tmp_path, np_array_with_different_number_of_axes):
    mmap = RaggedMmap.from_generator(tmp_path / "base", np_array_with_different_number_of_axes, batch_size=1)
    mmap = RaggedMmap(tmp_path / "base", wrapper_fn=lambda x: x, copy_before_wrapper_fn=False)
    for i in range(2):
        sample = mmap[i]
