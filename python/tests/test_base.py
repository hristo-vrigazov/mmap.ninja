import numpy as np

from mmap_ninja import base


def test_int_conversions():
    a = 17
    assert a == base._bytes_to_int(base._int_to_bytes(a))


def test_int_file_conversions(tmp_path):
    a = 17
    file = tmp_path / "int"
    base._int_to_file(a, file)
    assert a == base._file_to_int(file)


def test_large_num():
    base._int_to_bytes(6653643750000, fmt="<Q")


def test_str_conversions():
    a = "ugabuga"
    assert a == base._bytes_to_str(base._str_to_bytes(a))


def test_str_file_conversions(tmp_path):
    a = "asd"
    file = tmp_path / "str"
    base._str_to_file(a, file)
    assert a == base._file_to_str(file)


def test_shape_conversions():
    shape = (1, 9, 10, 1)
    assert shape == base._bytes_to_shape(base._shape_to_bytes(shape))


def test_shape_file(tmp_path):
    shape = (1, 4, 5)
    file = tmp_path / "shape"
    base._shape_to_file(shape, file)
    assert shape == base._file_to_shape(file)


def test_wrapper():
    arr = np.array([1, 2, 3])
    wrapped = base.Wrapped(arr, lambda x: -x)
    assert wrapped[0] == -1
    assert wrapped[1] == -2
    assert wrapped[2] == -3
    assert len(wrapped) == 3
