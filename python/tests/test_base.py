from mmap_ninja import base


def test_int_conversions():
    a = 17
    assert a == base.bytes_to_int32(base.int32_to_bytes(a))


def test_int_file_conversions(tmp_path):
    a = 17
    file = tmp_path / 'int'
    base.int_to_file(a, file)
    assert a == base.file_to_int(file)


def test_str_conversions():
    a = 'ugabuga'
    assert a == base.bytes_to_str(base.str_to_bytes(a))


def test_str_file_conversions(tmp_path):
    a = 'asd'
    file = tmp_path / 'str'
    base.str_to_file(a, file)
    assert a == base.file_to_str(file)


def test_shape_conversions():
    shape = (1, 9, 10, 1)
    assert shape == base.bytes_to_shape(base.shape_to_bytes(shape))


def test_shape_file(tmp_path):
    shape = (1, 4, 5)
    file = tmp_path / 'shape'
    base.shape_to_file(shape, file)
    assert shape == base.file_to_shape(file)

