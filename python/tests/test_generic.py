import pytest

from mmap_ninja import base, generic


def test_open_throws_exception_if_type_is_unknown(tmp_path):
    out_dir = tmp_path / "mmap"
    out_dir.mkdir(exist_ok=True)
    base._str_to_file("asd", out_dir / "type.ninja")

    with pytest.raises(ValueError):
        generic.open_existing(out_dir)
