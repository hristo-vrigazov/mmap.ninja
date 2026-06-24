import json
import os
import stat

import pytest

from mmap_ninja import generic
from mmap_ninja.string import StringsMmap


def test_base_case(tmp_path):
    list_of_strings = ["Torba", "Boiler", "a", "zele pitka", "", "popo"]

    memmap = StringsMmap.from_strings(tmp_path / "strings_memmap", list_of_strings, verbose=True)
    for i, string in enumerate(list_of_strings):
        assert string == memmap[i]
    memmap[:2] = ["Korbo", "Moiler"]
    memmap[2] = "b"
    assert list_of_strings[:2] != memmap[:2]
    assert list_of_strings[2] != memmap[2]
    memmap.close()


def test_open_existing(tmp_path):
    list_of_strings = ["Torba", "Boiler", "a", "zele pitka", "", "popo"]

    memmap = StringsMmap.from_strings(tmp_path / "strings_memmap", list_of_strings, verbose=True)
    memmap = StringsMmap(tmp_path / "strings_memmap")
    for i, string in enumerate(list_of_strings):
        assert string == memmap[i]
    memmap[:2] = ["Korbo", "Moiler"]
    memmap[2] = "b"
    assert list_of_strings[:2] != memmap[:2]
    assert list_of_strings[2] != memmap[2]
    memmap.close()


def test_extend(tmp_path):
    list_of_strings = ["Torba", "Boiler", "a", "zele pitka", "", "popo"]

    memmap = StringsMmap.from_strings(tmp_path / "strings_memmap", list_of_strings)
    memmap.extend(["new", "new2", "uga dunga"])
    assert len(memmap) == 9
    assert memmap[-1] == "uga dunga"
    assert memmap[-2] == "new2"
    assert memmap[-3] == "new"


def test_append(tmp_path):
    list_of_strings = ["Torba", "Boiler", "a", "zele pitka", "", "popo"]

    memmap = StringsMmap.from_strings(tmp_path / "strings_memmap", list_of_strings)
    memmap.append("new")
    assert len(memmap) == 7
    assert memmap[-1] == "new"


def generate_strs(n):
    for i in range(n):
        yield str(i)


@pytest.mark.parametrize("n", [30, 3])
def test_from_generator(tmp_path, n):
    memmap = StringsMmap.from_generator(tmp_path / "strings_memmap", generate_strs(n), 4, verbose=True)
    for i in range(n):
        assert str(i) == memmap[i]
    generic.open_existing(tmp_path / "strings_memmap")


def test_open_empty(tmp_path):
    memmap = StringsMmap(tmp_path / "strings_memmap")
    assert len(memmap) == 0
    memmap.append("Something")
    assert len(memmap) == 1


def test_index_error(tmp_path):
    memmap = StringsMmap(tmp_path / "strings_memmap")
    print(memmap)
    with pytest.raises(IndexError):
        memmap[0]


def test_json_wrapper(tmp_path):
    memmap = StringsMmap(tmp_path / "strings_memmap")
    memmap.append(json.dumps({"asdasd": "as"}))

    memmap = generic.open_existing(tmp_path / "strings_memmap", wrapper_fn=json.loads)
    print(memmap[0])


def test_read_only(tmp_path):
    out_path = tmp_path / 'read_only_str_memmap'
    list_of_strings = ['This is the first test string...', '... and the second one.']
    StringsMmap.from_strings(out_dir=out_path, strings=list_of_strings)
    # Make all memmap files read-only for the user
    for p in out_path.glob(r'**/*'):
        if not p.is_file():
            continue
        os.chmod(p, stat.S_IRUSR)
    # Open in read-only mode
    memmap = StringsMmap(out_dir=out_path, mode='rb')
    for i, string in enumerate(list_of_strings):
        assert string == memmap[i]


def test_empty_strings_mmap(tmp_path):
    out_path = tmp_path / 'empty_strings_mmap'
    strings_mmap = StringsMmap.from_strings(out_dir=out_path, strings=[])
    assert len(strings_mmap) == 0
    assert [] == strings_mmap[:]
    strings_mmap.append("icak")
    assert len(strings_mmap) == 1
    assert strings_mmap[0] == "icak"


@pytest.mark.parametrize("zstd_train_dictionary_size", [None, 256])
def test_zstd_compression(tmp_path, zstd_train_dictionary_size):
    base_strings = ["Torba", "Boiler", "a", "zele pitka", "x", "popo"]
    list_of_strings = [f"{s} sample {i}".ljust(100) for i in range(20) for s in base_strings]
    out_path = tmp_path / "zstd_strings"
    memmap = StringsMmap.from_strings(
        out_path,
        list_of_strings,
        zstd_level=3,
        zstd_train_dictionary_size=zstd_train_dictionary_size,
    )
    assert memmap[:] == list_of_strings


def test_zstd_reload(tmp_path):
    list_of_strings = ["Torba", "Boiler", "a", "zele pitka", "", "popo"]
    out_path = tmp_path / "zstd_reload"
    StringsMmap.from_strings(out_path, list_of_strings, zstd_level=3)
    memmap = StringsMmap(out_path)
    for i, s in enumerate(list_of_strings):
        assert memmap[i] == s


def test_zstd_extend(tmp_path):
    list_of_strings = ["Torba", "Boiler", "a", "zele pitka", "", "popo"]
    out_path = tmp_path / "zstd_extend"
    memmap = StringsMmap.from_strings(out_path, list_of_strings, zstd_level=3)
    memmap.extend(["new", "new2"])
    assert len(memmap) == 8
    assert memmap[-1] == "new2"
    assert memmap[-2] == "new"
    for i, s in enumerate(list_of_strings):
        assert memmap[i] == s


def test_zstd_set_raises(tmp_path):
    out_path = tmp_path / "zstd_set"
    memmap = StringsMmap.from_strings(out_path, ["hello", "world"], zstd_level=3)
    with pytest.raises(ValueError):
        memmap[0] = "other"

