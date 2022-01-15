import pytest

from mmap_ninja.string import StringsMmmap


def test_base_case(tmp_path):
    list_of_strings = [
        'Torba',
        'Boiler',
        'a',
        'zele pitka',
        '',
        'popo'
    ]

    memmap = StringsMmmap.from_strings(list_of_strings, tmp_path / 'strings_memmap', verbose=True)
    for i, string in enumerate(list_of_strings):
        assert string == memmap[i]
    memmap[:2] = ['Korbo', 'Moiler']
    memmap[2] = 'b'
    assert list_of_strings[:2] != memmap[:2]
    assert list_of_strings[2] != memmap[2]
    memmap.close()


def test_extend(tmp_path):
    list_of_strings = [
        'Torba',
        'Boiler',
        'a',
        'zele pitka',
        '',
        'popo'
    ]

    memmap = StringsMmmap.from_strings(list_of_strings, tmp_path / 'strings_memmap')
    memmap.extend(['new', 'new2', 'uga dunga'])
    assert len(memmap) == 9
    assert memmap[-1] == 'uga dunga'
    assert memmap[-2] == 'new2'
    assert memmap[-3] == 'new'


def test_append(tmp_path):
    list_of_strings = [
        'Torba',
        'Boiler',
        'a',
        'zele pitka',
        '',
        'popo'
    ]

    memmap = StringsMmmap.from_strings(list_of_strings, tmp_path / 'strings_memmap')
    memmap.append('new')
    assert len(memmap) == 7
    assert memmap[-1] == 'new'


def generate_strs(n):
    for i in range(n):
        yield str(i)


@pytest.mark.parametrize("n", [30, 3])
def test_from_generator(tmp_path, n):
    memmap = StringsMmmap.from_generator(generate_strs(n), tmp_path / 'strings_memmap', 4, verbose=True)
    for i in range(n):
        assert str(i) == memmap[i]
