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

    memmap = StringsMmmap.from_strings(list_of_strings, tmp_path / 'strings_memmap')
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


def generate_strs():
    for i in range(30):
        yield str(i)


def test_from_generator(tmp_path):
    memmap = StringsMmmap.from_generator(generate_strs(), tmp_path / 'strings_memmap', 4)
    for i in range(30):
        assert str(i) == memmap[i]

