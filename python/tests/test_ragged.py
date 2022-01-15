from mmap_ninja.ragged import RaggedMMap


def test_base_case(tmp_path):
    simple = list(range(4))
    mmap = RaggedMMap.from_lists(tmp_path / 'simple', simple)
    for i in range(4):
        assert i == mmap[i]


def test_open_existing_case(tmp_path):
    simple = list(range(4))
    mmap = RaggedMMap.from_lists(tmp_path / 'simple', simple)
    mmap = RaggedMMap.open_existing(tmp_path / 'simple')
    for i in range(4):
        assert i == mmap[i]
