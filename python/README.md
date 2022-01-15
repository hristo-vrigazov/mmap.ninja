## mmap.ninja Python API

Extension to `numpy.memmap`, which currently includes a `RaggedMemoryMap` (where the elements are of different shape),
and a `StringsMmmap`  (which is about 9 times faster than just storing your documents as text files).

Currently under active development, API is not yet stable.

Examples:

```python
from mmap_ninja.string import StringsMmmap
from pathlib import Path

tmp_path = Path('.')
memmap = StringsMmmap.open_existing(tmp_path / 'strings_memmap')
for i in range(len(memmap)):
    print(memmap[i])
memmap.close()
```


```python
from mmap_ninja.ragged import RaggedMMap
from pathlib import Path

tmp_path = Path('.')
memmap = RaggedMMap.open_existing(tmp_path / 'ragged')
for i in range(len(memmap)):
    print(memmap[i])
memmap.close()
```