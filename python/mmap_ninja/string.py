import mmap
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from mmap_ninja.base import bytes_to_str, str_to_bytes, sequence_of_strings_to_bytes
from mmap_ninja import numpy


class StringsMmmap:

    def __init__(self, data_file: Union[str, Path], starts, ends, mode='r+b'):
        self.data_file = Path(data_file)
        self.starts = starts
        self.ends = ends
        self.range = np.arange(len(starts))
        self.file = open(data_file, mode=mode)
        self.buffer = mmap.mmap(self.file.fileno(), 0)

    def get_multiple(self, item):
        indices = self.range[item]
        return [self.__getitem__(idx) for idx in indices]

    def get_single(self, item):
        start = self.starts[item]
        end = self.ends[item]
        return bytes_to_str(self.buffer[start:end])

    def __getitem__(self, item):
        if np.isscalar(item):
            return self.get_single(item)
        return self.get_multiple(item)

    def set_multiple(self, key, value):
        for i, idx in enumerate(self.range[key]):
            new_value: str = value[i]
            self.set_single(idx, new_value)

    def set_single(self, idx, new_value):
        start = self.starts[idx]
        end = self.ends[idx]
        self.buffer[start:end] = str_to_bytes(new_value)

    def __setitem__(self, key, value):
        if np.isscalar(key):
            return self.set_single(key, value)
        return self.set_multiple(key, value)

    def close(self):
        self.buffer.close()
        self.file.close()

    @classmethod
    def from_strings(cls, strings: Sequence[str], out_dir: Union[str, Path]):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        buffer, ends, starts = sequence_of_strings_to_bytes(strings)
        with open(out_dir / 'data.ninja', "wb") as f:
            f.write(buffer)
        numpy.from_ndarray(np.array(starts), out_dir / 'starts')
        numpy.from_ndarray(np.array(ends), out_dir / 'ends')
        return cls.open_existing(out_dir)

    @classmethod
    def open_existing(cls, out_dir: Union[str, Path], mode='r+b'):
        starts_np = numpy.open_existing(out_dir / 'starts', mode='r')
        ends_np = numpy.open_existing(out_dir / 'ends', mode='r')
        return cls(out_dir / 'data.ninja', starts_np, ends_np, mode=mode)




