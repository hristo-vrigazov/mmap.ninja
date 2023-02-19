import mmap
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from mmap_ninja import numpy, base
from mmap_ninja.base import _bytes_to_str, _str_to_bytes, _sequence_of_strings_to_bytes


class StringsMmap:
    def __init__(
        self,
        out_dir: Union[str, Path],
        mode="r+b",
        starts_key="starts",
        ends_key="ends",
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        data_file = out_dir / "data.ninja"
        self.data_file = Path(data_file)
        self.mode = mode
        self.out_dir = out_dir
        self.starts_key = starts_key
        self.ends_key = ends_key

        self.starts = numpy.open_existing(self.out_dir / self.starts_key, mode="r")
        self.ends = numpy.open_existing(self.out_dir / self.ends_key, mode="r")
        self.range = np.arange(len(self.starts), dtype=np.int64)
        self.file = open(data_file, mode=mode)
        self.buffer = mmap.mmap(self.file.fileno(), 0)

    def get_multiple(self, item):
        indices = self.range[item]
        return [self.__getitem__(idx) for idx in indices]

    def get_single(self, item):
        start = self.starts[item]
        end = self.ends[item]
        return _bytes_to_str(self.buffer[start:end])

    def __getitem__(self, item):
        if np.isscalar(item):
            return self.get_single(item)
        return self.get_multiple(item)

    def __setitem__(self, key, value):
        if np.isscalar(key):
            return self.set_single(key, value)
        return self.set_multiple(key, value)

    def __len__(self):
        return len(self.starts)

    def set_multiple(self, key, value):
        for i, idx in enumerate(self.range[key]):
            new_value: str = value[i]
            self.set_single(idx, new_value)

    def set_single(self, idx, new_value):
        start = self.starts[idx]
        end = self.ends[idx]
        self.buffer[start:end] = _str_to_bytes(new_value)

    def close(self):
        self.buffer.close()
        self.file.close()

    def extend(self, list_of_strings: Sequence[str], verbose=False):
        bytes_slices = _sequence_of_strings_to_bytes(list_of_strings, verbose=verbose)
        end = self.ends[-1]
        start_offsets = end + bytes_slices.starts
        end_offsets = end + bytes_slices.ends
        numpy.extend(self.starts, start_offsets)
        numpy.extend(self.ends, end_offsets)
        self.close()
        out_dir = self.data_file.parent
        with open(out_dir / "data.ninja", "ab") as data_file:
            data_file.write(bytes_slices.buffer)
            data_file.flush()

        self.starts = numpy.open_existing(self.out_dir / self.starts_key, mode="r")
        self.ends = numpy.open_existing(self.out_dir / self.ends_key, mode="r")
        self.range = np.arange(len(self.starts), dtype=np.int64)
        self.file = open(self.data_file, mode=self.mode)
        self.buffer = mmap.mmap(self.file.fileno(), 0)

    def append(self, string: str):
        self.extend([string])

    def __repr__(self):
        base_repr = super().__repr__()
        return f"{base_repr} of length: {len(self)}"

    @classmethod
    def from_strings(
        cls,
        out_dir: Union[str, Path],
        strings: Sequence[str],
        mode="r+b",
        starts_key="starts",
        ends_key="ends",
        verbose=False,
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        bytes_slices = _sequence_of_strings_to_bytes(strings, verbose=verbose)
        with open(out_dir / "data.ninja", "wb") as f:
            f.write(bytes_slices.buffer)
        base._str_to_file("string", out_dir / "type.ninja")
        numpy.from_ndarray(out_dir / starts_key, np.array(bytes_slices.starts, dtype=np.int64))
        numpy.from_ndarray(out_dir / ends_key, np.array(bytes_slices.ends, dtype=np.int64))
        return cls(out_dir, mode=mode, starts_key=starts_key, ends_key=ends_key)

    @classmethod
    def from_generator(cls, out_dir: Union[str, Path], sample_generator, batch_size: int, verbose=False, **kwargs):
        return base.from_generator_base(
            out_dir=out_dir,
            sample_generator=sample_generator,
            batch_size=batch_size,
            verbose=verbose,
            batch_ctor=cls.from_strings,
            **kwargs,
        )
