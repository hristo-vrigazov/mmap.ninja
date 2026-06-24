import mmap
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from mmap_ninja import base, numpy
from mmap_ninja.base import _bytes_to_str, _str_to_bytes, _sequence_of_strings_to_bytes, BytesSlices


def _compress_strings_to_bytes(strings: Sequence[str], compressor, verbose=False) -> BytesSlices:
    buffer = bytearray()
    starts = []
    ends = []
    if verbose:
        from tqdm import tqdm
        strings = tqdm(strings)
    for string in strings:
        compressed = compressor.compress(_str_to_bytes(string))
        starts.append(len(buffer))
        ends.append(len(buffer) + len(compressed))
        buffer.extend(compressed)
    return BytesSlices(bytes(buffer), starts, ends)


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

        self.starts = None
        self.ends = None
        self.range = None
        self.file = None
        self.buffer = None
        self.compressor = None
        self.decompressor = None

        if (self.out_dir / self.starts_key / "dtype.ninja").exists():
            self._reload_fields()

    def get_multiple(self, item):
        indices = self.range[item]
        return [self.__getitem__(idx) for idx in indices]

    def get_single(self, item):
        start = self.starts[item]
        end = self.ends[item]
        raw = bytes(self.buffer[start:end])
        if self.decompressor is not None:
            raw = self.decompressor.decompress(raw)
        return _bytes_to_str(raw)

    def __getitem__(self, item):
        if self.starts is None:
            if np.isscalar(item):
                raise IndexError(f"StringsMmap is empty!")
            return []
        if np.isscalar(item):
            return self.get_single(item)
        return self.get_multiple(item)

    def __setitem__(self, key, value):
        if np.isscalar(key):
            return self.set_single(key, value)
        return self.set_multiple(key, value)

    def __len__(self):
        if self.starts is None:
            return 0
        return len(self.starts)

    def set_multiple(self, key, value):
        for i, idx in enumerate(self.range[key]):
            new_value: str = value[i]
            self.set_single(idx, new_value)

    def set_single(self, idx, new_value):
        if self.compressor is not None:
            raise ValueError("In-place modification is not supported for compressed StringsMmap.")
        start = self.starts[idx]
        end = self.ends[idx]
        self.buffer[start:end] = _str_to_bytes(new_value)

    def close(self):
        self.buffer.close()
        self.file.close()

    def extend(self, list_of_strings: Sequence[str], verbose=False):
        if self.starts is None:
            StringsMmap.from_strings(self.out_dir, list_of_strings, verbose=verbose)
            self._reload_fields()
            return
        if self.compressor is not None:
            bytes_slices = _compress_strings_to_bytes(list_of_strings, self.compressor, verbose=verbose)
        else:
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
        self._reload_fields()

    def _reload_fields(self):
        self.starts = numpy.open_existing(self.out_dir / self.starts_key, mode="r")
        self.ends = numpy.open_existing(self.out_dir / self.ends_key, mode="r")
        self.range = np.arange(len(self.starts), dtype=np.int64)
        self.file = open(self.data_file, mode=self.mode)
        access = mmap.ACCESS_READ if self.mode == 'rb' else mmap.ACCESS_DEFAULT
        self.buffer = mmap.mmap(self.file.fileno(), 0, access=access)

        zstd_level_file = self.out_dir / "zstd_level.ninja"
        if zstd_level_file.exists():
            import zstandard
            level = int(base._file_to_str(zstd_level_file))
            zstd_dict = None
            zstd_dict_file = self.out_dir / "zstd_dict.ninja"
            if zstd_dict_file.exists():
                with open(zstd_dict_file, "rb") as f:
                    zstd_dict = zstandard.ZstdCompressionDict(f.read())
            self.compressor = zstandard.ZstdCompressor(level=level, dict_data=zstd_dict)
            self.decompressor = zstandard.ZstdDecompressor(dict_data=zstd_dict)
        else:
            self.compressor = None
            self.decompressor = None

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
        zstd_level: Optional[int] = None,
        zstd_train_dictionary_size: Optional[int] = None,
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        if len(strings) == 0:
            return cls(out_dir, mode=mode, starts_key=starts_key, ends_key=ends_key)
        if zstd_level is not None:
            import zstandard
            raw_bytes_list = [_str_to_bytes(s) for s in strings]
            zstd_dict = None
            if zstd_train_dictionary_size is not None:
                zstd_dict = zstandard.train_dictionary(zstd_train_dictionary_size, raw_bytes_list)
                with open(out_dir / "zstd_dict.ninja", "wb") as f:
                    f.write(zstd_dict.as_bytes())
            base._str_to_file(str(zstd_level), out_dir / "zstd_level.ninja")
            compressor = zstandard.ZstdCompressor(level=zstd_level, dict_data=zstd_dict)
            iterable = raw_bytes_list
            if verbose:
                from tqdm import tqdm
                iterable = tqdm(iterable)
            buf = bytearray()
            starts = []
            ends = []
            for raw in iterable:
                compressed = compressor.compress(raw)
                starts.append(len(buf))
                ends.append(len(buf) + len(compressed))
                buf.extend(compressed)
            bytes_slices = BytesSlices(bytes(buf), starts, ends)
        else:
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
