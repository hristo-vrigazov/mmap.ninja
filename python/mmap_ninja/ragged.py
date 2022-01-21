from pathlib import Path
from typing import Union, Sequence

import numpy as np

from mmap_ninja import numpy


class RaggedMmap:

    def __init__(self, out_dir: Union[str, Path],
                 mode='r',
                 wrapper_fn=None,
                 starts_key='starts',
                 ends_key='ends',
                 shapes_key='shapes',
                 flattened_shapes_key='flattened_shapes'):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        self.starts_key = starts_key
        self.ends_key = ends_key
        self.shapes_key = shapes_key
        self.flattened_shapes_key = flattened_shapes_key

        data_file = out_dir / 'data'
        self.data_file = Path(data_file)

        self.out_dir = out_dir
        self.wrapper_fn = wrapper_fn
        self.mode = mode

        self.memmap = numpy.open_existing(self.out_dir, mode=self.mode)
        self.starts = numpy.open_existing(self.out_dir / self.starts_key, mode=self.mode)
        self.ends = numpy.open_existing(self.out_dir / self.ends_key, mode=self.mode)
        self.shapes = numpy.open_existing(self.out_dir / self.shapes_key, mode=self.mode)
        self.flattened_shapes = numpy.open_existing(self.out_dir / self.flattened_shapes_key, mode=self.mode)
        self.range = np.arange(len(self.starts), dtype=np.int32)
        self.n = len(self.shapes)
        self.range = np.arange(self.n)

    def get_multiple(self, item):
        indices = self.range[item]
        return [self.__getitem__(idx) for idx in indices]

    def set_multiple(self, item, value):
        for i, idx in enumerate(self.range[item]):
            self.set_single(idx, value[i])

    def set_single(self, idx, value):
        start = self.starts[idx]
        end = self.ends[idx]
        value = np.asarray(value)
        self.memmap[start:end] = value

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

    def get_single(self, item):
        start = self.starts[item]
        end = self.ends[item]
        shape = self.shapes[item]
        res = self.memmap[start:end]
        if shape == (0,):
            res = np.asscalar(res)
        if self.wrapper_fn is not None:
            res = self.wrapper_fn(res)
        return res

    def extend(self, arrays: Sequence[np.ndarray]):
        numpy_bytes_slices = numpy.lists_of_ndarrays_to_bytes(arrays, self.memmap.dtype)
        numpy.extend(self.memmap, numpy_bytes_slices.buffer)
        end = self.ends[-1]
        numpy.extend(self.starts, end + numpy_bytes_slices.starts)
        numpy.extend(self.ends, end + numpy_bytes_slices.ends)
        numpy.extend(self.flattened_shapes, numpy_bytes_slices.flattened_shapes)
        numpy.extend(self.shapes, numpy_bytes_slices.shapes)

        self.memmap = numpy.open_existing(self.out_dir, mode=self.mode)
        self.starts = numpy.open_existing(self.out_dir / self.starts_key, mode=self.mode)
        self.ends = numpy.open_existing(self.out_dir / self.ends_key, mode=self.mode)
        self.shapes = numpy.open_existing(self.out_dir / self.shapes_key, mode=self.mode)
        self.flattened_shapes = numpy.open_existing(self.out_dir / self.flattened_shapes_key, mode=self.mode)
        self.range = np.arange(len(self.starts), dtype=np.int32)
        self.n = len(self.shapes)
        self.range = np.arange(self.n)

    @classmethod
    def from_lists(cls, out_dir: Union[str, Path], lists: Sequence[np.ndarray],
                   dtype=np.int64,
                   mode='r+',
                   wrapper_fn=None,
                   starts_key='starts',
                   ends_key='ends',
                   shapes_key='shapes',
                   flattened_shapes_key='flattened_shapes'):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        numpy_bytes_slices = numpy.lists_of_ndarrays_to_bytes(lists, dtype)
        numpy.from_ndarray(np.array(numpy_bytes_slices.starts, dtype=np.int32), out_dir / starts_key)
        numpy.from_ndarray(np.array(numpy_bytes_slices.ends, dtype=np.int32), out_dir / ends_key)
        numpy.from_ndarray(np.array(numpy_bytes_slices.shapes, dtype=np.int32), out_dir / shapes_key)
        numpy.from_ndarray(np.array(numpy_bytes_slices.flattened_shapes, dtype=np.int32),
                           out_dir / flattened_shapes_key)
        numpy.from_ndarray(np.array(numpy_bytes_slices.buffer, dtype=dtype), out_dir)
        return cls(out_dir=out_dir,
                   wrapper_fn=wrapper_fn,
                   mode=mode,
                   starts_key=starts_key,
                   ends_key=ends_key,
                   shapes_key=shapes_key,
                   flattened_shapes_key=flattened_shapes_key)

