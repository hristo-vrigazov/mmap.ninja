from pathlib import Path
from typing import Union

import numpy as np

from mmap_ninja import numpy


class RaggedMmap:

    def __init__(self, data_file: Union[str, Path],
                 starts,
                 ends,
                 shapes,
                 flattened_shapes,
                 wrapper_fn=None):
        self.data_file = Path(data_file)
        self.starts = starts
        self.ends = ends
        self.shapes = shapes
        self.wrapper_fn = wrapper_fn

        self.out_dir = data_file.parent
        self.range = np.arange(len(starts), dtype=np.int32)
        self.flattened_shapes = flattened_shapes
        self.n = len(self.shapes)
        self.range = np.arange(self.n)

        self.memmap = numpy.open_existing(self.out_dir, mode='r+')

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

    @classmethod
    def from_lists(cls, out_dir: Union[str, Path], lists, dtype=np.int64, wrapper_fn=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        offset = 0
        starts = []
        ends = []
        shapes = []
        arrs = []
        flattened_shapes = []
        for l in lists:
            arr = np.asarray(l, dtype=dtype)
            flattened = arr.ravel()
            starts.append(offset)
            ends.append(offset + len(flattened))
            flattened_shapes.append(len(flattened))
            shapes.append(arr.shape if len(arr.shape) > 0 else (0,))
            arrs.append(flattened)
            offset += len(flattened)
        buffer = np.concatenate(arrs)
        numpy.from_ndarray(np.array(starts, dtype=np.int32), out_dir / 'starts')
        numpy.from_ndarray(np.array(ends, dtype=np.int32), out_dir / 'ends')
        numpy.from_ndarray(np.array(shapes, dtype=np.int32), out_dir / 'shapes')
        numpy.from_ndarray(np.array(flattened_shapes, dtype=np.int32), out_dir / 'flattened_shapes')
        numpy.from_ndarray(np.array(buffer, dtype=dtype), out_dir)
        return cls(data_file=out_dir / 'data.ninja',
                   starts=starts,
                   ends=ends,
                   shapes=shapes,
                   flattened_shapes=flattened_shapes,
                   wrapper_fn=wrapper_fn)

    @classmethod
    def open_existing(cls, out_dir: Union[str, Path], wrapper_fn=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        starts = numpy.open_existing(out_dir / 'starts', mode='r')
        ends = numpy.open_existing(out_dir / 'ends', mode='r')
        shapes = numpy.open_existing(out_dir / 'shapes', mode='r')
        flattened_shapes = numpy.open_existing(out_dir / 'flattened_shapes', mode='r')
        return cls(data_file=out_dir / 'data',
                   starts=starts,
                   ends=ends,
                   shapes=shapes,
                   flattened_shapes=flattened_shapes,
                   wrapper_fn=wrapper_fn)
