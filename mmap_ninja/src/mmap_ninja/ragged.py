from copy import copy
from pathlib import Path
from typing import Union, Sequence

import numpy as np

from mmap_ninja import base, numpy


def _np_shape_extend(shapes, arr):
    numpy.extend(shapes, arr)


def _ragged_shape_extend(shapes, arr):
    shapes.extend(arr)


class RaggedMmap:
    def __init__(
        self,
        out_dir: Union[str, Path],
        wrapper_fn=None,
        mode="r",
        starts_key="starts",
        ends_key="ends",
        shapes_key="shapes",
        flattened_shapes_key="flattened_shapes",
        copy_before_wrapper_fn=True,
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        self.starts_key = starts_key
        self.ends_key = ends_key
        self.shapes_key = shapes_key
        self.flattened_shapes_key = flattened_shapes_key

        data_file = out_dir / "data"
        self.data_file = Path(data_file)

        self.out_dir = out_dir
        self.wrapper_fn = wrapper_fn
        self.mode = mode

        self.shapes_are_flat = None
        self.shapes_ctor = None
        self.memmap = None
        self.starts = None
        self.ends = None
        self.shapes = None
        self.flattened_shapes = None
        self.range = None
        self.n = None
        self.range = None
        self.shapes_extension_fn = None

        if (self.out_dir / "shapes_are_flat.ninja").exists():
            self._reload_fields()

        self.copy_before_wrapper_fn = copy_before_wrapper_fn

    def _reload_fields(self):
        self.shapes_are_flat = bool(base._file_to_int(self.out_dir / "shapes_are_flat.ninja"))
        self.shapes_ctor = numpy.open_existing if self.shapes_are_flat else RaggedMmap
        self.memmap = numpy.open_existing(self.out_dir, mode=self.mode)
        self.starts = numpy.open_existing(self.out_dir / self.starts_key, mode=self.mode)
        self.ends = numpy.open_existing(self.out_dir / self.ends_key, mode=self.mode)
        self.shapes = self.shapes_ctor(self.out_dir / self.shapes_key, mode=self.mode)
        self.flattened_shapes = numpy.open_existing(self.out_dir / self.flattened_shapes_key, mode=self.mode)
        self.range = np.arange(len(self.starts), dtype=np.int64)
        self.n = len(self.shapes)
        self.range = np.arange(self.n)
        self.shapes_extension_fn = _np_shape_extend if self.shapes_are_flat else _ragged_shape_extend

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
        if self.starts is None:
            raise IndexError(f"RaggedMmap is empty!")
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

    def get_single(self, item):
        start = self.starts[item]
        end = self.ends[item]
        shape = self.shapes[item]
        res = self.memmap[start:end]
        if self.shapes_are_flat and shape[0] == 0:
            if len(res) == 1:
                res = res.item()
        else:
            res = res.reshape(shape)
        if self.wrapper_fn is not None:
            if self.copy_before_wrapper_fn:
                res = copy(res)
            res = self.wrapper_fn(res)
        return res

    def append(self, array: np.ndarray):
        self.extend([array])

    def extend(self, arrays: Sequence[np.ndarray]):
        if self.starts is None:
            RaggedMmap.from_lists(self.out_dir, arrays)
            self._reload_fields()
            return
        numpy_bytes_slices = numpy._lists_of_ndarrays_to_bytes(arrays, self.memmap.dtype)
        numpy.extend(self.memmap, numpy_bytes_slices.buffer)
        end = self.ends[-1]
        numpy.extend(self.starts, end + numpy_bytes_slices.starts)
        numpy.extend(self.ends, end + numpy_bytes_slices.ends)
        numpy.extend(self.flattened_shapes, numpy_bytes_slices.flattened_shapes)
        self.shapes_extension_fn(self.shapes, numpy_bytes_slices.shapes)
        self._reload_fields()

    def __repr__(self):
        base_repr = super().__repr__()
        return f"{base_repr} of length: {len(self)}"

    @classmethod
    def from_lists(
        cls,
        out_dir: Union[str, Path],
        lists: Sequence[np.ndarray],
        dtype=None,
        mode="r+",
        wrapper_fn=None,
        starts_key="starts",
        ends_key="ends",
        shapes_key="shapes",
        flattened_shapes_key="flattened_shapes",
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        numpy_bytes_slices = numpy._lists_of_ndarrays_to_bytes(lists, dtype)
        numpy.from_ndarray(out_dir / starts_key, np.array(numpy_bytes_slices.starts, dtype=np.int64))
        numpy.from_ndarray(out_dir / ends_key, np.array(numpy_bytes_slices.ends, dtype=np.int64))
        shapes_are_flat = all([len(shape) == 1 for shape in numpy_bytes_slices.shapes])
        base._int_to_file(int(shapes_are_flat), out_dir / "shapes_are_flat.ninja")

        if shapes_are_flat:
            numpy.from_ndarray(out_dir / shapes_key, numpy_bytes_slices.shapes)
        else:
            RaggedMmap.from_lists(out_dir / shapes_key, numpy_bytes_slices.shapes)
        numpy.from_ndarray(
            out_dir / flattened_shapes_key, np.array(numpy_bytes_slices.flattened_shapes, dtype=np.int64)
        )
        numpy.from_ndarray(out_dir, np.array(numpy_bytes_slices.buffer))
        base._str_to_file("ragged", out_dir / "type.ninja")
        return cls(
            out_dir=out_dir,
            wrapper_fn=wrapper_fn,
            mode=mode,
            starts_key=starts_key,
            ends_key=ends_key,
            shapes_key=shapes_key,
            flattened_shapes_key=flattened_shapes_key,
        )

    @classmethod
    def from_generator(cls, out_dir: Union[str, Path], sample_generator, batch_size: int, verbose=False, **kwargs):
        return base.from_generator_base(
            out_dir=out_dir,
            sample_generator=sample_generator,
            batch_size=batch_size,
            verbose=verbose,
            batch_ctor=cls.from_lists,
            **kwargs,
        )

    @classmethod
    def from_indexable(cls, out_dir: Union[str, Path], indexable, batch_size: int, n_jobs=None, verbose=False, **kwargs):
        return base.from_indexable_base(
            out_dir=out_dir,
            indexable=indexable,
            batch_size=batch_size,
            n_jobs=n_jobs,
            verbose=verbose,
            batch_ctor=cls.from_lists,
            **kwargs,
        )
