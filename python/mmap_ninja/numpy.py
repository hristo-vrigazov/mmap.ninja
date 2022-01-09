from pathlib import Path
from typing import Union
from mmap_ninja import base


# See: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
import numpy as np


def save_mmap_kwargs(out_dir: Path,
                     dtype,
                     shape,
                     order):
    out_dir.mkdir(exist_ok=True)
    base.str_to_file(np.dtype(dtype).name, out_dir / f'dtype.ninja')
    base.shape_to_file(shape, out_dir / f'shape.ninja')
    base.str_to_file(order, out_dir / f'order.ninja')


def read_mmap_kwargs(out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    return {
        'dtype': base.file_to_str(out_dir / 'dtype.ninja'),
        'shape': base.file_to_shape(out_dir / 'shape.ninja'),
        'order': base.file_to_str(out_dir / 'order.ninja')
    }


class NumpyMmap:

    def __init__(self, memmap, dtype, shape, order):
        self.memmap = memmap
        self.dtype = dtype
        self.order = order
        self.shape = shape

    @classmethod
    def empty(cls, out_dir: Union[str, Path], dtype, shape, order):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        save_mmap_kwargs(out_dir, dtype, shape, order)
        memmap = np.memmap(str(out_dir / 'data.ninja'),
                           mode='w+',
                           dtype=dtype,
                           shape=shape,
                           order=order)
        return cls(memmap, dtype, shape, order)

    @classmethod
    def from_ndarray(cls, arr: np.ndarray, out_dir: Union[str, Path]):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        dtype = arr.dtype
        shape = arr.shape
        order = 'F' if np.isfortran(arr) else 'C'
        memmap = np.memmap(str(out_dir / 'data.ninja'),
                           mode='w+',
                           dtype=dtype,
                           shape=shape,
                           order=order)
        memmap[:] = arr
        save_mmap_kwargs(out_dir, dtype, shape, order)
        return cls(memmap, dtype, shape, order)

    @classmethod
    def open_existing(cls, out_dir: Union[str, Path], mode='r'):
        out_dir = Path(out_dir)
        kwargs = read_mmap_kwargs(out_dir)
        memmap = np.memmap(str(out_dir / 'data.ninja'),
                           mode=mode,
                           **kwargs)
        return cls(memmap, **kwargs)

