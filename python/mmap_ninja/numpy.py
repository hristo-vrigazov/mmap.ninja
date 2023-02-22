from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Tuple, Sequence, Optional, Dict

# See: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
import numpy as np
from mmap_ninja.base import from_generator_base

from mmap_ninja import base


def _create_if_not_exists(out_dir: Union[str, Path]) -> Path:
    """
    A helper mkdir that creates the directory if it doesn't exist.
    Returns the path to the directory.created.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


def _save_mmap_kwargs(out_dir: Path, dtype: Union[np.dtype, str], shape: Sequence[int], order: str) -> None:
    """
    Persists the arguments needed for initializing a ``np.memmap``, so that the user does not have to specify them.

    :param out_dir: The directory in which the memory map is persisted
    :param dtype: The ``np.dtype`` of the memory map, represented either by a string or directly by its type.
    :param shape: A sequence of integers representing the shape of the memory map.
    :param order: A string, representing the order - ``"F"`` or ``"C"``
    :return:
    """
    out_dir = _create_if_not_exists(out_dir)
    base._str_to_file("numpy", out_dir / "type.ninja")
    base._str_to_file(np.dtype(dtype).name, out_dir / f"dtype.ninja")
    base._shape_to_file(shape, out_dir / f"shape.ninja")
    base._str_to_file(order, out_dir / f"order.ninja")


def _read_mmap_kwargs(out_dir: Path) -> Dict:
    """
    Reads already persisted arguments needed for opening a ``np.memmap``.

    :param out_dir: The persistence directory of the ``np.memmap``.
    :return: A dictionary representing the ``kwargs`` needed for initialization.
    """
    out_dir = _create_if_not_exists(out_dir)
    return {
        "dtype": base._file_to_str(out_dir / "dtype.ninja"),
        "shape": base._file_to_shape(out_dir / "shape.ninja"),
        "order": base._file_to_str(out_dir / "order.ninja"),
    }


def _empty(
    out_dir: Union[str, Path],
    dtype: Union[str, np.dtype],
    shape: Sequence[int],
    order: str,
) -> np.memmap:
    """
    Creates an empty ``np.memmap``

    :param out_dir: The persistence directory
    :param dtype: The dtype of the array
    :param shape: The shape of the array
    :param order: The order, either ``"C"`` or ``"F"``
    :return:
    """
    out_dir = _create_if_not_exists(out_dir)
    _save_mmap_kwargs(out_dir, dtype, shape, order)
    memmap = np.memmap(str(out_dir / "data.ninja"), mode="w+", dtype=dtype, shape=shape, order=order)
    return memmap


def from_ndarray(out_dir: Union[str, Path], arr: np.ndarray) -> np.memmap:
    """
    Initializes a memory map, in which all samples should be of the same shape

    :param out_dir: The directory in which the memory map will be persisted
    :param arr: The numpy array which should be memory mapped.
    :return: The memory mapped file
    """
    arr = np.asarray(arr)
    out_dir = _create_if_not_exists(out_dir)
    dtype = arr.dtype
    shape = arr.shape
    order = "F" if np.isfortran(arr) else "C"
    memmap = np.memmap(str(out_dir / "data.ninja"), mode="w+", dtype=dtype, shape=shape, order=order)
    memmap[:] = arr
    _save_mmap_kwargs(out_dir, dtype, shape, order)
    return memmap


def _write_samples(
    memmap: Optional[np.memmap],
    out_dir: Path,
    samples: Sequence[np.ndarray],
    start: int,
    total: int,
) -> Tuple[np.memmap, int]:
    """
    Writes samples starting from a given start index in the memory map.
    Initializes the memory map if it is not yet initialized

    :param memmap: The memory map or ``None`` if not initialized yet
    :param out_dir: The directory in which the memory map is persisted
    :param samples: The samples that need to be written
    :param start: The start index at which the samples should be written
    :param total: Total number of samples. Used when initializing the memory map.
    :return: A tuple of the updated memory map, and the end offset at which the last sample was written.
    """
    arr = np.stack(samples)
    if memmap is None:
        dtype = arr.dtype
        shape = (total,) + arr.shape[1:]
        order = "F" if np.isfortran(arr) else "C"
        memmap = np.memmap(
            str(out_dir / "data.ninja"),
            mode="w+",
            dtype=dtype,
            shape=shape,
            order=order,
        )
    end = start + len(samples)
    memmap[start:end] = arr
    return memmap, end


def from_generator(out_dir: Union[str, Path], sample_generator, batch_size: int, n: int, verbose=False) -> np.memmap:
    """
    Create a numpy memory-map from a sample generator.

    :param sample_generator: A generator of the samples
    :param out_dir: The output directory
    :param batch_size: How often to flush to disk
    :param n: Total number of samples.
    :param verbose: Whether to show the progress bar.
    :return:
    """
    return from_generator_base(
        out_dir=out_dir,
        sample_generator=sample_generator,
        batch_size=batch_size,
        batch_ctor=from_ndarray,
        extend_fn=extend,
        verbose=verbose,
    )


def open_existing(out_dir: Union[str, Path], mode="r") -> np.memmap:
    """
    Open an already existing numpy array.

    :param out_dir: The output directory.
    :param mode: The mode with which to open the memory-mapped file.
    :return: The ``np.memmap`` object.
    """
    out_dir = Path(out_dir)
    kwargs = _read_mmap_kwargs(out_dir)
    memmap = np.memmap(str(out_dir / "data.ninja"), mode=mode, **kwargs)
    return memmap


def extend_dir(out_dir: Union[str, Path], arr: np.ndarray) -> None:
    """
    Extend an already existing memory map by adding new samples

    :param out_dir: The directory, in which the memory-mapped array is stored.
    :param arr: The numpy array of new samples
    :return:
    """
    arr = np.asarray(arr)
    out_dir = Path(out_dir)
    kwargs = _read_mmap_kwargs(out_dir)
    shape = kwargs["shape"]
    order = kwargs["order"]
    dtype = np.dtype(kwargs["dtype"])
    assert shape[1:] == arr.shape[1:], (
        f"Trying to append samples with incorrect shape: {arr.shape[1:]}, " f"expected: {shape[1:]}"
    )
    with open(out_dir / "data.ninja", "ab") as data_file:
        data_file.write(arr.astype(dtype).tobytes(order=order))
        data_file.flush()
    new_shape = (shape[0] + arr.shape[0], *shape[1:])
    base._shape_to_file(new_shape, out_dir / f"shape.ninja")


def extend(np_mmap: np.memmap, arr: np.ndarray) -> None:
    """
    Extend a numpy memory map with new samples

    :param np_mmap: The numpy memory map object
    :param arr: The new samples
    :return:
    """
    extend_dir(Path(np_mmap.filename).parent, arr)


def append(np_mmap: np.memmap, arr: np.ndarray) -> None:
    """
    Append a single sample to an already existing numpy array

    :param np_mmap:
    :param arr:
    """
    extend(np_mmap, np.expand_dims(np.asarray(arr), axis=0))


@dataclass
class NumpyBytesSlices:
    buffer: np.ndarray
    starts: List[int]
    ends: List[int]
    flattened_shapes: List[int]
    shapes: List[Tuple[int]]


def _lists_of_ndarrays_to_bytes(lists: Sequence[np.ndarray], dtype) -> NumpyBytesSlices:
    """
    Converts a list of numpy arrays into bytes.

    :param lists: The samples that have to be converted.
    :param dtype: The dtype of the arrays. If not provided, the dtype of the first sample will be used.
    :return: The numpy arrays as bytes
    """
    offset = 0
    starts = []
    ends = []
    shapes = []
    arrs = []
    flattened_shapes = []
    for l in lists:
        if dtype is None:
            arr = np.asarray(l)
            dtype = arr.dtype
        else:
            arr = np.asarray(l, dtype=dtype)
        flattened = arr.ravel()
        starts.append(offset)
        ends.append(offset + len(flattened))
        flattened_shapes.append(len(flattened))
        shapes.append(arr.shape if len(arr.shape) > 0 else (0,))
        arrs.append(flattened)
        offset += len(flattened)
    buffer = np.concatenate(arrs)
    return NumpyBytesSlices(buffer, starts, ends, flattened_shapes, shapes)
