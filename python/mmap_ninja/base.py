import struct
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, List


def _bytes_to_int(inp: bytes, fmt: str = "<i") -> int:
    """
    Utility function for converting bytes into int.

    :param inp: the ``bytes`` input
    :param fmt: the format of the integer. Check https://docs.python.org/3/library/struct.html#format-characters
    :return: The integer
    """
    return struct.unpack(fmt, inp)[0]


def _int_to_bytes(inp: int, fmt: str = "<i") -> bytes:
    """
    Stores an integer as bytes

    :param inp: The integer input
    :param fmt: the format to use when storing. Check https://docs.python.org/3/library/struct.html#format-characters
    :return:
    """
    return struct.pack(fmt, inp)


def _bytes_to_str(inp: bytes, encoding: str = "utf-8") -> str:
    """
    Reads bytes into a string.

    :param inp: The bytes array
    :param encoding: The encoding to use.
    :return: The string
    """
    return inp.decode(encoding)


def _str_to_bytes(inp: str, encoding: str = "utf-8") -> bytes:
    """
    Dumps string as bytes

    :param inp: The string input
    :param encoding: The encoding to use
    :return: the byte array
    """
    return inp.encode(encoding)


def _int_to_file(inp: int, file: Union[str, Path], *args, **kwargs) -> None:
    """
    Utility function, which writes an integer as bytes in a given file.

    :param inp: The integer to be stored
    :param file: The file in which the integer should be stored.
    :param args: additional positional args to be passed to ``int_to_bytes``
    :param kwargs: additional keyword args to be passed to ``int_to_bytes``
    :return:
    """
    with open(file, "wb") as out_file:
        out_file.write(_int_to_bytes(inp, *args, **kwargs))


def _file_to_int(file: Union[str, Path], *args, **kwargs) -> int:
    """
    Reads an integer from a file

    :param file: The file to read
    :param args: additional positional args to be passed to ``bytes_to_int``
    :param kwargs: additional keyword args to be passed to ``bytes_to_int``
    :return: the integer
    """
    with open(file, "rb") as in_file:
        return _bytes_to_int(in_file.read(), *args, **kwargs)


def _str_to_file(inp: str, file: Union[str, Path], *args, **kwargs) -> None:
    """
    Stores a string to a file

    :param inp: The string to be written
    :param file: The output file
    :param args: additional positional args to be passed to ``str_to_bytes``
    :param kwargs: additional keyword args to be passed to ``str_to_bytes``
    :return:
    """
    with open(file, "wb") as out_file:
        out_file.write(_str_to_bytes(inp, *args, **kwargs))


def _file_to_str(file: Union[str, Path], *args, **kwargs) -> str:
    """
    Read string from file.

    :param file: The file to read from
    :param args: additional positional args to be passed to ``bytes_to_str``
    :param kwargs: additional keyword args to be passed to ``bytes_to_str``
    :return: The string
    """
    with open(file, "rb") as in_file:
        return _bytes_to_str(in_file.read(), *args, **kwargs)


def _shape_to_bytes(shape: Sequence[int], fmt: str = "<Q") -> bytes:
    """
    Returns the shape of numpy array as a byte array.

    :param shape: The shape of the numpy array
    :param fmt: The format of the bytes
    :return: The bytes array
    """
    res = bytearray()
    for axis_len in shape:
        res.extend(_int_to_bytes(axis_len, fmt=fmt))
    return bytes(res)


def _bytes_to_shape(inp: bytes, step=8, fmt: str = "<Q") -> Sequence[int]:
    """
    Reads shape tuple from bytes.

    :param inp: the bytes array
    :param step: Number of bytes to read for an int
    :param fmt: The formatting used for the integer.
    :return: A tuple of integers, representing the shape.
    """
    res = []
    for start in range(0, len(inp) - 1, step):
        end = start + step
        res.append(_bytes_to_int(inp[start:end], fmt=fmt))
    return tuple(res)


def _shape_to_file(shape: Sequence[int], file: Union[str, Path]) -> None:
    """
    Write a shape sequence to a file

    :param shape: The shape to be dumped.
    :param file: The output file
    :return:
    """
    with open(file, "wb") as out_file:
        out_file.write(_shape_to_bytes(shape))


def _file_to_shape(file: Union[str, Path]) -> Sequence[int]:
    """
    Read shape from file

    :param file: The input file
    :return: the shape as a tuple
    """
    with open(file, "rb") as out_file:
        return _bytes_to_shape(out_file.read())


@dataclass
class BytesSlices:
    buffer: bytes
    starts: List[int]
    ends: List[int]


def _sequence_of_strings_to_bytes(strings: Sequence[str], verbose=False) -> BytesSlices:
    """
    Converts a sequence of strings to a bytes array

    :param strings: A sequence of strings.
    :param verbose: whether a tqdm progressbar should be shown.
    :return: the bytes slices
    """
    buffer = bytearray()
    starts = []
    ends = []
    if verbose:
        from tqdm import tqdm

        strings = tqdm(strings)
    for string in strings:
        arr = _str_to_bytes(string)
        starts.append(len(buffer))
        ends.append(len(buffer) + len(arr))
        buffer.extend(arr)
    return BytesSlices(bytes(buffer), starts, ends)


def from_generator_base(out_dir, sample_generator, batch_size, batch_ctor, extend_fn=None, **kwargs):
    """
    Creates an output from a generator, flushing every batch to disk.

    :param out_dir: The output directory.
    :param sample_generator: The generator of samples.
    :param batch_size: The batch size, which controls how often the output should be written to disk.
    :param batch_ctor: The constructor used to initialize the output.
    :param extend_fn: Functon to call when doing .extend. By default, this will call memmap.extend(samples)
    :param kwargs: Additional keyword arguments to be passed when initializing the output.
    :return:
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    samples = []
    memmap = None
    if kwargs.pop("verbose", False):
        from tqdm import tqdm

        sample_generator = tqdm(sample_generator)
    for sample in sample_generator:
        samples.append(sample)
        if len(samples) % batch_size != 0:
            continue
        if memmap is None:
            memmap = batch_ctor(out_dir, samples, **kwargs)
        else:
            if extend_fn is not None:
                extend_fn(memmap, samples)
            else:
                memmap.extend(samples)
        samples = []
    if len(samples) > 0:
        if memmap is None:
            memmap = batch_ctor(out_dir, samples, **kwargs)
        else:
            if extend_fn is not None:
                extend_fn(memmap, samples)
            else:
                memmap.extend(samples)
    return memmap


class Wrapped:
    def __init__(self, data, wrapper_fn):
        self.data = data
        self.wrapper_fn = wrapper_fn

    def __getitem__(self, item):
        return self.wrapper_fn(copy(self.data[item]))

    def __len__(self):
        return len(self.data)
