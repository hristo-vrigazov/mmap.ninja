import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, List


def bytes_to_int32(inp: bytes, byteorder="little", signed=True) -> int:
    return int.from_bytes(bytes=inp, byteorder=byteorder, signed=signed)


def int32_to_bytes(inp: int, fmt: str = '<i') -> bytes:
    return struct.pack(fmt, inp)


def bytes_to_str(inp: bytes, encoding: str = 'utf-8') -> str:
    return inp.decode(encoding)


def str_to_bytes(inp: str, encoding: str = 'utf-8') -> bytes:
    return inp.encode(encoding)


def int32_to_file(inp: int, file: Union[str, Path], *args, **kwargs):
    with open(file, 'wb') as out_file:
        out_file.write(int32_to_bytes(inp, *args, **kwargs))


def file_to_int32(file: Union[str, Path], *args, **kwargs) -> int:
    with open(file, 'rb') as in_file:
        return bytes_to_int32(in_file.read(), *args, **kwargs)


def str_to_file(inp: str, file: Union[str, Path], *args, **kwargs):
    with open(file, 'wb') as out_file:
        out_file.write(str_to_bytes(inp, *args, **kwargs))


def file_to_str(file: Union[str, Path], *args, **kwargs) -> str:
    with open(file, 'rb') as in_file:
        return bytes_to_str(in_file.read(), *args, **kwargs)


def shape_to_bytes(shape: Sequence[int]) -> bytes:
    res = bytearray()
    for axis_len in shape:
        res.extend(int32_to_bytes(axis_len))
    return bytes(res)


def bytes_to_shape(inp: bytes, step=4) -> Sequence[int]:
    res = []
    for start in range(0, len(inp) - 1, step):
        end = start + step
        res.append(bytes_to_int32(inp[start:end]))
    return tuple(res)


def shape_to_file(shape: Sequence[int], file: Union[str, Path]):
    with open(file, 'wb') as out_file:
        out_file.write(shape_to_bytes(shape))


def file_to_shape(file: Union[str, Path]) -> Sequence[int]:
    with open(file, 'rb') as out_file:
        return bytes_to_shape(out_file.read())


@dataclass
class BytesSlices:
    buffer: bytes
    starts: List[int]
    ends: List[int]


def sequence_of_strings_to_bytes(strings: Sequence[str], verbose=False) -> BytesSlices:
    buffer = bytearray()
    starts = []
    ends = []
    if verbose:
        from tqdm import tqdm
        strings = tqdm(strings)
    for string in strings:
        arr = str_to_bytes(string)
        starts.append(len(buffer))
        ends.append(len(buffer) + len(arr))
        buffer.extend(arr)
    return BytesSlices(bytes(buffer), starts, ends)


