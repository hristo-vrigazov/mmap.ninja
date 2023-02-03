from mmap_ninja import base
from mmap_ninja import numpy as np_ninja
from mmap_ninja.base import Wrapped
from mmap_ninja.string import StringsMmap
from mmap_ninja.ragged import RaggedMmap
from pathlib import Path
from typing import Union, Callable, Optional


def open_existing(out_dir: Union[str, Path], wrapper_fn: Optional[Callable] = None):
    out_dir = Path(out_dir)
    type_str = base._file_to_str(out_dir / "type.ninja")
    if type_str == "numpy":
        memmap = np_ninja.open_existing(out_dir)
        if wrapper_fn is not None:
            return Wrapped(memmap, wrapper_fn)
        return memmap
    if type_str == "ragged":
        return RaggedMmap(out_dir, wrapper_fn=wrapper_fn)
    if type_str == "string":
        return StringsMmap(out_dir)
    raise ValueError(f'Unknown type "{type_str}" while trying to open "{out_dir}" !')
