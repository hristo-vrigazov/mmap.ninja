__version__ = "0.7.3"

from .base import Wrapped
from .numpy import (
    from_ndarray as np_from_ndarray,
    from_generator as np_from_generator,
    open_existing as np_open_existing,
    extend_dir as np_extend_dir,
    extend as np_extend,
    append as np_append
)

from .generic import open_existing
from .ragged import RaggedMmap
from .string import StringsMmap
