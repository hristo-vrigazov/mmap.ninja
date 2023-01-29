# Public API

Here, you can find a full list of the things you can do with `mmap-ninja`.

### Wrapped

The `Wrapped` class allows you to lazily apply a function
whenever `__getitem__` is called.
For example:

```python
from mmap_ninja.base import Wrapped

wrapped = Wrapped(dataset, wrapper_fn=torch.tensor)

print(wrapped[14])
```

### Create a Numpy memmap from a Numpy array

The `mmap_ninja.numpy` module provides utilities for
initializing and reading from a `np.memmap`.
You can create a `np.memmap` from an `np.array` using 
this example:

```python
import numpy as np
from mmap_ninja import numpy as np_ninja

arr = np.random.randn(200, 224, 224, 3)
np_ninja.from_ndarray('imgs_mmap', arr)
```