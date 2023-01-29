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


### Create a Numpy memmap from a generator

Very often, you cannot load the whole dataset into memory.
For this reason, `mmap-ninja` provides a method for initializing
a `np.memmap` from a generator of samples, which flushes to disk
every `batch_size` samples. Optionally, a progress bar can be shown
during the conversion using `tqdm`, since the conversion usually takes a long time.

```python
import matplotlib.image as mpimg
import numpy as np
from mmap_ninja import numpy as np_ninja
from pathlib import Path
from os import listdir

imgs_dir = Path('./path_to_img_dir')

np_ninja.from_generator(
    out_dir='imgs_mmap',
    sample_generator=map(mpimg.imread, imgs_dir.iterdir()),
    batch_size=32,
    n=len(listdir(imgs_dir))
)
```

### Append new samples to a Numpy memmap

To append/extend new sample, just use the `np_ninja.extend`
or `np_ninja.extend_dir` method. The difference between the two methods
is the type of the input, one uses `np.memmap` directly, while the other
uses the directory, in which the `np.memmap` is persisted.

```python
import numpy as np

from mmap_ninja import numpy as np_ninja

arr = np.arange(3)
memmap = np_ninja.from_ndarray("growable", arr)
np_ninja.extend(memmap, np.arange(11, 13))
np_ninja.extend_dir("growable", np.arange(14, 16))
```

### Open existing Numpy memmap

Once you have created a `np.memmap`, you can open it
in later stages of the project using `np_ninja.open_existing`

```python
from mmap_ninja import numpy as np_ninja

memmap = np_ninja.open_existing("growable")
```