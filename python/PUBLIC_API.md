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

Once you have opened it, you can do all the usual numpy operations
on the `np.memmap`.

### Create a RaggedMmap from list of samples

If your samples are of different shapes, then you should use
`RaggedMmap`. If they all fit into memory, you can use the
`from_lists` class method:

```python
import numpy as np
from mmap_ninja.ragged import RaggedMmap

simple = [np.array([11, 13, -1, 17]), np.array([2, 3, 4, 19]), np.array([90, 12])]
RaggedMmap.from_lists("simple", simple)
```

### Create a RaggedMmap from a generator

If your samples don't fit into memory, you can initialize
it from a generator, which yields samples (a sample is a `np.ndarray`).
This method flushes to disk every `batch_size` samples. Optionally, a progress bar can be shown
during the conversion using `tqdm`, since the conversion usually takes a long time.

```python
import matplotlib.pyplot as plt
from mmap_ninja.ragged import RaggedMmap
from pathlib import Path

img_path = Path('<PATH TO IMAGE DATASET>')
val_images = RaggedMmap.from_generator(
    out_dir='val_images', 
    sample_generator=map(plt.imread, img_path.iterdir()), 
    batch_size=1024, 
    verbose=True
)
```

### Open an existing RaggedMmap

Once the `RaggedMmap` has been created, just open it using its
constructor:

```python
import numpy as np
from mmap_ninja.ragged import RaggedMmap

images = RaggedMmap('val_images')
assert isinstance(np.ndarray, images[4])
```

For convenience, you can also pass in a wrapper function to be applied
to every sample after `__getitem__`.

```python
import numpy as np
import torch
from mmap_ninja.ragged import RaggedMmap

images = RaggedMmap('val_images', wrapper_fn=torch.tensor)
assert isinstance(np.ndarray, images[4])
```

### Append new samples to a RaggedMmap

To append a single sample, use `RaggedMmap.append`.

To append multiple samples, use `RaggedMmap.extend`.

```python
import numpy as np
from mmap_ninja.ragged import RaggedMmap

mmap = RaggedMmap('samples')
new_samples = [np.array([123, -1]), np.array([-1, 0, 123, 92, 12])]
mmap.extend(new_samples)
```