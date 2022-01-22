## mmap.ninja Python API

### Memory mapping text documents

You can create a new `StringsMmmap` from one of its class methods: `StringsMmmap.from_strings`,
`StringsMmap.from_generator`. 
Once it's created, you can open it by just supplying the path to the memory map.

An example of creating a memory map for the [sklearn's 20newsgroups dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html):

```python
from mmap_ninja.string import StringsMmmap
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
memmap = StringsMmmap.from_strings('20newsgroups', data['data'], verbose=True)
```

Opening an already existing `StringsMmmap`:
```python
from mmap_ninja.string import StringsMmmap
texts = StringsMmmap('20newsgroups')
print(texts[123]) # Prints the 123-th text
```

You can also extend an already existing memory map easily by using the `.extend` method.

In the table show the time needed for initial loading, 100 iterations over the [sklearn's 20newsgroups dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html),
the memory usage of every method and the disk usage.

|                |   Initial load (s) |   Time for iteration (s) | Memory usage (GB)   | Disk usage (GB)   |
|:---------------|-------------------:|-------------------------:|:--------------------|:------------------|
| in_memory      |           0.174626 |                 0.068995 | 0.09 MB             | 45 MB             |
| ragged_mmap    |           0.003701 |                 2.052659 | 0.07 MB             | 22 MB             |
| read_from_disk |           0.000000 |                13.996738 | 0.07 MB             | 45 MB             |


You can see that once created, the `StringsMmap` is nearly **7 times** faster compared to reading `.txt` files
from disk one by one.
Moreover, it takes **2 times** less disk space (this is true only for `StringsMmap`, in general for other types the memory map
would take more disk space).
This makes the `StringsMmmap` a fantastic choice for your NLP, text-based machine learning datasets!

### Memory mapping images with different shapes

You can create a new `RaggedMmmap` from one of its class methods: `RaggedMmmap.from_lists`, 
`RaggedMmap.from_generator`.

Create a memory map from generator, flushing to disk every 1024 images (so that you don't have to keep it all in memory at once):

```python
import matplotlib.pyplot as plt
from mmap_ninja.ragged import RaggedMmap
from pathlib import Path

coco_path = Path('/home/hvrigazov/data/coco/val2017')
val_images = RaggedMmap.from_generator('val_images', 
                                       map(plt.imread, coco_path.iterdir()), 
                                       batch_size=1024, 
                                       verbose=True)
```

Once created, you can open the map by simply supplying the path to the memory map:
```python
from mmap_ninja.ragged import RaggedMmap

training_images = RaggedMmap('val_images')
print(training_images[3]) # Prints the ndarray image, e.g. with shape (387, 640, 3)
```

You can also extend an already existing memory map easily by using the `.extend` method.

In the table show the time needed for initial loading, one iteration over the COCO validation 2017 dataset,
the memory usage of every method and the disk usage.


|                  |   Initial load (s) |   Time for iteration (s) | Memory usage (GB)   | Disk usage (GB)   |
|:-----------------|-------------------:|-------------------------:|:--------------------|:------------------|
| in_memory        |           1.356077 |                 0.000403 | 3.818741 GB         | 3.819034 GB       |
| ragged_mmap      |           0.002054 |                 0.057858 | 0.001144 GB         | 3.819114 GB       |
| imread_from_disk |           0.000000 |                22.208385 | 0.001144 GB         | 0.758753 GB       |

You can see that once created, the `RaggedMmmap` is **383 times** faster for iterating over the 
dataset.
It does require 4 times more disk space though, so if you are willing to trade 4 times more disk space
for **383 times** speedup (and less memory usage), you definitely should use the `RaggedMmap`!

This makes the `RaggedMmmap` a fantastic choice for your computer vision, image-based machine learning datasets!
