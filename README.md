[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

# mmap.ninja


Install with:

```bash
pip install mmap_ninja
```

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-WMtVyfxx2aUMeV7vlG48Ia27-5cxnrS?usp=sharing)
[![Build Status](https://app.travis-ci.com/hristo-vrigazov/mmap.ninja.svg?branch=master)](https://app.travis-ci.com/hristo-vrigazov/mmap.ninja)
[![codecov](https://codecov.io/gh/hristo-vrigazov/mmap.ninja/branch/master/graph/badge.svg?token=YUCO0KJONB)](https://codecov.io/gh/hristo-vrigazov/mmap.ninja)
[![PyPI download month](https://img.shields.io/pypi/dm/mmap_ninja.svg)](https://pypi.python.org/pypi/mmap_ninja/)
[![PyPi version](https://badgen.net/pypi/v/mmap_ninja/)](https://pypi.com/project/mmap_ninja)
[![PyPI license](https://img.shields.io/pypi/l/mmap_ninja.svg)](https://pypi.python.org/pypi/mmap_ninja/)

![Alt text](./carbon_ragged.svg)

Accelerate the iteration over your machine learning dataset by up to **20 times** !

`mmap_ninja` is a library for storing your datasets in memory-mapped files,
which leads to a dramatic speedup in the training time.

The only dependencies are `numpy` and `tqdm`.

You can use `mmap_ninja` with any training framework (such as `Tensorflow`, `PyTorch`, `MxNet`), etc.,
as it stores your dataset as a memory-mapped numpy array.



## What is it?

A **memory mapped file** is a file that is physically present on disk in a way that the correlation between the file
and the memory space permits applications to treat the mapped portions as if it were primary memory, allowing very fast
I/O!

When working on a machine learning project, one of the most time-consuming parts is the model's training.
However, a large portion of the training time actually consists of just iterating over your dataset and filesystem I/O!

This library, `mmap_ninja` provides high-level, easy to use, well tested API for using memory maps for your 
datasets, reducing the time needed for training.


Memory maps would usually take a little more disk space though, so if you are willing to trade some disk space
for fast filesystem to memory I/O, this is your library!

All memory map classes are initialized first (once per project) - usually this is done
from a generator or an in-memory object (check the [use cases](#use-cases)).

For example, the classmethod for creating a `RaggedMmap` from a generator looks like this:

```python
@classmethod
def from_generator(
        cls,
        out_dir: Union[str, Path],
        sample_generator,
        batch_size: int,
        verbose=False,
        **kwargs
):
```

where `out_dir` is the directory to persist the memory map, the `samples_generator` is the generator
of the samples (duh), and `batch_size` is the maximum number of samples to be kept at once in memory
before flushing to disk. `verbose` either shows a progress bar or does not.

Once created, they provide the following highly performant methods:

```python

def __getitem__(self, item):
def __len__(self):
def append(self, array: np.ndarray):
def extend(self, arrays: Sequence[np.ndarray]):
```

## Use cases

| Use case                 | Notebook                                                                                                                                                             | Benchmark                                                 | Class/Module                                |
|:-------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|:--------------------------------------------|
| List of image files      | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-WMtVyfxx2aUMeV7vlG48Ia27-5cxnrS?usp=sharing) | [COCO 2017](#memory-mapping-images-with-different-shapes) | `from mmap_ninja.ragged import RaggedMmap`  |
| List of text files       | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18bEwylFwx4owMpb-RAkJZS_9JrrUcFd7?usp=sharing) | [20 newsgroups](#memory-mapping-text-documents)           | `from mmap_ninja.string import StringsMmap` |
| Flat array (e.g. labels) | Coming soon!                                                                                                                                                         | Coming soon!                                              | `from mmap_ninja import numpy as np_ninja`   |

### Memory mapping images with different shapes

You can create a new `RaggedMmmap` from one of its class methods: `RaggedMmmap.from_lists`, 
`RaggedMmap.from_generator`.

Create a memory map from generator, flushing to disk every 1024 images (so that you don't have to keep it all in memory at once):

```python
import matplotlib.pyplot as plt
from mmap_ninja.ragged import RaggedMmap
from pathlib import Path

coco_path = Path('<PATH TO IMAGE DATASET>')
val_images = RaggedMmap.from_generator(
    out_dir='val_images', 
    sample_generator=map(plt.imread, coco_path.iterdir()), 
    batch_size=1024, 
    verbose=True
)
```

Once created, you can open the map by simply supplying the path to the memory map:
```python
from mmap_ninja.ragged import RaggedMmap

val_images = RaggedMmap('val_images')
print(val_images[3]) # Prints the ndarray image, e.g. with shape (387, 640, 3)
```

You can also extend an already existing memory map easily by using the `.extend` method.

In the table show the time needed for initial loading, one iteration over the COCO validation 2017 dataset,
the memory usage of every method and the disk usage.


|                  |   Initial load (s) |   Time for iteration (s) | Memory usage (GB)   | Disk usage (GB)   |
|:-----------------|-------------------:|-------------------------:|:--------------------|:------------------|
| in_memory        |           1.356077 |                 0.000403 | 3.818741 GB         | 3.819034 GB       |
| ragged_mmap      |           0.002054 |                 0.057858 | 0.001144 GB         | 3.819114 GB       |
| imread_from_disk |           0.000000 |                22.208385 | 0.001144 GB         | 0.758753 GB       |

You can see that once created, the `RaggedMmap` is **383 times** faster for iterating over the 
dataset.
It does require 4 times more disk space though, so if you are willing to trade 4 times more disk space
for **383 times** speedup (and less memory usage), you definitely should use the `RaggedMmap`!

This makes the `RaggedMmap` a fantastic choice for your computer vision, image-based machine learning datasets!

### Memory mapping text documents

You can create a new `StringsMmmap` from one of its class methods: `StringsMmmap.from_strings`,
`StringsMmap.from_generator`. 
Once it's created, you can open it by just supplying the path to the memory map.

An example of creating a memory map for the [sklearn's 20newsgroups dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html):

```python
from mmap_ninja.string import StringsMmap
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
memmap = StringsMmap.from_strings('20newsgroups', data['data'], verbose=True)
```

Opening an already existing `StringsMmmap`:

```python
from mmap_ninja.string import StringsMmap

texts = StringsMmap('20newsgroups')
print(texts[123])  # Prints the 123-th text
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



[//]: # (### Directory of jpg files)

[//]: # ()
[//]: # (A directory of `.jpg` files - very popular &#40;no need to do anything&#41;.)

[//]: # ()
[//]: # (Pros and cons:)

[//]: # ()
[//]: # (:heavy_plus_sign: No need to do additional work after downloading the dataset)

[//]: # ()
[//]: # (:heavy_plus_sign: Can open a random image from the dataset easily based on its filename)

[//]: # ()
[//]: # (:heavy_minus_sign: It's super slow :hourglass_flowing_sand:. Like, really, )

[//]: # (really slow and you waste all that time on every epoch, on every model you train!)

[//]: # ()
[//]: # (More information is coming soon!)
