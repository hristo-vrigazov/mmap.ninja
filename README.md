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

Accelerate the iteration over your machine learning dataset by up to **20 times** !

`mmap_ninja` is a library for storing your datasets in memory-mapped files,
which leads to a dramatic speedup in the training time.

The only dependencies are `numpy` and `tqdm`.

## What is it?

When working on a machine learning project, one of the most time-consuming parts is the model's training.
However, a large portion of the training time actually consists of just iterating over your dataset and filesystem I/O!

This library, `mmap_ninja` provides high-level, easy to use, well tested API for using memory maps for your 
datasets, reducing the time needed for training.

## Use cases

| Use case                | Initialize                                                       |
|:------------------------|:-----------------------------------------------------------------|
| I have an image dataset | [Markdown example](#memory-mapping-images-with-different-shapes) |
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

You can see that once created, the `RaggedMmmap` is **383 times** faster for iterating over the 
dataset.
It does require 4 times more disk space though, so if you are willing to trade 4 times more disk space
for **383 times** speedup (and less memory usage), you definitely should use the `RaggedMmap`!

This makes the `RaggedMmmap` a fantastic choice for your computer vision, image-based machine learning datasets!


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
