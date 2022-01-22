## mmap.ninja Python API

### Memory mapping text documents



### Memory mapping images with different shapes

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
You can see that once created, the `RaggedMmmap` is **383 times** faster for iterating over the 
dataset.
It does require 4 times more disk space though, so if you are willing to trade 4 times more disk space
for **383 times** speedup, you can use the `RaggedMmap`.

|                  |   Initial load (s) |   Time for iteration (s) | Memory usage (GB)   | Disk usage (GB)   |
|:-----------------|-------------------:|-------------------------:|:--------------------|:------------------|
| in_memory        |           1.356077 |                 0.000403 | 3.818741 GB         | 3.819034 GB       |
| ragged_mmap      |           0.002054 |                 0.057858 | 0.001144 GB         | 3.819114 GB       |
| imread_from_disk |           0.000000 |                22.208385 | 0.001144 GB         | 0.758753 GB       |