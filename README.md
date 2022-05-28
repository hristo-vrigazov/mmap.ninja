# mmap.ninja

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-WMtVyfxx2aUMeV7vlG48Ia27-5cxnrS?usp=sharing)
[![PyPI download month](https://img.shields.io/pypi/dm/mmap_ninja.svg)](https://pypi.python.org/pypi/mmap_ninja/)
[![PyPi version](https://badgen.net/pypi/v/mmap_ninja/)](https://pypi.com/project/mmap_ninja)
[![PyPI license](https://img.shields.io/pypi/l/mmap_ninja.svg)](https://pypi.python.org/pypi/mmap_ninja/)

Accelerate your machine learning training by up to **400 times** !

`mmap_ninja` is a library for storing your datasets in memory-mapped files,
which leads to a dramatic speedup in the training time.

The only dependencies are `numpy` and `tqdm`.

## What is it?

Let's say that you have a dataset of images for training your ML model.

For the sake of example, let's use the `Oxford Pets` segmentation dataset.

How would you store the images? You have several options:


### Directory of jpg files

A directory of `.jpg` files - very popular (no need to do anything).

Pros and cons:

:heavy_plus_sign: No need to do additional work after downloading the dataset

:heavy_plus_sign: Can open a random image from the dataset easily based on its filename

:heavy_minus_sign: It's super slow :hourglass_flowing_sand:. Like, really, 
really slow and you waste all that time on every epoch, on every model you train!

