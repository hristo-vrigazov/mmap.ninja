# mmap.ninja

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-WMtVyfxx2aUMeV7vlG48Ia27-5cxnrS?usp=sharing)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)


Accelerate your machine learning training by up to **400 times** !

`mmap_ninja` is a library for storing your datasets in memory-mapped files,
which leads to a dramatic speedup in the training time.

The only dependency is `numpy`.

## What is it?

Let's say that you have a dataset of images for training your ML model.

How would you store the images? You have several options:


### Directory of jpg files

A directory of `.jpg` files - very popular (no need to do anything).

Pros: :heavy_plus_sign:

- No need to do additional work after downloading the dataset
- Can open a random image from the dataset easily based on its filename

Cons:

2. Load them in memory at the start of the training - very slow startup, also you need A LOT OF memory.
3. Store them as `.npy` files - but what do we do with the different shapes of the images?