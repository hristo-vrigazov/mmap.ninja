.. mmap_ninja documentation master file, created by
   sphinx-quickstart on Sat Jun 18 16:34:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mmap_ninja's documentation!
======================================

``mmap-ninja`` allows you to persist a list of numpy arrays (of varying shapes) in a memory-mapped format,
so you can randomly lookup an element of this list very fast. It's only dependencies are
``numpy`` and optionally ``tqdm`` (for verbose output).

``mmap-ninja`` is very tiny, <2k lines.

Everything you need to know about ``mmap-ninja`` is on this page.

In deep learning projects, ``mmap-ninja`` significantly speeds up the reading of the samples
at the cost of more disk space. The idea is that you convert your samples once per project into a
memory-mapped format, and then you can read from it very fast. You can also append to an existing memory map.

The benefits of ``mmap-ninja`` (almost no memory usage, very low CPU usage, because it eliminates the need
for parallel workers) come at the cost of more disk space, so if you are willing to trade disk space for memory and CPU,
this is your library :)


Samples of different shape
====================================

In the examples below, ``sample_generator`` is a sequence of ``numpy`` arrays.

Initialize once per project::

   from mmap_ninja.ragged import RaggedMmap
   from pathlib import Path

   val_images = RaggedMmap.from_generator(
       out_dir='<OUTPUT PATH>',
       sample_generator=sample_generator,
       batch_size=1024,
       verbose=True
   )


Once created, you can open the map by simply supplying the path to the memory map::

   from mmap_ninja.ragged import RaggedMmap

   val_images = RaggedMmap('<OUTPUT PATH>')
   print(val_images[3]) # Prints the ndarray

Reading from a ``RaggedMmap`` is several hundred times faster than reading ``.jpg`` files from disk.
Check https://github.com/hristo-vrigazov/mmap.ninja#memory-mapping-images-with-different-shapes for benchmarks and
Colab notebooks.

String samples
====================================

In the example below, we convert a list of strings to a ``StringsMmap``.

Initialize once per project::

   from mmap_ninja.string import StringsMmap

   memmap = StringsMmap.from_generator(
      out_dir='<OUTPUT PATH>',
      sample_generator=generate_strs(),
      batch_size=4,
      verbose=True
   )


Once created, you can open the map by simply supplying the path to the memory map::

   from mmap_ninja.string import StringsMmap

   texts = StringsMmap('20newsgroups')
   print(texts[123])  # Prints the 123-th text


Samples of the same shape
=========================

Initialize once per project::

   from mmap_ninja import numpy as np_ninja

   memmap = np_ninja.from_generator(
      out_dir=tmp_path / 'generator',
      sample_generator=simple_gen(),
      batch_size=4,
      n=30,
      verbose=True
   )

Once created, you can open the map by simply supplying the path to the memory map::

   from mmap_ninja import numpy as np_ninja

   memmap = np_ninja.open_existing(tmp_path / 'growable')
   print(memmap[123])  # Prints the 123-th sample

.. toctree::
   :maxdepth: 2
   :caption: API reference:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


