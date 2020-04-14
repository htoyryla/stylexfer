neural-imagen
=============

THIS IS A FORK BY @htoyryla
FOR EDUCATIONAL AND EXPERIMENTAL PURPOSES
OF AN ORIGINAL REPO BY @alexjc

NO ATTEMPT IS MADE TO RETAIN COMPATIBILITY WITH THE ORIGINAL

This repository includes:

1. A library of building blocks for state-of-the-art image synthesis.
2. Reference implementations of popular deep learning algorithms.

----

Reference Implementations
=========================

In the examples folder, you'll find a documented implementation of neural style transfer based on the following:

* `A Neural Algorithm of Artistic Style <https://arxiv.org/abs/1508.06576>`_, Gatys et al. 2015.
* `Improving the Neural Algorithm of Artistic Style <https://arxiv.org/abs/1605.04603>`_, Novak & Nikulin, 2016.
* `Stable and Controllable Neural Synthesi <https://arxiv.org/abs/1701.08893>`_, Risser et al, 2017.


Requirements
------------


pytorch (I am using 1.1)
torchvision 

progressbar2

---to be completed ----


Quick start
-----------

To do basic style transfer, give

.. code:: bash

    python stylexfer.py --style style1-512.png --content tallinna800.png   --output out.png  --scales 3 --iterations 500 --style-multiplier 1e+6

You can then go on to experiment

* the effect for style-multiplier
* different style and content images 

The size of the images DOES matter. The output image will be the size as the content image. Initially it is best to use style images of roughly the same size. Changing the style image resolution will enlarge or decrease the size of the style patterns produced. 

Usage
-----

1. Texture Synthesis
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    python stylexfer.py --style style1-512.png --output-size 512x512 --output textured.png 



2. Style Transfer
~~~~~~~~~~~~~~~~~

.. code:: bash

    python stylexfer.py --content tallinna800.png --style style1-512.png --output stylexferred.png


Options
-------

You will likely need to experiment with the default options to obtain good results:

* ``--scales=N``: Coarse-to-fine rendering with downsampled images.
* ``--iterations=N``: Number of steps to run the optimizer at each scale.
* ``--style-layers A,B,C,D``: Specify convolution layers of VGG19 manually, by default ``1_2,2_2,3_3,4_3,5_3``.
* ``--style-weights a,b,c,d``: Override loss weights for style layers, by default ``1.0`` for each.
* ``--content-layers E F``: Specify convolution layers of VGG19 manually, by default ``4_1`` for ``relu4_1``.
* ``--content-weights e f``: Override loss weight for content layers, by default ``1.0``.
* ``--seed image.png``: Provide a starting image for the optimization.
* ``--seed <integer>``: Give a random seed manually (for reproducibility)

Changing content layers has not yet been tested.


COMING:

Scripts for processing video frames or folders of images.
