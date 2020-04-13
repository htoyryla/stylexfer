neural-imagen
=============

THIS IS A FORK OF AN ORIGINAL REPO BY @alexjc
FOR EDUCATIONAL AND EXPERIMENTAL PURPOSES

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


Usage
-----

To do basic style transfer, give

.. code:: bash

    python examples/iterative2.py --style hameenkatu222.png --content tallinna600x800.png   --output koe2.png  --scales 3 --iterations 500 --style-multiplier 1e+6




1. Texture Synthesis
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    python examples/iterative2.py --style texture.png --output-size 256x256 --output generated1.png 


2. Image Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    python examples/iterative2.py --content image.png --output generated2.png


3. Style Transfer
~~~~~~~~~~~~~~~~~

.. code:: bash

    python examples/iterative2.py --content image.png --style texture.png --output generated3.png


Options
-------

You will likely need to experiment with the default options to obtain good results:

* ``--scales=N``: Coarse-to-fine rendering with downsampled images.
* ``--iterations=N``: Number of steps to run the optimizer at each scale.
* ``--style-layers A,B,C,D``: Specify convolution layers of VGG19 manually, by default ``1_2,2_2,3_3,4_3,5_3`.
* ``--style-weights a,b,c,d``: Override loss weights for style layers, by default ``1.0`` for each.
* ``--content-layers E F``: Specify convolution layers of VGG19 manually, by default ``4_`` for ``relu4_1``.
* ``--content-weights e f``: Override loss weight for content layers, by default ``1.0``.
* ``--seed image.png``: Provide a starting image for the optimization.
* ``--seed <integer>``: Give a random seed manually (for reproducibility)

COMING:

Scripts for processing video frames or folders of images.
