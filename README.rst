stylexfer
=============

THIS IS A FORK BY @htoyryla
FOR EDUCATIONAL AND EXPERIMENTAL PURPOSES
OF AN ORIGINAL REPO BY @alexjc

NO ATTEMPT IS MADE TO RETAIN COMPATIBILITY WITH THE ORIGINAL

Focus here is on style transfer applications rather than in texture generation.

Requirements
------------


pytorch (I am using 1.4)
torchvision 

progressbar2

moviepy (for processing video frames)

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

You can now adjust style size with --style-size parameter.

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
* ``--style-size xhw``: Resize style image, e.g. 400x600 (height x width).
* ``--content-size xhw``: Resize content image, e.g. 400x600 (height x width).
* ``--device <device>``: 'cuda' or 'cpu', default is cuda.

Image sizes
-----------

When working with your own images, you probably will see errors like this:

RuntimeError: The size of tensor a (404) must match the size of tensor b (405) at non-singleton dimension 3

This happens because the images are processed at multiple resolutions, like 1/8, 1/4, 1/2 and 1/1 of the full size. Therefore image dimensions should be multiples of e.g. 8 (depending on the value of scales).

When this happens, you can use --content-size and --style-size to give dimensions that are safe (you don't have to go and resize your image files).


Using multiple content images
------------------------------

.. code:: bash

    python stylexfer.py --content tallinna800.png,ffurt05.jpg --content-size 600x800 --style style2.png --style-size 600x800 --style-multiplier 1e4 --iterations 400 --output stylexferred.png --save-every 50

All content images have to be the same size, which is here achieved by specifying the content-size explicitly. 

Variable iterations
-------------------

By default the same number of iterations is run at each scale. It is also possible to give the iterations for each scale explicitly:

.. code:: bash

    python stylexfer.py --content tallinna800.png --content-size 600x800 --style style1-512.png --style-multiplier 1e5 --scales 4 --iterations 50 50 100 400 --output stylexferred.png --save-every 50

Here we have used --save-every 50 to be able to view the intermediate results (stored in output/ folder).

Processing video
----------------

Install moviepy

Extract frames from video with v2frames.py:

.. code:: bash

    python v2frames.py --video /path/to/your/videofile --output_dir frames/
    
To extract only a part, specify start and end positions /in seconds):

.. code:: bash

    python v2frames.py --video /work3/tools/mmovie/movie.mp4 --start 10 --end 15 --output_dir frames/ 

Run stylexfer in series/cascade mode

.. code:: bash

    python stylexfer.py --style style1.png  --content frames/l-%d.jpg  --content-size 480x640 --style-size 480x640 --output output/processed-%d.png --scales 3 --iterations 300 --style-multiplier 1e+5 --seed-random 765 --series --cascade --start 12633 --howmany 16

Because of the series flag, this will convert 16 successively numbered frames starting from frames/l-12633.jpg and place the converted frames in output/ . Make sure that the required folders exist. Note the format of content and output filenames: %d will be replaced by the actual frame number.

The cascade flag reduces processing time by seeding each successive frame from the output of the previous.

To ensure maximum consistence between processed frames, give a random seed (any integer) manually, as in --seed-random 765 in the example above.

A video can then by created from the converted frames with a suitable tool. Instructions for ffmpeg will be included here as soon as possible.
 


