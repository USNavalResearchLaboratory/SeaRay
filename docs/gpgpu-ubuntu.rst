Ubuntu 18.04 Advanced Install
================================

Before starting, follow the steps in :doc:`generic-install`.

Install Compilers
-----------------

It is a good idea to install both GCC and LLVM compilers first.

	#. :samp:`sudo apt update`
	#. :samp:`sudo apt install gcc`
	#. :samp:`sudo apt install llvm clang`

Support for CPU via pocl
-------------------------

Even if you get your GPU working using drivers from NVIDIA or AMD, it may still be useful to install :samp:`pocl` so you can run calculations on the CPU.  One advantage of this is that the CPU usually has access to more memory.

	#. Install ``pocl`` system wide

		* :samp:`sudo apt install pocl-opencl-icd libpocl2 ocl-icd-opencl-dev`

	#. Remove the bad environment and recreate it without ``pocl``.

		* :samp:`conda remove -n {NAME} --all`
		* :samp:`conda create -n {NAME} -c conda-forge pyopencl scipy matplotlib`
		* :samp:`conda activate {NAME}`

	#. Copy the ICD registry files from system-wide installation to the Anaconda environment

		* :samp:`cp /etc/OpenCL/vendors/* {path_to_anaconda}/envs/{NAME}/etc/OpenCL/vendors/`

Support for NVIDIA Graphics
----------------------------

It is possible to install all the necessary packages using ``apt`` (no need to visit NVIDIA website).

	#. :samp:`sudo apt update`
	#. :samp:`sudo add-apt-repository ppa:graphics-drivers/ppa`
	#. :samp:`sudo apt install nvidia-driver-{XXX}`

		* Replace :samp:`{XXX}` with the version of your choice.  As of this writing the latest is 396.  Get a current list using :samp:`apt search nvidia-driver`.
		* As an alternative :samp:`sudo ubuntu-drivers autoinstall` is supposed to automatically select a suitable version.

	#. :samp:`sudo apt update`
	#. Copy the ICD registry files from the root environment to the Anaconda environment

		* :samp:`sudo cp /etc/OpenCL/vendors/* {path_to_anaconda}/envs/{NAME}/etc/OpenCL/vendors/`


Support for AMD Graphics
-------------------------

It is possible to install all the necessary packages using ``apt`` (no need to visit AMD website).

	#. :samp:`sudo apt update`
	#. :samp:`sudo add-apt-repository ppa:oibaf/graphics-drivers`
	#. :samp:`sudo apt install mesa-opencl-icd`
	#. :samp:`sudo apt update`
	#. Copy the ICD registry files from the root environment to the Anaconda environment

		* :samp:`sudo cp /etc/OpenCL/vendors/* {path_to_anaconda}/envs/{NAME}/etc/OpenCL/vendors/`


Display Recovery
------------------

Installing graphics drivers in Linux can sometimes cause you to lose your display.  If this happens, try to switch to console mode by pressing :samp:`Ctrl-Alt-F2` (you may have to try different function keys).  If this succeeds you can issue the following commands to rollback the graphics driver:

	#. :samp:`sudo apt install ppa-purge`
	#. Purge the drivers from the appropriate repositories

		* :samp:`ppa-purge ppa:graphics-drivers/ppa`
		* :samp:`ppa-purge ppa:oibaf/graphics-drivers`

	#. Reboot using :samp:`sudo reboot`

Of course upon doing this SeaRay GPU support may be lost.

TeX for premium plot labels
---------------------------

If you want the nicest looking plot labels you have to install a TeX distribution.

	#. :samp:`sudo apt install texlive`
	#. :samp:`sudo apt install texlive-publishers`
	#. :samp:`sudo apt install dvipng`
	#. Uncomment the line :samp:`mpl.rcParams['text.usetex'] = True` near the top of :samp:`ray_plotter.py`.

Advanced 3D Plotting
---------------------------

The SeaRay plotter supports :samp:`matplotlib` and/or :samp:`mayavi` for 3d plotting. The 3D capabilities of :samp:`matplotlib` are at present nonideal (e.g., depth is not properly rendered in all cases). If you want robust 3D plots you should install :samp:`mayavi`.

As of this writing the best way to install :samp:`mayavi` into a conda environment is with ``pip`` rather than the ``conda`` tool.  In some cases ``mayavi`` and ``matplotlib`` step on each other.  If this happens you may need separate environments for each.  The plotter is written to sense which library is available and react accordingly.

	#. Activate your environment.
	#. :samp:`pip install mayavi`

Interactive Notebooks
----------------------

	#. Activate your environment.
	#. :samp:`conda install jupyter`
	#. Create a directory :samp:`~/.jupyter/custom/` and copy :samp:`{raysroot}/extras/custom.css` to the new directory.
	#. If there are problems with Jupyter notebooks any or all of the following may be tried:

		* :samp:`conda install widgetsnbextension={n}`, where :samp:`{n}` is some older version.
		* :samp:`conda install ipywidgets`
		* :samp:`jupyter nbextension install --py --sys-prefix widgetsnbextension`
		* :samp:`jupyter nbextension enable --py --sys-prefix widgetsnbextension`
