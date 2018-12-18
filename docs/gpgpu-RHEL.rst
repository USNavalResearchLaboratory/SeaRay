CentOS 7.5 Advanced Install
=============================

Before starting, follow the steps in :doc:`generic-install`.

.. Warning::

	Installing video drivers on Linux can sometimes cause you to lose your display.  Recovery is often difficult.  If you cannot afford for this to happen you should take steps to backup your system.

Support for CPU via pocl
-------------------------

As of this writing there seems to be no suitable repository containing an RPM for pocl.  The only way to gain pocl functionality seems to be with the ``conda`` tool.

Support for NVIDIA Graphics
----------------------------

It is possible to install all the necessary packages using ``yum`` (no need to visit NVIDIA website).

	#. :samp:`sudo yum update`
	#. :samp:`sudo yum install ocl-icd clinfo`
	#. Perform internet search to find instructions for installing ``ELRepo``, and carry out.
	#. :samp:`sudo yum install kmod-nvidia`
	#. Reboot the system
	#. Copy the ICD registry files from the root environment to the Anaconda environment

		* :samp:`sudo cp /etc/OpenCL/vendors/* {path_to_anaconda}/envs/{NAME}/etc/OpenCL/vendors/`

	#. :samp:`clinfo` should give a listing of platforms and devices, if the installation succeeded.

Support for AMD Graphics
-------------------------

As of this writing not recommended.  You can try to use the ``amdgpu`` installer from the AMD website, but this is vulnerable to breakage after a kernel update.  Keep an eye on ``ELRepo`` and ``RPMFusion`` repositories for a more suitable alternative.

TeX for premium plot labels
---------------------------

If you want the nicest looking plot labels you have to install a TeX distribution.

	#. :samp:`sudo yum install texlive`
	#. :samp:`sudo yum install dvipng`
	#. You may need ``anyfontsize.sty``

		* Search for ``anyfontsize.sty`` on the internet and download it
		* Create the directory ``texmf/tex/latex/local`` in your home directory
		* Copy ``anyfontsize.sty`` into the new directory

	#. Uncomment the line :samp:`mpl.rcParams['text.usetex'] = True` near the top of :samp:`ray_plotter.py`.

Advanced 3D Plotting
---------------------------

The SeaRay plotter supports :samp:`matplotlib` and/or :samp:`mayavi` for 3d plotting. The 3D capabilities of :samp:`matplotlib` are at present nonideal (e.g., depth is not properly rendered in all cases). If you want robust 3D plots you should install :samp:`mayavi`.

As of this writing the best way to install :samp:`mayavi` into a conda environment is with ``pip`` rather than the ``conda`` tool.  In some cases ``mayavi`` and ``matplotlib`` step on each other.  If this happens you may need separate environments for each.  The plotter is written to sense which library is available and react accordingly.

	#. Activate your environment.
	#. :samp:`pip install mayavi`

Interactive Notebooks
----------------------

	#. If your environment is not already activated, activate it as above.
	#. :samp:`conda install jupyter nb_conda`
	#. Create a directory :samp:`~/.jupyter/custom/` and copy :samp:`{raysroot}/extras/custom.css` to the new directory.
	#. If there are problems with Jupyter notebooks any or all of the following may be tried:

		* :samp:`conda install widgetsnbextension={n}`, where :samp:`{n}` is some older version.
		* :samp:`conda install ipywidgets`
		* :samp:`jupyter nbextension install --py --sys-prefix widgetsnbextension`
		* :samp:`jupyter nbextension enable --py --sys-prefix widgetsnbextension`
