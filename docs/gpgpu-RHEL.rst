CentOS 7.5 Advanced Install
=============================

Before starting, follow the steps in :doc:`generic-install`.

Support for AMD Graphics
-------------------------

.. Warning::
	Installing graphics drivers in linux can result in losing the display manager.  Recovering functionality is usually possible, but requires considerable expertise.  You may want to take steps to backup the system before attempting the driver installation.

Graphics drivers can change rapidly, so internet searches may figure prominently into your installation effort.  As of this writing the AMD driver to use is Radeon or Radeon Pro, with the typical package name ``amdgpu`` or ``amdgpu-pro``.  Also as of this writing the installation has been known to fail due to a dependency problem.

Due to the difficulties we do not attempt to give specific instructions.  You will have to try to follow guidance on the AMD website.

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
	#. :samp:`conda install jupyter`
	#. :samp:`conda install nb_conda`
	#. :samp:`conda install -c conda-forge widgetsnbextension`
	#. If there are problems with Jupyter notebooks the following may be tried:

		* :samp:`jupyter nbextension install --py --sys-prefix widgetsnbextension`
		* :samp:`jupyter nbextension enable --py --sys-prefix widgetsnbextension`

	#. Create a directory :samp:`~/.jupyter/custom/` and copy :samp:`{raysroot}/docs/config-files/custom.css` to the new directory.
