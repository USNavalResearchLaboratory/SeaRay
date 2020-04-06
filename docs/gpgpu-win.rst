Windows Advanced Install
=========================

Before starting, follow the steps in :doc:`generic-install`.

Support for CPU via Intel OpenCL
--------------------------------

Under construction.

Support for Proprietary Graphics
--------------------------------

If SeaRay fails to find your graphics card:

	#. Update to the latest drivers,
	#. If using NVIDIA try installing CUDA developer tools
	#. Activate the conda environment (if not already active)
	#. :samp:`conda install -c conda-forge ocl-icd-system`
	#. See also :doc:`troubleshooting-ocl`.

TeX for premium plot labels
---------------------------

If you want the nicest looking plot labels you have to install a TeX distribution. One distribution for Windows is ``MiKTeX``.

	#. Search for ``miktex`` and run the basic installer.
	#. You may be prompted to perform further installations as ``MiKTeX`` gets used.
	#. Uncomment the line :samp:`mpl.rcParams['text.usetex'] = True` near the top of :samp:`ray_plotter.py`.

Advanced 3D Plotting
---------------------------

The SeaRay plotter supports :samp:`matplotlib` and/or :samp:`mayavi` for 3d plotting. The 3D capabilities of :samp:`matplotlib` are at present nonideal (e.g., depth is not properly rendered in all cases). If you want robust 3D plots you should install :samp:`mayavi`.

 In some cases ``mayavi`` and ``matplotlib`` step on each other.  If this happens you may need separate environments for each.  The plotter is written to sense which library is available and react accordingly.

	#. Install Visual Studio Community Edition
	#. Open the Anaconda prompt (or PowerShell if configured).
	#. Activate your ``conda`` environment.
	#. :samp:`conda install -c conda-forge mayavi`

Interactive Notebooks
----------------------

	#. Activate your ``conda`` environment.
	#. :samp:`conda install jupyter`
	#. :samp:`jupyter notebook --generate-config`
	#. Create a directory :samp:`~/.jupyter/custom/` and copy :samp:`{raysroot}/extras/custom.css` to the new directory.
	#. If there are problems with Jupyter notebooks any or all of the following may be tried:

		* :samp:`conda install widgetsnbextension={n}`, where :samp:`{n}` is some older version.
		* :samp:`conda install ipywidgets`
		* :samp:`jupyter nbextension install --py --sys-prefix widgetsnbextension`
		* :samp:`jupyter nbextension enable --py --sys-prefix widgetsnbextension`
