CentOS 8 Advanced Install
=============================

Before starting, follow the steps in :doc:`generic-install`.

.. warning::

	Installing video drivers on Linux can sometimes cause you to lose your display.  Recovery is often difficult.  If you cannot afford for this to happen you should take steps to backup your system.

Support for NVIDIA Graphics
----------------------------

#. Prepare for EPEL:
	* For CentOS, type ``sudo dnf config-manager --set-enabled PowerTools``
	* For RHEL, type ``ARCH=$( /bin/arch )`` followed by ``sudo subscription-manager repos --enable "codeready-builder-for-rhel-8-${ARCH}-rpms"``

#. Go to `EPEL <https://fedoraproject.org/wiki/EPEL>`_ and install.  As of this writing there is a link, ``epel-release-latest-8``, that runs a graphical installer.
#. Go to `RPM Fusion <https://rpmfusion.org/Configuration>`_ and install the ``nonfree`` repository for RHEL 8 or compatible (there is no charge, ``nonfree`` refers to license restrictions).  The link runs a graphical installer.
#. Type ``sudo dnf install akmod-nvidia``
	* This automatic kernel module recompiles automatically when a new Linux kernel is installed (e.g. during a system update).  After restarting you must allow extra time for the kernel module to compile.  There could be a long delay before the login screen appears.

#. Restart the system, allow extra time for this restart.
#. ``sudo dnf install xorg-x11-drv-nvidia-cuda``
#. ``sudo dnf install ocl-icd-devel``
#. Activate the conda environment (if not already active)
#. :samp:`conda install -c conda-forge ocl-icd-system`

Support for AMD Graphics
-------------------------

#. Download the driver for your GPU from the AMD website.
#. Follow the instructions from AMD to install the driver **with OpenCL**.
	* As of this writing, the link is `here <https://amdgpu-install.readthedocs.io>`_.  You must specify a command line option ``--opencl=pal`` or ``--opencl=legacy`` depending on your GPU.  You must also use the ``amdgpu-pro-install`` variant of the install script.

#. Activate the conda environment (if not already active)
#. :samp:`conda install -c conda-forge ocl-icd-system`


TeX for premium plot labels
---------------------------

If you want the nicest looking plot labels you have to install a TeX distribution.

#. :samp:`sudo dnf install texlive`
#. :samp:`sudo dnf install dvipng`
#. You may need ``anyfontsize.sty``
	* Search for ``anyfontsize.sty`` on the internet and download it
	* Create the directory ``texmf/tex/latex/local`` in your home directory
	* Copy ``anyfontsize.sty`` into the new directory

#. Uncomment the line :samp:`mpl.rcParams['text.usetex'] = True` near the top of :samp:`ray_plotter.py`.

Advanced 3D Plotting
---------------------------

The SeaRay plotter supports :samp:`matplotlib` and/or :samp:`mayavi` for 3d plotting. The 3D capabilities of :samp:`matplotlib` are at present nonideal (e.g., depth is not properly rendered in all cases). If you want robust 3D plots you should install :samp:`mayavi`.

In some cases ``mayavi`` and ``matplotlib`` step on each other.  If this happens you may need separate environments for each.  The plotter is written to sense which library is available and react accordingly.

#. Activate your environment.
#. :samp:`conda install -c conda-forge mayavi`

Interactive Notebooks
----------------------

#. If your environment is not already activated, activate it as above.
#. :samp:`conda install jupyter ipympl`
#. Create a directory :samp:`~/.jupyter/custom/` and copy :samp:`{raysroot}/extras/custom.css` to the new directory.
#. If there are problems with Jupyter notebooks any or all of the following may be tried:
	* :samp:`conda install widgetsnbextension={n}`, where :samp:`{n}` is some older version.
	* :samp:`conda install ipywidgets`
	* :samp:`jupyter nbextension install --py --sys-prefix widgetsnbextension`
	* :samp:`jupyter nbextension enable --py --sys-prefix widgetsnbextension`
