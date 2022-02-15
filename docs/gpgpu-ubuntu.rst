Ubuntu 20.04 Advanced Install
================================

Before starting, follow the steps in :doc:`generic-install`.

Install Compilers
-----------------

It is a good idea to install both GCC and LLVM compilers first.

#. :samp:`sudo apt update`
#. :samp:`sudo apt install gcc`
#. :samp:`sudo apt install llvm clang`

System-wide pocl (optional)
---------------------------

You can install ``pocl`` using ``apt`` for all users.  If you wish to do this proceed as follows.

#. Install ``pocl`` system wide
	* :samp:`sudo apt install pocl-opencl-icd libpocl2 ocl-icd-opencl-dev`

#. If you created a generic environment already, recreate it without ``pocl`` as follows
	* :samp:`conda remove -n {NAME} --all`
	* :samp:`conda create -n {NAME} -c conda-forge pyopencl scipy matplotlib`
	* :samp:`conda activate {NAME}`
	* :samp:`conda install -c conda-forge ocl-icd-system`

Support for NVIDIA Graphics
----------------------------

#. :samp:`sudo apt update`
#. :samp:`sudo apt install nvidia-opencl-icd nvidia-opencl-dev`
#. :samp:`sudo apt update`
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

#. :samp:`sudo apt install texlive`
#. :samp:`sudo apt install texlive-publishers`
#. :samp:`sudo apt install dvipng`
#. Uncomment the line :samp:`mpl.rcParams['text.usetex'] = True` near the top of :samp:`ray_plotter.py`.

Advanced 3D Plotting
---------------------------

The SeaRay plotter supports :samp:`matplotlib` and/or :samp:`mayavi` for 3d plotting. The 3D capabilities of :samp:`matplotlib` are at present nonideal (e.g., depth is not properly rendered in all cases). If you want robust 3D plots you should install :samp:`mayavi`.

In some cases ``mayavi`` and ``matplotlib`` step on each other.  If this happens you may need separate environments for each.  The plotter is written to sense which library is available and react accordingly.

#. Activate your environment.
#. :samp:`conda install -c conda-forge mayavi`

Interactive Notebooks
----------------------

#. Activate your environment.
#. :samp:`conda install jupyter ipympl`
#. Create a directory :samp:`~/.jupyter/custom/` and copy :samp:`{raysroot}/extras/custom.css` to the new directory.
#. If there are problems with Jupyter notebooks any or all of the following may be tried:
	* :samp:`conda install widgetsnbextension={n}`, where :samp:`{n}` is some older version.
	* :samp:`conda install ipywidgets`
	* :samp:`jupyter nbextension install --py --sys-prefix widgetsnbextension`
	* :samp:`jupyter nbextension enable --py --sys-prefix widgetsnbextension`
