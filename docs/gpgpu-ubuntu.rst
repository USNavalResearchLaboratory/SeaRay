Ubuntu 20.04 Advanced Install
================================

Before starting, follow the steps in :doc:`generic-install`.

Install Compilers
-----------------

It is a good idea to install both GCC and LLVM compilers first.

#. :samp:`sudo apt update`
#. :samp:`sudo apt install gcc`
#. :samp:`sudo apt install llvm clang`

Support for NVIDIA Graphics
----------------------------

#. :samp:`sudo apt update`
#. :samp:`sudo apt install nvidia-opencl-icd nvidia-opencl-dev`
#. :samp:`sudo apt update`
#. Activate the conda environment (if not already active)
#. :samp:`conda install ocl-icd-system`

Support for AMD Graphics
-------------------------

#. Download the driver for your GPU from the AMD website.
#. Follow the instructions from AMD to install the driver **with OpenCL**.
	* As of this writing, the link is `here <https://amdgpu-install.readthedocs.io>`_.  You must specify a command line option ``--opencl=pal`` or ``--opencl=legacy`` depending on your GPU.  You must also use the ``amdgpu-pro-install`` variant of the install script.

#. Activate the conda environment (if not already active)
#. :samp:`conda install ocl-icd-system`
