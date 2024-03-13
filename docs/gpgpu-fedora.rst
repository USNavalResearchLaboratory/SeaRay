Fedora Advanced Install
=============================

Before starting, follow the steps in :doc:`generic-install`.

Support for NVIDIA Graphics
----------------------------

#. Prepare for EPEL:
	* ``sudo dnf config-manager --set-enabled PowerTools``

#. Go to `EPEL <https://fedoraproject.org/wiki/EPEL>`_ and install.  As of this writing there is a link, ``epel-release-latest-8``, that runs a graphical installer.
#. Go to `RPM Fusion <https://rpmfusion.org/Configuration>`_ and install the ``nonfree`` repository for RHEL 8 or compatible (there is no charge, ``nonfree`` refers to license restrictions).  The link runs a graphical installer.
#. Type ``sudo dnf install akmod-nvidia``
	* This automatic kernel module recompiles automatically when a new Linux kernel is installed (e.g. during a system update).  After restarting you must allow extra time for the kernel module to compile.  There could be a long delay before the login screen appears.

#. Restart the system, allow extra time for this restart.
#. ``sudo dnf install xorg-x11-drv-nvidia-cuda``
#. ``sudo dnf install ocl-icd-devel``
#. Activate the conda environment (if not already active)
#. :samp:`conda install ocl-icd-system`

Support for AMD Graphics
-------------------------

#. Download the driver for your GPU from the AMD website.
#. Follow the instructions from AMD to install the driver **with OpenCL**.
	* As of this writing, the link is `here <https://amdgpu-install.readthedocs.io>`_.  You must specify a command line option ``--opencl=pal`` or ``--opencl=legacy`` depending on your GPU.  You must also use the ``amdgpu-pro-install`` variant of the install script.

#. Activate the conda environment (if not already active)
#. :samp:`conda install ocl-icd-system`
