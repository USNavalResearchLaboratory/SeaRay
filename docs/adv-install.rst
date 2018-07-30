Advanced Installation
======================

General Notes
-------------

The generic installation typically does not support GPGPU devices (Mac OS is the exception).
SeaRay uses OpenCL to offload computations to GPGPU or other devices.  In principle, OpenCL can be used for any modern GPGPU from either AMD or NVIDIA, as well as other types of devices such as MIC or FPGA.  In practice one needs to find a driver that plays with OpenCL, and in the case of SeaRay, PyOpenCL in particular.

Specific OS and Device Combinations
-----------------------------------

.. toctree::
	:maxdepth: 2

	gpgpu-win-amd
	gpgpu-ubuntu-amd
	gpgpu-RHEL-amd
	gpgpu-ubuntu-nvidia
	gpgpu-RHEL-nvidia
