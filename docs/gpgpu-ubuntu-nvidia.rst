Ubuntu 16.04 with NVIDIA Devices
================================

.. caution::

	We assume SeaRay has already been installed according to the documentation, with no steps omitted.

Graphics drivers can change rapidly, so internet searches may figure prominently into your installation effort.
NVIDIA GPGPU devices use CUDA software, which includes OpenCL.  This typically supports NVIDIA devices only.

	#. Find the Debian local install file for NVIDIA CUDA (use internet search)

		* OS=linux, Arch=x86_64, Dist=Ubuntu, Vers=16.04, Installer=deb(local)
		
	#. Do NOT use the runfile
	#. Navigate to downloaded file in the terminal
	#. :samp:`sudo dpkg -i {downloaded_file}`
	#. :samp:`sudo apt update`
	#. :samp:`sudo apt install cuda`
	#. :samp:`sudo apt update`
	#. Note: don't install opencl-headers, it may conflict with CUDA.  If already installed, use :samp:`sudo apt remove opencl-headers`
