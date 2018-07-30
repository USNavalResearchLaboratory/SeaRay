RHEL 7.3 with AMD Devices
=============================

.. caution::

	We assume SeaRay has already been installed according to the documentation, with no steps omitted.

Graphics drivers can change rapidly, so internet searches may figure prominently into your installation effort.
As of this writing the driver to use for AMD is the radeon-pro driver.  This typically supports AMD workstation class GPU and AMD or Intel CPU.  However, AMD gaming class GPU or APU may not be supported.

  #. :samp:`sudo yum install opencl-headers`
  #. Find and download the radeon-pro software for RHEL 7.3 (use internet search)
  #. Navigate to downloaded archive
  #. :samp:`tar -Jxvf {downloaded_file}`
  #. Navigate into unpacked directory
  #. :samp:`sudo ./amd-gpu-install`
  #. Reboot the system
  #. The radeon-pro files should be in :samp:`/opt`
  #. Edit :samp:`~/.bashrc`
  #. Add line :samp:`export PATH=/opt/amdgpu-pro/bin:$PATH`
  #. Remember, no spaces around equals sign.
  #. Open a new terminal window
  #. :samp:`clinfo`
  #. If the above command gives a device listing which includes your CPU and your GPU the installation likely succeeded.
