Running on the Desktop
======================

.. caution::

	This document assumes you have followed the installation instructions precisely.  Once installed, running SeaRay should be the same for any UNIX-based desktop system.

.. tip::

	You can issue UNIX like commands in Windows via the PowerShell.  Otherwise, use the Anaconda prompt and replace the forward slash with the backslash in directory paths.

OpenCL Background
-----------------

In order to run SeaRay, it is useful to understand a little about OpenCL. SeaRay uses a Python wrapper for OpenCL, called PyOpenCL, to accelerate certain computations.  OpenCL is designed to interface with arbitrary computing devices, especially multi-core CPU and general purpose GPU (GPGPU) devices.  For this reason the user is invited to specify a device to use for these accelerations.  This specification is optional, but if SeaRay tries to choose a device on its own, and that device turns out not to be appropriate, the run may fail.  The two failure modes most likely are (i) not enough memory on the device, and (ii) not enough floating point precision on the device.

Running an Example
------------------

	#. Activate your virtual environment (see :doc:`generic-install`)
	#. Pick some example from :samp:`{raysroot}/examples`.
	#. For definiteness, let us use :samp:`{raysroot}/examples/eikonal/parabola.py`
	#. Open a terminal window and navigate to :samp:`{raysroot}`
	#. :samp:`python rays.py list`
	#. The above command lists the hardware acceleration platforms and devices available on your system.  A device may be available only within a given platform.  If there is more than one platform, choose the one you would like to use, and pick out some unique part of its name, such as :samp:`{cuda}`.  Case does not matter.  Similarly, if there is more than one device, choose some unique part of its name, such as :samp:`{titan}`
	#. :samp:`python rays.py run file=examples/parabola.py platform={cuda} device={titan}`
	#. This copies the :samp:`parabola.py` example file to the :samp:`{raysroot}` directory as :samp:`inputs.py` and runs the calculation.  If you do not specify a file, SeaRay will use whatever :samp:`inputs.py` is in :samp:`{raysroot}`.  It is best practice to never directly edit :samp:`inputs.py`.
	#. When the run is finished, you should have several output files in :samp:`{raysroot}/out`.  The output files are simply pickled numpy arrays.
	#. Let us plot the results using the SeaRay plotter.  The plotter is not interactive, but allows for a fairly high degree of control using command line options. You can get a help screen by executing :samp:`python ray_plotter.py` with no arguments.
	#. :samp:`python ray_plotter.py out/test o3d`
	#. You should see a 3D rendering of the ray orbits reflecting off an off-axis parabola, as in Fig. 1 below (assuming :samp:`matplotlib` environment).  When you are done looking close the plot window.
	#. :samp:`python ray_plotter.py out/test det`
	#. This should produce an image of the radiation intensity 1 mm upstream of the focal point and exactly at the focal point, as in Fig. 2 below.

.. figure:: parabola.png
	:scale: 50 %

	Fig. 1 --- ray orbits from parabolic mirror example

.. figure:: parabola-spots.png
	:scale: 50 %

	Fig. 2 --- Intensity in eikonal plane and at best focus
