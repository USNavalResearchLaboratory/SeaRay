Running on the Desktop
======================

.. warning::

	This document assumes you have followed the installation instructions precisely.  Once installed, running SeaRay should be the same for any UNIX-based desktop system.

.. tip::

	You can issue UNIX like commands in Windows via the PowerShell.  Otherwise, use the Anaconda prompt and replace the forward slash with the backslash in directory paths.

OpenCL Background
-----------------

SeaRay uses a Python wrapper for OpenCL, called PyOpenCL, to accelerate certain computations.  OpenCL is designed to interface with arbitrary computing devices, especially multi-core CPU and general purpose GPU (GPGPU) devices.  The user can specify a particular device to use, or allow SeaRay to select one.  If there are errors related to memory or floating point precision you can try a different device.

Running the Tests
-----------------

You should test the installation by running the unit tests::

	python -m pytest

If any of the tests fail consider checking out a different version.

Running a Ray Example
---------------------

#. Activate your virtual environment (see :doc:`generic-install`)
#. Pick some example from :samp:`{raysroot}/examples/eikonal/`.
#. For definiteness, let us use :samp:`{raysroot}/examples/eikonal/parabola.py`
#. Open a terminal window and navigate to :samp:`{raysroot}`
#. :samp:`python rays.py list`
#. The above command lists the hardware acceleration platforms and devices available on your system.  A device may be available only within a given platform.  If there is more than one platform, choose the one you would like to use, and pick out some unique part of its name, such as :samp:`{cuda}`.  Case does not matter.  Similarly, if there is more than one device, choose some unique part of its name, such as :samp:`{titan}`
#. :samp:`python rays.py run file=examples/eikonal/parabola.py platform={cuda} device={titan}`
#. This copies the :samp:`parabola.py` example file to the :samp:`{raysroot}` directory as :samp:`inputs.py` and runs the calculation.  If you do not specify a file, SeaRay will use whatever :samp:`inputs.py` is in :samp:`{raysroot}`.  It is best practice to never directly edit :samp:`inputs.py`.
#. When the run is finished, you should have several output files in :samp:`{raysroot}/out`.  The output files are simply pickled numpy arrays.
#. Let us plot the results using the SeaRay plotter.  The plotter is not interactive, but allows for a fairly high degree of control using command line options. You can get a help screen by executing :samp:`python plotter.py` with no arguments.
#. :samp:`python plotter.py out/test o3d`
#. You should see a 3D rendering of the ray orbits reflecting off an off-axis parabola, as in Fig. 1 below (assuming :samp:`matplotlib` environment).  When you are done looking close the plot window.
#. :samp:`python plotter.py out/test det=1,2/0,0/0.1`
#. This should produce an image of the radiation intensity at the focal point, as in Fig. 2 below.

.. figure:: parabola.png
	:alt: parabola
	:scale: 50 %

	Fig. 1 --- ray orbits from parabolic mirror example

.. figure:: parabola-spots.png
	:alt: parabola spots
	:scale: 50 %

	Fig. 2 --- Intensity at best focus

Running a Wave Example
----------------------

#. Activate your virtual environment (see :doc:`generic-install`)
#. Run the example case :samp:`{raysroot}/examples/paraxial/air-fil.py` following the same general procedure as above.
#. Wave runs typically take longer, although this one is fairly quick.  You should see some text based progress indicators as the wave propagation is calculated.  The time stepper is adaptive, so varying amounts of work may be done between diagnostic planes.
#. Run the Jupyter notebook :samp:`viewer.ipynb` using your favorite notebook interface (Chrome, VS Code, etc.).
#. For this example you should not need to change the source code.  Generally, if output files are saved under a different location you have to change the value of ``base_diagnostic``.  Note also that as of this writing, the normalizing length is hard coded in the notebook.
#. Run the notebook (e.g. select ``Run All`` from the ``Cell`` menu).  Advance the z-slider to observe the pulse evolution.

.. figure:: air-filament.png
	:alt: filament
	:scale: 70 %

	Fig. 3 --- Interactive viewer with results from ``paraxial/air-fil.py`` example.
