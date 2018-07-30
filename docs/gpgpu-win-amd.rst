Windows with AMD Graphics
=========================

As of this writing, there is a problem with Anaconda support for :samp:`scipy.spatial`, so we will use the native python :samp:`venv` system for managing environments, and :samp:`pip` for installing packages.

Install the latest AMD driver for your graphics card or APU
AMD has a tool to automatically pick the right driver. As of this writing it is unclear whether we need to install the AMD APP SDK as well (which seems to have disappeared).

Install Python for Windows
--------------------------

	#. Search for the latest Python 3 for Windows and run the installer.  As of this writing it can be found `here <https://www.python.org/downloads/>`_.

		#. Make sure to select 32 or 64 bit version as appropriate.
		#. Select the checkbox to set the PATH variable.
		#. Press the button to install Python.
		#. Respond to the prompts until complete.

	#. Go to the search box in the task bar and type :samp:`command` and press Enter.
	#. From the resulting command prompt window type :samp:`python`.
	#. If the installation was successful you should be given the Python prompt.
	#. Exit Python by typing :samp:`exit()`.

Environment and Basic Packages
------------------------------

	#. Choose a directory :samp:`{VEPATH}` for your virtual environment.  Create the directory if it does not exist already.  If you want to put your environment in the working directory you can omit :samp:`{VEPATH}\\` in the following.
	#. Choose a name :samp:`{VENAME}` for the environment
	#. :samp:`python -m venv {VEPATH}\\{VENAME}`
	#. Activate the virtual environment.

		#. :samp:`{VEPATH}\\{VENAME}\\Scripts\\activate.bat`

	#. You are now in an isolated virtual environment.  Anytime you open a new terminal the environment must be activated.
	#. :samp:`python -m pip install --upgrade pip`
	#. :samp:`python -m pip install scipy`
	#. :samp:`python -m pip install matplotlib`

Install PyOpenCL
----------------

	#. Search web for :samp:`python binaries for windows`. You are looking for Cristoph Gohlke's precompiled binaries, and in particular, the precompiled PyOpenCL package.

		#. Locate the PyOpenCL binary that goes with the version of Python you installed.  As of this writing the correct binary is :samp:`pyopencl-2018.1.1+cl21-cp36-cp36m-win_amd64.whl` (assuming 64 bit Python for Windows was installed above, and assuming hardware and driver support OpenCL 2.1).  Let us denote your particular binary as :samp:`{pyopencl-binary}`.
		#. Click on :samp:`{pyopencl-binary}` to download it.

	#. If you closed the command prompt window, open a new one and activate the virtual environment.
	#. :samp:`python -m pip install mako cffi`
	#. Navigate to the location where you downloaded :samp:`{pyopencl-binary}`
	#. :samp:`python -m pip install {pyopencl-binary}`

Check the Plotter
-----------------

Since we have not installed LaTeX, it has to be disabled in the Plotter.  Edit :samp:`ray_plotter.py` and change the value of :samp:`mpl.rcParams['text.usetex']` from :samp:`True` to :samp:`False`, if necessary.
