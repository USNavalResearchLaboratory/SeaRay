SeaRay Installation without Anaconda
====================================

You can install SeaRay without Anaconda using the usual python tools, :samp:`venv` for managing environments, and :samp:`pip` for installing packages.  Note however that at present, the intention is to devote more time to making sure things work in Anaconda (i.e., you may be on your own if you use native python tools).

Root OS Packages
----------------

Unlike Anaconda, native python environments deal only with python packages.  You may find you have to install packages into the native operating system. If your OS uses Debian packages (e.g., Ubuntu), then this is done with  :samp:`sudo apt install {package_name}`.  If your OS uses RPM packages (e.g., CentOS), then this is done with :samp:`sudo yum install {package_name}`.

Environment and Basic Packages
------------------------------

	#. Choose a directory :samp:`{VEPATH}` for your virtual environment.  Create the directory if it does not exist already.
	#. Choose a name :samp:`{VENAME}` for the environment
	#. :samp:`python3 -m venv {VEPATH}/{VENAME}`
	#. :samp:`source {VEPATH}/{VENAME}/bin/activate`
	#. You are now in an isolated virtual environment.  Anytime you open a new terminal the environment must be activated.  You may want to define an environment variable to reduce the amount of typing, e.g., :samp:`export PY={VEPATH}/{VENAME}/bin/activate`.  Then you just type :samp:`source $PY` to activate the environment.
	#. Note that in a virtual environment installing packages with :samp:`sudo` is unnecessary since the user owns everything.
	#. :samp:`pip3 install scipy`
	#. :samp:`pip3 install matplotlib`
	#. :samp:`pip3 install jupyter`
	#. :samp:`pip3 install widgetsnbextension`
	#. :samp:`jupyter nbextension install --py widgetsnbextension --sys-prefix`
	#. :samp:`jupyter nbextension enable widgetsnbextension --py --sys-prefix`

Install PyOpenCL
----------------

	#. Open a terminal and activate the virtual environment
	#. :samp:`pip3 install mako`
	#. Depending on your OpenCL platform (e.g., :samp:`pocl`, :samp:`AMDAPP`, :samp:`CUDA`) you may have to take additional steps here.

		* For CUDA, :samp:`export LIBRARY_PATH=/usr/local/cuda/lib64`
		* For pocl, :samp:`pip3 install pocl`

	#. :samp:`pip3 install pyopencl`
