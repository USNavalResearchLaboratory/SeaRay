Generic SeaRay Installation
===========================

We will use Anaconda to create an environment suitable for SeaRay.  Ideally this insulates the installation from platform dependencies, although SeaRay uses technologies that stress the dependency chain, and the effect of this can vary by platform.  For PyOpenCL hardware acceleration, there are generic options that should work for most CPUs.  For GPUs it may be necessary to install special drivers.

.. note::
	Windows is different.  If you are using Windows see :doc:`adv-install`

Environment and Basic Packages
------------------------------

  #. If you already have Anaconda3 installed, skip the next 3 steps.  It is possible you can use your existing environment and skip this entire section: but to be safe create a new environment as detailed below.
  #. Download Miniconda3 installer from internet
  #. Navigate to downloaded file
  #. :samp:`bash {filename}`, where :samp:`{filename}` is the file that you just downloaded
  #. Choose a name for your environment, denoted :samp:`{NAME}`
  #. :samp:`conda create -n {NAME}`
  #. :samp:`source activate {NAME}`
  #. You are now in an isolated conda environment.  The environment must be activated each time you open a new terminal window.
  #. :samp:`conda install scipy`
  #. :samp:`conda install matplotlib`

Install PyOpenCL
------------------------------

This is the hard part.  PyOpenCL requires, in essence, that a driver be present which allows the software to interact with parallel hardware.  In the case of a CPU, the driver can be generic.  Using either :samp:`AMDAPP` or :samp:`pocl` avoids the need to install vendor specific drivers.  This may not give the best possible performance, but makes for a simpler (but still sometimes painful) installation process.  If you want to take advantage of a GPU you must install more specific driver software (see :doc:`adv-install`).

Option 1: Use Mac OS
,,,,,,,,,,,,,,,,,,,,,,,,

In this case you just have to install PyOpenCL itself.  This should give you access to both your CPU and GPU.

	#. If your environment is not already activated, activate it as above.
	#. :samp:`conda install -c conda-forge pyopencl`

Option 2: Use AMDAPP
,,,,,,,,,,,,,,,,,,,,,,,,

	#. Find AMDAPP SDK on the web and download.
	#. Navigate to the downloaded file
	#. :samp:`tar -xjvf {name of downloaded file}`
	#. :samp:`sudo bash {name of unpacked file}`
	#. Accept the defaults and wait for the installer to finish. Open a new terminal window to update the path (close the old window)
	#. Test the driver by running :samp:`clinfo`.
	#. If clinfo gives an error, execute :samp:`sudo cp /opt/AMDAPP-3.0/lib/x86_64/sdk/* /usr/lib`

		* If the AMDAPP version is different you may have to modify the above command accordingly, by looking in :samp:`/opt` to see what was actually installed.

	#. :samp:`source activate {NAME}`
	#. Let the path of your miniconda directory be denoted {min}. Enter the following command to put the installable client driver (ICD) registry files where Anaconda can find them:

		* :samp:`cp /etc/OpenCL/vendors/*.icd {min}/envs/{NAME}/etc/openCL/vendors`

	#. :samp:`conda install -c conda-forge pyopencl`


Option 3: Use pocl
,,,,,,,,,,,,,,,,,,,,,,,,

Unfortunately, as of this writing, installing :samp:`pocl` is not always a turnkey operation. The following seems to work on CentOS but not Ubuntu.  The trouble may have to do with the delicacy of the relationship between :samp:`pocl` and the compiler infrastructure :samp:`LLVM` that it relies on.

  #. If your environment is not already activated, activate it as above.
  #. :samp:`conda install -c conda-forge pocl`
  #. :samp:`conda install -c conda-forge pyopencl`


TeX for premium plot labels
---------------------------

If you want the nicest looking plot labels you have to install a TeX distribution.  If you don't need this, then comment out the line :samp:`mpl.rcParams['text.usetex'] = True` near the top of :samp:`ray_plotter.py`.

	#. :samp:`sudo apt install texlive`
	#. :samp:`sudo apt install texlive-publishers`
	#. :samp:`sudo apt install dvipng`


Advanced 3D Plotting
---------------------------

The SeaRay plotter supports :samp:`matplotlib` and/or :samp:`mayavi` for 3d plotting. The 3D capabilities of :samp:`matplotlib` are at present nonideal (e.g., depth is not properly rendered in all cases). If you want robust 3D plots you should install :samp:`mayavi`.

	#. Open a new terminal. We will use a different environment because as of this writing, the 3D plotting package :samp:`mayavi` needs Python2.
	#. Also as of this writing, we have to specify a specific vtk version.
	#. :samp:`conda create -n maya python=2.7 vtk=6.3.0 scipy mayavi`
	#. :samp:`source activate maya`
	#. :samp:`conda install -c conda-forge pyopencl`

Interactive Notebooks
----------------------

	#. If your environment is not already activated, activate it as above.
	#. :samp:`conda install jupyter`
	#. :samp:`conda install nb_conda`
	#. :samp:`conda install -c conda-forge widgetsnbextension`
	#. If there are problems with Jupyter notebooks the following may be tried:

		* :samp:`jupyter nbextension install --py --sys-prefix widgetsnbextension`
		* :samp:`jupyter nbextension enable --py --sys-prefix widgetsnbextension`

	#. Create a directory :samp:`~/.jupyter/custom/` and copy :samp:`{raysroot}/docs/config-files/custom.css` to the new directory.
