Generic SeaRay Installation
===========================

We will use Anaconda to create an environment suitable for SeaRay.  Ideally this insulates the installation from platform dependencies, although SeaRay uses technologies that stress the dependency chain, and the effect of this can vary by platform.  Installing PyOpenCL (for hardware acceleration) is generally the most troublesome part.

Environment and Basic Packages
------------------------------

	#. If you already have Anaconda3 installed, skip the next 3 steps.  It is possible you can use your existing environment and skip this entire section: but to be safe create a new environment as detailed below.
	#. Download Miniconda3 installer from internet
	#. Navigate to downloaded file

		* For linux or MacOS, execute :samp:`bash {filename}`, where :samp:`{filename}` is the file that you just downloaded
		* For Windows run the graphical installer and accept the defaults.  When finished use the Anaconda prompt that should be available in the ``Start`` menu to issue the commands below.

	#. :samp:`conda update conda`
	#. Choose a name for your environment, denoted :samp:`{NAME}`
	#. :samp:`conda create -n {NAME}`
	#. :samp:`conda activate {NAME}`

		* We are adopting the use of ``conda`` (rather than ``source``) to activate the environment.  You may be prompted to modify your login files.
		* This command puts you in an isolated conda environment.  This command must be issued each time you open a new terminal window, in order to use the environment.

Install PyOpenCL
------------------------------

This is the hard part.  PyOpenCL requires, in essence, that a driver be present which allows the software to interact with parallel hardware.

MacOS and Windows Users
,,,,,,,,,,,,,,,,,,,,,,,,

For Windows, update graphics drivers to the vendor's latest version.

	#. If your environment is not already activated, activate it as above.
	#. :samp:`conda install -c conda-forge pyopencl scipy matplotlib`

For MacOS, you should gain access to both CPU and GPU.  For Windows, you should gain access to your GPU.

Linux Users
,,,,,,,,,,,,,,,,,,,,,,,,

We will use the generic :samp:`pocl` driver to support multi-core CPU parallelism.  If you want to take advantage of a GPU you must install more specific driver software (see :doc:`adv-install`).  Unfortunately, as of this writing, installing :samp:`pocl` is not always a turnkey operation. In theory the following is supposed to work.  If it fails see :doc:`adv-install`.

	#. If your environment is not already activated, activate it as above.
	#. :samp:`conda install -c conda-forge pocl pyopencl scipy matplotlib`

Getting SeaRay Components
-------------------------

To copy the SeaRay components to your local computer perform the following procedure:

	#. Open a terminal window (Anaconda prompt on Windows)
	#. Test to see if you have Git installed by executing :samp:`git --version`
	#. Install Git if necessary.

		* Anaconda --- :samp:`conda install git`
		* CentOS/RHEL/SL --- :samp:`sudo yum install git`
		* Homebrew --- :samp:`brew install git`
		* MacPorts --- :samp:`sudo port install git`
		* Ubuntu --- :samp:`sudo apt install git`

	#. Navigate to the directory where you want to install SeaRay (you don't need to make an enclosing directory).
	#. :samp:`git clone https://github.com/USNavalResearchLaboratory/searay.git`
	#. Checkout a stable version

		* :samp:`git tag --list` displays tagged commits.
		* Select a tag without a letter suffix for the highest stability.
		* :samp:`git checkout {vers}`, where :samp:`{vers}` is the selected tag.

	#. If you like you can give the SeaRay root directory another name, we will call it :samp:`{raysroot}` from now on.

Advanced Installation Preview
------------------------------

At this point you should have enough to run SeaRay simulations and view the data with the SeaRay plotter.  If you want to activate more features, see :doc:`adv-install`.  The additional features include:

	* Higher performance parallelism
	* Premium plot labels using TeX
	* Advanced 3D plots using mayavi
	* Interactive Jupyter notebooks
