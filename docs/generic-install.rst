Generic SeaRay Installation
===========================

Prepare Anaconda
---------------------

If you are already using anaconda3 and are confident in your environment you can skip this section.

#. Download Miniconda3 installer from internet
#. Navigate to downloaded file
	* For linux or MacOS, execute :samp:`bash {filename}`, where :samp:`{filename}` is the file that you just downloaded
	* For Windows run the graphical installer and accept the defaults.  When finished use the Anaconda prompt that should be available in the ``Start`` menu to issue the commands below.
#. :samp:`conda update conda`
#. :samp:`conda init`
	* If you want to use the Windows PowerShell run :samp:`conda init powershell`

Environment and Basic Packages
------------------------------

#. Choose a name for your environment, denoted :samp:`{NAME}`
#. :samp:`conda create -n {NAME} -c conda-forge pocl pyopencl scipy matplotlib pillow pytest`
	* The ``pocl`` package provides generic OpenCL support.  Depending on the system configuration it may be possible to omit this.
#. :samp:`conda activate {NAME}`
	* This command puts you in an isolated conda environment.  This command must be issued each time you open a new terminal window, in order to use the environment.

If the ``pyopencl`` package (for hardware acceleration) seems to be causing problems it may help to read :doc:`troubleshooting-ocl`.

Getting SeaRay Components
-------------------------

To copy the SeaRay components to your local computer perform the following procedure:

#. Open a terminal window (Anaconda prompt on Windows)
#. Test to see if you have Git installed by executing :samp:`git --version`
#. Install Git if necessary.
	* Anaconda --- :samp:`conda install git`
	* CentOS/RHEL --- :samp:`sudo yum install git`
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
