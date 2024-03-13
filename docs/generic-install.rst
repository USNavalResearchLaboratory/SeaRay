Generic SeaRay Installation
===========================

Install Miniforge3
---------------------

The ``pyopencl`` developers recommend using Miniforge as opposed to Anaconda or Miniconda.

#. Uninstall miniconda/anaconda, or else be prepared to add path specifiers from time to time
#. Internet search for ``miniforge``
#. Follow installation guide for your platform
#. :samp:`conda update conda`
#. :samp:`conda init`
	* If you want to use PowerShell run :samp:`conda init powershell`


Environment and Basic Packages
------------------------------

#. Choose a name for your environment, denoted :samp:`{NAME}`
#. :samp:`conda create -n {NAME} pyopencl scipy matplotlib jupyter ipympl pillow pytest python=3.10`
	* As of this writing there is an issue with OpenSSL, it may help to force miniforge to use a specific version, e.g., add ``openssl=3.2.0`` to all install commands.
#. :samp:`conda activate {NAME}`
	* This command puts you in an isolated conda environment.  This command must be issued each time you open a new terminal window, in order to use the environment.

Getting SeaRay Components
-------------------------

To copy the SeaRay components to your local computer perform the following procedure:

#. Open a terminal window (Anaconda prompt on Windows)
#. Test to see if you have Git installed by executing :samp:`git --version`
#. Install Git if necessary.
	* Anaconda --- :samp:`conda install git`
	* Fedora/RHEL --- :samp:`sudo dnf install git`
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

Install Drivers
---------------

* Windows + AMD --- install specific drivers for the video card
* Windows + NVIDIA --- install CUDA developer tools
* Windows + Intel CPU --- install Intel CPU Runtime for OpenCL
* Linux + GPU --- a lot of variation, search internet
* Mac/Linux + CPU --- activate environment, then :samp:`conda install pocl`
* Mac/Linux in general --- try ``conda install ocl-icd-system`` to add system-wide drivers to conda environment.

.. sidebar::
	The ``pocl`` package (portable OpenCL) is a generic way to get OpenCL support for a wide range of devices.

Optional Components
---------------------------

#. If you want the nicest looking plot labels you may want to install a TeX distribution.
	* Search internet to find instructions for your operating system.
	* Uncomment the line :samp:`mpl.rcParams['text.usetex'] = True` near the top of :samp:`plotter.py`.
#. If you want the best 3D plots you may want to install ``mayavi``
	* Activate your environment.
	* :samp:`conda install mayavi`
	* The plotter automatically senses its presence
