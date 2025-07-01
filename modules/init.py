import pyopencl
import logging
import pathlib

class fft:
	"""contains kernel programs for Fourier and Hankel transformation"""
	def __init__(self,program: pyopencl.Program):
		self.FFT = program.FFT
		self.IFFT = program.IFFT
		self.RFFT = program.RFFT
		self.IRFFT = program.IRFFT
		self.DtSpectral = program.DtSpectral
		self.iDtSpectral = program.iDtSpectral
		self.Filter = program.Filter
		self.SoftTimeWindow = program.SoftTimeWindow
		self.FFT_axis1 = program.FFT_axis1
		self.FFT_axis2 = program.FFT_axis2
		self.IFFT_axis1 = program.IFFT_axis1
		self.IFFT_axis2 = program.IFFT_axis2
		self.RootVolumeMultiply = program.RootVolumeMultiply
		self.RootVolumeDivide = program.RootVolumeDivide
		self.RadialTransform = program.RadialTransform
		self.InverseRadialTransform = program.InverseRadialTransform

class uppe:
	"""contains kernel programs for UPPE module"""
	def __init__(self,program: pyopencl.Program):
		self.RKStage = program.RKStage
		self.RKFinish = program.RKFinish
		self.SourcePolarization = program.SourcePolarization
		self.PropagateLinear = program.PropagateLinear
		self.VectorPotentialToElectricField = program.VectorPotentialToElectricField
		self.CurrentToODERHS = program.CurrentToODERHS
		self.AddKerrChi = program.AddKerrChi

class paraxial:
	"""contains kernel programs for paraxial module"""
	def __init__(self,program: pyopencl.Program):
		self.SourcePolarization = program.SourcePolarization
		self.PropagateLinear = program.PropagateLinear
		self.CurrentToODERHS = program.CurrentToODERHS
		self.AddKerrChi = program.AddKerrChi
		#self.AddNonuniformChi = program.AddNonuniformChi
		self.AddPlasmaCurrent = program.AddPlasmaCurrent
		self.LoadModulus = program.LoadModulus
		self.LoadStepSize = program.LoadStepSize

class ionization:
	"""contains kernel programs for ionization"""
	def __init__(self,program: pyopencl.Program):
		self.ComputePlasmaDensity = program.ComputePlasmaDensity
		self.EnvelopeRate = program.EnvelopeRate
		self.ExplicitRate = program.ExplicitRate
		self.EnvelopeRateSeries = program.EnvelopeRateSeries
		self.ExplicitRateSeries = program.ExplicitRateSeries

class rotations:
	"""contains kernel programs for molecular rotations"""
	def __init__(self,program: pyopencl.Program):
		self.AddChiUPPE = program.AddChiUPPE
		self.AddChiParaxial = program.AddChiParaxial

class caustic:
	"""contains kernel programs for analyzing caustics"""
	def __init__(self,program: pyopencl.Program):
		self.divergence = program.divergence
		self.laplacian = program.laplacian

class cl_refs:
	"""All OpenCL functionality is accessed through this object."""
	def __init__(self,c: pyopencl.Context, d: pyopencl.Device, q: pyopencl.CommandQueue):
		self.ctx = c
		self.dev = d
		self.q = q
		self.loc_size_1d = None
		self.loc_size_2d = None
		self.loc_size_3d = None
		self.prog = {}
	def setup_workgroups(self,input_dict):
		"""If the input dictionary specified any work group parameters, capture them.
		N.b. each kernel invocation has to explicitly reference these, and objects have to explicitly call this function."""
		self.loc_size_1d = None if 'work group 1d' not in input_dict else input_dict['work group 1d']
		self.loc_size_2d = None if 'work group 2d' not in input_dict else input_dict['work group 2d']
		self.loc_size_3d = None if 'work group 3d' not in input_dict else input_dict['work group 3d']
	def program(self,basename):
		"""Retrieve the program with filename = basename (no extension).
		The program must have been previously added with add_program.

		:param str basename: the filename of the OpenCL program, without extension."""
		return self.prog[basename]
	def add_program(self,dir,prog_filename,plugin=''):
		"""Compile an OpenCL program and add it to the internal dictionary.

		:param str dir: directory where kernel file is found
		:param str prog_filename: filename of OpenCL program, extension optional
		:param str plugin: lines of code to add to beginning of the OpenCL program"""

		# Remove cl file extension if necessary
		if prog_filename[-3:]=='.cl':
			basename = prog_filename[:-3]
		else:
			basename = prog_filename
		# Read the OpenCL program file into memory
		path = pathlib.Path(dir,basename+'.cl')
		program_file = open(path, 'r')
		program_text = program_file.read()

		# Install the plugin
		program_text = plugin + program_text
		# Enable double precision
		program_text = '#ifdef cl_khr_fp64 \n#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n #elif defined(cl_amd_fp64) \n#pragma OPENCL EXTENSION cl_amd_fp64 : enable \n #endif \n' + program_text

		# Build OpenCL program file
		program = pyopencl.Program(self.ctx, program_text)
		program.build()

		self.prog[basename] = program

	def create_all_kernels(self):
		"""Create all universal kernels, some other kernels have plugin headers
		and have to be created per object."""
		self.fft = fft(self.program('fft'))
		self.uppe = uppe(self.program('uppe'))
		self.paraxial = paraxial(self.program('paraxial'))
		self.ionization = ionization(self.program('ionization'))
		self.rotations = rotations(self.program('rotations'))
		self.caustic = caustic(self.program('caustic'))

def list_opencl():
	print('PyOpenCL version',pyopencl.VERSION)
	for platform in pyopencl.get_platforms():
		print('Platform:',platform.name)
		for dev in platform.get_devices(pyopencl.device_type.ALL):
			print('  ',dev.name)
			print('      ',dev.version)
			print('      {:0.1f} GB , '.format(dev.global_mem_size/1e9)+str(dev.native_vector_width_float*32)+' bit vectors')

def setup_opencl(argv):
	"""Process command line arguments and setup OpenCL
	argv = command line argument list
	Returns:
	cl_refs = class containing device, context, and queue
	args = command line arguments stripped of OpenCL items"""

	device_search_string = ''
	device_id = -1
	platform_search_string = ''
	platform_id = -1
	args = []
	for arg in argv:
		CL_related = False
		if arg=='run':
			CL_related = True
		if arg.split('=')[0]=='device':
			device_search_string = arg.split('=')[1]
			if device_search_string.isdigit():
				device_id = int(device_search_string)
			CL_related = True
		if arg.split('=')[0]=='platform':
			platform_search_string = arg.split('=')[1]
			if platform_search_string.isdigit():
				platform_id = int(platform_search_string)
			CL_related = True
		if CL_related==False:
			args.append(arg)

	platform_list = pyopencl.get_platforms()
	if platform_id>=0 and platform_id<=10:
		platform = platform_list[platform_id]
	else:
		found_platform = False
		for test in platform_list:
			if platform_search_string.lower() in test.get_info(pyopencl.platform_info.NAME).lower():
				platform = test
				found_platform = True
		if not found_platform:
			print('Could not find requested platform')
			exit(1)

	device_list = platform.get_devices(pyopencl.device_type.ALL)
	if device_id>=0 and device_id<=10:
		device = device_list[device_id]
	else:
		found_device = False
		for test in device_list:
			if device_search_string.lower() in test.get_info(pyopencl.device_info.NAME).lower():
				device = test
				found_device = True
		if not found_device:
			print('Could not find requested device')
			exit(1)

	ctx = pyopencl.Context([device])
	print('Device = ',device.get_info(pyopencl.device_info.NAME))
	print('Device Memory = ',device.get_info(pyopencl.device_info.GLOBAL_MEM_SIZE)/1e9,' GB')

	# Check for double precision support. If not available exit.
	ext = device.get_info(pyopencl.device_info.EXTENSIONS)
	if not ('cl_APPLE_fp64_basic_ops' in ext or 'cl_khr_fp64' in ext or 'cl_amd_fp64' in ext):
		logging.critical("device does not appear to support double precision\n")
		exit(1)

	# Create the OpenCL command queue and kernel
	queue = pyopencl.CommandQueue(ctx)

	return cl_refs(ctx,device,queue),args
