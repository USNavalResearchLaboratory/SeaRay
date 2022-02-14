import pyopencl

class cl_refs:
	"""All OpenCL functionality is accessed through this object."""
	def __init__(self,c,d,q):
		self.ctx = c
		self.dev = d
		self.q = q
		self.prog = {}
	def program(self,basename):
		"""Retrieve the program with filename = basename (no extension).
		The program must have been previously added with add_program.

		:param str basename: the filename of the OpenCL program, without extension."""
		return self.prog[basename]
	def add_program(self,prog_filename,plugin=''):
		"""Compile an OpenCL program and add it to the internal dictionary.

		:param str prog_filename: filename of OpenCL program, extension optional
		:param str plugin: lines of code to add to beginning of the OpenCL program"""

		# Remove cl file extension if necessary
		if prog_filename[-3:]=='.cl':
			basename = prog_filename[:-3]
		else:
			basename = prog_filename
		# Read the OpenCL program file into memory
		program_file = open(basename+'.cl', 'r')
		program_text = program_file.read()

		# Install the plugin
		program_text = plugin + program_text
		# Enable double precision
		program_text = '#ifdef cl_khr_fp64 \n#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n #elif defined(cl_amd_fp64) \n#pragma OPENCL EXTENSION cl_amd_fp64 : enable \n #endif \n' + program_text

		# Build OpenCL program file
		program = pyopencl.Program(self.ctx, program_text)
		program.build()

		self.prog[basename] = program

def setup_opencl(argv):
	"""Process command line arguments and setup OpenCL
	argv = command line argument list
	Returns:
	cl_refs = class containing device, context, and queue
	args = command line arguments stripped of OpenCL items"""

	if argv[1]=='list':
		print('PyOpenCL version',pyopencl.VERSION)
		for platform in pyopencl.get_platforms():
			print('Platform:',platform.name)
			for dev in platform.get_devices(pyopencl.device_type.ALL):
				print('  ',dev.name)
				print('      ',dev.version)
				print('      {:0.1f} GB , '.format(dev.global_mem_size/1e9)+str(dev.native_vector_width_float*32)+' bit vectors')
		exit(0)

	device_search_string = ''
	device_id = -1
	platform_search_string = ''
	platform_id = -1
	optimization_iterations = 1
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
	    print("\nFatal error: Device does not appear to support double precision\n")
	    exit(1)

	# Create the OpenCL command queue and kernel
	queue = pyopencl.CommandQueue(ctx)

	return cl_refs(ctx,device,queue),args
