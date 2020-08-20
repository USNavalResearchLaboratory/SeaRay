from time import time
import os
import glob
import sys
import shutil
import numpy as np
import pyopencl

import init
import dispersion
import ray_kernel

if len(sys.argv)==1:
	print('==========BEGIN HELP FOR SEARAY==========')
	print('Version: 0.8.3')
	print('Usage: rays.py cmd [file=<name>] [device=<dev_str>] [platform=<plat_str>] [iterations=<n>]')
	print('Arguments in square brackets are optional.')
	print('cmd = list --- displays all platforms and devices')
	print('cmd = run --- executes calculation')
	print('<name> = path of input file, if not given inputs.py must be in working directory.')
	print('<dev_str> = something in desired device name, or numerical id')
	print('<plat_str> = something in desired OpenCL platform, or numerical id')
	print('large numbers (>10) are assumed to be part of a name rather than an id')
	print('defaults are the last platform/device in the list')
	print('<n> = iterations to use in optimization (default=1)')
	print('==========END HELP FOR SEARAY==========')
	exit(1)

# Error check command line arguments
valid_arg_keys = ['run','list','file','device','platform','iterations']
for arg in sys.argv[1:]:
	if arg.split('=')[0] not in valid_arg_keys:
		raise SyntaxError('The argument <'+arg+'> was not understood.')

# Set up OpenCL
print('--------------------')
print('Accelerator Hardware')
print('--------------------')
cl,args = init.setup_opencl(sys.argv)
cl.add_program('fft')
cl.add_program('uppe')
cl.add_program('paraxial')
cl.add_program('caustic')
cl.add_program('ionization')

# Get input file
for arg in args:
	if arg.split('=')[0]=='file':
		shutil.copyfile(arg.split('=')[1],'inputs.py')
import inputs

# Add the outer list if necessary
if type(inputs.sim)==dict:
	inputs.sim = [inputs.sim]
	inputs.wave = [inputs.wave]
	inputs.ray = [inputs.ray]
	inputs.optics = [inputs.optics]
	inputs.diagnostics = [inputs.diagnostics]

# Pre-simulation cleaning
if inputs.diagnostics[0]['clean old files']:
	file_list = glob.glob(inputs.diagnostics[0]['base filename']+'*.npy')
	for f in file_list:
		os.remove(f)

# Run one case for each set of dictionaries
for irun in range(len(inputs.sim)):

	print('-------------')
	print('Begin Run',irun+1)
	print('-------------')
	print(inputs.sim[irun]['message'])
	time1 = time()

	output_path = os.path.dirname(inputs.diagnostics[irun]['base filename'])
	if not os.path.exists(output_path):
		print('INFO: creating output directory',output_path)
		os.mkdir(output_path)

	print('\nSetting up optics...\n')
	for opt_dict in inputs.optics[irun]:
		print('Initializing',opt_dict['object'].name)
		opt_dict['object'].Initialize(opt_dict)
		opt_dict['object'].InitializeCL(cl,opt_dict)

	print('\nSetting up rays and orbits...')
	if len(inputs.ray[irun])>1:
		raise ValueError('Only one ray box allowed at present.')
	xp,eikonal,vg = ray_kernel.init(inputs.wave[irun],inputs.ray[irun][0])
	orbit_dict = ray_kernel.setup_orbits(xp,eikonal,inputs.ray[irun][0],inputs.diagnostics[irun],inputs.optics[irun])
	micro_action_0 = ray_kernel.GetMicroAction(xp,eikonal,vg)
	xp0 = np.copy(xp)
	eikonal0 = np.copy(eikonal)

	print('\nStart propagation...\n')
	print('Initial micro-action = {:.3g}\n'.format(micro_action_0))

	for opt_dict in inputs.optics[irun]:
		opt_dict['object'].Propagate(xp,eikonal,vg,orb=orbit_dict)

	print('\nStart diagnostic reports...\n')

	if len(inputs.sim)>1:
		basename = inputs.diagnostics[irun]['base filename'] + '_' + str(irun)
	else:
		basename = inputs.diagnostics[irun]['base filename']

	if not inputs.diagnostics[irun]['suppress details']:
		ray_kernel.SyncSatellites(xp,vg)
		np.save(basename+'_xp0',xp0)
		np.save(basename+'_xp',xp)
		np.save(basename+'_eikonal',eikonal)
		np.save(basename+'_orbits',orbit_dict['data'])

	for opt_dict in inputs.optics[irun]:
		opt_dict['object'].Report(basename,inputs.sim[irun]['mks_length'])

	time2 = time()
	print('Completed in {:.1f} seconds'.format(time2-time1))
