from time import time
import os
import glob
import sys
import numpy as np
import pyopencl

import inputs
import init
import dispersion
import ray_kernel

if len(sys.argv)==1:
	print('==========BEGIN HELP FOR SEARAY==========')
	print('Usage: rays.py cmd [device=string] [platform=string] [iterations=n]')
	print('Arguments in square brackets are optional.')
	print('cmd = list --- displays all platforms and devices')
	print('cmd = run --- executes calculation')
	print('  (inputs.py must be in working directory)')
	print('string = something in desired device/platform name, or numerical id')
	print('large numbers (>10) are assumed to be part of a name rather than an id')
	print('defaults are the last platform/device in the list')
	print('iterations = iterations to use in optimization (default=1)')
	print('==========END HELP FOR SEARAY==========')
	exit(1)

# Pre-simulation cleaning
if inputs.diagnostics[0]['clean old files']:
	file_list = glob.glob(inputs.diagnostics[0]['base filename']+'*.npy')
	for f in file_list:
		os.remove(f)

# Set up OpenCL
print('--------------------')
print('Accelerator Hardware')
print('--------------------')
cl,args = init.setup_opencl(sys.argv)

# Run one case for each set of dictionaries
# Typical use would be to change the frequency each time
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

	# Initialize optical elements
	for opt_dict in inputs.optics[irun]:
		print('Initializing',opt_dict['object'].name)
		opt_dict['object'].Initialize(opt_dict)
		if 'integrator' in opt_dict:
			opt_dict['object'].InitializeCL(cl,opt_dict)

	# Create rays and orbits and save initial state
	xp,eikonal,vg = ray_kernel.init(inputs.wave[irun],inputs.ray[irun])
	orbit_dict = ray_kernel.setup_orbits(xp,eikonal,inputs.ray[irun],inputs.diagnostics[irun],inputs.optics[irun])
	micro_action_0 = ray_kernel.GetMicroAction(xp,eikonal,vg)
	xp0 = np.copy(xp)
	eikonal0 = np.copy(eikonal)

	print('\nStart ray propagation...\n')
	print('Initial micro-action = {:.3g}\n'.format(micro_action_0))

	for opt_dict in inputs.optics[irun]:
		opt_dict['object'].Propagate(xp,eikonal,vg,orb=orbit_dict)

	if len(inputs.sim)>1:
		basename = inputs.diagnostics[irun]['base filename'] + '_' + str(i)
	else:
		basename = inputs.diagnostics[irun]['base filename']

	if not inputs.diagnostics[irun]['suppress details']:
		np.save(basename+'_xp0',xp0)
		np.save(basename+'_xp',xp)
		np.save(basename+'_eikonal',eikonal)
		np.save(basename+'_orbits',orbit_dict['data'])

	for opt_dict in inputs.optics[irun]:
		opt_dict['object'].Report(basename,inputs.sim[irun]['mks_length'])

	time2 = time()
	print('Completed in {:.1f} seconds'.format(time2-time1))
