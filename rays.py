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
	print('==========ENDHELP FOR SEARAY==========')
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
for i in range(len(inputs.sim)):

	print('-------------')
	print('Begin Run',i+1)
	print('-------------')
	print(inputs.sim[i]['message'])
	time1 = time()

	# Create rays and orbits and save initial state
	xp,eikonal = ray_kernel.init(inputs.wave[i],inputs.ray[i])
	xp0 = np.copy(xp)
	eikonal0 = np.copy(eikonal)
	num_bundles = xp.shape[0]

	# Initialize objects
	for opt_dict in inputs.optics[i]:
		print('Initializing',opt_dict['object'].name)
		opt_dict['object'].Initialize(opt_dict)
		if 'integrator' in opt_dict:
			opt_dict['object'].InitializeCL(cl,opt_dict)

	# Setup orbits
	N_bund = inputs.ray[i]['number']
	N_orb = inputs.diagnostics[i]['orbit rays']
	num_orbits = N_orb[0]*N_orb[1]*N_orb[2]
	if N_bund[0]<N_orb[0] or N_bund[1]<N_orb[1] or N_bund[2]<N_orb[2]:
		print('ERROR: more orbits than rays.')
		exit(1)
	orbit_points = 1 # always have initial point
	for opt_dict in inputs.optics[i]:
		orbit_points += opt_dict['object'].OrbitPoints()
	orbits = np.zeros((orbit_points,num_orbits,12))
	list1 = np.outer(np.array(list(range(int(N_bund[0]/N_orb[0]/2),N_bund[0],int(N_bund[0]/N_orb[0])))).astype(np.int),np.ones(N_orb[1]).astype(np.int))
	list2 = np.outer(np.ones(N_orb[0]).astype(np.int),np.array(list(range(int(N_bund[1]/N_orb[1]/2),N_bund[1],int(N_bund[1]/N_orb[1])))).astype(np.int))
	offsets = (list1*N_bund[1]*N_bund[2] + list2*N_bund[2]).flatten()
	list3 = np.outer(np.ones(offsets.shape).astype(np.int),np.array(list(range(int(N_bund[2]/N_orb[2]/2),N_bund[2],int(N_bund[2]/N_orb[2])))).astype(np.int))
	bundle_list = offsets + list3.flatten()
	#bundle_list = np.random.choice(range(num_bundles),num_orbits).tolist()
	xp_selector = (bundle_list,[0],)
	eik_selector = (bundle_list,)
	orbits[0,:,:8] = xp[xp_selector]
	orbits[0,:,8:] = eikonal[eik_selector]
	orbit_dict = { 'data' : orbits , 'xpsel': xp_selector , 'eiksel' : eik_selector , 'idx' : 1}

	print('Start ray propagation...')

	for opt_dict in inputs.optics[i]:
		opt_dict['object'].Propagate(xp,eikonal,orb=orbit_dict)

	if len(inputs.sim)>1:
		basename = inputs.diagnostics[i]['base filename'] + '_' + str(i)
	else:
		basename = inputs.diagnostics[i]['base filename']

	if not inputs.diagnostics[i]['suppress details']:
		np.save(basename+'_xp0',xp0)
		np.save(basename+'_xp',xp)
		np.save(basename+'_eikonal',eikonal)
		np.save(basename+'_orbits',orbits)

	for opt_dict in inputs.optics[i]:
		opt_dict['object'].Report(basename,inputs.sim[i]['mks_length'])

	time2 = time()
	print('Completed in {:.1f} seconds'.format(time2-time1))
