import sys
sys.path.append('modules')
from time import time
import os
import glob
import shutil
import numpy as np
import logging
import json

import modules.base as base
import modules.init as init
import modules.ray_kernel as ray_kernel

def help():
	print('==========BEGIN HELP FOR SEARAY==========')
	print('Version: 1.0.0a3')
	print('Usage: rays.py cmd [log=<loglevel>] [file=<name>] [device=<dev_str>] [platform=<plat_str>]')
	print('Arguments in square brackets are optional.')
	print('cmd = list --- displays all platforms and devices')
	print('cmd = run --- executes calculation')
	print('<loglevel> = logging level: debug, info, warning, error, critical')
	print('<name> = path of input file, if not given inputs.py must be in working directory.')
	print('<dev_str> = something in desired device name, or numerical id')
	print('<plat_str> = something in desired OpenCL platform, or numerical id')
	print('large numbers (>10) are assumed to be part of a name rather than an id')
	print('defaults are the last platform/device in the list')
	print('==========END HELP FOR SEARAY==========')

###############################
# EXTERNAL CALLER ENTRY POINT #
###############################

def run(cmd_line,sim,sources,optics,diagnostics):
	'''Run a SeaRay simulation.  This is invoked automatically if `rays.py` is run from
	the command line.  It can also be called from the user's own control program.
	
	:param list cmd_line: command line arguments excluding 0,1
	:param dict sim: simulation object
	:param list sources: list of source objects
	:param list optics: list of optics
	:param dict diagnostics: diagnostic object'''

	# Error check command line arguments
	valid_arg_keys = ['log','device','platform']
	for arg in cmd_line:
		if arg.split('=')[0] not in valid_arg_keys:
			raise SyntaxError('The argument <'+arg+'> was not understood.')

	# Set up OpenCL
	print('--------------------')
	print('Accelerator Hardware')
	print('--------------------')
	cl,args = init.setup_opencl(sys.argv)
	cl.add_program('kernels','fft')
	cl.add_program('kernels','uppe')
	cl.add_program('kernels','paraxial')
	cl.add_program('kernels','caustic')
	cl.add_program('kernels','ionization')
	cl.add_program('kernels','rotations')

	# Setup logging
	log_level = logging.WARNING
	for arg in args:
		if arg.split('=')[0]=='log':
			log_level = getattr(logging, arg.split('=')[1].upper())
	logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', datefmt='%m/%d/%y %H:%M:%S', level=log_level)

	# verify top level inputs
	if not isinstance(sim,dict):
		raise TypeError('sim should be a dictionary')
	base.check_list_of_dict(sources)
	base.check_list_of_dict(optics)
	if not isinstance(diagnostics,dict):
		raise TypeError('diagnostics should be a dictionary')

	# Pre-simulation cleaning
	if diagnostics['clean old files']:
		file_list = glob.glob(diagnostics['base filename']+'*.npy')
		file_list += glob.glob(diagnostics['base filename']+'*.json')
		for f in file_list:
			os.remove(f)

	print(sim['message'])
	time1 = time()

	output_path = os.path.dirname(diagnostics['base filename'])
	if not os.path.exists(output_path):
		logging.info('creating output directory %s',output_path)
		os.mkdir(output_path)

	print('\nSetting up optics...\n')
	for opt_dict in optics:
		print('Initializing',opt_dict['object'].name)
		opt_dict['object'].Initialize(opt_dict)
		opt_dict['object'].InitializeCL(cl,opt_dict)

	print('\nSetting up rays and orbits...')
	if len(sources)>1:
		raise ValueError('Only one source allowed at present.')
	xp,eikonal,vg = ray_kernel.init(sources[0])
	orbit_dict = ray_kernel.setup_orbits(xp,eikonal,sources[0]['rays'],diagnostics,optics)
	micro_action_0 = ray_kernel.GetMicroAction(xp,eikonal,vg)
	if not diagnostics['suppress details']:
		xp0 = np.copy(xp)
		eikonal0 = np.copy(eikonal)

	print('\nStart propagation...\n')
	print('Initial micro-action = {:.3g}\n'.format(micro_action_0))

	for opt_dict in optics:
		print('\nEncountering',opt_dict['object'].name)
		opt_dict['object'].Propagate(xp,eikonal,vg,orb=orbit_dict)

	print('\nStart diagnostic reports...\n')

	basename = diagnostics['base filename']
	with open(basename+'_sim.json','w') as f:
		json.dump(sim,f)
	with open(basename+'_sources.json','w') as f:
		json.dump(sources,f)
	with open(basename+'_diagnostics.json','w') as f:
		json.dump(diagnostics,f)
	
	if not diagnostics['suppress details']:
		ray_kernel.SyncSatellites(xp,vg)
		np.save(basename+'_eikonal0',eikonal0)
		np.save(basename+'_xp0',xp0)
		np.save(basename+'_xp',xp)
		np.save(basename+'_eikonal',eikonal)
		np.save(basename+'_orbits',orbit_dict['data'])

	for opt_dict in optics:
		opt_dict['object'].Report(basename,sim['mks_length'])

	time2 = time()
	print('Completed in {:.1f} seconds'.format(time2-time1))

############################
# COMMAND LINE ENTRY POINT #
############################

if __name__ == "__main__":
	if len(sys.argv)==1 or sys.argv[1]=='help':
		help()
		exit(0)

	valid_arg_keys = ['run','list','log','file','device','platform']
	for arg in sys.argv[1:]:
		if arg.split('=')[0] not in valid_arg_keys:
			raise SyntaxError('The argument <'+arg+'> was not understood.')
	
	if sys.argv[1]=='list':
		init.list_opencl()
		exit(0)
	
	if sys.argv[1]=='run':
		args = sys.argv[2:]
		reduced_args = []
		for arg in args:
			if arg.split('=')[0]=='file':
				shutil.copyfile(arg.split('=')[1],'inputs.py')
			elif arg!='run' and arg!='list':
				reduced_args += [arg]
		import inputs
		run(reduced_args,inputs.sim,inputs.sources,inputs.optics,inputs.diagnostics)
	else:
		raise SyntaxError('The first positional command was not understood.')