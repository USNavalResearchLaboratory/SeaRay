'''
Module: :samp:`ray_kernel`
--------------------------

This module is the primary computational engine for ray tracing with eikonal field data.
'''
import numpy as np
import vec3 as v3
import pyopencl
import pyopencl.array as cl_array
import numpy.random
import grid_tools

def SyncSatellites(xp,vg):
	sync = (xp[...,0:1] - xp[...,0:1,0:1])*vg
	xp[...,:4] -= sync
	return sync

def UnsyncSatellites(xp,sync):
	xp[...,:4] += sync

def RepairSatellites(dt):
	for sat in range(1,7):
		ds = dt[...,sat]
		bad = np.logical_or(np.isnan(ds),np.isinf(ds))
		sel = np.where(np.logical_or(bad,ds<0.0))[0]
		dt[sel,sat] = dt[sel,0]

def ExtractRays(impact,xp,eik,vg):
	return xp[impact,...],eik[impact,...],vg[impact,...]

def extract_rays(impact,xp,vg):
	return xp[impact,...],vg[impact,...]

def UpdateRays(impact,xp,eik,vg,xps,eiks,vgs):
	xp[impact,...] = xps
	eik[impact,...] = eiks
	vg[impact,...] = vgs

def update_rays(impact,xp,vg,xps,vgs):
	xp[impact,...] = xps
	vg[impact,...] = vgs

def ComputeRayVolumes(xp,vg):
	sync = SyncSatellites(xp,vg)
	# Compute the volume
	v1 = xp[...,1,1:4] - xp[...,2,1:4]
	v2 = xp[...,3,1:4] - xp[...,4,1:4]
	v3 = xp[...,5,1:4] - xp[...,6,1:4]
	v1xv2_1 = v1[...,1]*v2[...,2] - v1[...,2]*v2[...,1]
	v1xv2_2 = v1[...,2]*v2[...,0] - v1[...,0]*v2[...,2]
	v1xv2_3 = v1[...,0]*v2[...,1] - v1[...,1]*v2[...,0]
	vol = np.abs(v1xv2_1*v3[...,0] + v1xv2_2*v3[...,1] + v1xv2_3*v3[...,2])
	UnsyncSatellites(xp,sync)
	return vol

def TestStep(ds,xp,vg):
	# len(ds.shape) = len(xp.shape or vg.shape)-1
	xp[...,:4] += ds[...,np.newaxis]*vg

def FullStep(ds,xp,eik,vg):
	# len(ds.shape or eik.shape) = len(xp.shape or vg.shape)-1
	V0 = ComputeRayVolumes(xp,vg)
	xp[...,:4] += ds[...,np.newaxis]*vg
	eik[...,0] += ds[...,0]*(xp[...,0,5]*vg[...,0,1] + xp[...,0,6]*vg[...,0,2] + xp[...,0,7]*vg[...,0,3])
	V1 = ComputeRayVolumes(xp,vg)
	eik[...,1] *= np.sqrt(V0/V1)
	eik[...,2] *= np.sqrt(V0/V1)
	eik[...,3] *= np.sqrt(V0/V1)

def GetMicroAction(xp,eik,vg):
	'''Numerical accuracy metric, should be conserved.'''
	volume = ComputeRayVolumes(xp,vg)
	a2 = eik[...,1]**2 + eik[...,2]**2 + eik[...,3]**2
	return np.sum(a2*volume)

def GetTransversality(xp,eik):
	'''Numerical accuracy metric, should be unity in isotropic media.'''
	ans = np.cross(xp[...,0,5:8],eik[...,1:4])
	ans = np.einsum('...i,...i',ans,ans) + np.finfo(np.double).tiny
	ans /= np.einsum('...i,...i',xp[...,0,5:8],xp[...,0,5:8])
	ans /= np.einsum('...i,...i',eik[...,1:4],eik[...,1:4]) + np.finfo(np.double).tiny
	ans = np.sqrt(np.mean(ans))
	return ans

def AddFrequencyDimension(spatial_dims):
	if len(spatial_dims)==3:
		return (1,) + spatial_dims
	else:
		return spatial_dims

def AddFrequencyRange(carrier_freq,spatial_range):
	if len(spatial_range)==6:
		return (0.99*carrier_freq,1.01*carrier_freq) + spatial_range
	else:
		return spatial_range

def PulseSpectrum(xp,box,N,pulse_length,w0):
	'''Weight the rays based their frequency.  Assume transform limited pulse.
	Spectral amplitude is matched with time domain amplitude by performing a test FFT.
	If the lower frequency bound is zero, real carrier resolved fields are used.

	:param numpy.ndarray xp: Ray phase space, primary ray frequency must be loaded
	:param tuple box: The first two elements are the frequency bounds
	:param tuple N: The first element is the number of frequency nodes
	:param pulse_length: The Gaussian pulse width (1/e amplitude)
	:param w0: The central frequency of the wave

	:returns: weight factors with shape (bundles,)'''
	sigma_w = 2.0/pulse_length
	freq_range = grid_tools.cyclic_nodes(box[0],box[1],N[0])
	# Create a unit-height Gaussian spectral amplitude
	envelope = np.exp(-(freq_range-w0)**2/sigma_w**2)
	if box[0]==0.0:
		# Intepret as real carrier resolved field
		envelope = np.fft.irfft(envelope)
	else:
		# Interpret as spectral envelope putting negative frequencies last
		envelope = np.fft.ifftshift(envelope)
		# Get the corresponding time domain envelope
		envelope = np.fft.ifft(envelope)
	coeff = 1/np.max(np.abs(envelope))
	return coeff*np.exp(-(xp[:,0,4]-w0)**2/sigma_w**2)

def ConfigureRayOrientation(xp,eikonal,vg,wave_dict_list):
	'''Uses the first wave dictionary to orient the rays.
	This implies that superposition waves must be oriented the same way.'''
	A4 = np.array(wave_dict_list[0]['a0'])
	K4 = np.array(wave_dict_list[0]['k0'])
	F4 = np.array(wave_dict_list[0]['focus'])
	orientation = v3.basis()
	orientation.Create(A4[1:],K4[1:])
	orientation.ExpressRaysInStdBasis(xp,eikonal,vg)
	xp[:,:,1:4] += F4[1:4]

def SphericalWave(xp,eikonal,vg,wave):
	A4 = np.array(wave['a0'])
	K4 = np.array(wave['k0'])
	R4 = np.array(wave['r0'])
	F4 = np.array(wave['focus'])
	SG = np.array(wave['supergaussian exponent'])
	# tf = time to reach focus, positive is incoming ray
	# Transformation of ray box:
	# Interpret z-tf as the new radial position
	# Carry over angles from the plane directly
	# This orients satellites in the new system
	# N.b. this does alter the spot size by cos(theta0)
	tf = F4[0]
	xp[...,3] -= tf # put focus at origin
	ro = np.sqrt(xp[...,1]**2 + xp[...,2]**2 + xp[...,3]**2)
	r = xp[...,3]
	phi = np.arctan2(xp[...,2],xp[...,1])
	sgn = np.abs(r)/r # positive is outgoing ray
	theta0 = np.pi/2 - sgn*np.pi/2
	theta = np.arccos(r/ro)
	dtheta = np.arcsin(R4[1]/tf)
	xp[...,1] = np.abs(r) * np.sin(theta) * np.cos(phi)
	xp[...,2] = np.abs(r) * np.sin(theta) * np.sin(phi)
	xp[...,3] = np.abs(r) * np.cos(theta)

	xp[:,:,5] = sgn * xp[:,:,4] * np.sin(theta) * np.cos(phi)
	xp[:,:,6] = sgn * xp[:,:,4] * np.sin(theta) * np.sin(phi)
	xp[:,:,7] = sgn * xp[:,:,4] * np.cos(theta)
	vg[...] = xp[...,4:8]/xp[...,4:5]
	eikonal[:,0] = xp[:,0,4] * xp[:,0,0]

	amag = np.sqrt(np.dot(A4[1:4],A4[1:4]))
	amag = amag * (-tf/r) * np.exp(-(theta-theta0)**SG/dtheta**SG)
	az = sgn * amag * np.sin(theta) * np.cos(phi)
	return np.sqrt(amag[:,0]**2 - az[:,0]**2),az

def ParaxialWave(xp,eikonal,vg,wave):
	A4 = np.array(wave['a0'])
	K4 = np.array(wave['k0'])
	R4 = np.array(wave['r0'])
	F4 = np.array(wave['focus'])
	SG = np.array(wave['supergaussian exponent'])
	try:
		phase = wave['phase']
	except KeyError:
		phase = 0.0
	# Set up in wave basis where propagation is +z and polarization is +x
	amag = np.sqrt(np.dot(A4[1:4],A4[1:4]))
	xp[:,:,7] = xp[:,:,4]
	eikonal[:,0] = xp[:,0,7]*xp[:,0,3]
	vg[...] = xp[...,4:8]/xp[...,4:5]
	ax = amag*np.ones(eikonal.shape[:-1])
	if SG!=2:
		r2 = xp[:,0,1]**2 + xp[:,0,2]**2
		ax *= np.exp(-r2**(SG/2)/R4[1]**SG - xp[:,0,3]**2/R4[3]**2)
	else:
		ax *= np.exp(-xp[:,0,1]**2/R4[1]**2 - xp[:,0,2]**2/R4[2]**2 - xp[:,0,3]**2/R4[3]**2)
	return ax,0.0

def load_rays_xw(xp,bundle_radius,N,box,loading_coordinates):
	'''Load the rays in the z=0 plane in a regular pattern.'''
	num_bundles = N[0]*N[1]*N[2]*N[3]
	o0 = np.ones(N[0])
	o1 = np.ones(N[1])
	o2 = np.ones(N[2])
	o3 = np.ones(N[3])
	# load frequencies to respect FFT conventions
	grid0 = grid_tools.cyclic_nodes(box[0],box[1],N[0])
	grid1 = grid_tools.cell_centers(box[2],box[3],N[1])
	if loading_coordinates=='cartesian':
		grid2 = grid_tools.cell_centers(box[4],box[5],N[2])
	else:
		grid2 = grid_tools.cyclic_nodes(box[4],box[5],N[2])
	grid3 = grid_tools.cell_centers(box[6],box[7],N[3])

	if box[0]==0.0:
		grid0 = grid0[1:]
		o0 = o0[1:]
		num_bundles -= N[1]*N[2]*N[3]

	# Load the primary rays in configuration+w space

	if loading_coordinates=='cartesian':
		xp[:,0,0] = 0.0
		xp[:,0,1] = np.einsum('i,j,k,l',o0,grid1,o2,o3).reshape(num_bundles)
		xp[:,0,2] = np.einsum('i,j,k,l',o0,o1,grid2,o3).reshape(num_bundles)
		xp[:,0,3] = np.einsum('i,j,k,l',o0,o1,o2,grid3).reshape(num_bundles)
		xp[:,0,4] = np.einsum('i,j,k,l',grid0,o1,o2,o3).reshape(num_bundles)
	else:
		xp[:,0,0] = 0.0
		xp[:,0,1] = np.einsum('i,j,k,l',o0,grid1,np.cos(grid2),o3).reshape(num_bundles)
		xp[:,0,2] = np.einsum('i,j,k,l',o0,grid1,np.sin(grid2),o3).reshape(num_bundles)
		xp[:,0,3] = np.einsum('i,j,k,l',o0,o1,o2,grid3).reshape(num_bundles)
		xp[:,0,4] = np.einsum('i,j,k,l',grid0,o1,o2,o3).reshape(num_bundles)

	# Load the satellite rays in configuration+w space

	xp[:,1,:] = xp[:,0,:]
	xp[:,2,:] = xp[:,0,:]
	xp[:,3,:] = xp[:,0,:]
	xp[:,4,:] = xp[:,0,:]
	xp[:,5,:] = xp[:,0,:]
	xp[:,6,:] = xp[:,0,:]

	xp[:,1,1] += bundle_radius[1]
	xp[:,2,1] -= bundle_radius[1]

	xp[:,3,2] += bundle_radius[2]
	xp[:,4,2] -= bundle_radius[2]

	xp[:,5,3] += bundle_radius[3]
	xp[:,6,3] -= bundle_radius[3]

def relaunch_rays(xp,eikonal,vg,A,vol_dict):
	'''Use wave data to create a new ray distribution.
	The wave data is stored as A[w,x,y,z]'''
	# Assume vacuum for now
	ampl = np.abs(A[...,-1])
	phasex = np.unwrap(np.angle(A[...,-1]),axis=1)
	phasey = np.unwrap(np.angle(A[...,-1]),axis=2)
	if phasex.shape[1]>1:
		kx = np.gradient(phasex,axis=1)
	else:
		kx = np.zeros(phasex.shape)
	if phasey.shape[2]>1:
		ky = np.gradient(phasey,axis=2)
	else:
		ky = np.zeros(phasey.shape)
	# Rays keep their original frequency and transverse positions.
	# Frequency shifts are still accounted for because amplitude may change.
	xp[...,0] += vol_dict['size'][2] # How to handle ray time?
	xp[...,3] += vol_dict['size'][2]
	xp[...,5] = 0.0 # Need 2D gather of kx
	xp[...,6] = 0.0 # Need 2D gather of ky
	xp[...,7] = np.sqrt(xp[...,4]**2 - xp[...,5]**2 - xp[...,6]**2)
	eikonal[...,0] = 0.0 # Need 2D gather of phase
	eikonal[...,1] = 1.0 # Need 2D gather of amplitude
	eikonal[...,2] = 0.0
	eikonal[...,3] = 0.0
	vg[...,0] = 1.0
	vg[...,1] = 0.0
	vg[...,2] = 0.0
	vg[...,3] = 1.0

def init(wave_dict_list,ray_dict):
	'''Use dictionaries from input file to create initial ray distribution.
	Vacuum is assumed as the initial environment.
	Rays are packed in bundles of 7.  If there are Nb bundles, there are Nb*7 rays.

	:returns: xp,eikonal,vg
	:rtype: numpy.ndarray((Nb,7,8)),numpy.ndarray((Nb,4)),numpy.ndarray((Nb,7,4))

	The 8 elements of xp are x0,x1,x2,x3,k0,k1,k2,k3.
	The 4 elements of eikonal are phase,ax,ay,az.
	The 4 elements of vg are 1,vx,vy,vz.'''
	if type(wave_dict_list)==dict:
		wave_dict_list = [wave_dict_list]
	N = AddFrequencyDimension(ray_dict['number'])
	box = AddFrequencyRange(wave_dict_list[0]['k0'][0],ray_dict['box'])

	# Set up host storage
	if box[0]==0.0:
		num_bundles = (N[0]-1)*N[1]*N[2]*N[3]
	else:
		num_bundles = N[0]*N[1]*N[2]*N[3]
	xp = np.zeros((num_bundles,7,8)).astype(np.double)
	eikonal = np.zeros((num_bundles,4)).astype(np.double)
	vg = np.zeros((num_bundles,7,4)).astype(np.double)

	# Spatial and frequency loading
	load_rays_xw(xp,ray_dict['bundle radius'],N,box,ray_dict['loading coordinates'])

	# Use wave description to set wavenumber, phase, and amplitude
	# All primary rays and satellite rays must be loaded into configuration space first
	# Ray configuration will be transformed into orientation of wave

	for wave_dict in wave_dict_list:
		if wave_dict['focus'][0]==0.0:
			ax,az = ParaxialWave(xp,eikonal,vg,wave_dict)
		else:
			ax,az = SphericalWave(xp,eikonal,vg,wave_dict)
		freq_weights = PulseSpectrum(xp,box,N,wave_dict['r0'][0],wave_dict['k0'][0])
		eikonal[...,1] += ax * freq_weights
		eikonal[...,3] += az * freq_weights
	ConfigureRayOrientation(xp,eikonal,vg,wave_dict_list)

	return xp,eikonal,vg

def setup_orbits(xp,eikonal,ray_dict,diag_dict,opt_list):
	N_bund = AddFrequencyDimension(ray_dict['number'])
	N_orb = AddFrequencyDimension(diag_dict['orbit rays'])
	num_bundles = N_bund[0]*N_bund[1]*N_bund[2]*N_bund[3]
	num_orbits = N_orb[0]*N_orb[1]*N_orb[2]*N_orb[3]
	if N_bund[0]<N_orb[0] or N_bund[1]<N_orb[1] or N_bund[2]<N_orb[2] or N_bund[3]<N_orb[3]:
		print('ERROR: more orbits than rays.')
		exit(1)
	orbit_points = 1 # always have initial point
	for opt_dict in opt_list:
		orbit_points += opt_dict['object'].OrbitPoints()
	orbits = np.zeros((orbit_points,num_orbits,12))
	o = []
	l = []
	for i in range(4):
		o.append(np.ones(N_orb[i]).astype(np.int))
		l.append(np.array(list(range(int(N_bund[i]/N_orb[i]/2),N_bund[i],int(N_bund[i]/N_orb[i])))).astype(np.int))
	idx0 = np.einsum('i,j,k,l',l[0],o[1],o[2],o[3])
	idx1 = np.einsum('i,j,k,l',o[0],l[1],o[2],o[3])
	idx2 = np.einsum('i,j,k,l',o[0],o[1],l[2],o[3])
	idx3 = np.einsum('i,j,k,l',o[0],o[1],o[2],l[3])
	stride3 = 1
	stride2 = stride3*N_bund[3]
	stride1 = stride2*N_bund[2]
	stride0 = stride1*N_bund[1]
	bundle_list = (idx0*stride0 + idx1*stride1 + idx2*stride2 + idx3*stride3).flatten()
	#bundle_list = np.random.choice(range(num_bundles),num_orbits).tolist()
	xp_selector = (bundle_list,[0],)
	eik_selector = (bundle_list,)
	orbits[0,:,:8] = xp[xp_selector]
	orbits[0,:,8:] = eikonal[eik_selector]
	orbit_dict = { 'data' : orbits , 'xpsel': xp_selector , 'eiksel' : eik_selector , 'idx' : 1}
	return orbit_dict

def track(cl,xp,eikonal,vol_dict,orb):
	"""cl = OpenCL reference class
	xp,eikonal = all rays, kernel must handle outliers
	Assumes satellites are synchronized"""

	# Set up host storage
	num_bundles = xp.shape[0]

	# Set up device storage
	xp_dev = pyopencl.array.to_device(cl.q,xp)
	eikonal_dev = pyopencl.array.to_device(cl.q,eikonal)
	cl.q.finish()

	stepNow = 0
	num_pars = num_bundles*7
	num_active = num_pars
	while num_active>0 and stepNow<vol_dict['steps']:

		# Push the particles through multiple cycles

		cl.program('ray_integrator').Symplectic(cl.q,
			(num_bundles,),
			None,
			xp_dev.data,
			eikonal_dev.data,
			np.double(vol_dict['dt']),
			np.int32(vol_dict['subcycles']))
		cl.q.finish()

		stepNow += vol_dict['subcycles']
		if orb['idx']!=0:
			xp[...] = xp_dev.get().reshape(num_bundles,7,8)
			eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)
			orb['data'][orb['idx'],:,:8] = xp[orb['xpsel']]
			orb['data'][orb['idx'],:,8:] = eikonal[orb['eiksel']]
			orb['idx'] += 1

	xp[...] = xp_dev.get().reshape(num_bundles,7,8)
	eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)

def track_RIC(cl,xp,eikonal,dens,vol_dict,orb):
	"""cl = OpenCL reference class
	xp,eikonal = all rays, kernel must handle outliers
	Assumes satellites are synchronized"""

	# Set up host storage
	num_bundles = xp.shape[0]

	# Set up device storage
	xp_dev = pyopencl.array.to_device(cl.q,xp)
	eikonal_dev = pyopencl.array.to_device(cl.q,eikonal)
	dens_dev = pyopencl.array.to_device(cl.q,dens)
	cl.q.finish()

	stepNow = 0
	num_pars = num_bundles*7
	num_active = num_pars
	while num_active>0 and stepNow<vol_dict['steps']:

		# Push the particles through multiple cycles

		cl.program('ray_in_cell').Symplectic(cl.q,
						(num_bundles,),
						None,
						xp_dev.data,
						eikonal_dev.data,
						dens_dev.data,
						np.double(vol_dict['dt']),
						np.int32(vol_dict['subcycles']))
		cl.q.finish()

		stepNow += vol_dict['subcycles']
		if orb['idx']!=0:
			xp[...] = xp_dev.get().reshape(num_bundles,7,8)
			eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)
			orb['data'][orb['idx'],:,:8] = xp[orb['xpsel']]
			orb['data'][orb['idx'],:,8:] = eikonal[orb['eiksel']]
			orb['idx'] += 1

	xp[...] = xp_dev.get().reshape(num_bundles,7,8)
	eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)

def gather(cl,xp,dens):
	"""Return density at location of each ray (primary + satellites)."""
	# Set up host storage
	ans = np.zeros((xp.shape[0],xp.shape[1]))
	# Set up device storage
	xp_dev = pyopencl.array.to_device(cl.q,xp)
	dens_dev = pyopencl.array.to_device(cl.q,dens)
	ans_dev = pyopencl.array.to_device(cl.q,ans)
	cl.q.finish()
	# Run the kernel
	cl.program('ray_in_cell').GetDensity(cl.q,(xp.shape[0],xp.shape[1]),None,xp_dev.data,dens_dev.data,ans_dev.data)
	cl.q.finish()
	return ans_dev.get().reshape((xp.shape[0],xp.shape[1]))
