'''
Module: :samp:`ray_kernel`
--------------------------

This module is the primary computational engine for ray tracing with eikonal field data.
'''
import numpy as np
import vec3 as v3
import pyopencl
import grid_tools
import base

def SyncSatellites(xp,vg):
	sync = (xp[...,0:1] - xp[...,0:1,0:1])*vg
	xp[...,:4] -= sync
	return sync

def UnsyncSatellites(xp,sync):
	xp[...,:4] += sync

def RepairSatellites(dt):
	for sat in range(1,4):
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
	# Compute the volume of the ray tetrahedron
	v1 = xp[...,1,1:4] - xp[...,0,1:4]
	v2 = xp[...,2,1:4] - xp[...,0,1:4]
	v3 = xp[...,3,1:4] - xp[...,0,1:4]
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
		# Interpret as spectral envelope
		# Put negative frequencies last for this calculation
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
	xp[...,3] -= F4[0]
	orientation.ExpressRaysInStdBasis(xp,eikonal,vg)
	xp[:,:,1:4] += F4[1:4]

def SphericalWave(xp,eikonal,vg,wave):
	A4 = np.array(wave['a0'])
	K4 = np.array(wave['k0'])
	R4 = np.array(wave['r0'])
	F4 = np.array(wave['focus'])
	SG = np.array(wave['supergaussian exponent'])
	# Take ray surface to be referenced to the origin, and +z propagation direction.
	# The center of the sphere is at z = F4[0] = time to reach focus (can be negative)
	# We do not want to change the positions on the ray surface, only amplitude and wavevector.
	xperp = np.sqrt(xp[...,1]**2 + xp[...,2]**2)
	R = np.sqrt(F4[0]**2 + xperp**2)
	phi = np.arctan2(xp[...,2],xp[...,1])
	theta = np.arccos(-F4[0]/R) # 0<theta<pi, theta=0 is a point at the +z tip of the sphere
	dtheta = np.arcsin(R4[1]/F4[0]) # constant

	# Start by assuming outward propagating wave for either sign of F4[0]
	xp[:,:,5] = xp[:,:,4] * np.sin(theta) * np.cos(phi)
	xp[:,:,6] = xp[:,:,4] * np.sin(theta) * np.sin(phi)
	xp[:,:,7] = xp[:,:,4] * np.cos(theta)
	# If F4[0] is positive we want to have an incoming wave
	xp[...,5:8] *= -np.sign(F4[0])
	vg[...] = xp[...,4:8]/xp[...,4:5]
	eikonal[:,0] = -np.sign(F4[0])*(R[:,0]-F4[0])*xp[:,0,4]
	xp[...,0] -= np.sign(F4[0])*(R[:,0]-F4[0])[:,np.newaxis]
	theta0 = np.pi*(1 + np.sign(F4[0]))/2
	amag = np.sqrt(np.dot(A4[1:4],A4[1:4]))
	amag *= (F4[0]/R[:,0]) * np.exp(-(theta[:,0]-theta0)**SG/dtheta**SG)
	# Get vector from eikonal gauge condition dot(k,a) = 0
	ax = amag / np.sqrt(1+xp[:,0,5]**2/xp[:,0,7]**2)
	az = -ax*xp[:,0,5]/xp[:,0,7]
	return ax,az

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
	vg[...] = xp[...,4:8]/xp[...,4:5]
	eikonal[:,0] = np.einsum('...i,...i',xp[:,0,1:4],xp[:,0,5:8])
	ax = amag*np.ones(eikonal.shape[:-1])
	if SG!=2:
		r2 = xp[:,0,1]**2 + xp[:,0,2]**2
		ax *= np.exp(-r2**(SG/2)/R4[1]**SG - xp[:,0,3]**2/R4[3]**2)
	else:
		ax *= np.exp(-xp[:,0,1]**2/R4[1]**2 - xp[:,0,2]**2/R4[2]**2 - xp[:,0,3]**2/R4[3]**2)
	return ax,0.0

def load_rays_xw(xp,bundle_radius,N,box,loading_coordinates):
	'''Load the rays in the z=0 plane in a regular pattern.'''
	num_bundles = N[0]*N[1]*N[2]
	o0 = np.ones(N[0])
	o1 = np.ones(N[1])
	o2 = np.ones(N[2])
	# load frequencies to respect FFT conventions
	grid0 = grid_tools.cyclic_nodes(box[0],box[1],N[0])
	grid1 = grid_tools.cell_centers(box[2],box[3],N[1])
	if loading_coordinates=='cartesian':
		grid2 = grid_tools.cell_centers(box[4],box[5],N[2])
	else:
		grid2 = grid_tools.cyclic_nodes(box[4],box[5],N[2])

	if box[0]==0.0:
		grid0 = grid0[1:]
		o0 = o0[1:]
		num_bundles -= N[1]*N[2]

	# Load the primary rays in configuration+w space

	if loading_coordinates=='cartesian':
		xp[:,0,0] = 0.0
		xp[:,0,1] = np.einsum('i,j,k',o0,grid1,o2).reshape(num_bundles)
		xp[:,0,2] = np.einsum('i,j,k',o0,o1,grid2).reshape(num_bundles)
		xp[:,0,3] = np.zeros(num_bundles)
		xp[:,0,4] = np.einsum('i,j,k',grid0,o1,o2).reshape(num_bundles)
	else:
		xp[:,0,0] = 0.0
		xp[:,0,1] = np.einsum('i,j,k',o0,grid1,np.cos(grid2)).reshape(num_bundles)
		xp[:,0,2] = np.einsum('i,j,k',o0,grid1,np.sin(grid2)).reshape(num_bundles)
		xp[:,0,3] = np.zeros(num_bundles)
		xp[:,0,4] = np.einsum('i,j,k',grid0,o1,o2).reshape(num_bundles)

	# Load the satellite rays in configuration+w space

	xp[:,1,:] = xp[:,0,:]
	xp[:,2,:] = xp[:,0,:]
	xp[:,3,:] = xp[:,0,:]

	xp[:,1,1] += bundle_radius[1]
	xp[:,1,3] -= bundle_radius[3]

	xp[:,2,1] -= bundle_radius[1] * np.cos(np.pi/3)
	xp[:,2,2] -= bundle_radius[2] * np.sin(np.pi/3)
	xp[:,2,3] -= bundle_radius[3]

	xp[:,3,1] -= bundle_radius[1] * np.cos(np.pi/3)
	xp[:,3,2] += bundle_radius[2] * np.sin(np.pi/3)
	xp[:,3,3] -= bundle_radius[3]

def init(wave_dict_list,ray_dict):
	'''Use dictionaries from input file to create initial ray distribution.
	Vacuum is assumed as the initial environment.
	Rays are packed in bundles of 4.  If there are Nb bundles, there are Nb*4 rays.

	:returns: xp,eikonal,vg
	:rtype: numpy.ndarray((Nb,4,8)),numpy.ndarray((Nb,4)),numpy.ndarray((Nb,4,4))

	The 8 elements of xp are x0,x1,x2,x3,k0,k1,k2,k3.
	The 4 elements of eikonal are phase,ax,ay,az.
	The 4 elements of vg are 1,vx,vy,vz.'''
	N = ray_dict['number']
	box = ray_dict['box']
	brad = ray_dict['bundle radius']
	base.check_ray_tuple(N)
	base.check_ray_tuple(box,True)
	base.check_vol_tuple(brad)

	# Set up host storage
	if box[0]==0.0:
		# Use real fields suitable for UPPE
		num_bundles = (N[0]-1)*N[1]*N[2]
	else:
		# Use complex fields suitable for Paraxial
		num_bundles = N[0]*N[1]*N[2]
	xp = np.zeros((num_bundles,4,8)).astype(np.double)
	eikonal = np.zeros((num_bundles,4)).astype(np.double)
	vg = np.zeros((num_bundles,4,4)).astype(np.double)

	# Spatial and frequency loading
	load_rays_xw(xp,brad,N,box,ray_dict['loading coordinates'])

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
	N_bund = ray_dict['number']
	N_orb = diag_dict['orbit rays']
	base.check_ray_tuple(N_bund)
	base.check_ray_tuple(N_orb)
	num_orbits = N_orb[0]*N_orb[1]*N_orb[2]
	if N_bund[0]<N_orb[0] or N_bund[1]<N_orb[1] or N_bund[2]<N_orb[2]:
		print('ERROR: more orbits',N_orb,'than rays',N_bund)
		exit(1)
	orbit_points = 1 # always have initial point
	for opt_dict in opt_list:
		orbit_points += opt_dict['object'].OrbitPoints()
	orbits = np.zeros((orbit_points,num_orbits,12))
	o = []
	l = []
	for i in range(3):
		o.append(np.ones(N_orb[i]).astype(int))
		l.append(np.array(list(range(int(N_bund[i]/N_orb[i]/2),N_bund[i],int(N_bund[i]/N_orb[i])))).astype(int))
	idx0 = np.einsum('i,j,k',l[0],o[1],o[2])
	idx1 = np.einsum('i,j,k',o[0],l[1],o[2])
	idx2 = np.einsum('i,j,k',o[0],o[1],l[2])
	stride2 = 1
	stride1 = stride2*N_bund[2]
	stride0 = stride1*N_bund[1]
	bundle_list = (idx0*stride0 + idx1*stride1 + idx2*stride2).flatten()
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
	num_pars = num_bundles*4
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
			xp[...] = xp_dev.get().reshape(num_bundles,4,8)
			eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)
			orb['data'][orb['idx'],:,:8] = xp[orb['xpsel']]
			orb['data'][orb['idx'],:,8:] = eikonal[orb['eiksel']]
			orb['idx'] += 1

	xp[...] = xp_dev.get().reshape(num_bundles,4,8)
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
	num_pars = num_bundles*4
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
			xp[...] = xp_dev.get().reshape(num_bundles,4,8)
			eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)
			orb['data'][orb['idx'],:,:8] = xp[orb['xpsel']]
			orb['data'][orb['idx'],:,8:] = eikonal[orb['eiksel']]
			orb['idx'] += 1

	xp[...] = xp_dev.get().reshape(num_bundles,4,8)
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
	cl.program('ray_in_cell').GetRelDensity(cl.q,(xp.shape[0],xp.shape[1]),None,xp_dev.data,dens_dev.data,ans_dev.data)
	cl.q.finish()
	return ans_dev.get().reshape((xp.shape[0],xp.shape[1]))
