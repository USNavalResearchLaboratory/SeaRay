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
	eik[...,0] += ds[...,0]*(xp[...,0,5]*vg[...,0,1] + xp[...,0,6]*vg[...,0,2] + xp[...,0,7]*vg[...,0,3] - xp[...,0,4]*vg[...,0,0])
	V1 = ComputeRayVolumes(xp,vg)
	eik[...,1] *= np.sqrt(V0/V1)
	eik[...,2] *= np.sqrt(V0/V1)
	eik[...,3] *= np.sqrt(V0/V1)

def GetMicroAction(xp,eik,vg):
	volume = ComputeRayVolumes(xp,vg)
	a2 = eik[...,1]**2 + eik[...,2]**2 + eik[...,3]**2
	return np.sum(a2*volume)

def AddFrequencyDimension(spatial_dims):
	if len(spatial_dims)==3:
		return (1,) + spatial_dims
	else:
		return spatial_dims

def AddFrequencyRange(freq_range,spatial_range):
	if len(spatial_range)==6:
		return freq_range + spatial_range
	else:
		return spatial_range

def PulseSpectrum(xp,eikonal,wave):
	'''Weight the rays based their frequency.  Assume transform limited pulse.'''
	dtau = wave['r0'][0]
	w0 = wave['k0'][0]
	dw = 2.0/dtau
	coeff = dtau/np.sqrt(2.0)
	weights = coeff*np.exp(-(xp[:,0,4]-w0)**2/dw**2)
	eikonal[:,1] *= weights
	eikonal[:,2] *= weights
	eikonal[:,3] *= weights

def SphericalWave(xp,eikonal,vg,wave):
	orientation = v3.basis()
	A4 = np.array(wave['a0'])
	K4 = np.array(wave['k0'])
	R4 = np.array(wave['r0'])
	F4 = np.array(wave['focus'])
	SG = np.array(wave['supergaussian exponent'])
	# tf = time to reach focus, positive is incoming ray
	tf = F4[0]
	xp[:,:,3] -= tf # put focus at origin
	r2 = xp[:,:,1]**2 + xp[:,:,2]**2 + xp[:,:,3]**2
	rho2 = xp[:,:,1]**2 + xp[:,:,2]**2
	dz = xp[:,:,3]
	phi = np.arctan2(xp[:,:,2],xp[:,:,1])
	sgn = np.abs(dz)/dz # positive is outgoing ray
	r = sgn*np.sqrt(r2)
	theta0 = np.pi/2 - sgn*np.pi/2
	theta = np.arccos(dz/np.sqrt(r2))
	dtheta = np.arcsin(R4[1]/tf)
	tau0 = tf
	tau = xp[:,:,0] - r
	dtau = R4[0]
	amag = np.sqrt(np.dot(A4[1:4],A4[1:4]))
	amag = amag * np.exp(-(theta-theta0)**SG/dtheta**SG) * np.exp(-(tau-tau0)**2/dtau**2)
	az = sgn * amag * np.sin(theta) * np.cos(phi)
	xp[:,:,5] = sgn * xp[:,:,4] * np.sin(theta) * np.cos(phi)
	xp[:,:,6] = sgn * xp[:,:,4] * np.sin(theta) * np.sin(phi)
	xp[:,:,7] = sgn * xp[:,:,4] * np.cos(theta)
	eikonal[:,0] = xp[:,0,4] * (tau0-tau[:,0])
	eikonal[:,1] = np.sqrt(amag[:,0]**2 - az[:,0]**2)
	eikonal[:,2] = 0
	eikonal[:,3] = az[:,0]
	vg[...] = xp[...,4:8]/xp[...,4:5]
	orientation.Create(A4[1:],K4[1:])
	orientation.ExpressRaysInStdBasis(xp,eikonal,vg)
	xp[:,:,1:4] += F4[1:4]

def ParaxialWave(xp,eikonal,vg,wave):
	orientation = v3.basis()
	A4 = np.array(wave['a0'])
	K4 = np.array(wave['k0'])
	R4 = np.array(wave['r0'])
	F4 = np.array(wave['focus'])
	SG = np.array(wave['supergaussian exponent'])
	# Set up in wave basis where propagation is +z and polarization is +x
	amag = np.sqrt(np.dot(A4[1:4],A4[1:4]))
	xp[:,:,7] = xp[:,:,4]
	eikonal[:,0] = xp[:,0,4]*(xp[:,0,3] - xp[:,0,0])
	eikonal[:,1:4] = np.array([amag,0.0,0.0])
	if SG!=2:
		r2 = xp[:,0,1]**2 + xp[:,0,2]**2
		eikonal[:,1] *= np.exp(-r2**(SG/2)/R4[1]**SG - xp[:,0,3]**2/R4[3]**2)
	else:
		eikonal[:,1] *= np.exp(-xp[:,0,1]**2/R4[1]**2 - xp[:,0,2]**2/R4[2]**2 - xp[:,0,3]**2/R4[3]**2)
	vg[...] = xp[...,4:8]/xp[...,4:5]
	orientation.Create(A4[1:],K4[1:])
	orientation.ExpressRaysInStdBasis(xp,eikonal,vg)
	xp[:,:,1:4] += F4[1:4]

def init(wave_dict,ray_dict):
	"""Use dictionaries from input file to create initial ray distribution"""
	w0 = wave_dict['k0'][0]
	N = AddFrequencyDimension(ray_dict['number'])
	box = AddFrequencyRange((0.9*w0,1.1*w0),ray_dict['box'])

	# Set up host storage
	num_bundles = N[0]*N[1]*N[2]*N[3]
	xp = np.zeros((num_bundles,7,8)).astype(np.double)
	eikonal = np.zeros((num_bundles,4)).astype(np.double)
	vg = np.zeros((num_bundles,7,4)).astype(np.double)

	# Initialize Rays
	# It is assumed they start in vacuum
	# Packing of xp : xp[bundle,bundle_element,components]
	# components : x0,x1,x2,x3,k0,k1,k2,k3 : x0=time : k0=energy
	o0 = np.ones(N[0])
	o1 = np.ones(N[1])
	o2 = np.ones(N[2])
	o3 = np.ones(N[3])
	grid0 = np.linspace(box[0],box[1],N[0]+2)[1:-1]
	if ray_dict['loading coordinates']=='cartesian':
		grid1 = np.linspace(box[2],box[3],N[1]+2)[1:-1]
		grid2 = np.linspace(box[4],box[5],N[2]+2)[1:-1]
	else:
		grid1 = np.linspace(box[2]+(box[3]-box[2])/N[1]/1000,box[3],N[1])
		grid2 = np.linspace(box[4],box[5]-(box[5]-box[4])/N[2],N[2])
	grid3 = np.linspace(box[6],box[7],N[3]+2)[1:-1]

	# Load the primary rays in configuration+w space

	if ray_dict['loading coordinates']=='cartesian':
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

	xp[:,1,1] += ray_dict['bundle radius'][1]
	xp[:,2,1] -= ray_dict['bundle radius'][1]

	xp[:,3,2] += ray_dict['bundle radius'][2]
	xp[:,4,2] -= ray_dict['bundle radius'][2]

	xp[:,5,3] += ray_dict['bundle radius'][3]
	xp[:,6,3] -= ray_dict['bundle radius'][3]

	# Use wave description to set wavenumber, phase, and amplitude
	# All primary rays and satellite rays must be loaded into configuration space first
	# Ray configuration will be transformed into orientation of wave

	if wave_dict['focus'][0]==0.0:
		ParaxialWave(xp,eikonal,vg,wave_dict)
	else:
		SphericalWave(xp,eikonal,vg,wave_dict)
	if N[0]>1:
		PulseSpectrum(xp,eikonal,wave_dict)

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

def track(queue,pusher_kernel,xp,eikonal,vol_dict,orb):
	"""queue = OpenCL command queue
	pusher_kernel = OpenCL kernel function to push particles
	xp,eikonal = all rays, kernel must handle outliers
	Assumes satellites are synchronized"""

	# Set up host storage
	num_bundles = xp.shape[0]

	# Set up device storage
	xp_dev = pyopencl.array.to_device(queue,xp)
	eikonal_dev = pyopencl.array.to_device(queue,eikonal)
	queue.finish()

	stepNow = 0
	num_pars = num_bundles*7
	num_active = num_pars
	while num_active>0 and stepNow<vol_dict['steps']:

		# Push the particles through multiple cycles

		pusher_kernel(	queue,
						(num_bundles,),
						None,
						xp_dev.data,
						eikonal_dev.data,
						np.double(vol_dict['dt']),
						np.int32(vol_dict['subcycles']))
		queue.finish()

		stepNow += vol_dict['subcycles']
		if orb['idx']!=0:
			xp[...] = xp_dev.get().reshape(num_bundles,7,8)
			eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)
			orb['data'][orb['idx'],:,:8] = xp[orb['xpsel']]
			orb['data'][orb['idx'],:,8:] = eikonal[orb['eiksel']]
			orb['idx'] += 1

	xp[...] = xp_dev.get().reshape(num_bundles,7,8)
	eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)

def track_RIC(queue,pusher_kernel,xp,eikonal,dens,vol_dict,orb):
	"""queue = OpenCL command queue
	pusher_kernel = OpenCL kernel function to push particles
	xp,eikonal = all rays, kernel must handle outliers
	Assumes satellites are synchronized"""

	# Set up host storage
	num_bundles = xp.shape[0]

	# Set up device storage
	xp_dev = pyopencl.array.to_device(queue,xp)
	eikonal_dev = pyopencl.array.to_device(queue,eikonal)
	dens_dev = pyopencl.array.to_device(queue,dens)
	queue.finish()

	stepNow = 0
	num_pars = num_bundles*7
	num_active = num_pars
	while num_active>0 and stepNow<vol_dict['steps']:

		# Push the particles through multiple cycles

		pusher_kernel(	queue,
						(num_bundles,),
						None,
						xp_dev.data,
						eikonal_dev.data,
						dens_dev.data,
						np.double(vol_dict['dt']),
						np.int32(vol_dict['subcycles']))
		queue.finish()

		stepNow += vol_dict['subcycles']
		if orb['idx']!=0:
			xp[...] = xp_dev.get().reshape(num_bundles,7,8)
			eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)
			orb['data'][orb['idx'],:,:8] = xp[orb['xpsel']]
			orb['data'][orb['idx'],:,8:] = eikonal[orb['eiksel']]
			orb['idx'] += 1

	xp[...] = xp_dev.get().reshape(num_bundles,7,8)
	eikonal[...] = eikonal_dev.get().reshape(num_bundles,4)

def gather(queue,gather_kernel,xp,dens):
	"""Return density at location of each ray (primary + satellites).
	queue = OpenCL command queue
	gather_kernel = OpenCL kernel function to gather grid data"""
	# Set up host storage
	ans = np.zeros((xp.shape[0],xp.shape[1]))
	# Set up device storage
	xp_dev = pyopencl.array.to_device(queue,xp)
	dens_dev = pyopencl.array.to_device(queue,dens)
	ans_dev = pyopencl.array.to_device(queue,ans)
	queue.finish()
	# Run the kernel
	gather_kernel(queue,(xp.shape[0],xp.shape[1]),None,xp_dev.data,dens_dev.data,ans_dev.data)
	queue.finish()
	return ans_dev.get().reshape((xp.shape[0],xp.shape[1]))
