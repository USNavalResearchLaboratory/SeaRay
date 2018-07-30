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

def ComputeRayVolumes(xp):
	v1 = xp[...,1,1:4] - xp[...,2,1:4]
	v2 = xp[...,3,1:4] - xp[...,4,1:4]
	v3 = xp[...,5,1:4] - xp[...,6,1:4]
	v1xv2_1 = v1[...,1]*v2[...,2] - v1[...,2]*v2[...,1]
	v1xv2_2 = v1[...,2]*v2[...,0] - v1[...,0]*v2[...,2]
	v1xv2_3 = v1[...,0]*v2[...,1] - v1[...,1]*v2[...,0]
	vol = np.abs(v1xv2_1*v3[...,0] + v1xv2_2*v3[...,1] + v1xv2_3*v3[...,2])
	return vol

def TestStep(ds,xp,vg):
	# ds has an element for each bundle
	# have to reshape ds to satisfy python broadcasting rules
	xp[...,:4] += ds[...,np.newaxis,np.newaxis]*vg

def FullStep(ds,xp,eik,vg):
	# ds has an element for each bundle
	V0 = ComputeRayVolumes(xp)
	# have to reshape ds to satisfy python broadcasting rules
	xp[...,:4] += ds[...,np.newaxis,np.newaxis]*vg
	eik[...,0] += ds*(xp[...,0,5]*vg[...,0,1] + xp[...,0,6]*vg[...,0,2] + xp[...,0,7]*vg[...,0,3] - xp[...,0,4]*vg[...,0,0])
	V1 = ComputeRayVolumes(xp)
	eik[...,1] *= np.sqrt(V0/V1)
	eik[...,2] *= np.sqrt(V0/V1)
	eik[...,3] *= np.sqrt(V0/V1)

def SphericalWave(xp,eikonal,wave):
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
	xp[:,:,4] = K4[0]
	xp[:,:,5] = sgn * K4[0] * np.sin(theta) * np.cos(phi)
	xp[:,:,6] = sgn * K4[0] * np.sin(theta) * np.sin(phi)
	xp[:,:,7] = sgn * K4[0] * np.cos(theta)
	eikonal[:,0] = K4[0] * (tau0-tau[:,0])
	eikonal[:,1] = np.sqrt(amag[:,0]**2 - az[:,0]**2)
	eikonal[:,2] = 0
	eikonal[:,3] = az[:,0]
	orientation.Create(A4[1:],K4[1:])
	orientation.ExpressRaysInStdBasis(xp,eikonal)
	xp[:,:,1:4] += F4[1:4]

def ParaxialWave(xp,eikonal,wave):
	orientation = v3.basis()
	A4 = np.array(wave['a0'])
	K4 = np.array(wave['k0'])
	R4 = np.array(wave['r0'])
	F4 = np.array(wave['focus'])
	SG = np.array(wave['supergaussian exponent'])
	# Set up in wave basis where propagation is +z and polarization is +x
	amag = np.sqrt(np.dot(A4[1:4],A4[1:4]))
	xp[:,:,4:8] = np.array([K4[0],0.0,0.0,K4[0]])
	eikonal[:,1:4] = np.array([amag,0.0,0.0])
	eikonal[:,0] = K4[0]*(xp[:,0,3] - xp[:,0,0])
	if SG!=2:
		r2 = xp[:,0,1]**2 + xp[:,0,2]**2
		eikonal[:,1] *= np.exp(-r2**(SG/2)/R4[1]**SG - xp[:,0,3]**2/R4[3]**2)
	else:
		eikonal[:,1] *= np.exp(-xp[:,0,1]**2/R4[1]**2 - xp[:,0,2]**2/R4[2]**2 - xp[:,0,3]**2/R4[3]**2)
	orientation.Create(A4[1:],K4[1:])
	orientation.ExpressRaysInStdBasis(xp,eikonal)
	xp[:,:,1:4] += F4[1:4]

def init(wave_dict,ray_dict):
	"""Use dictionaries from input file to create initial ray distribution"""
	# Set up host storage
	Nx = ray_dict['number'][0]
	Ny = ray_dict['number'][1]
	Nz = ray_dict['number'][2]
	num_bundles = Nx*Ny*Nz
	xp = np.zeros((num_bundles,7,8)).astype(np.double)
	eikonal = np.zeros((num_bundles,4)).astype(np.double)

	# Initialize Rays
	# It is assumed they start in vacuum
	# Packing of xp : xp[bundle,bundle_element,components]
	# components : x0,x1,x2,x3,k0,k1,k2,k3 : x0=time : k0=energy
	box = ray_dict['box']
	xp[:,0,0] = 0.0
	if ray_dict['loading coordinates']=='cartesian':
		grid1 = np.linspace(box[0],box[1],Nx+2)[1:-1]
		grid2 = np.linspace(box[2],box[3],Ny+2)[1:-1]
	else:
		grid1 = np.linspace(box[0]+(box[1]-box[0])/(Nx*100),box[1],Nx)
		grid2 = np.linspace(box[2],box[3]-(box[3]-box[2])/Ny,Ny)
	grid3 = np.linspace(box[4],box[5],Nz+2)[1:-1]

	# Load the primary rays in configuration space

	if ray_dict['loading coordinates']=='cartesian':
		xp[:,0,1] = np.outer(grid1,np.outer(np.ones(Ny),np.ones(Nz))).reshape(num_bundles)
		xp[:,0,2] = np.outer(np.ones(Nx),np.outer(grid2,np.ones(Nz))).reshape(num_bundles)
		xp[:,0,3] = np.outer(np.ones(Nx),np.outer(np.ones(Ny),grid3)).reshape(num_bundles)
	else:
		xp[:,0,1] = np.outer(np.outer(grid1,np.cos(grid2)),np.ones(Nz)).reshape(num_bundles)
		xp[:,0,2] = np.outer(np.outer(grid1,np.sin(grid2)),np.ones(Nz)).reshape(num_bundles)
		xp[:,0,3] = np.outer(np.ones(Nx),np.outer(np.ones(Ny),grid3)).reshape(num_bundles)

	# Load the satellite rays in configuration space

	xp[:,1,:] = xp[:,0,:]
	xp[:,2,:] = xp[:,0,:]
	xp[:,3,:] = xp[:,0,:]
	xp[:,4,:] = xp[:,0,:]
	xp[:,5,:] = xp[:,0,:]
	xp[:,6,:] = xp[:,0,:]

	xp[:,1,1] += .001*wave_dict['r0'][1]
	xp[:,2,1] -= .001*wave_dict['r0'][1]

	xp[:,3,2] += .001*wave_dict['r0'][2]
	xp[:,4,2] -= .001*wave_dict['r0'][2]

	xp[:,5,3] += .001*wave_dict['r0'][3]
	xp[:,6,3] -= .001*wave_dict['r0'][3]

	# Use wave description to set wavenumber, phase, and amplitude
	# All primary rays and satellite rays must be loaded into configuration space first
	# Ray configuration will be transformed into orientation of wave

	if wave_dict['focus'][0]==0.0:
		ParaxialWave(xp,eikonal,wave_dict)
	else:
		SphericalWave(xp,eikonal,wave_dict)

	return xp,eikonal

def track(queue,pusher_kernel,xp,eikonal,vol_dict,orb):
	"""queue = OpenCL command queue
	pusher_kernel = OpenCL kernel function to push particles
	xp,eikonal = all rays, kernel must handle outliers"""

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
	xp,eikonal = all rays, kernel must handle outliers"""

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
