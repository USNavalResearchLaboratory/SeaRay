'''
Module: :samp:`paraxial_kernel`
-------------------------------

This module is the primary computational engine for advancing paraxial wave equations.
'''
import numpy as np
from numpy.fft import fft
from numpy.fft import ifft
import pyopencl
import caustic_tools
import grid_tools
import ionization

class source_cluster:
	'''This class gathers references to host and device storage
	that are used during the ODE integrator right-hand-side evaluation.'''
	def __init__(self,queue,a,ng,w,kz,k0,L):
		# Following shape calculations have redundancy.
		# Keep this for parity with case of real fields.
		# N.b. we must not assume anything about ordering of frequency array
		if w.shape[0]==1:
			bandwidth = 1.0
		else:
			bandwidth = (np.max(w) - np.min(w))*(1 + 1/(w.shape[0]-1))
		self.dt = 2*np.pi/bandwidth
		self.L = L
		self.freqs = a.shape[0]
		self.steps = self.freqs
		self.transverse_shape = (a.shape[1],a.shape[2])
		self.freq_domain_shape = (self.freqs,) + self.transverse_shape
		self.time_domain_shape = (self.steps,) + self.transverse_shape
		self.dphi = 1.0
		self.rel_amin = 1e-2
		# Device arrays
		self.q00_dev = pyopencl.array.to_device(queue,a)
		self.q0_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.complex)
		self.qw_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.complex)
		self.Aw_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.complex)
		self.At_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.complex)
		self.Et_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.complex)
		self.ng_dev= pyopencl.array.to_device(queue,ng)
		self.w_dev = pyopencl.array.to_device(queue,w)
		self.kz_dev = pyopencl.array.to_device(queue,kz)
		self.k0_dev = pyopencl.array.to_device(queue,k0)
		self.ne_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.J_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.complex)

def update_current(cl,src,dchi,chi3,ionizer):
	'''Get current density from A[w,x,y]'''
	ionizer.ResetParameters(timestep=src.dt)

	# Setup some shorthand
	shp2 = src.transverse_shape
	shp3 = src.Aw_dev.shape
	wdev = src.w_dev.data
	Awdev = src.Aw_dev.data
	Atdev = src.At_dev.data
	Etdev = src.Et_dev.data
	nedev = src.ne_dev.data
	ngdev = src.ng_dev.data
	Jdev = src.J_dev.data

	# Form time domain potential and field
	# N.b. time index runs backwards due to FFT conventions
	src.Et_dev[...] = src.Aw_dev
	src.At_dev[...] = src.Aw_dev
	cl.program('fft').DtSpectral(cl.q,shp3,None,Etdev,wdev,np.double(-1.0))
	cl.program('fft').IFFT(cl.q,shp2,None,Etdev,np.int32(src.freqs))
	cl.program('fft').IFFT(cl.q,shp2,None,Atdev,np.int32(src.freqs))

	# Accumulate the source terms
	# First handle plasma formation
	ionizer.RateCL(cl,shp3,nedev,Etdev,True)
	ionizer.GetPlasmaDensityCL(cl,shp3,nedev,ngdev)

	# Current due to nonuniform and Kerr susceptibility
	cl.program('paraxial').SetKerrPolarization(cl.q,shp3,None,Jdev,Etdev,np.double(chi3))
	cl.program('fft').FFT(cl.q,shp2,None,Jdev,np.int32(src.steps))
	#cl.program('paraxial').AddNonuniformChi(cl.q,shp3,None,Jdev,Ewdev,dchi)
	cl.program('fft').DtSpectral(cl.q,shp3,None,Jdev,wdev,np.double(1.0))
	cl.program('fft').IFFT(cl.q,shp2,None,Jdev,np.int32(src.freqs))

	# Plasma current
	cl.program('paraxial').AddPlasmaCurrent(cl.q,shp3,None,Jdev,Atdev,nedev)
	cl.program('fft').FFT(cl.q,shp2,None,Jdev,np.int32(src.steps))

def load_source(z,cl,T,src,dchi,chi3,ionizer,return_dz=False):
	'''We are trying to advance q(z;w,kx,ky) = exp(-i*kz*z)*A(z;w,kx,ky).
	We have an equation in the form dq/dz = S(z,q).
	This loads src.J_dev with S(z,q), using q = src.qw_dev

	:param double z: the integration variable, propagation distance.
	:param cl_refs cl: OpenCL reference bundle
	:param TransverseModeTool T: used for transverse mode transformations
	:param source_cluster src: stores references to OpenCL buffers
	:param numpy.array dchi: nonuniform part of susceptibility in representation (w,x,y) as type complex
	:param double chi3: nonlinear susceptibility
	:param Ionization ionizer: class for encapsulating ionization models'''
	# Setup some shorthand
	shp3 = src.Aw_dev.shape
	qwdev = src.qw_dev.data
	Awdev = src.Aw_dev.data
	Jdev = src.J_dev.data
	kzdev = src.kz_dev.data
	k0dev = src.k0_dev.data
	dzdev = src.ne_dev.data # re-use ne for the step size estimate
	src.Aw_dev[...] = src.qw_dev
	cl.program('paraxial').PropagateLinear(cl.q,shp3,None,Awdev,kzdev,np.double(z))
	T.rspacex(src.Aw_dev)
	update_current(cl,src,dchi,chi3,ionizer)
	T.kspacex(src.J_dev)
	cl.program('paraxial').CurrentToODERHS(cl.q,shp3,None,Jdev,kzdev,k0dev,np.double(z))
	if return_dz:
		cl.program('paraxial').LoadModulus(cl.q,shp3,None,qwdev,dzdev)
		amin = src.rel_amin * pyopencl.array.max(src.ne_dev).get()
		cl.program('paraxial').LoadStepSize(cl.q,shp3,None,qwdev,Jdev,dzdev,np.double(src.L),np.double(src.dphi),np.double(amin))
		return pyopencl.array.min(src.ne_dev).get()

def finish_iteration(z,cl,src):
	'''Load qw and q00 with the updated potential (q=a in the new plane)

	:param double z: the integration variable, propagation distance.
	:param cl_refs cl: OpenCL reference bundle
	:param source_cluster src: stores references to OpenCL buffers'''
	# Setup some shorthand
	shp3 = src.Aw_dev.shape
	qwdev = src.qw_dev.data
	kzdev = src.kz_dev.data
	cl.program('paraxial').PropagateLinear(cl.q,shp3,None,qwdev,kzdev,np.double(z))
	src.q00_dev[...] = src.qw_dev

def finish(cl,T,src,dchi,chi3,ionizer):
	'''Finalize data at the end of the iterations and retrieve.

	:param cl_refs cl: OpenCL reference bundle
	:param TransverseModeTool T: used for transverse mode transformations
	:param source_cluster src: stores references to OpenCL buffers
	:param numpy.array dchi: nonuniform part of susceptibility in representation (w,x,y) as type complex
	:param double chi3: nonlinear susceptibility
	:param Ionization ionizer: class for encapsulating ionization models'''
	src.Aw_dev[...] = src.qw_dev
	T.rspacex(src.Aw_dev)
	update_current(cl,src,dchi,chi3,ionizer)
	return src.Aw_dev.get(),src.J_dev.get(),np.fft.fft(src.ne_dev.get(),axis=0)

def propagator(cl,ctool,a,chi,chi3,dens,ionizer,n0,ng,L):
	'''Advance a[w,x,y] to a new z plane using paraxial wave equation.
	Works in either Cartesian or cylindrical geometry.
	If cylindrical, intepret x as radial and y as azimuthal.

	:param cl_refs cl: OpenCL reference bundle
	:param CausticTool ctool: contains grid info and transverse mode tool
	:param double vwin: speed of the pulse frame variable
	:param numpy.array a: the vector potential in representation (w,x,y) as type complex
	:param numpy.array chi: the susceptibility at reference density with shape (Nw,) as type complex
	:param double chi3: the nonlinear susceptibility
	:param numpy.array dens: the density normalized to the reference with shape (Nx,Ny)
	:param Ionization ionizer: class for encapsulating ionization models
	:param double n0: the reference index of refraction
	:param double ng: the reference group index
	:param double L: distance to the new z plane'''
	w,x,y,ext = ctool.GetGridInfo()
	T = ctool.GetTransverseTool()
	N = (w.shape[0],x.shape[0],y.shape[0],1)
	w0 = w[int(N[0]/2)]

	# Form uniform contribution to susceptibility
	chi0 = np.copy(chi)
	chi0[np.where(np.real(chi)>0.0)] *= np.min(dens)
	chi0[np.where(np.real(chi)<0.0)] *= np.max(dens)
	# Form spatial perturbation to susceptibility - must be positive
	dchi = np.einsum('i,jk',chi,dens) - chi0[...,np.newaxis,np.newaxis]
	# Form the linear propagator
	dn = n0-ng
	kperp2 = T.kr2()
	fw = w**2*(1+chi0-ng**2)-w0**2*dn**2-2*w*w0*ng*dn
	kappa2 = fw[...,np.newaxis,np.newaxis] - kperp2[np.newaxis,...]
	k0 = (w0*dn+w*ng)[...,np.newaxis,np.newaxis]
	kz = 0.5*kappa2/k0

	T.AllocateDeviceMemory(a.shape)

	# Reorder frequencies for FFT processing
	a = np.fft.ifftshift(a,axes=0)
	kz = np.fft.ifftshift(kz,axes=0)
	k0 = np.fft.ifftshift(k0,axes=0)
	w = np.fft.ifftshift(w,axes=0)
	dchi = np.fft.ifftshift(dchi,axes=0)

	# Advance a(w,kx,ky) using RK4.
	# Define q(z;w,kx,ky) = exp(-i*kz*z)*a(z;w,kx,ky).  Note q=a in initial plane.
	src = source_cluster(cl.q,a,dens,w,kz,k0,L)
	T.kspacex(src.q00_dev)
	src.qw_dev[...] = src.q00_dev
	z = 0.0
	iterations = 0

	while z<L and (z-L)**2 > (L/1e6)**2:
		print('.',end='',flush=True)
		iterations += 1
		dz = load_source(0.0,cl,T,src,dchi,chi3,ionizer,return_dz=True) # load k1
		if z+dz>L:
			dz = L-z
		if (L-z)/dz > 10:
			print(int((L-z)/dz),end='',flush=True)
		src.q0_dev = src.q00_dev + dz*src.J_dev/6
		src.qw_dev = src.q00_dev + dz*src.J_dev/2
		load_source(0.5*dz,cl,T,src,dchi,chi3,ionizer) # load k2
		src.q0_dev += dz*src.J_dev/3
		src.qw_dev = src.q00_dev + dz*src.J_dev/2
		load_source(0.5*dz,cl,T,src,dchi,chi3,ionizer) # load k3
		src.q0_dev += dz*src.J_dev/3
		src.qw_dev = src.q00_dev + dz*src.J_dev
		load_source(dz,cl,T,src,dchi,chi3,ionizer) # load k4
		src.qw_dev = src.q0_dev + dz*src.J_dev/6
		finish_iteration(dz,cl,src)
		z += dz

	a,J,ne = finish(cl,T,src,dchi,chi3,ionizer)

	T.FreeDeviceMemory()

	# Sort frequencies
	a = np.fft.fftshift(a,axes=0)
	J = np.fft.fftshift(J,axes=0)
	ne = np.fft.fftshift(ne,axes=0)

	return a,J,ne,4*iterations

def track(cl,xp,eikonal,vg,vol_dict):
	'''Propagate paraxial waves using eikonal data as a boundary condition.
	The volume must be oriented so the polarization axis is x (linear polarization only).

	:param numpy.array xp: ray phase space with shape (bundles,rays,8)
	:param numpy.array eikonal: ray eikonal data with shape (bundles,4)
	:param numpy.array vg: ray group velocity with shape (bundles,rays,4)
	:param dictionary vol_dict: input file dictionary for the volume'''
	band = vol_dict['frequency band']
	size = (band[1]-band[0],) + vol_dict['size']
	N = vol_dict['wave grid']
	# N[3] is the number of diagnostic planes, including the initial plane
	diagnostic_steps = N[3]-1
	subcycles = vol_dict['subcycles']
	steps = diagnostic_steps*subcycles
	field_planes = steps + 1
	Vol = vol_dict['object']

	powersof2 = [2**i for i in range(32)]
	if N[0] not in powersof2:
		raise ValueError('Paraxial propagator requires 2**n w-nodes')
	if N[1] not in powersof2:
		raise ValueError('Paraxial propagator requires 2**n x-nodes')
	if N[2] not in powersof2:
		raise ValueError('Paraxial propagator requires 2**n y-nodes')
	try:
		window_speed = vol_dict['window speed']
	except:
		window_speed = 1.0
	try:
		chi3 = vol_dict['chi3']
	except:
		chi3 = 0.0

	# Capture the rays
	if vol_dict['wave coordinates']=='cartesian':
		field_tool = caustic_tools.FourierTool(N,band,(0,0,0),size[1:],cl)
	else:
		field_tool = caustic_tools.BesselBeamTool(N,band,(0,0,0),size[1:],cl)

	w_nodes,x1_nodes,x2_nodes,plot_ext = field_tool.GetGridInfo()
	A = np.zeros(N).astype(np.complex)
	J = np.zeros(N).astype(np.complex)
	ne = np.zeros(N).astype(np.complex)
	A0,dom3d = field_tool.GetBoundaryFields(xp[:,0,:],eikonal,1)
	n0 = np.mean(np.sqrt(np.einsum('...i,...i',xp[:,0,5:8],xp[:,0,5:8]))/xp[:,0,4])
	ng = 1.0/np.mean(np.sqrt(np.einsum('...i,...i',vg[:,0,1:4],vg[:,0,1:4])))

	# Setup the wave propagation domain
	chi = vol_dict['dispersion inside'].chi(w_nodes).astype(np.complex)
	try:
		ionizer = vol_dict['ionizer']
	except KeyError:
		ionizer = ionization.Ionization(1.0,1.0,1.0,1.0)
	dens_nodes = grid_tools.cell_centers(-size[3]/2,size[3]/2,steps)
	field_walls = grid_tools.cell_walls(dens_nodes[0],dens_nodes[-1],steps)
	diagnostic_walls = np.linspace(-size[3]/2,size[3]/2,N[3])
	dz = field_walls[1]-field_walls[0]
	Dz = diagnostic_walls[1]-diagnostic_walls[0]
	dom4d = np.concatenate((dom3d,[field_walls[0],field_walls[-1]]))

	# Step through the domain
	# Strategy to get density plane is to re-use ray gather system
	# This works as long as the shape of xp is (*,*,8)
	xp_eff = np.zeros((N[1],N[2],8))
	if vol_dict['wave coordinates']=='cartesian':
		xp_eff[...,1] = np.outer(x1_nodes,np.ones(N[2]))
		xp_eff[...,2] = np.outer(np.ones(N[1]),x2_nodes)
	else:
		xp_eff[...,1] = np.outer(x1_nodes,np.cos(x2_nodes))
		xp_eff[...,2] = np.outer(x1_nodes,np.sin(x2_nodes))
	A[...,0] = A0
	J[...,0] = 0.0
	ne[...,0] = 0.0
	for k in range(diagnostic_steps):
		print('Advancing to diagnostic plane',k+1)
		rhs_evals = 0
		for s in range(subcycles):
			if subcycles>1:
				if s==0:
					print('  subcycling *',end='',flush=True)
				else:
					print('*',end='',flush=True)
			xp_eff[...,3] = dens_nodes[k*subcycles + s]
			dens = vol_dict['object'].GetDensity(xp_eff)
			A0,J0,ne0,evals = propagator(cl,field_tool,A0,chi,chi3,dens,ionizer,n0,ng,dz)
			rhs_evals += evals
		print('',rhs_evals,'evaluations of j(w,kx,ky)')
		A[...,k+1] = A0
		J[...,k+1] = J0
		ne[...,k+1] = ne0

	# Finish by relaunching rays and returning wave data
	field_tool.RelaunchRays(xp,eikonal,vg,A[...,-1],size[3])
	return A,J,ne,dom4d
