'''
Module: :samp:`uppe_kernel`
-------------------------------

This module is the primary computational engine for advancing unidirectional pulse propagation equations (UPPE).
'''
import numpy as np
import scipy.integrate
import pyopencl
import caustic_tools
import grid_tools
import ionization

class source_cluster:
	'''This class gathers references to host and device storage
	that are used during the ODE integrator right-hand-side evaluation.'''
	def __init__(self,queue,a,ng,w,kz):
		bandwidth = w[-1] - w[0] + w[1] - w[0]
		self.dt = 2*np.pi/bandwidth
		self.freqs = a.shape[0]
		self.steps = self.freqs*2-2
		self.transverse_shape = (a.shape[1],a.shape[2])
		self.freq_domain_shape = (self.freqs,) + self.transverse_shape
		self.time_domain_shape = (self.steps,) + self.transverse_shape
		# Host arrays
		self.kz = kz
		self.Et = np.zeros(self.time_domain_shape)
		self.J = np.zeros(self.freq_domain_shape).astype(np.complex)
		self.P = np.zeros(self.time_domain_shape)
		self.ne = np.zeros(self.time_domain_shape)
		# Device arrays
		self.Ew_dev = pyopencl.array.to_device(queue,a)
		self.ng_dev= pyopencl.array.to_device(queue,ng)
		self.w_dev = pyopencl.array.to_device(queue,w)
		self.kz_dev = pyopencl.array.to_device(queue,kz)
		self.ne_dev = pyopencl.array.to_device(queue,self.ne)
		self.P_dev = pyopencl.array.to_device(queue,self.P)
		self.J_dev = pyopencl.array.to_device(queue,self.J)
		self.Et_dev = pyopencl.array.to_device(queue,self.Et)
	def SetField(self,queue,a):
		'''Send the vector potential to the compute device

		:param class queue: OpenCL queue
		:param numpy.array a: vector potential in representation (w,x,y)'''
		# The device array will be converted to electric field first thing upon calling update_current
		self.Ew_dev = pyopencl.array.to_device(queue,a)
	def GetCurrent(self,queue):
		'''Retrieve current density from compute device

		:param class queue: OpenCL queue
		:returns: current density in representation (w,x,y)'''
		return self.J_dev.get()
	def GetPlasma(self,queue):
		'''Retrieve electron density from compute device

		:param class queue: OpenCL queue
		:returns: electron density in representation (w,x,y)'''
		return np.fft.rfft(self.ne_dev.get(),axis=0)

def update_current(cl,src,dchi,chi3,ionizer):
	'''Get current density from a[w,x,y]'''
	ionizer.ResetParameters(timestep=src.dt)

	# Form electric field in the time domain
	# N.b. time index runs backwards due to FFT conventions
	cl.program('uppe').PotentialToField(cl.q,src.freq_domain_shape,None,src.Ew_dev.data,src.w_dev.data)
	cl.program('fft').IRFFT(cl.q,src.transverse_shape,None,src.Ew_dev.data,src.Et_dev.data,np.int32(src.freqs))

	# Accumulate the source terms
	# First handle plasma formation
	ionizer.InstantaneousRateCL(src.ne_dev,src.Et_dev,cl.q,cl.program('uppe').ComputeRate)
	ionizer.GetPlasmaDensityCL(src.ng_dev,src.ne_dev,cl.q,cl.program('uppe').ComputePlasmaDensity)
	cl.program('uppe').ComputePlasmaPolarization(cl.q,src.transverse_shape,None,src.P_dev.data,
		src.Et_dev.data,src.ne_dev.data,np.double(0.01),np.double(0.1),np.double(src.dt),np.int32(src.steps))

	# Nonuniform susceptibility
	#cl.program('uppe').AddNonuniformPolarization(cl.q,src.freq_domain_shape,None,src.P_dev,src.Ew_dev.data,dchi_dev.data)

	# Kerr effect
	#cl.program('uppe').AddKerrPolarization(cl.q,src.time_domain_shape,None,src.P_dev.data,src.Et_dev.data,np.double(chi3))

	# Get the current density in the frequency domain
	cl.program('fft').RFFT(cl.q,src.transverse_shape,None,src.P_dev.data,src.J_dev.data,np.int32(src.steps))
	cl.program('uppe').PolarizationToCurrent(cl.q,src.freq_domain_shape,None,src.J_dev.data,src.w_dev.data)

def uppe_rhs(z,q,cl,T,refs,src,dchi,chi3,ionizer):
	'''During UPPE step we are trying to advance q(z;w,kx,ky) = exp(-i*kz*z)*A(z;w,kx,ky).
	We have an equation in the form dq/dz = S(z,q).  This is a callback that allows some
	ODE integrator to evaluate S(z,q).

	:param double z: the integration variable, propagation distance.
	:param numpy.array q: the reduced vector potential in representation (w,kx,ky)
	:param cl_refs cl: OpenCL reference bundle
	:param TransverseModeTool T: used for transverse mode transformations
	:param array refs: tuple of references to device data saved by the tool ``T``
	:param source_cluster src: stores references to OpenCL buffers particular to UPPE
	:param numpy.array dchi: nonuniform part of susceptibility in representation (w,x,y)
	:param double chi3: nonlinear susceptibility
	:param Ionization ionizer: class for encapsulating ionization models'''
	a = np.exp(1j*src.kz*z)*q.reshape(src.kz.shape)
	src.SetField(cl.q,a)
	T.rspacex(refs[0],refs[1],refs[2],src.Ew_dev)
	update_current(cl,src,dchi,chi3,ionizer)
	T.kspacex(refs[0],refs[1],refs[2],src.J_dev)
	J = src.GetCurrent(cl.q)
	S = 0.5j*np.exp(-1j*src.kz*z)*J/np.real(src.kz)
	return S.flatten()

def propagator(cl,ctool,vwin,a,chi,chi3,dens,ionizer,dz):
	'''Advance a[w,x,y] to a new z plane using UPPE method.
	Works in either Cartesian or cylindrical geometry.
	If cylindrical, intepret x as radial and y as azimuthal.

	:param cl_refs cl: OpenCL reference bundle
	:param CausticTool ctool: contains grid info and transverse mode tool
	:param double vwin: speed of the pulse frame variable
	:param numpy.array a: the vector potential in representation (w,x,y)
	:param numpy.array chi: the susceptibility at reference density with shape (Nw,)
	:param numpy.array dens: the density normalized to the reference with shape (Nx,Ny)
	:param Ionization ionizer: class for encapsulating ionization models
	:param double dz: step size'''
	w,x,y,ext = ctool.GetGridInfo()
	T = ctool.GetTransverseTool()
	N = (w.shape[0],x.shape[0],y.shape[0],1)
	if w[0]!=0.0:
		raise ValueError('UPPE propagator requires frequency range [0,...]')
	if N[0]==1:
		raise ValueError('UPPE propagator cannot be used for monochromatic radiation.')

	# Form uniform contribution to susceptibility
	chi0 = np.copy(chi)
	chi0[np.where(np.real(chi)>0.0)] *= np.min(dens)
	chi0[np.where(np.real(chi)<0.0)] *= np.max(dens)
	# Form spatial perturbation to susceptibility - must be positive
	dchi = np.einsum('i,jk',chi,dens) - chi0[...,np.newaxis,np.newaxis]
	# Form the linear phase advance; phase advance = kz*dz
	w2n2 = w**2*(1+chi0)
	kz = np.sqrt(0j + w2n2[...,np.newaxis,np.newaxis] - T.kr2()[np.newaxis,...])
	kz[np.where(np.imag(kz)<0)] *= -1
	kz[np.where(np.real(kz)==0.0)] = 1.0
	# The Galilean pulse frame transformation; phase advance = kGalileo*dz
	kGalileo = -w[...,np.newaxis,np.newaxis]/vwin

	# Advance a(w,kx,ky) using ODE solver.
	# Define q(z;w,kx,ky) = exp(-i*kz*z)*a(z;w,kx,ky).  Note q=a in initial plane.
	a = T.kspace(a)
	src = source_cluster(cl.q,a,dens,w,kz)
	refs = T.GetDeviceRefs(a)
	dqdz = lambda z,q : uppe_rhs(z,q,cl,T,refs,src,dchi,chi3,ionizer)
	sol = scipy.integrate.solve_ivp(dqdz,[0.0,dz],a.flatten(),t_eval=[dz],rtol=1e-3,atol=1e-5)

	# Get the diagnostic quantities at the new z-plane
	a = T.rspace(sol.y[...,-1].reshape(kz.shape)*np.exp(1j*(kz+kGalileo)*dz))
	src.SetField(cl.q,a)
	update_current(cl,src,dchi,chi3,ionizer)
	return a,src.GetCurrent(cl.q),src.GetPlasma(cl.q),sol.nfev

def track(cl,xp,eikonal,vg,vol_dict):
	'''Propagate unidirectional fully dispersive waves using eikonal data as a boundary condition.
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

	# Setup the wave propagation domain
	chi = vol_dict['dispersion inside'].chi(w_nodes)
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
					print('  subcycling .',end='',flush=True)
				else:
					print('.',end='',flush=True)
			xp_eff[...,3] = dens_nodes[k*subcycles + s]
			dens = vol_dict['object'].GetDensity(xp_eff)
			A0,J0,ne0,evals = propagator(cl,field_tool,window_speed,A0,chi,chi3,dens,ionizer,dz)
			A0[:4,...] = 0.0
			rhs_evals += evals
		print('',rhs_evals,'evaluations of j(w,kx,ky)')
		A[...,k+1] = A0
		J[...,k+1] = J0
		ne[...,k+1] = ne0

	# Return the wave amplitude
	# Rays are re-launched externally
	return A,J,ne,dom4d
