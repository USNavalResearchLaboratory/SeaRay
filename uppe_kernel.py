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
	def __init__(self,queue,a,ng,w,kz,kg,L,NL_band):
		bandwidth = w[-1] - w[0] + w[1] - w[0]
		self.dt = 2*np.pi/bandwidth
		self.dw = w[1] - w[0]
		self.L = L
		self.freqs = a.shape[0]
		self.steps = self.freqs*2-2
		self.transverse_shape = (a.shape[1],a.shape[2])
		self.freq_domain_shape = (self.freqs,) + self.transverse_shape
		self.time_domain_shape = (self.steps,) + self.transverse_shape
		self.dphi = 0.1
		self.rel_amin = 1e-2
		# Device arrays
		self.qw_dev = pyopencl.array.to_device(queue,a)
		self.qi_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.complex)
		self.qf_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.complex)
		self.Aw_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.complex)
		self.At_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.Et_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.ng_dev= pyopencl.array.to_device(queue,ng)
		self.w_dev = pyopencl.array.to_device(queue,w)
		self.kz_dev = pyopencl.array.to_device(queue,kz)
		self.kg_dev = pyopencl.array.to_device(queue,kg)
		self.ne_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.Jt_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.Jw_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.complex)
		self.dz_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.double)
		# Setup filters
		filter = np.ones(self.freqs).astype(np.double)
		cut_on_idx = int(NL_band[0]/self.dw)
		cut_off_idx = int(NL_band[1]/self.dw)
		filter[:cut_on_idx] = 0.0
		filter[cut_off_idx:] = 0.0
		self.NLFilt_dev = pyopencl.array.to_device(queue,filter)
		filter = np.ones(self.freqs).astype(np.double)
		filter[:4] = 0.0
		self.conditioner_dev = pyopencl.array.to_device(queue,filter)

def condition_field(cl,src):
	'''Impose filters or other conditions on field'''
	sw = src.freq_domain_shape
	qwdev = src.qw_dev.data
	cond = src.conditioner_dev.data
	# Filter DC and very low frequencies
	cl.program('fft').Filter(cl.q,sw,None,qwdev,cond)

def update_current(cl,src,dchi,chi3,ionizer):
	'''Get current density from A[w,x,y]'''
	ionizer.ResetParameters(timestep=src.dt)

	# Setup some shorthand
	s2 = src.transverse_shape
	st = src.time_domain_shape
	sw = src.freq_domain_shape
	wdev = src.w_dev.data
	Awdev = src.Aw_dev.data
	Atdev = src.At_dev.data
	Etdev = src.Et_dev.data
	nedev = src.ne_dev.data
	ngdev = src.ng_dev.data
	Jtdev = src.Jt_dev.data
	Jwdev = src.Jw_dev.data

	# Form time domain fields
	# N.b. time index runs backwards due to FFT conventions
	src.Jw_dev[...] = src.Aw_dev # use Jw as temporary so Aw is not destroyed
	cl.program('fft').IRFFT(cl.q,s2,None,Jwdev,Atdev,np.int32(src.freqs))
	cl.program('fft').DtSpectral(cl.q,sw,None,Jwdev,wdev,np.double(-1.0))
	cl.program('fft').IRFFT(cl.q,s2,None,Jwdev,Etdev,np.int32(src.freqs))

	# Accumulate the source terms
	# First handle plasma formation
	ionizer.RateCL(cl,st,nedev,Etdev,False)
	ionizer.GetPlasmaDensityCL(cl,st,nedev,ngdev)

	# Kerr effect
	src.Jt_dev = src.Et_dev**2 # use Jt and Jw as temporaries in computing <E.E>
	cl.program('fft').RFFT(cl.q,s2,None,Jtdev,Jwdev,np.int32(src.steps))
	cl.program('fft').Filter(cl.q,sw,None,Jwdev,src.NLFilt_dev.data)
	cl.program('fft').IRFFT(cl.q,s2,None,Jwdev,Jtdev,np.int32(src.freqs))
	cl.program('uppe').SetKerrPolarization(cl.q,st,None,Jtdev,Etdev,np.double(chi3))
	# Switch to current density
	cl.program('fft').RFFT(cl.q,s2,None,Jtdev,Jwdev,np.int32(src.steps))
	cl.program('fft').DtSpectral(cl.q,sw,None,Jwdev,wdev,np.double(1.0))
	cl.program('fft').IRFFT(cl.q,s2,None,Jwdev,Jtdev,np.int32(src.freqs))

	# Plasma current
	cl.program('fft').SoftTimeWindow(cl.q,st,None,nedev,np.int32(4),np.int32(64))
	cl.program('uppe').AddPlasmaCurrent(cl.q,st,None,Jtdev,Atdev,nedev)
	cl.program('fft').RFFT(cl.q,s2,None,Jtdev,Jwdev,np.int32(src.steps))

def load_source(z,cl,T,src,dchi,chi3,ionizer,return_dz=False,dzmin=1.0):
	'''We are trying to advance q(z;w,kx,ky) = exp(-i*kz*z)*A(z;w,kx,ky).
	We have an equation in the form dq/dz = S(z,q).
	This loads src.Jw_dev with S(z,q), using src.qw_dev

	:param double z: the integration variable, propagation distance.
	:param cl_refs cl: OpenCL reference bundle
	:param TransverseModeTool T: used for transverse mode transformations
	:param source_cluster src: stores references to OpenCL buffers particular to UPPE
	:param numpy.array dchi: nonuniform part of susceptibility in representation (w,x,y) as type complex
	:param double chi3: nonlinear susceptibility
	:param Ionization ionizer: class for encapsulating ionization models
	:param bool return_dz: return step size estimate if true'''
	# Setup some shorthand
	sw = src.freq_domain_shape
	kzdev = src.kz_dev.data
	kgdev = src.kg_dev.data
	Awdev = src.Aw_dev.data
	qwdev = src.qw_dev.data
	Jwdev = src.Jw_dev.data
	dzdev = src.dz_dev.data
	src.Aw_dev[...] = src.qw_dev
	cl.program('uppe').PropagateLinear(cl.q,sw,None,Awdev,kzdev,kgdev,np.double(z))
	T.rspacex(src.Aw_dev)
	update_current(cl,src,dchi,chi3,ionizer)
	T.kspacex(src.Jw_dev)
	cl.program('uppe').CurrentToODERHS(cl.q,sw,None,Jwdev,kzdev,kgdev,np.double(z))
	if return_dz:
		if np.isnan(pyopencl.array.sum(src.qw_dev).get()):
			print('!',end='',flush=True)
		cl.program('paraxial').LoadModulus(cl.q,sw,None,qwdev,dzdev)
		amin = src.rel_amin * pyopencl.array.max(src.dz_dev).get()
		cl.program('paraxial').LoadStepSize(cl.q,sw,None,qwdev,Jwdev,dzdev,np.double(src.L),np.double(src.dphi),np.double(amin))
		dz = pyopencl.array.min(src.dz_dev).get()
		if dz<dzmin:
			dz = dzmin
		if z+dz>src.L:
			dz = src.L-z
		else:
			print('.',end='',flush=True)
		if (src.L-z)/dz > 10:
			print(int((src.L-z)/dz),end='',flush=True)
		return dz

def finish(cl,T,src,dchi,chi3,ionizer):
	'''Finalize data at the end of the iterations and retrieve.

	:param cl_refs cl: OpenCL reference bundle
	:param TransverseModeTool T: used for transverse mode transformations
	:param source_cluster src: stores references to OpenCL buffers
	:param numpy.array dchi: nonuniform part of susceptibility in representation (w,x,y) as type complex
	:param double chi3: nonlinear susceptibility
	:param Ionization ionizer: class for encapsulating ionization models'''
	# Setup some shorthand
	sw = src.Aw_dev.shape
	Awdev = src.Aw_dev.data
	kzdev = src.kz_dev.data
	kgdev = src.kg_dev.data
	src.Aw_dev[...] = src.qw_dev
	cl.program('uppe').PropagateLinear(cl.q,sw,None,Awdev,kzdev,kgdev,np.double(src.L))
	T.rspacex(src.Aw_dev)
	update_current(cl,src,dchi,chi3,ionizer)
	return src.Aw_dev.get(),src.Jw_dev.get(),np.fft.rfft(src.ne_dev.get(),axis=0)

def propagator(cl,ctool,vwin,a,chi,chi3,dens,ionizer,NL_band,L,dzmin):
	'''Advance a[w,x,y] to a new z plane using UPPE method.
	Works in either Cartesian or cylindrical geometry.
	If cylindrical, intepret x as radial and y as azimuthal.

	:param cl_refs cl: OpenCL reference bundle
	:param CausticTool ctool: contains grid info and transverse mode tool
	:param double vwin: speed of the pulse frame variable
	:param numpy.array a: the vector potential in representation (w,x,y) as type complex
	:param numpy.array chi: the susceptibility at reference density with shape (Nw,) as type complex
	:param numpy.array dens: the density normalized to the reference with shape (Nx,Ny)
	:param Ionization ionizer: class for encapsulating ionization models
	:param tuple NL_band: filter applied to nonlinear source terms
	:param double L: distance to the new z plane
	:param doulbe dzmin: minimum step size'''
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

	T.AllocateDeviceMemory(a.shape)

	# Advance a(w,kx,ky) using RK4.
	# Define q(z;w,kx,ky) = exp(-i*kz*z)*a(z;w,kx,ky).  Note q=a in initial plane.

	src = source_cluster(cl.q,a,dens,w,kz,kGalileo,L,NL_band)
	T.kspacex(src.qw_dev)
	z = 0.0
	iterations = 0
	sw = src.qw_dev.shape
	qwdev = src.qw_dev.data
	qidev = src.qi_dev.data
	qfdev = src.qf_dev.data
	Jwdev = src.Jw_dev.data

	while z<L and (z-L)**2 > (L/1e6)**2:
		src.qi_dev[...] = src.qw_dev
		src.qf_dev[...] = src.qw_dev

		dz = load_source(z,cl,T,src,dchi,chi3,ionizer,return_dz=True,dzmin=dzmin) # load k1
		cl.program('uppe').RKStage(cl.q,sw,None,qwdev,qidev,qfdev,Jwdev,np.double(dz/6),np.double(dz/2))
		condition_field(cl,src)

		load_source(z+0.5*dz,cl,T,src,dchi,chi3,ionizer) # load k2
		cl.program('uppe').RKStage(cl.q,sw,None,qwdev,qidev,qfdev,Jwdev,np.double(dz/3),np.double(dz/2))
		condition_field(cl,src)

		load_source(z+0.5*dz,cl,T,src,dchi,chi3,ionizer) # load k3
		cl.program('uppe').RKStage(cl.q,sw,None,qwdev,qidev,qfdev,Jwdev,np.double(dz/3),np.double(dz))
		condition_field(cl,src)

		load_source(z+dz,cl,T,src,dchi,chi3,ionizer) # load k4
		cl.program('uppe').RKFinish(cl.q,sw,None,qwdev,qfdev,Jwdev,np.double(dz/6))
		condition_field(cl,src)

		iterations += 1
		z += dz

	a,J,ne = finish(cl,T,src,dchi,chi3,ionizer)

	T.FreeDeviceMemory()

	return a,J,ne,4*iterations

def track(cl,xp,eikonal,vg,vol_dict):
	'''Propagate unidirectional fully dispersive waves using eikonal data as a boundary condition.
	The volume must be oriented so the polarization axis is x (linear polarization only).

	:param numpy.array xp: ray phase space with shape (bundles,rays,8)
	:param numpy.array eikonal: ray eikonal data with shape (bundles,4)
	:param numpy.array vg: ray group velocity with shape (bundles,rays,4)
	:param dictionary vol_dict: input file dictionary for the volume'''
	band = vol_dict['frequency band']
	NL_band = vol_dict['nonlinear band']
	size = (band[1]-band[0],) + vol_dict['size']
	N = vol_dict['wave grid']
	# N[3] is the number of diagnostic planes, including the initial plane
	diagnostic_steps = N[3]-1
	subcycles = vol_dict['subcycles']
	steps = diagnostic_steps*subcycles
	field_planes = steps + 1

	powersof2 = [2**i for i in range(32)]
	if N[0]-1 not in powersof2:
		raise ValueError('UPPE propagator requires 2**n+1 w-nodes')
	if N[1] not in powersof2:
		raise ValueError('UPPE propagator requires 2**n x-nodes')
	if N[2] not in powersof2:
		raise ValueError('UPPE propagator requires 2**n y-nodes')
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
	chi = vol_dict['dispersion inside'].chi(w_nodes).astype(np.complex)
	try:
		ionizer = vol_dict['ionizer']
	except KeyError:
		ionizer = ionization.Ionization(1.0,1.0,1.0,1.0)
	dens_nodes = grid_tools.cell_centers(-size[3]/2,size[3]/2,steps)
	field_walls = grid_tools.cell_walls(dens_nodes[0],dens_nodes[-1],steps)
	diagnostic_walls = np.linspace(-size[3]/2,size[3]/2,N[3])
	dzmin = vol_dict['minimum step']
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
			A0,J0,ne0,evals = propagator(cl,field_tool,window_speed,A0,chi,chi3,dens,ionizer,NL_band,dz,dzmin)
			rhs_evals += evals
		print('',rhs_evals,'evaluations of j(w,kx,ky)')
		A[...,k+1] = A0
		J[...,k+1] = J0
		ne[...,k+1] = ne0

	# Finish by relaunching rays and returning UPPE data
	field_tool.RelaunchRays(xp,eikonal,vg,A[...,-1],size[3])
	return A,J,ne,dom4d
