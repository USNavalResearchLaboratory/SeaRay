'''
Module: :samp:`uppe_kernel`
-------------------------------

This module is the primary computational engine for advancing unidirectional pulse propagation equations (UPPE).
The call stack is as follows:

* `track` is called externally
* `track` repeatedly calls `propagator` to advance to the end of the volume
* `propagator` advances to the next diagnostic plane by repeatedly calling

	- `load_source`
	- RK4 functions
	- `condition_field`

* `load_source` has these steps

	- propagate linearly
	- transform (kx,ky) -> (x,y)
	- call `update_current`
	- transform (x,y) -> (kx,ky)
	- estimate step size if requested
'''
import logging
import numpy as np
import pyopencl
import caustic_tools
import grid_tools
import ionization
import rotations

class Material:
	'''This class manages host and device storage describing the material's density and susceptibility.
	It also computes the generator of linear axial translations, kz.'''
	def __init__(self,queue,ctool,N,coords,vol,chi,chi3,nref,vg):
		''':param class queue: OpenCL command queue
		:param CausticTool ctool: contains grid info and transverse mode tool
		:param tuple N: dimensions of the field grid
		:param str coords: if string not "cartesian" cylindrical is used
		:param class vol: the volume object
		:param numpy.array chi: the susceptibility at reference density with shape (Nw,) as type complex
		:param double chi3: the nonlinear susceptibility at the reference density
		:param double nref: the reference density in plasma units
		:param double vg: the window velocity'''
		w,x1,x2,ext = ctool.GetGridInfo()
		self.q = queue
		self.T = ctool.GetTransverseTool()
		self.N = N
		self.vol = vol
		self.chi = chi
		self.w = w
		self.redchi3 = chi3/nref
		self.n_ref = nref
		self.vg = vg
		self.xp_eff = np.zeros((N[1],N[2],8))
		# Strategy to get density plane is to re-use ray gather system
		# This works as long as the shape of xp is (*,*,8)
		if coords=='cartesian':
			self.xp_eff[...,1] = np.outer(x1,np.ones(N[2]))
			self.xp_eff[...,2] = np.outer(np.ones(N[1]),x2)
		else:
			self.xp_eff[...,1] = np.outer(x1,np.cos(x2))
			self.xp_eff[...,2] = np.outer(x1,np.sin(x2))
		self.ng_dev = pyopencl.array.empty(queue,N[1:3],np.double)
		self.kz_dev = pyopencl.array.empty(queue,N[:3],np.cdouble)
		self.kg_dev = pyopencl.array.empty(queue,(N[0],1,1),np.double)
	def UpdateMaterial(self,z):
		self.xp_eff[...,3] = z
		# Get the density relative to the reference density for scaling of chi
		dens = self.vol.GetRelDensity(self.xp_eff)
		# Form uniform contribution to susceptibility, chi0(w), which can be different for each w
		chi0 = np.copy(self.chi)
		# For w such that chi(w)>0 we choose the smallest density
		# Here it is useful to remember that np.where is selecting frequencies, not spatial points.
		chi0[np.where(np.real(self.chi)>0.0)] *= np.min(dens)
		# Fow w such that chi(w)<0 we choose the largest density
		chi0[np.where(np.real(self.chi)<0.0)] *= np.max(dens)
		# Form spatial perturbation to susceptibility - must be positive
		self.dchi = np.einsum('i,jk',self.chi,dens) - chi0[...,np.newaxis,np.newaxis]
		# Form the linear propagator
		w2n2 = self.w**2*(1+chi0)
		kz = np.sqrt(0j + w2n2[...,np.newaxis,np.newaxis] - self.T.kr2()[np.newaxis,...])
		kz[np.where(np.imag(kz)<0)] *= -1
		kz[np.where(np.real(kz)==0.0)] = 1.0
		# The Galilean pulse frame transformation; phase advance = kGalileo*dz
		kGalileo = -self.w[...,np.newaxis,np.newaxis]/self.vg
		# Put gas density in simulation units for ionization calculations
		self.ng_dev.set(self.n_ref*dens,queue=self.q)
		self.kz_dev.set(kz,queue=self.q)
		self.kg_dev.set(kGalileo,queue=self.q)

class Source:
	'''This class gathers references to host and device storage
	that are used during the ODE integrator right-hand-side evaluation.'''
	def __init__(self,queue,a,w,zi,L,NL_band):
		self.dt = np.pi/np.max(w)
		self.dw = w[1] - w[0]
		self.zi = zi
		self.L = L
		self.freqs = a.shape[0]
		self.steps = self.freqs*2-2
		self.transverse_shape = (a.shape[1],a.shape[2])
		self.freq_domain_shape = (self.freqs,) + self.transverse_shape
		self.time_domain_shape = (self.steps,) + self.transverse_shape
		self.dphi = 0.4
		self.rel_amin = 1e-2
		# Device arrays
		self.qw_dev = pyopencl.array.to_device(queue,a)
		self.qi_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.cdouble)
		self.qf_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.cdouble)
		self.Ew_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.cdouble)
		self.Et_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.w_dev = pyopencl.array.to_device(queue,w)
		self.scratchw_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.cdouble)
		self.ne_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.chi_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.Jt_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.Jw_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.cdouble)
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

def update_current(cl,src,mat,ionizer: ionization.Ionizer,rotator: rotations.Rotator):
	'''Get current density from A[w,x,y]'''

	# Setup some shorthand
	s2 = src.transverse_shape
	st = src.time_domain_shape
	sw = src.freq_domain_shape
	wdev = src.w_dev.data
	scratch = src.scratchw_dev.data
	Ewdev = src.Ew_dev.data
	Etdev = src.Et_dev.data
	nedev = src.ne_dev.data
	chidev = src.chi_dev.data
	ngasdev = mat.ng_dev.data
	Jtdev = src.Jt_dev.data
	Jwdev = src.Jw_dev.data

	# Form time domain fields
	# N.b. time index runs backwards due to FFT conventions
	cl.program('fft').IRFFT(cl.q,s2,None,Ewdev,Etdev,np.int32(src.freqs))

	# Accumulate the source terms
	# First handle plasma formation
	if ionizer==None:
		src.ne_dev.fill(0.0,queue=cl.q)
		src.Jw_dev.fill(0.0,queue=cl.q)
	else:
		ionizer.RateCL(cl,st,nedev,Etdev,False)
		ionizer.GetPlasmaDensityCL(cl,st,nedev,ngasdev,src.dt)
		cl.program('fft').SoftTimeWindow(cl.q,st,None,nedev,np.int32(4),np.int32(64))
		src.Jt_dev[...] = (src.ne_dev*src.Et_dev).copy(queue=cl.q)
		cl.program('fft').RFFT(cl.q,s2,None,Jtdev,Jwdev,np.int32(src.steps))
		cl.program('fft').iDtSpectral(cl.q,sw,None,Jwdev,wdev,np.double(0.1),np.double(1.0))

	# Add currents due to Raman
	src.chi_dev.fill(0.0,queue=cl.q)
	if rotator!=None:
		rotator.AddChi(cl,st,chidev,Etdev,ngasdev,src.dt,True)

	# Kerr effect
	cl.program('uppe').AddKerrChi(cl.q,st,None,chidev,Etdev,ngasdev,np.double(mat.redchi3))

	# Form polarization, then current, then filter
	cl.program('uppe').SourcePolarization(cl.q,st,None,Jtdev,chidev,Etdev)
	cl.program('fft').RFFT(cl.q,s2,None,Jtdev,scratch,np.int32(src.steps))
	cl.program('fft').DtSpectral(cl.q,sw,None,scratch,wdev,np.double(1.0))
	src.Jw_dev[...] = (src.Jw_dev+src.scratchw_dev).copy(queue=cl.q)
	cl.program('fft').Filter(cl.q,sw,None,Jwdev,src.NLFilt_dev.data)

def load_source(z0,z,cl,T,src,mat,ionizer,rotator,return_dz=False,dzmin=1.0):
	'''We have an equation in the form dq/dz = S(z,q).
	This loads src.Jw_dev with S(z,q), using src.qw_dev

	:param double z0: global coordinate of the start of the subcycle region
	:param double z: local coordinate of evaluation point referenced to z0
	:param cl_refs cl: OpenCL reference bundle
	:param TransverseModeTool T: used for transverse mode transformations
	:param Source src: stores references to OpenCL buffers used in RHS evaluations
	:param Material mat: stores references to OpenCL buffers used in Linear propagation
	:param Ionization ionizer: class for encapsulating ionization models
	:param Rotator rotator: class for encapsulating molecular rotations
	:param bool return_dz: return step size estimate if true'''
	# Setup some shorthand
	sw = src.freq_domain_shape
	wdev = src.w_dev.data
	kzdev = mat.kz_dev.data
	kgdev = mat.kg_dev.data
	Ewdev = src.Ew_dev.data
	qwdev = src.qw_dev.data
	Jwdev = src.Jw_dev.data
	dzdev = src.dz_dev.data
	src.Ew_dev[...] = src.qw_dev.copy(queue=cl.q)
	cl.program('uppe').PropagateLinear(cl.q,sw,None,Ewdev,kzdev,kgdev,np.double(z0-src.zi+z))
	cl.program('uppe').VectorPotentialToElectricField(cl.q,sw,None,Ewdev,kzdev)
	T.rspacex(src.Ew_dev)
	update_current(cl,src,mat,ionizer,rotator)
	T.kspacex(src.Jw_dev)
	cl.program('uppe').CurrentToODERHS(cl.q,sw,None,Jwdev,kzdev,kgdev,np.double(z0-src.zi+z))
	if return_dz:
		cl.program('paraxial').LoadModulus(cl.q,sw,None,qwdev,dzdev)
		if np.isnan(pyopencl.array.sum(src.dz_dev,queue=cl.q).get(queue=cl.q)):
			print('!',end='',flush=True)
		amin = src.rel_amin * pyopencl.array.max(src.dz_dev,queue=cl.q).get(queue=cl.q)
		cl.program('paraxial').LoadStepSize(cl.q,sw,None,qwdev,Jwdev,dzdev,np.double(src.L),np.double(src.dphi),np.double(amin))
		dz = pyopencl.array.min(src.dz_dev,queue=cl.q).get(queue=cl.q)
		if dz<dzmin:
			dz = dzmin
		if z+dz>src.L:
			dz = src.L-z
		else:
			print('.',end='',flush=True)
		if (src.L-z)/dz > 10:
			print(int((src.L-z)/dz),end='',flush=True)
		return dz

def finish(cl,T,src,mat,ionizer,rotator,zf):
	'''Finalize data at the end of the iterations and retrieve.

	:param cl_refs cl: OpenCL reference bundle
	:param TransverseModeTool T: used for transverse mode transformations
	:param source_cluster src: stores references to OpenCL buffers
	:param Material mat: stores references to OpenCL buffers used in Linear propagation
	:param Ionization ionizer: class for encapsulating ionization models
	:param Rotator rotator: class for encapsulating molecular rotations
	:param double zf: global coordinate of the final position'''
	# Setup some shorthand
	sw = src.Ew_dev.shape
	qwdev = src.qw_dev.data
	Ewdev = src.Ew_dev.data
	kzdev = mat.kz_dev.data
	kgdev = mat.kg_dev.data
	cl.program('uppe').PropagateLinear(cl.q,sw,None,qwdev,kzdev,kgdev,np.double(zf-src.zi))
	T.rspacex(src.qw_dev)
	src.Ew_dev[...] = src.qw_dev.copy(queue=cl.q)
	cl.program('uppe').VectorPotentialToElectricField(cl.q,sw,None,Ewdev,kzdev)
	update_current(cl,src,mat,ionizer,rotator)
	return src.qw_dev.get(queue=cl.q), \
		src.Jw_dev.get(queue=cl.q), \
		np.fft.rfft(src.chi_dev.get(queue=cl.q),axis=0), \
		np.fft.rfft(src.ne_dev.get(queue=cl.q),axis=0)

def propagator(cl,ctool,vg,a,mat,ionizer,rotator,NL_band,subcycles,zi,zf,dzmin):
	'''Advance a[mu,w,x,y] to a new z plane using UPPE method; mu = 4-vector index.
	Works in either Cartesian or cylindrical geometry.
	If cylindrical, intepret x as radial and y as azimuthal.
	The RK4 stepper is advancing q(z;w,kx,ky) = exp(-i*(kz+kg)*z)*a(z;w,kx,ky).
	In the time domain the kg(w) wavevector results in a(t-vg*z,x,y), i.e., gives us the pulse frame.

	:param cl_refs cl: OpenCL reference bundle
	:param CausticTool ctool: contains grid info and transverse mode tool
	:param double vg: the window velocity
	:param numpy.array a: the vector potential in representation (mu,w,x,y) as type complex
	:param Material mat: stores references to OpenCL buffers used in Linear propagation
	:param Ionization ionizer: class for encapsulating ionization models
	:param Rotator rotator: class for encapsulating molecular rotations
	:param tuple NL_band: filter applied to nonlinear source terms
	:param int subcycles: number of density evaluations along the path
	:param double zi: initial z position
	:param double zf: final z position
	:param doulbe dzmin: minimum step size'''
	w,x,y,ext = ctool.GetGridInfo()
	T = ctool.GetTransverseTool()
	N = (w.shape[0],x.shape[0],y.shape[0],1)
	if w[0]!=0.0:
		raise ValueError('UPPE propagator requires frequency range [0,...]')
	if N[0]==1:
		raise ValueError('UPPE propagator cannot be used for monochromatic radiation.')

	T.AllocateDeviceMemory(a.shape)

	L = (zf-zi)/subcycles
	src = Source(cl.q,a,w,zi,L,NL_band)
	T.kspacex(src.qw_dev)
	iterations = 0
	sw = src.qw_dev.shape
	qwdev = src.qw_dev.data
	qidev = src.qi_dev.data
	qfdev = src.qf_dev.data
	Jwdev = src.Jw_dev.data

	for sub in range(subcycles):
		print('*',end='',flush=True)
		z0 = zi + sub*L # global coordinate of start of subcycle
		z = 0.0 # local coordinate within this subcycle
		mat.UpdateMaterial(z0+0.5*L)
		while z<L and (z-L)**2 > (L/1e6)**2:
			src.qi_dev[...] = src.qw_dev.copy(queue=cl.q)
			src.qf_dev[...] = src.qw_dev.copy(queue=cl.q)

			dz = load_source(z0,z,cl,T,src,mat,ionizer,rotator,return_dz=True,dzmin=dzmin) # load k1
			cl.program('uppe').RKStage(cl.q,sw,None,qwdev,qidev,qfdev,Jwdev,np.double(dz/6),np.double(dz/2))
			condition_field(cl,src)

			load_source(z0,z+0.5*dz,cl,T,src,mat,ionizer,rotator) # load k2
			cl.program('uppe').RKStage(cl.q,sw,None,qwdev,qidev,qfdev,Jwdev,np.double(dz/3),np.double(dz/2))
			condition_field(cl,src)

			load_source(z0,z+0.5*dz,cl,T,src,mat,ionizer,rotator) # load k3
			cl.program('uppe').RKStage(cl.q,sw,None,qwdev,qidev,qfdev,Jwdev,np.double(dz/3),np.double(dz))
			condition_field(cl,src)

			load_source(z0,z+dz,cl,T,src,mat,ionizer,rotator) # load k4
			cl.program('uppe').RKFinish(cl.q,sw,None,qwdev,qfdev,Jwdev,np.double(dz/6))
			condition_field(cl,src)

			iterations += 1
			z += dz

	a,J,rot,ne = finish(cl,T,src,mat,ionizer,rotator,zf)

	T.FreeDeviceMemory()

	return a,J,rot,ne,4*iterations

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
	A = np.zeros(N).astype(np.cdouble)
	J = np.zeros(N).astype(np.cdouble)
	chiNL = np.zeros(N).astype(np.cdouble)
	ne = np.zeros(N).astype(np.cdouble)

	powersof2 = [2**i for i in range(32)]
	if N[0]-1 not in powersof2:
		raise ValueError('UPPE propagator requires 2**n+1 w-nodes')
	if N[1] not in powersof2:
		raise ValueError('UPPE propagator requires 2**n x-nodes')
	if N[2] not in powersof2:
		raise ValueError('UPPE propagator requires 2**n y-nodes')
	try:
		window_speed = vol_dict['window speed']
	except KeyError:
		window_speed = 1.0
	try:
		chi3 = vol_dict['chi3']
	except KeyError:
		chi3 = 0.0
	try:
		full_relaunch = vol_dict['full relaunch']
	except KeyError:
		full_relaunch = False

	if vol_dict['wave coordinates']=='cartesian':
		field_tool = caustic_tools.FourierTool(N,band,(0,0,0),size[1:],cl)
	else:
		field_tool = caustic_tools.BesselBeamTool(N,band,(0,0,0),size[1:],cl,vol_dict['radial modes'])
	w_nodes,x1_nodes,x2_nodes,dom3d = field_tool.GetGridInfo()

	if 'incoming wave' not in vol_dict:
		logging.info('Start UPPE propagation using ray data.')
		logging.warning('Polarization information is lost upon entering UPPE region.')
		A[...,0] = field_tool.GetBoundaryFields(xp[:,0,:],eikonal,1)
	else:
		logging.info('Start UPPE propagation using wave data')
		A[...,0] = vol_dict['incoming wave'](w_nodes,x1_nodes,x2_nodes)

	# Setup the wave propagation medium
	chi = vol_dict['dispersion inside'].chi(w_nodes).astype(np.cdouble)
	ionizer = None
	rotator = None
	if 'ionizer' in vol_dict.keys() and vol_dict['ionizer']!=None:
		ionizer = ionization.Ionizer(vol_dict['ionizer'])
	if 'rotator' in vol_dict.keys() and vol_dict['rotator']!=None:
		rotator = vol_dict['rotator']

	mat = Material(cl.q,field_tool,N,vol_dict['wave coordinates'],vol_dict['object'],chi,chi3,vol_dict['density reference'],window_speed)

	# Setup the wave propagation domain
	dens_nodes = grid_tools.cell_centers(-size[3]/2,size[3]/2,steps)
	field_walls = grid_tools.cell_walls(dens_nodes[0],dens_nodes[-1],steps)
	diagnostic_walls = np.linspace(-size[3]/2,size[3]/2,N[3])
	dzmin = vol_dict['minimum step']
	dom4d = np.concatenate((dom3d,[field_walls[0],field_walls[-1]]))

	# Step through the domain
	A0 = np.copy(A[...,0])
	for k in range(diagnostic_steps):
		zi = diagnostic_walls[k]
		zf = diagnostic_walls[k+1]
		print('Advancing to diagnostic plane',k+1)
		A0,J0,rot0,ne0,evals = propagator(cl,field_tool,window_speed,A0,mat,ionizer,rotator,NL_band,subcycles,zi,zf,dzmin)
		print('',evals,'evaluations of j(w,kx,ky)')
		A[...,k+1] = A0
		J[...,k+1] = J0
		chiNL[...,k+1] = rot0
		ne[...,k+1] = ne0

	# Finish by relaunching rays and returning UPPE data
	if full_relaunch:
		field_tool.RelaunchRays1(xp,eikonal,vg,A[...,-1],size[3],vol_dict['dispersion inside'])
	else:
		field_tool.RelaunchRays(xp,eikonal,vg,A[...,-1],size[3],vol_dict['dispersion inside'])
	return A,J,chiNL,ne,dom4d
