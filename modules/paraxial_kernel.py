'''
Module: :samp:`paraxial_kernel`
-------------------------------

This module is the primary computational engine for advancing paraxial wave equations.
The call stack is as follows:

* `track` is called externally
* `track` repeatedly calls `propagator` to advance to the end of the volume
* `propagator` repeatedly calls `load_source` and RK4 functions to advance to the next diagnostic plane
* `load_source` has these steps

	- propagate linearly
	- transform (kx,ky) -> (x,y)
	- call `update_current`
	- transform (x,y) -> (kx,ky)
	- estimate step size if requested
'''
import numpy as np
import pyopencl
import init
import caustic_tools
import grid_tools
import ionization
import rotations
import logging
from base import base_volume_interface

class Material:
	'''This class manages host and device storage describing the material's density and susceptibility.
	It also computes the generator of linear axial translations, kz.'''
	def __init__(self,queue,
			  ctool: caustic_tools.CausticTool,
			  N: tuple,
			  coords: str,
			  vol: base_volume_interface,
			  chi: np.ndarray,
			  chi3: np.double,
			  nref: np.double,
			  w0: np.double,
			  ng: np.double,
			  n0: np.double):
		''':param class queue: OpenCL command queue
		:param CausticTool ctool: contains grid info and transverse mode tool
		:param tuple N: dimensions of the field grid
		:param str coords: if string not "cartesian" cylindrical is used
		:param class vol: the volume object
		:param numpy.array chi: the susceptibility at reference density with shape (Nw,) as type complex
		:param double chi3: the nonlinear susceptibility at the reference density
		:param double nref: the reference density in simulation units
		:param double w0: the reference frequency used in describing the envelope
		:param double ng: the reference group index used in describing the envelope
		:param double n0: the reference phase index used in describing the envelope'''
		logging.info("creating material for paraxial solver")
		w,x1,x2,ext = ctool.GetGridInfo()
		self.q = queue
		self.T = ctool.GetTransverseTool()
		self.N = N
		self.vol = vol
		self.chi = np.fft.ifftshift(chi,axes=0)
		self.w = np.fft.ifftshift(w,axes=0)
		self.redchi3 = chi3/nref
		self.n_ref = nref
		self.w0 = w0
		self.ng = ng
		self.n0 = n0
		self.xp_eff = np.zeros((N[1],N[2],8))
		# Strategy to get density plane is to re-use ray gather system
		# This works as long as the shape of xp is (*,*,8)
		if coords=='cartesian':
			self.xp_eff[...,1] = np.outer(x1,np.ones(N[2]))
			self.xp_eff[...,2] = np.outer(np.ones(N[1]),x2)
		else:
			self.xp_eff[...,1] = np.outer(x1,np.cos(x2))
			self.xp_eff[...,2] = np.outer(x1,np.sin(x2))
		self.ngas_dev = pyopencl.array.empty(queue,N[1:3],np.double)
		self.kz_dev = pyopencl.array.empty(queue,N[:3],np.cdouble)
		self.k0_dev = pyopencl.array.empty(queue,(N[0],1,1),np.double)
	def UpdateMaterial(self,z: np.double):
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
		# self.dchi = np.einsum('i,jk',self.chi,dens) - chi0[...,np.newaxis,np.newaxis]
		# Form the linear propagator
		w = self.w
		w0 = self.w0
		n0 = self.n0
		ng = self.ng
		dn = self.n0 - self.ng
		kperp2 = self.T.kr2()
		fw = w**2*(1+chi0-ng**2) - w0**2*dn**2 - 2*w*w0*ng*dn
		kappa2 = fw[...,np.newaxis,np.newaxis] - kperp2[np.newaxis,...]
		k0 = (w0*dn+w*ng)[...,np.newaxis,np.newaxis]
		kz = 0.5*kappa2/k0
		# Put gas density in simulation units for ionization calculations
		self.ngas_dev.set(self.n_ref*dens,queue=self.q)
		self.kz_dev.set(kz,queue=self.q)
		self.k0_dev.set(k0,queue=self.q)

class Source:
	'''This class gathers references to host and device storage
	that are used during the ODE integrator right-hand-side evaluation.'''
	def __init__(self,queue,a: np.ndarray,w: np.ndarray,zi: np.double,L: np.double):
		# Following shape calculations have redundancy.
		# Keep this for parity with case of real fields.
		# N.b. we must not assume anything about ordering of frequency array
		if w.shape[0]==1:
			bandwidth = 1.0
		else:
			bandwidth = (np.max(w) - np.min(w))*(1 + 1/(w.shape[0]-1))
		self.dt = 2*np.pi/bandwidth
		self.zi = zi
		self.L = L
		self.freqs = a.shape[0]
		self.steps = self.freqs
		self.transverse_shape = (a.shape[1],a.shape[2])
		self.freq_domain_shape = (self.freqs,) + self.transverse_shape
		self.time_domain_shape = (self.steps,) + self.transverse_shape
		self.dphi = 0.4
		self.rel_amin = 1e-2
		# Device arrays
		self.qw_dev = pyopencl.array.to_device(queue,a)
		self.qi_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.cdouble)
		self.qf_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.cdouble)
		self.Aw_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.cdouble)
		self.At_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.cdouble)
		self.Et_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.cdouble)
		self.w_dev = pyopencl.array.to_device(queue,w)
		self.ne_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.chi_dev = pyopencl.array.empty(queue,self.time_domain_shape,np.double)
		self.J_dev = pyopencl.array.empty(queue,self.freq_domain_shape,np.cdouble)

def update_current(cl: init.cl_refs,
				   src: Source,
				   mat: Material,
				   ionizer: ionization.Ionizer,
				   rotator: rotations.Rotator):
	'''Get current density from A[w,x,y]'''

	# Setup some shorthand
	shp2 = src.transverse_shape
	shp3 = src.Aw_dev.shape
	wdev = src.w_dev.data
	Atdev = src.At_dev.data
	Etdev = src.Et_dev.data
	nedev = src.ne_dev.data
	chidev = src.chi_dev.data
	ngasdev = mat.ngas_dev.data
	Jdev = src.J_dev.data

	# Form time domain potential and field
	# N.b. time index runs backwards due to FFT conventions
	src.Et_dev[...] = src.Aw_dev.copy(queue=cl.q)
	src.At_dev[...] = src.Aw_dev.copy(queue=cl.q)
	src.ne_dev.fill(0.0,queue=cl.q)
	src.chi_dev.fill(0.0,queue=cl.q)
	cl.program('fft').DtSpectral(cl.q,shp3,None,Etdev,wdev,np.double(-1.0))
	cl.program('fft').IFFT(cl.q,shp2,None,Etdev,np.int32(src.freqs))
	cl.program('fft').IFFT(cl.q,shp2,None,Atdev,np.int32(src.freqs))

	# First handle plasma formation
	if ionizer!=None:
		ionizer.RateCL(cl,shp3,nedev,Etdev,True)
		ionizer.GetPlasmaDensityCL(cl,shp3,nedev,ngasdev,src.dt)
	# Effective susceptibility due to nonlinear terms
	if rotator!=None:
		rotator.AddChi(cl,shp3,chidev,Etdev,ngasdev,src.dt,False)
	cl.program('paraxial').AddKerrChi(cl.q,shp3,None,chidev,Etdev,ngasdev,np.double(mat.redchi3))
	# Form polarization due to Raman and Kerr susceptibility
	# We are putting polarization into Jdev which will later become current density
	cl.program('paraxial').SourcePolarization(cl.q,shp3,None,Jdev,chidev,Etdev)
	#src.J_dev.fill(0.0,queue=cl.q) # uncomment to force linear propagation
	# Now go to frequency domain to add non-uniform but dispersive contributions,
	# and also take time derivative in frequency space to get J
	cl.program('fft').FFT(cl.q,shp2,None,Jdev,np.int32(src.steps))
	#cl.program('paraxial').AddNonuniformChi(cl.q,shp3,None,Jdev,Ewdev,dchi)
	cl.program('fft').DtSpectral(cl.q,shp3,None,Jdev,wdev,np.double(1.0))
	# Plasma current
	cl.program('fft').IFFT(cl.q,shp2,None,Jdev,np.int32(src.freqs))
	cl.program('paraxial').AddPlasmaCurrent(cl.q,shp3,None,Jdev,Atdev,nedev)
	cl.program('fft').FFT(cl.q,shp2,None,Jdev,np.int32(src.steps))

def load_source(z0: np.double,
				z: np.double,
				cl: init.cl_refs,
				T: grid_tools.TransverseModeTool,
				src: Source,
				mat: Material,
				ionizer: ionization.Ionizer,
				rotator: rotations.Rotator,
				return_dz=False):
	'''We have an equation in the form dq/dz = S(z,q).
	This loads src.J_dev with S(z,q), using q = src.qw_dev

	:param double z0: global coordinate of the start of the subcycle region
	:param double z: local coordinate of evaluation point referenced to z0
	:param cl_refs cl: OpenCL reference bundle
	:param TransverseModeTool T: used for transverse mode transformations
	:param Source src: stores references to OpenCL buffers used in RHS evaluations
	:param Material mat: stores references to OpenCL buffers used in Linear propagation
	:param Ionization ionizer: class for encapsulating ionization models
	:param Rotator rotator: class for encapsulating molecular rotations
	:param bool return_dz: if true return the step size estimate'''
	# Setup some shorthand
	shp3 = src.Aw_dev.shape
	qwdev = src.qw_dev.data
	Awdev = src.Aw_dev.data
	Jdev = src.J_dev.data
	kzdev = mat.kz_dev.data
	k0dev = mat.k0_dev.data
	dzdev = src.ne_dev.data # re-use ne for the step size estimate
	src.Aw_dev[...] = src.qw_dev.copy(queue=cl.q)
	cl.program('paraxial').PropagateLinear(cl.q,shp3,None,Awdev,kzdev,np.double(z0-src.zi+z))
	T.rspacex(src.Aw_dev)
	update_current(cl,src,mat,ionizer,rotator)
	T.kspacex(src.J_dev)
	cl.program('paraxial').CurrentToODERHS(cl.q,shp3,None,Jdev,kzdev,k0dev,np.double(z0-src.zi+z))
	if return_dz:
		cl.program('paraxial').LoadModulus(cl.q,shp3,None,qwdev,dzdev)
		if np.isnan(pyopencl.array.sum(src.ne_dev,queue=cl.q).get(queue=cl.q)):
			print('!',end='',flush=True)
		amin = src.rel_amin * pyopencl.array.max(src.ne_dev,queue=cl.q).get(queue=cl.q)
		cl.program('paraxial').LoadStepSize(cl.q,shp3,None,qwdev,Jdev,dzdev,np.double(src.L),np.double(src.dphi),np.double(amin))
		dz = pyopencl.array.min(src.ne_dev,queue=cl.q).get(queue=cl.q)
		if z+dz>src.L:
			dz = src.L-z
		else:
			print('.',end='',flush=True)
		if (src.L-z)/dz > 10:
			print(int((src.L-z)/dz),end='',flush=True)
		return dz

def finish(cl: init.cl_refs,
		   T: grid_tools.TransverseModeTool,
		   src: Source,
		   mat: Material,
		   ionizer: ionization.Ionizer,
		   rotator: rotations.Rotator,
		   zf: np.double) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
	'''Finalize data at the end of the iterations and retrieve.

	:param cl_refs cl: OpenCL reference bundle
	:param TransverseModeTool T: used for transverse mode transformations
	:param Source src: stores references to OpenCL buffers used in RHS evaluations
	:param Material mat: stores references to OpenCL buffers used in Linear propagation
	:param Ionization ionizer: class for encapsulating ionization models
	:param Rotator rotator: class for encapsulating molecular rotations
	:param double zf: global coordinate of the final position'''
	shp3 = src.Aw_dev.shape
	Awdev = src.Aw_dev.data
	kzdev = mat.kz_dev.data
	src.Aw_dev[...] = src.qw_dev.copy(queue=cl.q)
	cl.program('paraxial').PropagateLinear(cl.q,shp3,None,Awdev,kzdev,np.double(zf-src.zi))
	T.rspacex(src.Aw_dev)
	update_current(cl,src,mat,ionizer,rotator)
	return src.Aw_dev.get(queue=cl.q), \
		src.J_dev.get(queue=cl.q), \
		np.fft.fft(src.chi_dev.get(queue=cl.q),axis=0), \
		np.fft.fft(src.ne_dev.get(queue=cl.q),axis=0)

def propagator(cl: init.cl_refs,
			   ctool: caustic_tools.CausticTool,
			   a: np.ndarray,
			   mat: Material,
			   ionizer: ionization.Ionizer,
			   rotator: rotations.Rotator,
			   subcycles: int,
			   zi: np.double,
			   zf: np.double):
	'''Advance a[w,x,y] to a new z plane using paraxial wave equation.
	Works in either Cartesian or cylindrical geometry.
	If cylindrical, intepret x as radial and y as azimuthal.
	The RK4 stepper is advancing q(z;w,kx,ky) = exp(-i*kz*z)*a(z;w,kx,ky).

	:param cl_refs cl: OpenCL reference bundle
	:param CausticTool ctool: contains grid info and transverse mode tool
	:param numpy.array a: the vector potential in representation (w,x,y) as type complex
	:param Material mat: stores references to OpenCL buffers used in Linear propagation
	:param Ionization ionizer: class for encapsulating ionization models
	:param Rotator rotator: class for encapsulating molecular rotations
	:param int subcycles: number of density evaluations along the path
	:param double zi: initial z position
	:param double zf: final z position'''
	w,x,y,ext = ctool.GetGridInfo()
	T = ctool.GetTransverseTool()
	N = (w.shape[0],x.shape[0],y.shape[0],1)

	T.AllocateDeviceMemory(a.shape)

	# Reorder frequencies for FFT processing
	a = np.fft.ifftshift(a,axes=0)
	w = np.fft.ifftshift(w,axes=0)

	L = (zf-zi)/subcycles
	src = Source(cl.q,a,w,zi,L)
	T.kspacex(src.qw_dev)
	iterations = 0
	sw = src.qw_dev.shape
	qwdev = src.qw_dev.data
	qidev = src.qi_dev.data
	qfdev = src.qf_dev.data
	Jwdev = src.J_dev.data

	for sub in range(subcycles):
		print('*',end='',flush=True)
		z0 = zi + sub*L # global coordinate of start of subcycle
		z = 0.0 # local coordinate within this subcycle
		mat.UpdateMaterial(z0+0.5*L)
		while z<L and (z-L)**2 > (L/1e6)**2:
			src.qi_dev[...] = src.qw_dev.copy(queue=cl.q)
			src.qf_dev[...] = src.qw_dev.copy(queue=cl.q)

			dz = load_source(z0,z,cl,T,src,mat,ionizer,rotator,return_dz=True) # load k1
			cl.program('uppe').RKStage(cl.q,sw,None,qwdev,qidev,qfdev,Jwdev,np.double(dz/6),np.double(dz/2))

			load_source(z0,z+0.5*dz,cl,T,src,mat,ionizer,rotator) # load k2
			cl.program('uppe').RKStage(cl.q,sw,None,qwdev,qidev,qfdev,Jwdev,np.double(dz/3),np.double(dz/2))

			load_source(z0,z+0.5*dz,cl,T,src,mat,ionizer,rotator) # load k3
			cl.program('uppe').RKStage(cl.q,sw,None,qwdev,qidev,qfdev,Jwdev,np.double(dz/3),np.double(dz))

			load_source(z0,z+dz,cl,T,src,mat,ionizer,rotator) # load k4
			cl.program('uppe').RKFinish(cl.q,sw,None,qwdev,qfdev,Jwdev,np.double(dz/6))

			iterations += 1
			z += dz

	a,J,chiNL,ne = finish(cl,T,src,mat,ionizer,rotator,zf)

	T.FreeDeviceMemory()

	# Sort frequencies
	a = np.fft.fftshift(a,axes=0)
	J = np.fft.fftshift(J,axes=0)
	chiNL = np.fft.fftshift(chiNL,axes=0)
	ne = np.fft.fftshift(ne,axes=0)

	return a,J,chiNL,ne,4*iterations

def track(cl,xp,eikonal,vg,vol_dict: dict):
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
	A = np.zeros(N).astype(np.cdouble)
	J = np.zeros(N).astype(np.cdouble)
	chiNL = np.zeros(N).astype(np.cdouble)
	ne = np.zeros(N).astype(np.cdouble)

	powersof2 = [2**i for i in range(32)]
	if N[0] not in powersof2:
		raise ValueError('Paraxial propagator requires 2**n w-nodes')
	if N[1] not in powersof2:
		raise ValueError('Paraxial propagator requires 2**n x-nodes')
	if N[2] not in powersof2:
		raise ValueError('Paraxial propagator requires 2**n y-nodes')
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
		logging.info('Start paraxial propagation using ray data.')
		logging.warning('Polarization information is lost upon entering paraxial region.')
		A[...,0] = field_tool.GetBoundaryFields(xp[:,0,:],eikonal,1)
	else:
		logging.info('Start paraxial propagation using wave data')
		A[...,0] = vol_dict['incoming wave'](w_nodes,x1_nodes,x2_nodes)

	w0 = w_nodes[int(N[0]/2)]
	#n0 = np.mean(np.sqrt(np.einsum('...i,...i',xp[:,0,5:8],xp[:,0,5:8]))/xp[:,0,4])
	n0 = 1/window_speed
	#ng = 1.0/np.mean(np.sqrt(np.einsum('...i,...i',vg[:,0,1:4],vg[:,0,1:4])))
	ng = 1/window_speed

	# Setup the wave propagation medium
	chi = vol_dict['dispersion inside'].chi(w_nodes).astype(np.cdouble)
	ionizer = None
	rotator = None
	if 'ionizer' in vol_dict.keys() and vol_dict['ionizer']!=None:
		ionizer = ionization.Ionizer(vol_dict['ionizer'])
	if 'rotator' in vol_dict.keys() and vol_dict['rotator']!=None:
		rotator = vol_dict['rotator']

	mat = Material(cl.q,field_tool,N,vol_dict['wave coordinates'],vol_dict['object'],chi,chi3,vol_dict['density reference'],w0,ng,n0)

	# Setup the wave propagation domain
	dens_nodes = grid_tools.cell_centers(-size[3]/2,size[3]/2,steps)
	field_walls = grid_tools.cell_walls(dens_nodes[0],dens_nodes[-1],steps)
	diagnostic_walls = np.linspace(-size[3]/2,size[3]/2,N[3])
	dom4d = np.concatenate((dom3d,[field_walls[0],field_walls[-1]]))

	# Step through the domain
	A0 = np.copy(A[...,0])
	for k in range(diagnostic_steps):
		zi = diagnostic_walls[k]
		zf = diagnostic_walls[k+1]
		print('Advancing to diagnostic plane',k+1)
		A0,J0,chiNL0,ne0,evals = propagator(cl,field_tool,A0,mat,ionizer,rotator,subcycles,zi,zf)
		print('',evals,'evaluations of j(w,kx,ky)')
		A[...,k+1] = A0
		J[...,k+1] = J0
		chiNL[...,k+1] = chiNL0
		ne[...,k+1] = ne0

	# Finish by relaunching rays and returning wave data
	if full_relaunch:
		field_tool.RelaunchRays1(xp,eikonal,vg,A[...,-1],size[3],vol_dict['dispersion inside'])
	else:
		field_tool.RelaunchRays(xp,eikonal,vg,A[...,-1],size[3],vol_dict['dispersion inside'])
	return A,J,chiNL,ne,dom4d
