'''
Module: :samp:`uppe_kernel`
-------------------------------

This module is the primary computational engine for advancing unidirectional pulse propagation equations (UPPE).
'''
import numpy as np
import caustic_tools
import grid_tools
import ionization

def current_density(a,dens,dchi,chi3,ionizer,w):
	J = np.zeros(a.shape).astype(np.complex)
	bandwidth = w[-1] - w[0] + w[1] - w[0]
	dt = 2*np.pi/bandwidth
	# Get electric field in the time domain
	Et = np.fft.irfft(1j*w[...,np.newaxis,np.newaxis]*a,axis=0)
	# Accumulate the source terms
	# First handle plasma formation
	rate = ionizer.InstantaneousRate(np.abs(Et)+.001)
	ne = ionizer.GetPlasmaDensity(dens,rate,dt)
	J[1:,...] += 1j*np.fft.rfft(ne*Et,axis=0)[1:,...]/w[1:,np.newaxis,np.newaxis]
	nratio = (dens - ne)/dens
	# Nonuniform susceptibility
	dP = dchi * (w**2)[...,np.newaxis,np.newaxis] * a
	J += np.fft.rfft(nratio * np.fft.irfft(dP,axis=0),axis=0)
	# Kerr effect
	P3 = np.fft.rfft(nratio * chi3 * Et**3,axis=0)
	J += -1j*w[...,np.newaxis,np.newaxis]*P3
	return J,np.fft.rfft(ne,axis=0)

def propagator(T,vwin,a0,chi,chi3,dens,ionizer,dz):
	'''Advance a0[w,x,y] to a new z plane using UPPE method.
	Works in either Cartesian or cylindrical geometry.
	If cylindrical, intepret x as radial and y as azimuthal.

	:param class T: instance of TransverseModeTool class
	:param double vwin: speed of the pulse frame variable
	:param numpy.array a: the field to propagate with shape (Nw,Nx,Ny)
	:param numpy.array chi: the susceptibility at reference density with shape (Nw,)
	:param numpy.array dens: the density normalized to the reference with shape (Nx,Ny)
	:param class ionizer: instance of an ionization class
	:param double dz: step size'''
	w,x,y,ext = T.GetGridInfo()
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

	# Trial step
	J,ne = current_density(a0,dens,dchi,chi3,ionizer,w)
	J = T.kspace(J)
	a = T.kspace(a0)
	a += 0.25j*J*dz/np.real(kz)
	a *= np.exp(0.5j*(kz+kGalileo)*dz)
	a = T.rspace(a)

	# Full step - also manipulate source for diagnostic purposes
	J,ne = current_density(a,dens,dchi,chi3,ionizer,w)
	J = T.kspace(J)
	a = T.kspace(a0)
	a *= np.exp(1j*(kz+kGalileo)*dz)
	J *= np.exp(0.5j*(kz+kGalileo)*dz)
	a += 0.5j*J*dz/np.real(kz)
	a = T.rspace(a)
	J = T.rspace(J)

	return a,J,ne

def track(xp,eikonal,vg,vol_dict):
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
	Vol = vol_dict['object']

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
		field_tool = caustic_tools.FourierTool(N,band,(0,0,0),size[1:],Vol.queue,Vol.transform_k)
	else:
		field_tool = caustic_tools.BesselBeamTool(N,band,(0,0,0),size[1:],Vol.queue,Vol.transform_k)

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
		for s in range(subcycles):
			xp_eff[...,3] = dens_nodes[k*subcycles + s]
			dens = vol_dict['object'].GetDensity(xp_eff)
			A0,J0,ne0 = propagator(field_tool,window_speed,A0,chi,chi3,dens,ionizer,dz)
			try:
				A0 *= vol_dict['damping filter'](w_nodes)[:,np.newaxis,np.newaxis]
			except KeyError:
				A0 = A0
		A[...,k+1] = A0
		J[...,k+1] = J0
		ne[...,k+1] = ne0

	# Return the wave amplitude
	# Rays are re-launched externally
	return A,J,ne,dom4d
