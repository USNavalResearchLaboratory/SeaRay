'''
Module: :samp:`uppe_kernel`
-------------------------------

This module is the primary computational engine for advancing unidirectional pulse propagation equations (UPPE).
'''
import numpy as np
from numpy.fft import fft
from numpy.fft import ifft
import caustic_tools
import grid_tools

def propagator(a0,chi,chi3,dens,w,x,y,dz):
	'''Advance a0[w,x,y] to a new z plane using UPPE method.

	:param numpy.array a: the field to propagate with shape (Nw,Nx,Ny)
	:param numpy.array chi: the susceptibility at reference density with shape (Nw,)
	:param numpy.array dens: the density normalized to the reference with shape (Nx,Ny)
	:param numpy.array w: frequency nodes with shape (Nw,)
	:param numpy.array x: spatial nodes with shape (Nx,)
	:param numpy.array y: spatial nodes with shape (Ny,)
	:param double dz: step size'''
	a = np.copy(a0)
	N = (w.shape[0],x.shape[0],y.shape[0],1)
	if N[0]==1:
		raise ValueError('UPPE propagator cannot be used for monochromatic radiation.')
	else:
		bandwidth = w[-1] - w[0] + w[1] - w[0]
		dt = 2*np.pi/bandwidth
	dx = (dt,x[1]-x[0],y[1]-y[0],dz)
	# Form uniform contribution to susceptibility
	chi0 = np.copy(chi)
	chi0[np.where(np.real(chi)>0.0)] *= np.min(dens)
	chi0[np.where(np.real(chi)<0.0)] *= np.max(dens)
	# Form spatial perturbation to susceptibility - must be positive
	dchi = np.einsum('i,jk',chi,dens) - chi0[...,np.newaxis,np.newaxis]
	kx = 2.0*np.pi*np.fft.fftfreq(N[1],d=dx[1])
	ky = 2.0*np.pi*np.fft.fftfreq(N[2],d=dx[2])
	kperp2 = np.outer(kx**2,np.ones(N[2])) + np.outer(np.ones(N[1]),ky**2)
	w2n2 = w**2*(1+chi0)
	kz = np.sqrt(0j + w2n2[...,np.newaxis,np.newaxis] - kperp2[np.newaxis,...])
	kz[np.where(np.imag(kz)<0)] *= -1

	# linear half-step in kx-ky plane
	a = np.fft.fft(np.fft.fft(a,axis=1),axis=2)
	a *= np.exp(0.5j*kz*dz)
	a = np.fft.ifft(np.fft.ifft(a,axis=2),axis=1)

	# perturbative full step in x-y plane
	# 1. nonuniformity
	dkz = np.sqrt(0j + dchi * (w**2)[...,np.newaxis,np.newaxis])
	a *= np.exp(1j*dkz*dz)
	# 2. Kerr effect
	failsafe = 1e-4*np.max(np.abs(a))*np.sign(np.real(a))
	j3 = -1j*w[...,np.newaxis,np.newaxis]*a # E(w)
	j3 = np.fft.ifft(j3,axis=0)/dt # E(t)
	j3 = dt*np.fft.fft(chi3*j3**3,axis=0) # P(w)
	j3 = -1j*w[...,np.newaxis,np.newaxis]*j3 # j3(w)
	dkz = np.sqrt(j3 / (a + failsafe))
	a *= np.exp(1j*dkz*dz)

	# linear half-step kx-ky plane
	a = np.fft.fft(np.fft.fft(a,axis=1),axis=2)
	a *= np.exp(0.5j*kz*dz)
	a = np.fft.ifft(np.fft.ifft(a,axis=2),axis=1)

	return a

def axisymmetric_propagator(H,a,chi,chi3,dens,w,dz):
	'''Advance a[w,r] to a new z plane using UPPE method in cylindrical coordinates.

	:param object H: Hankel transform tool
	:param numpy.array a: the field to propagate with shape (Nw,Nr)
	:param numpy.array chi: the susceptibility at reference density with shape (Nw,)
	:param numpy.array dens: the density normalized to the reference with shape (Nr,)
	:param numpy.array w: frequency nodes with shape (Nw,)
	:param double dz: step size'''
	Nw = w.shape[0]
	if Nw==1:
		raise ValueError('UPPE propagator cannot be used for monochromatic radiation.')
	else:
		bandwidth = w[-1] - w[0] + w[1] - w[0]
		dt = 2*np.pi/bandwidth
	# Form uniform contribution to susceptibility
	chi0 = np.copy(chi)
	chi0[np.where(np.real(chi)>0.0)] *= np.min(dens)
	chi0[np.where(np.real(chi)<0.0)] *= np.max(dens)
	# Form spatial perturbation to susceptibility
	dchi = np.outer(chi,dens) - chi0[...,np.newaxis]
	kperp2 = H.kr2()
	w2n2 = w**2*(1+chi0)
	kz = np.sqrt(0j + w2n2[...,np.newaxis] - kperp2[np.newaxis,...])
	kz[np.where(np.imag(kz)<0)] *= -1

	# linear half-step in kr space
	a = H.kspace(a)
	a *= np.exp(0.5j*kz*dz)
	a = H.rspace(a)

	# linear half-step in kr space
	a = H.kspace(a)
	a *= np.exp(0.5j*kz*dz)
	a = H.rspace(a)

	return a

def track(xp,eikonal,vg,vol_dict):
	'''Propagate unidirectional fully dispersive waves using eikonal data as a boundary condition.
	The volume must be oriented so the polarization axis is x (linear polarization only).

	:param numpy.array xp: ray phase space with shape (bundles,rays,8)
	:param numpy.array eikonal: ray eikonal data with shape (bundles,4)
	:param numpy.array vg: ray group velocity with shape (bundles,rays,4)
	:param dictionary vol_dict: input file dictionary for the volume'''
	band = vol_dict['frequency band']
	size = (band[1]-band[0],) + vol_dict['size']
	N = vol_dict['grid points']

	# Capture the rays
	field_tool = caustic_tools.FourierTool(N,band,(0,0,0),vol_dict['size'])
	w_nodes,x_nodes,y_nodes,plot_ext = field_tool.GetGridInfo()
	A = np.zeros(N).astype(np.complex)
	A[...,0],dom3d = field_tool.GetBoundaryFields(xp[:,0,:],eikonal,1)

	# Setup the wave propagation domain
	chi = vol_dict['dispersion inside'].chi(w_nodes)
	dens_nodes = grid_tools.cell_centers(-size[3]/2,size[3]/2,N[3]-1)
	field_walls = grid_tools.cell_walls(-size[3]/2,size[3]/2,N[3])
	dz = field_walls[1]-field_walls[0]
	dom4d = np.concatenate((dom3d,[field_walls[0],field_walls[-1]]))

	# Step through the domain
	# Strategy to get density plane is to re-use ray gather system
	# This works as long as the shape of xp is (*,*,8)
	xp_eff = np.zeros((N[1],N[2],8))
	xp_eff[...,1] = np.outer(x_nodes,np.ones(N[2]))
	xp_eff[...,2] = np.outer(np.ones(N[1]),y_nodes)
	for k,z in enumerate(dens_nodes):
		xp_eff[...,3] = z
		dens = vol_dict['object'].GetDensity(xp_eff)
		A[...,k+1] = propagator(A[...,k],chi,0.0,dens,w_nodes,x_nodes,y_nodes,dz)
		# transformation into pulse frame - in frequency domain it is a phase shift
		galilean = np.einsum('i,j,k->ijk',w_nodes,np.ones(N[1]),np.ones(N[2]))
		A[...,k+1] *= np.exp(-1j*galilean*dz)

	# Return the wave amplitude
	# Rays are re-launched externally
	return A,dom4d
