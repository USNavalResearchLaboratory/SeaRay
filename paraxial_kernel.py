'''
Module: :samp:`paraxial_kernel`
-------------------------------

This module is the primary computational engine for advancing paraxial wave equations.
'''
import numpy as np
from numpy.fft import fft
from numpy.fft import ifft
from scipy.linalg import eig_banded
import scipy.sparse.linalg
import caustic_tools
import grid_tools

pulse_dict = { 'exp' : lambda x: np.exp(-x**2),
	'sech' : lambda x: 1/np.cosh(x) }

def quintic_step(x):
	if x<0:
		return 0.0
	elif x<1:
		return 10*x**3 - 15*x**4 + 6*x**5;
	else:
		return 1.0

def plasma_channel_dens(r,z,zperiod,ne0,nc):
	# create a plasma channel with given "betatron" period
	vg = 1/np.sqrt(1-ne0/nc)
	Omega = 2*np.pi*vg/zperiod
	ne2 = Omega**2 * nc
	r2 = r**2
	zfact = 1
	ans = ne0 + ne2*r2
	return ans*zfact

def propagator(a0,chi,dens,n0,ng,w,x,y,dz,long_pulse_approx):
	'''Advance a0[w,x,y] to a new z plane using paraxial wave equation.

	:param numpy.array a: the field to propagate with shape (Nw,Nx,Ny)
	:param numpy.array chi: the susceptibility at reference density with shape (Nw,)
	:param numpy.array dens: the density normalized to the reference with shape (Nx,Ny)
	:param double n0: the reference index of refraction
	:param double ng: the reference group index
	:param numpy.array w: frequency nodes with shape (Nw,)
	:param numpy.array x: spatial nodes with shape (Nx,)
	:param numpy.array y: spatial nodes with shape (Ny,)
	:param double dz: step size
	:param bool long_pulse_approx: used a simplified nonlinear treatment suitable for long pulses'''
	a = np.copy(a0)
	N = (w.shape[0],x.shape[0],y.shape[0],1)
	w0 = w[int(N[0]/2)]
	if N[0]==1 or long_pulse_approx:
		dt = 0.0
	else:
		bandwidth = w[-1] - w[0] + w[1] - w[0]
		dt = 2*np.pi/bandwidth
	dx = (dt,x[1]-x[0],y[1]-y[0],dz)
	dn = n0-ng
	# Form uniform contribution to susceptibility
	chi0 = np.min(dens)*chi
	# Form spatial perturbation to susceptibility
	dchi = np.einsum('i,jk',chi,dens) - chi0[...,np.newaxis,np.newaxis]
	kx = 2.0*np.pi*np.fft.fftfreq(N[1],d=dx[1])
	ky = 2.0*np.pi*np.fft.fftfreq(N[2],d=dx[2])
	kperp2 = np.outer(kx**2,np.ones(N[2])) + np.outer(np.ones(N[1]),ky**2)
	fw = w**2*(1+chi0-ng**2)-w0**2*dn**2-2*w*w0*ng*dn
	f = fw[...,np.newaxis,np.newaxis] - kperp2[np.newaxis,...]
	g = (2j*(w0*dn+w*ng))[...,np.newaxis,np.newaxis]

	# linear half-step
	a = np.fft.fft(np.fft.fft(a,axis=1),axis=2)
	a *= np.exp(-0.5*f*dz/g)
	a = np.fft.ifft(np.fft.ifft(a,axis=2),axis=1)

	# linear but nonuniform step
	h = dchi * (w**2)[...,np.newaxis,np.newaxis]
	a *= np.exp(-h*dz/g)

	# nonlinear step (assumes medium is plasma)
	a = np.fft.ifft(np.fft.ifftshift(a,axes=0),axis=0)
	a2 = np.abs(a)**2
	if dx[0]==0.0:
		# Dropping tau derivative allows for simple exponential solution
		# assuming we can evaluate |a|^2 at the old z
		a *= np.exp(1j*dens[np.newaxis,...]*a2*dz/(8*n0*w0))
	else:
		# Solve a tridiagonal system for each temporal strip
		for i in range(N[1]):
			for j in range(N[2]):
				T1 = np.ones(N[0]-1)*ng/dx[0]
				T2 = 2j*w0*n0+0.125*dz*dens[i,j]*a2[:,i,j]
				T3 = -np.ones(N[0]-1)*ng/dx[0]
				T = scipy.sparse.diags([T1,T2,T3],[-1,0,1]).tocsc()
				b = np.roll(a[:,i,j],1)*ng/dx[0] - np.roll(a[:,i,j],-1)*ng/dx[0]
				b += a[:,i,j]*(2j*w0*n0-0.125*dz*dens[i,j]*a2[:,i,j])
				a[:,i,j] = scipy.sparse.linalg.spsolve(T,b)
	a = np.fft.fftshift(np.fft.fft(a,axis=0),axes=0)

	# linear half-step
	a = np.fft.fft(np.fft.fft(a,axis=1),axis=2)
	a *= np.exp(-0.5*f*dz/g)
	a = np.fft.ifft(np.fft.ifft(a,axis=2),axis=1)

	return a

def axisymmetric_propagator(H,a,chi,dens,n0,ng,w,dz,long_pulse_approx):
	'''Advance a[w,r] to a new z plane using paraxial wave equation in cylindrical coordinates.

	:param object H: Hankel transform tool
	:param numpy.array a: the field to propagate with shape (Nw,Nr)
	:param numpy.array chi: the susceptibility at reference density with shape (Nw,)
	:param numpy.array dens: the density normalized to the reference with shape (Nr,)
	:param double n0: the reference index of refraction
	:param double ng: the reference group index
	:param numpy.array w: frequency nodes with shape (Nw,)
	:param double dz: step size
	:param bool long_pulse_approx: used a simplified nonlinear treatment suitable for long pulses'''
	Nw = w.shape[0]
	w0 = w[int(Nw/2)]
	dn = n0-ng
	# Form uniform contribution to susceptibility
	chi0 = np.min(dens)*chi
	# Form spatial perturbation to susceptibility
	dchi = np.outer(chi,dens) - chi0[...,np.newaxis]
	kperp2 = H.kr2()
	fw = w**2*(1+chi0-ng**2)-w0**2*dn**2-2*w*w0*ng*dn
	f = fw[...,np.newaxis] - kperp2[np.newaxis,...]
	g = (2j*(w0*dn+w*ng))[...,np.newaxis]

	# linear half-step
	a = H.kspace(a)
	a *= np.exp(-0.5*f*dz/g)
	a = H.rspace(a)

	# linear but nonuniform step
	h = dchi * (w**2)[...,np.newaxis]
	a *= np.exp(-h*dz/g)

	# nonlinear step (assumes medium is plasma)
	# a = np.fft.ifft(np.fft.ifftshift(a,axes=0),axis=0)
	# a2 = np.abs(a)**2
	# if long_pulse_approx:
	# 	# Dropping tau derivative allows for simple exponential solution
	# 	# assuming we can evaluate |a|^2 at the old z
	# 	a *= np.exp(1j*dens[np.newaxis,...]*a2*dz/(8*n0*w0))
	# else:
	# 	# Solve a tridiagonal system for each temporal strip
	# 	for i in range(a.shape[1]):
	# 		T1 = np.ones(Nw-1)*ng/dx[0]
	# 		T2 = 2j*w0*n0+0.125*dz*dens[i]*a2[:,i]
	# 		T3 = -np.ones(Nw-1)*ng/dx[0]
	# 		T = scipy.sparse.diags([T1,T2,T3],[-1,0,1]).tocsc()
	# 		b = np.roll(a[:,i],1)*ng/dx[0] - np.roll(a[:,i],-1)*ng/dx[0]
	# 		b += a[:,i]*(2j*w0*n0-0.125*dz*dens[i]*a2[:,i])
	# 		a[:,i] = scipy.sparse.linalg.spsolve(T,b)
	# a = np.fft.fftshift(np.fft.fft(a,axis=0),axes=0)

	# linear half-step
	a = H.kspace(a)
	a *= np.exp(-0.5*f*dz/g)
	a = H.rspace(a)

	return a

def track(xp,eikonal,vg,vol_dict):
	'''Propagate paraxial waves using eikonal data as a boundary condition.
	The volume must be oriented so the polarization axis is x (linear polarization only).

	:param numpy.array xp: ray phase space with shape (bundles,rays,8)
	:param numpy.array eikonal: ray eikonal data with shape (bundles,4)
	:param numpy.array vg: ray group velocity with shape (bundles,rays,4)
	:param dictionary vol_dict: input file dictionary for the volume'''
	band = vol_dict['frequency band']
	bandwidth = band[1]-band[0]
	wc = 0.5*(band[0]+band[1])
	center = (wc,0.0,0.0,0.0)
	size = (bandwidth,) + vol_dict['size']
	N = vol_dict['grid points']

	# Capture the rays
	field_tool = caustic_tools.FourierTool(N,size)
	w_nodes,x_nodes,y_nodes,plot_ext = field_tool.GetGridInfo()
	A = np.zeros(N).astype(np.complex)
	A[...,0],dom3d = field_tool.GetBoundaryFields(np.array(list(center)),xp[:,0,:],eikonal,1)
	n0 = np.mean(np.sqrt(np.einsum('...i,...i',xp[:,0,5:8],xp[:,0,5:8]))/xp[:,0,4])
	ng = 1.0/np.mean(np.sqrt(np.einsum('...i,...i',vg[:,0,1:4],vg[:,0,1:4])))

	# Setup the wave propagation domain
	w_nodes += wc
	chi = vol_dict['dispersion inside'].chi(w_nodes)
	dens_nodes = grid_tools.cell_centers(-size[3]/2,size[3]/2,N[3]-1)
	field_nodes = grid_tools.cell_walls(dens_nodes[0],dens_nodes[-1],N[3]-1)
	dz = field_nodes[1]-field_nodes[0]

	# Step through the domain
	# Strategy to get density plane is to re-use ray gather system
	# This works as long as the shape of xp is (*,*,8)
	xp_eff = np.zeros((N[1],N[2],8))
	xp_eff[...,1] = np.outer(x_nodes,np.ones(N[2]))
	xp_eff[...,2] = np.outer(np.ones(N[1]),y_nodes)
	for k,z in enumerate(dens_nodes):
		xp_eff[...,3] = z
		dens = vol_dict['object'].GetDensity(xp_eff)
		A[...,k+1] = propagator(A[...,k],chi,dens,n0,ng,w_nodes,x_nodes,y_nodes,dz,True)

	# Return the wave amplitude
	# Rays are re-launched externally
	return A
