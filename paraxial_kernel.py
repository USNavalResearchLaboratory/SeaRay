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

def propagator(T,a0,chi,dens,n0,ng,dz,long_pulse_approx):
	'''Advance a0[w,x,y] to a new z plane using paraxial wave equation.
	Works in either Cartesian or cylindrical geometry.
	If cylindrical, intepret x as radial and y as azimuthal.

	:param class T: instance of TransverseModeTool class
	:param numpy.array a: the field to propagate with shape (Nw,Nx,Ny)
	:param numpy.array chi: the susceptibility at reference density with shape (Nw,)
	:param numpy.array dens: the density normalized to the reference with shape (Nx,Ny)
	:param double n0: the reference index of refraction
	:param double ng: the reference group index
	:param double dz: step size,
	:param bool long_pulse_approx: use a simplified nonlinear treatment suitable for long pulses'''
	a = np.copy(a0)
	w,x,y,ext = T.GetGridInfo()
	N = (w.shape[0],x.shape[0],y.shape[0],1)
	w0 = w[int(N[0]/2)]
	if N[0]==1:
		dt = 1.0
	else:
		bandwidth = w[-1] - w[0] + w[1] - w[0]
		dt = 2*np.pi/bandwidth
	dx = (dt,x[1]-x[0],y[1]-y[0],dz)
	dn = n0-ng
	# Form uniform contribution to susceptibility
	chi0 = np.copy(chi)
	chi0[np.where(np.real(chi)>0.0)] *= np.min(dens)
	chi0[np.where(np.real(chi)<0.0)] *= np.max(dens)
	# Form spatial perturbation to susceptibility
	dchi = np.einsum('i,jk',chi,dens) - chi0[...,np.newaxis,np.newaxis]
	kperp2 = T.kr2()
	fw = w**2*(1+chi0-ng**2)-w0**2*dn**2-2*w*w0*ng*dn
	f = fw[...,np.newaxis,np.newaxis] - kperp2[np.newaxis,...]
	g = (2j*(w0*dn+w*ng))[...,np.newaxis,np.newaxis]

	# linear half-step
	a = T.kspace(a)
	a *= np.exp(-0.5*f*dz/g)
	a = T.rspace(a)

	# linear but nonuniform step
	h = dchi * (w**2)[...,np.newaxis,np.newaxis]
	a *= np.exp(-h*dz/g)

	# nonlinear step (assumes medium is plasma)
	a = np.fft.ifft(np.fft.ifftshift(a,axes=0),axis=0)/dx[0]
	a2 = np.abs(a)**2
	if long_pulse_approx or N[0]==1:
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
	a = dx[0]*np.fft.fftshift(np.fft.fft(a,axis=0),axes=0)

	# linear half-step
	a = T.kspace(a)
	a *= np.exp(-0.5*f*dz/g)
	a = T.rspace(a)

	return a

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
	A0,dom3d = field_tool.GetBoundaryFields(xp[:,0,:],eikonal,1)
	n0 = np.mean(np.sqrt(np.einsum('...i,...i',xp[:,0,5:8],xp[:,0,5:8]))/xp[:,0,4])
	ng = 1.0/np.mean(np.sqrt(np.einsum('...i,...i',vg[:,0,1:4],vg[:,0,1:4])))

	# Setup the wave propagation domain
	chi = vol_dict['dispersion inside'].chi(w_nodes)
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
	for k in range(diagnostic_steps):
		print('Advancing to diagnostic plane',k+1)
		for s in range(subcycles):
			xp_eff[...,3] = dens_nodes[k*subcycles + s]
			dens = vol_dict['object'].GetDensity(xp_eff)
			A0 = propagator(field_tool,A0,chi,dens,n0,ng,dz,True)
			try:
				A0 *= vol_dict['damping filter'](w_nodes)[:,np.newaxis,np.newaxis]
			except KeyError:
				A0 = A0
		A[...,k+1] = A0

	# Finish by relaunching rays and returning wave data
	field_tool.RelaunchRays(xp,eikonal,vg,A[...,-1],size[3])
	return A,dom4d
