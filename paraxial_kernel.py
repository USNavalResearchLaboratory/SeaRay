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

def plasma_lens_dens(r,z,f,L,n0,nc):
	h0 = np.sqrt(1-n0/nc)
	k2 = 1/(f*L)
	r2 = r**2
	zfact = np.exp(-pow(z/(0.5*L),8.0))
	ans = n0 + r2*(f*f*k2*h0-1.0)/(f*f)
	#ans += 0.25*r2*r2*(2*h0*k2/(f*f)-h0*h0*k2*k2)
	#ans -= r2*r2*r2*h0*k2/(6.0*f*f*f*f)
	return ans*zfact

def H00_envelope(psi,rho,z,tau0,rho0,w0,k0,pulse_shape):
	# psi = k0*z - w0*t
	zR = 0.5*k0*rho0**2
	Rci = z/(z**2+zR**2)
	rhoz = rho0*np.sqrt(1+z**2/zR**2)
	Phi = 0.5*k0*rho**2*Rci - np.arctan(z/zR)
	Phie = psi + 0.5*k0*rho**2*Rci + np.arctan(z/zR)
	return pulse_dict[pulse_shape](Phie/(w0*tau0))*np.exp(-rho**2/rhoz**2)*(rho0/rhoz)*np.exp(1j*Phi)

def FFT_w(dti,num):
	# Square of this is the proper eigenvalue of the FD laplacian
	# For small frequencies it corresponds to the wave frequency
	i_list = np.arange(0,num)
	sgn = np.ones(num)
	sgn[np.int(num/2)+1:] = -1.0
	return sgn*np.sqrt(2.0*dti*dti*(1.0 - np.cos(2.0*np.pi*i_list/num)));

class HankelTransformTool:
	def __init__(self,Nr,dr):
		r_list = np.arange(0,Nr)*dr + 0.5*dr
		A1 = 2*np.pi*(r_list-0.5*dr)
		A2 = 2*np.pi*(r_list+0.5*dr)
		V = np.pi*((r_list+0.5*dr)**2 - (r_list-0.5*dr)**2)
		T = np.zeros((3,Nr))
		T1 = A1/(dr*V)
		T2 = -(A1 + A2)/(dr*V)
		T3 = A2/(dr*V)
		self.Lambda = np.sqrt(V)
		# Boundary conditions
		T2[0] += T1[0]
		T2[Nr-1] -= T3[Nr-1]
		# Symmetrize the matrix
		# S = Lambda * T * Lambda^-1
		# This is the root-volume weighting
		T1[1:] *= self.Lambda[1:]/self.Lambda[:-1]
		T3[:-1] *= self.Lambda[:-1]/self.Lambda[1:]
		a_band_upper = np.zeros((2,Nr))
		a_band_upper[0,:] = T1 # T3->T1 thanks to scipy packing and symmetry
		a_band_upper[1,:] = T2
		self.vals,self.Hi = eig_banded(a_band_upper)
	def kspace(self,a):
		W = np.outer(self.Lambda,np.ones(a.shape[1]))
		return np.dot(self.Hi.T,W*a)/W
	def rspace(self,a):
		W = np.outer(self.Lambda,np.ones(a.shape[1]))
		return np.dot(self.Hi,W*a)/W
	def kr2(self):
		return -self.vals

def wave_action(a,dr,dt):
	# simplified action, for now
	r_list = np.arange(0,a.shape[0])*dr + 0.5*dr
	V = dt*np.pi*((r_list+0.5*dr)**2 - (r_list-0.5*dr)**2)
	W = np.outer(V,np.ones(a.shape[1]))
	return np.sum(np.abs(a)**2 * W)

def propagator(H,a,chiw0,chiw,n0,ng,w0,dr,dt,dz):
	# a[rho,t], chiw0[w], chiw[r,w]
	long_pulse_approx = True
	Nr = a.shape[0]
	Nt = a.shape[1]
	dn = n0-ng
	a = np.fft.fft(a,axis=1)
	w = w0 + FFT_w(1/dt,Nt)
	kr2 = np.outer(H.kr2(),np.ones(Nt))
	f = w**2*(1+chiw0-ng**2)-w0**2*dn**2-2*w*w0*ng*dn-kr2
	g = 2j*(w0*dn+w*ng)

	# linear half-step
	a = H.kspace(a)
	a *= np.exp(-0.5*f*dz/g)
	a = H.rspace(a)

	# linear but nonuniform step
	h = w**2*chiw
	a *= np.exp(-h*dz/g)

	# nonlinear step (assumes medium is plasma)
	wp2 = w0**2 * (chiw[:,0] + chiw0[0])
	a = np.fft.ifft(a,axis=1)
	if long_pulse_approx:
		# Dropping tau derivative allows for simple exponential solution
		# assuming we can evaluate |a|^2 at the old z
		a *= np.exp(1j*np.outer(wp2,np.ones(Nt))*np.abs(a)**2*dz/(8*n0*w0))
	else:
		# Solve a tridiagonal system for each temporal strip
		for ir in range(Nr):
			T1 = np.ones(Nt-1)*ng/dt
			T2 = 2j*w0*n0+0.125*dz*wp2[ir]*np.abs(a[ir,:])**2
			T3 = -np.ones(Nt-1)*ng/dt
			T = scipy.sparse.diags([T1,T2,T3],[-1,0,1]).tocsc()
			b = np.roll(a[ir,:],1)*ng/dt - np.roll(a[ir,:],-1)*ng/dt
			b += a[ir,:]*(2j*w0*n0-0.125*dz*wp2[ir]*np.abs(a[ir,:])**2)
			a[ir,:] = scipy.sparse.linalg.spsolve(T,b)
	a = np.fft.fft(a,axis=1)

	# linear half-step
	a = H.kspace(a)
	a *= np.exp(-0.5*f*dz/g)
	a = H.rspace(a)

	a = np.fft.ifft(a,axis=1)
	return a
