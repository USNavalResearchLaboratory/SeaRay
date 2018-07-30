import numpy as np
from scipy import constants as C

# chi(w) returns susceptibility at a reference density

class Vacuum:
	Dxk = '-dot4(k,k)'
	def vg(self,xp):
		vg = np.copy(xp[...,:4])
		vg[...,0] = 1.0
		vg[...,1] = xp[...,5]/xp[...,4]
		vg[...,2] = xp[...,6]/xp[...,4]
		vg[...,3] = xp[...,7]/xp[...,4]
		return vg
	def kz(self,w,kx,ky):
		return np.sqrt(0j + w**2 - kx**2 - ky**2)
	def chi(self,w):
		return np.zeros(w.shape)

class ColdPlasma(Vacuum):
	Dxk = 'wp2(x) - dot4(k,k)'
	def chi(self,w):
		return -1/w**2

class ParaxialPlasma:
	Dxk = '0.5*(wp2(x)+k.s1*k.s1+k.s2*k.s2)/k.s3 + k.s3 - k.s0'
	def vg(self,xp):
		vg = np.copy(xp[...,:4])
		vg[...,0] = 1.0
		vg[...,1] = xp[...,5]/xp[...,7]
		vg[...,2] = xp[...,6]/xp[...,7]
		vg[...,3] = 2.0 - xp[...,4]/xp[...,7]
		return vg
	def kz(self,w,kx,ky):
		return 0.5*w*(1+np.sqrt(0j+1-2*(kx**2+ky**2)/w**2))
	def chi(self,w):
		return -1/w**2

class FresnelPlasma(ParaxialPlasma):
	def kz(self,w,kx,ky):
		return w*(1-0.5*(kx**2+ky**2)/w**2)

class LiquidWater:
	# Sellmeier formula from appl. opt. 46 (18), 3811 (2007)
	# T = 21.5 deg C
	def __init__(self,mks_length):
		# need the frequency normalization in rad/s
		self.w1 = C.c/mks_length
	def vg(self,xp):
		vg = np.copy(xp[...,:4])
		k2 = xp[...,5]**2 + xp[...,6]**2 + xp[...,7]**2
		vg[...,0] = 1.0
		vg[...,1] = xp[...,4]*xp[...,5]/k2
		vg[...,2] = xp[...,4]*xp[...,6]/k2
		vg[...,3] = xp[...,4]*xp[...,7]/k2
		return vg
	def chi(self,w):
		A = np.array([5.689e-1,1.720e-1,2.063e-2,1.124e-1])
		l2 = np.array([5.110e-3,1.825e-2,2.624e-2,1.068e1])
		# form wavelength in microns
		l0 = 1e6*2.0*np.pi*C.c/(w*self.w1)
		# in the following np.einsum works like np.outer, except without flattening
		numerator = np.einsum('...,k',l0**2,A)
		try:
			denominator = np.einsum('...,k',l0**2,np.ones(l2.shape)) - np.einsum('...,k',np.ones(l0.shape),l2)
		except AttributeError:
			denominator = np.einsum('...,k',l0**2,np.ones(l2.shape)) - np.einsum('...,k',np.ones(1),l2)
		mat = numerator / denominator
		return np.sum(mat,axis=-1)

class BK7:
	# Sellmeier formula from https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
	def __init__(self,mks_length):
		# need the frequency normalization in rad/s
		self.w1 = C.c/mks_length
	def vg(self,xp):
		vg = np.copy(xp[...,:4])
		k2 = xp[...,5]**2 + xp[...,6]**2 + xp[...,7]**2
		vg[...,0] = 1.0
		vg[...,1] = xp[...,4]*xp[...,5]/k2
		vg[...,2] = xp[...,4]*xp[...,6]/k2
		vg[...,3] = xp[...,4]*xp[...,7]/k2
		return vg
	def chi(self,w):
		A = np.array([1.03961212,0.231792344,1.01046945])
		l2 = np.array([0.00600069867,0.0200179144,103.560653])
		# form wavelength in microns
		l0 = 1e6*2.0*np.pi*C.c/(w*self.w1)
		# in the following np.einsum works like np.outer, except without flattening
		numerator = np.einsum('...,k',l0**2,A)
		try:
			denominator = np.einsum('...,k',l0**2,np.ones(l2.shape)) - np.einsum('...,k',np.ones(l0.shape),l2)
		except AttributeError:
			denominator = np.einsum('...,k',l0**2,np.ones(l2.shape)) - np.einsum('...,k',np.ones(1),l2)
		mat = numerator / denominator
		return np.sum(mat,axis=-1)
