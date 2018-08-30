'''
Module :samp:`dispersion`
-------------------------

Dispersion objects are used to specify a dispersion relation inside a ``volume`` object, or on either side of a ``surface`` object.  Each dispersion object provides a function ``chi(w)`` that returns the susceptibility at a frequency or an array of susceptibilities corresponding to an array of frequencies.  For compressible media, the susceptibility is given at a specific reference density.

Dispersion objects must also provide ``Dxk(dens)``, which returns a string containing OpenCL code that computes the local dispersion function :math:`D(x,k)` (the ray Hamiltonian, up to a parameterization).  The argument ``dens`` is a string containing an OpenCL function call that retrieves the density relative to the reference density.

It is simple to add a new medium to SeaRay if the dispersion relation can be put in Sellmeier form.  To do so derive the new object from ``sellmeier_medium`` and provide the arrays of coefficients.  See, e.g., the BK7 object for an example.
'''
import numpy as np
from scipy import constants as C

# chi(w) returns susceptibility at a reference density

class Vacuum:
	def Dxk(self,dens):
		return 'return -dot4(k,k);\n'
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
	def Dxk(self,dens):
		return 'return ' + dens + ' - dot4(k,k);\n'
	def chi(self,w):
		return -1/w**2

class ParaxialPlasma:
	def Dxk(self,dens):
		return 'return 0.5*(' + dens + ' + k.s1*k.s1 + k.s2*k.s2)/k.s3 + k.s3 - k.s0;\n'
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

class weak_dispersion:
	def vg(self,xp):
		vg = np.copy(xp[...,:4])
		k2 = xp[...,5]**2 + xp[...,6]**2 + xp[...,7]**2
		vg[...,0] = 1.0
		vg[...,1] = xp[...,4]*xp[...,5]/k2
		vg[...,2] = xp[...,4]*xp[...,6]/k2
		vg[...,3] = xp[...,4]*xp[...,7]/k2
		return vg

class sellmeier_medium(weak_dispersion):
	def __init__(self,mks_length):
		# need the frequency normalization in rad/s
		self.w1 = C.c/mks_length
		# sellmeier units are A[dimensionless] and l2[microns^2]
		# setting all terms to 0.0 results in vacuum
		self.A = np.array([0.0])
		self.l2 = np.array([0.0])
	def Dxk(self,dens):
		terms = self.A.shape[0]
		cl_code = 'const double w2 = k.s0*k.s0;\n'
		cl_code += 'const double l20 = ' + str((1e6*2.0*np.pi*C.c/self.w1)**2) + '/w2;\n'
		cl_code += 'const double A['+str(terms)+'] = {' + str(self.A[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.A[i+1])
		cl_code += '};\n'
		cl_code += 'const double l2['+str(terms)+'] = {' + str(self.l2[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.l2[i+1])
		cl_code += '};\n'
		cl_code += 'const double chi = A[0]*l20/(l20-l2[0])'
		for i in range(terms-1):
			cl_code += ' + A['+str(i+1)+']*l20/(l20-l2['+str(i+1)+'])'
		cl_code += ';\n'
		cl_code += 'return -dot4(k,k)-w2*chi*' + dens + ';\n'
		return cl_code
	def chi(self,w):
		terms = self.A.shape
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		# form squared wavelength in microns^2
		l20 = (1e6*2.0*np.pi*C.c/(w*self.w1))**2
		# in the following np.einsum works like np.outer, except without flattening
		numerator = np.einsum('...,k',l20,self.A)
		denominator = np.einsum('...,k',l20,np.ones(terms)) - np.einsum('...,k',np.ones(freqs),self.l2)
		mat = numerator / denominator
		return np.sum(mat,axis=-1)

class sellmeier_medium_alt1(weak_dispersion):
	def __init__(self,mks_length):
		# need the frequency normalization in rad/s
		self.w1 = C.c/mks_length
		# sellmeier units are B[microns^-2] and C[microns^-2]
		# setting all terms to 0.0 results in vacuum
		self.B = np.array([0.0])
		self.C = np.array([0.0])
	def Dxk(self,dens):
		terms = self.B.shape[0]
		cl_code = 'const double w2 = k.s0*k.s0;\n'
		cl_code += 'const double l2i = ' + str((self.w1/(1e6*2.0*np.pi*C.c))**2) + '*w2;\n'
		cl_code += 'const double B['+str(terms)+'] = {' + str(self.B[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.B[i+1])
		cl_code += '};\n'
		cl_code += 'const double C['+str(terms)+'] = {' + str(self.C[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.C[i+1])
		cl_code += '};\n'
		cl_code += 'const double chi = B[0]/(C[0]-l2i)'
		for i in range(terms-1):
			cl_code += ' + B['+str(i+1)+']/(C['+str(i+1)+']-l2i)'
		cl_code += ';\n'
		cl_code += 'return -dot4(k,k)-w2*chi*' + dens + ';\n'
		return cl_code
	def chi(self,w):
		terms = self.B.shape
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		# form inverse squared wavelength in microns^-2
		l2i = (1e6*2.0*np.pi*C.c/(w*self.w1))**-2
		# in the following np.einsum works like np.outer, except without flattening
		numerator = np.einsum('...,k',np.ones(freqs),self.B)
		denominator = np.einsum('...,k',np.ones(freqs),self.C) - np.einsum('...,k',l2i,np.ones(terms))
		mat = numerator / denominator
		return np.sum(mat,axis=-1)

class Air(sellmeier_medium_alt1):
	# Sellmeier formula from Eq. 2, A.A. Voronin & A.M. Zheltikov, Nat. Sci. Rep. DOI:10.1038/srep46111
	# THIS IS THE SIMPLIFIED ONE valid from 0.2 to 1.0 microns
	def __init__(self,mks_length):
		self.w1 = C.c/mks_length
		self.B = np.array([0.05792105,0.00167917])
		self.C = np.array([238.0185,57.362])

class LiquidWater(sellmeier_medium):
	# Sellmeier formula from appl. opt. 46 (18), 3811 (2007)
	# T = 21.5 deg C
	def __init__(self,mks_length):
		self.w1 = C.c/mks_length
		self.A = np.array([5.689e-1,1.720e-1,2.063e-2,1.124e-1])
		self.l2 = np.array([5.110e-3,1.825e-2,2.624e-2,1.068e1])

class BK7(sellmeier_medium):
	# Sellmeier formula from https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
	def __init__(self,mks_length):
		self.w1 = C.c/mks_length
		self.A = np.array([1.03961212,0.231792344,1.01046945])
		self.l2 = np.array([0.00600069867,0.0200179144,103.560653])
