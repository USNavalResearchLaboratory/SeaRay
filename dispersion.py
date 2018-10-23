'''
Module :samp:`dispersion`
-------------------------

Dispersion objects are used to specify a dispersion relation inside a ``volume`` object, or on either side of a ``surface`` object.  The simplest way to use the dispersion module is to reference predefined materials.  The available materials are listed in the following example::

	mks_length = 0.8e-6/(2*np.pi)
	air = dispersion.Air(mks_length)
	water = dispersion.LiquidWater(mks_length)
	glass = dispersion.BK7(mks_length)

It is simple to add a new medium to SeaRay if the dispersion relation can be put in Sellmeier form.  To do so derive the new object from ``sellmeier_medium`` and provide the arrays of coefficients.  See, e.g., the BK7 object for an example.  You can also create a Sellmeier medium on the fly.

Required Methods
,,,,,,,,,,,,,,,,,,

For dispersion that cannot be put in Sellmeier form you have to provide the following low level methods:

.. py:method:: chi(w)

	Calculate the susceptibility at a given frequency or set of frequenices.  For compressible media, the susceptibility is given at a specific reference density.

	:param double w: the frequency or array frequencies to evaluate
	:returns: susceptibility or array of susceptibilities corresponding to ``w``

.. py:method:: Dxk(dens)

	Creates a string containing OpenCL code that computes the local dispersion function

	:math:`D(x,k)`

	for use in ray tracing through non uniform media (this is the ray Hamiltonian, up to a parameterization).

	:param str dens: this string must contain an OpenCL function call that retrieves the density relative to the reference density.
	:returns: String containing OpenCL code that computes the dispersion function

.. py:method:: vg(xp)

	:param numpy.array xp: The phase space data of the rays with shape (bundles,rays,8)
	:returns: the group velocity of the rays with shape (bundles,rays,4)

Dispersion Classes
,,,,,,,,,,,,,,,,,,
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
	def chi(self,w):
		return -1/w**2

class isotropic_medium:
	def vg(self,xp):
		vg = np.copy(xp[...,:4])
		chi = self.chi(xp[...,4])
		dchidw = self.dchidw(xp[...,4])
		denom = xp[...,4]*(1.0 + chi + 0.5*xp[...,4]*dchidw)
		vg[...,0] = 1.0
		vg[...,1] = xp[...,5]/denom
		vg[...,2] = xp[...,6]/denom
		vg[...,3] = xp[...,7]/denom
		return vg

class sellmeier_medium(isotropic_medium):
	r'''Dispersion object supporting susceptibility in the form:

	:math:`\chi = \sum_i\frac{A_i \lambda^2}{\lambda^2-L_i}`'''
	def __init__(self,mks_length,A,L):
		'''Construct the dispersion object.

		:param float mks_length: the unit of length in meters
		:param numpy.array A: dimensionless coefficients in sum
		:param numpy.array L: terms in sum with dimension of microns^2

		Example::

			x1 = 0.8e-6/(2*np.pi)
			A = np.array([0.0])
			L = np.array([0.0])
			vacuum = dispersion.sellmeier_medium(x1,A,L)'''
		self.A = A
		# Translation of L to a normalized coefficient in terms of normalized frequency
		self.D = L*1e-12 / mks_length**2 / (2*np.pi)**2
	def Dxk(self,dens):
		terms = self.A.shape[0]
		cl_code = 'const double w2 = k.s0*k.s0;\n'
		cl_code += 'const double A['+str(terms)+'] = {' + str(self.A[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.A[i+1])
		cl_code += '};\n'
		cl_code += 'const double D['+str(terms)+'] = {' + str(self.D[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.D[i+1])
		cl_code += '};\n'
		cl_code += 'const double chi = A[0]/(1.0-D[0]*w2)'
		for i in range(terms-1):
			cl_code += ' + A['+str(i+1)+']/(1.0-D['+str(i+1)+']*w2)'
		cl_code += ';\n'
		cl_code += 'return -dot4(k,k)-w2*chi*' + dens + ';\n'
		return cl_code
	def chi(self,w):
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		numerator = np.einsum('...,k',np.ones(freqs),self.A)
		denominator = 1 - np.einsum('...,k',w**2,self.D)
		mat = numerator / denominator
		return np.sum(mat,axis=-1)
	def dchidw(self,w):
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		numerator = np.einsum('...,k',2*w,self.A*self.D)
		denominator = (1.0 - np.einsum('...,k',w**2,self.D))**2
		mat = numerator / denominator
		return np.sum(mat,axis=-1)

class sellmeier_medium_alt1(isotropic_medium):
	r'''Dispersion object supporting susceptibility in the form:

	:math:`\chi = \sum_i\frac{B_i}{C_i-\lambda^{-2}}`

	Similar in all respects to ``sellmeier_medium``.'''
	def __init__(self,mks_length,B,C):
		# normalized coefficients, in terms of normalized frequency
		self.Bn = (2.0*np.pi)**2 * mks_length**2 * 1e12*B
		self.Cn = (2.0*np.pi)**2 * mks_length**2 * 1e12*C
	def Dxk(self,dens):
		terms = self.B.shape[0]
		cl_code = 'const double w2 = k.s0*k.s0;\n'
		cl_code += 'const double B['+str(terms)+'] = {' + str(self.Bn[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.B[i+1])
		cl_code += '};\n'
		cl_code += 'const double C['+str(terms)+'] = {' + str(self.Cn[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.C[i+1])
		cl_code += '};\n'
		cl_code += 'const double chi = B[0]/(C[0]-w2)'
		for i in range(terms-1):
			cl_code += ' + B['+str(i+1)+']/(C['+str(i+1)+']-w2)'
		cl_code += ';\n'
		cl_code += 'return -dot4(k,k)-w2*chi*' + dens + ';\n'
		return cl_code
	def chi(self,w):
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		numerator = np.einsum('...,k',np.ones(freqs),self.Bn)
		denominator = np.einsum('...,k',np.ones(freqs),self.Cn) - np.einsum('...,k',w**2,np.ones(terms))
		mat = numerator / denominator
		return np.sum(mat,axis=-1)
	def dchidw(self,w):
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		numerator = np.einsum('...,k',2*w,self.Bn)
		denominator = (np.einsum('...,k',np.ones(freqs),self.Cn) - np.einsum('...,k',w**2,np.ones(terms)))**2
		mat = numerator / denominator
		return np.sum(mat,axis=-1)

class Air(sellmeier_medium_alt1):
	# Sellmeier formula from Eq. 2, A.A. Voronin & A.M. Zheltikov, Nat. Sci. Rep. DOI:10.1038/srep46111
	# THIS IS THE SIMPLIFIED ONE valid from 0.2 to 1.0 microns
	def __init__(self,mks_length):
		super().__init__(mks_length,
			B=np.array([0.05792105,0.00167917]),
			C=np.array([238.0185,57.362]))

class LiquidWater(sellmeier_medium):
	# Sellmeier formula from appl. opt. 46 (18), 3811 (2007)
	# T = 21.5 deg C
	def __init__(self,mks_length):
		super().__init__(mks_length,
			A=np.array([5.689e-1,1.720e-1,2.063e-2,1.124e-1]),
			L=np.array([5.110e-3,1.825e-2,2.624e-2,1.068e1]))

class BK7(sellmeier_medium):
	# Sellmeier formula from https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
	def __init__(self,mks_length):
		super().__init__(mks_length,
			A=np.array([1.03961212,0.231792344,1.01046945]),
			L=np.array([0.00600069867,0.0200179144,103.560653]))
