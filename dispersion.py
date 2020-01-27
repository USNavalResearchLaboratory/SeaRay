'''
Module :samp:`dispersion`
-------------------------

Dispersion objects are used to specify a dispersion relation inside a ``volume`` object, or on either side of a ``surface`` object.  The simplest way to use the dispersion module is to reference predefined materials.  The available materials are listed in the following example::

	mks_length = 0.8e-6/(2*np.pi)
	air1 = dispersion.SimpleAir(mks_length)
	air2 = dispersion.DryAir(mks_length)
	air3 = dispersion.HumidAir(mks_length,0.4)
	water = dispersion.LiquidWater(mks_length)
	glass = dispersion.BK7(mks_length)
	salt = dispersion.NaCl(mks_length)

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

	Retrieve the group velocity for rays with phase space data xp.  Must be written to accept variable number of dimensions (use ellipsis notation).

	:param numpy.array xp: The phase space data of the rays with shape (...,8)
	:returns: the group velocity of the rays with shape (...,4)

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
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		return np.zeros(freqs)
	def dchidw(self,w):
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		return np.zeros(freqs)
	def GroupVelocityMagnitude(self,w):
		try:
			freqs = w.shape
		except AttributeError:
			freqs = (1,)
		test_xp = np.zeros(freqs+(8,))
		test_xp[...,4] = w
		test_xp[...,5] = w*np.real(np.sqrt(1+self.chi(w)))
		v = self.vg(test_xp)[...,1:]
		if freqs==(1,):
			v = v[0,...]
		return np.sqrt(np.einsum('...i,...i',v,v))


class ColdPlasma(Vacuum):
	def Dxk(self,dens):
		return 'return ' + dens + ' - dot4(k,k);\n'
	def chi(self,w):
		return -1/w**2
	def dchidw(self,w):
		return 2/w**3

class ParaxialPlasma(Vacuum):
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
	def dchidw(self,w):
		return 2/w**3

class attenuator(Vacuum):
	def __init__(self,mks_length):
		self.mks_length = mks_length
		self.alpha = []
		self.wstop1 = []
		self.wstop2 = []
	def reset_opacity(self):
		self.alpha = []
		self.wstop1 = []
		self.wstop2 = []
	def add_opacity_region(self,alpha,wavelength1,wavelength2):
		'''Add a phenomenological opacity region.

		:param double alpha: attenuation in nepers/meter
		:param double wavelength1: lower wavelength in meters
		:param double wavelength2: upper wavelength in meters'''
		w1 = 2*np.pi*self.mks_length/wavelength2
		w2 = 2*np.pi*self.mks_length/wavelength1
		self.alpha.append(alpha*self.mks_length)
		self.wstop1.append(w1)
		self.wstop2.append(w2)
	def chi(self,w):
		a = np.array(self.alpha)
		w1 = np.array(self.wstop1)
		w2 = np.array(self.wstop2)
		try:
			freqs = w.shape
			squeeze = False
		except AttributeError:
			w = np.array([w])
			freqs = w.shape
			squeeze = True
		terms = a.shape[0]
		if terms>0:
			weps = 1e-4*np.min(w1+w2)
			opacity = np.zeros(w.shape+a.shape).astype(np.complex)
			wx = np.einsum('...,k',w,np.ones(terms))
			w1x = np.einsum('...,k',np.ones(freqs),w1)
			w2x = np.einsum('...,k',np.ones(freqs),w2)
			sel = np.where(np.logical_and(wx>w1x,wx<w2x))
			# imag(chi) = 2*real(n)*alpha/w, here we approximate n=1
			opacity[sel] = 1j*np.einsum('...,k',1/(weps+w),2*a)[sel]
			ans = np.sum(opacity,axis=-1)
		else:
			ans = np.zeros(w.shape)
		if squeeze:
			ans = np.squeeze(ans)
		return ans

class isotropic_medium(attenuator):
	def vg(self,xp):
		vg = np.copy(xp[...,:4])
		chi = self.chi(xp[...,4])
		dchidw = self.dchidw(xp[...,4])
		denom = xp[...,4]*(1.0 + chi + 0.5*xp[...,4]*dchidw)
		vg[...,0] = 1.0
		vg[...,1] = np.real(xp[...,5]/denom)
		vg[...,2] = np.real(xp[...,6]/denom)
		vg[...,3] = np.real(xp[...,7]/denom)
		return vg

class sellmeier_medium(isotropic_medium):
	r'''Dispersion object supporting susceptibility in the form:

	:math:`\chi = \sum_i\frac{A_i \lambda^2}{\lambda^2-L_i}`'''
	def __init__(self,mks_length,A,L):
		'''Construct the dispersion object.

		:param float mks_length: the simulation unit of length in meters
		:param numpy.array A: dimensionless coefficients in sum
		:param numpy.array L: terms in sum with dimension of microns^2

		Example::

			x1 = 0.8e-6/(2*np.pi)
			A = np.array([0.0])
			L = np.array([0.0])
			vacuum = dispersion.sellmeier_medium(x1,A,L)'''
		super().__init__(mks_length)
		self.A = A
		# Translation of L to a normalized coefficient in terms of normalized frequency
		self.D = L*1e-12 / self.mks_length**2 / (2*np.pi)**2
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
		opacity = super().chi(w)
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		numerator = np.einsum('...,k',np.ones(freqs),self.A)
		denominator = 1 - np.einsum('...,k',w**2,self.D)
		mat = numerator / denominator
		return opacity + np.sum(mat,axis=-1)
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

	:math:`\chi = \sum_i\frac{B_i}{C_i-\lambda^{-2}}`'''
	def __init__(self,mks_length,B,C):
		'''Construct the dispersion object.

		:param float mks_length: the simulation unit of length in meters
		:param numpy.array B: terms in sum with dimension of microns^-2
		:param numpy.array C: terms in sum with dimension of microns^-2

		Example::

			x1 = 0.8e-6/(2*np.pi)
			B = np.array([0.0])
			C = np.array([0.0])
			vacuum = dispersion.sellmeier_medium_alt1(x1,B,C)'''
		super().__init__(mks_length)
		# normalized coefficients, in terms of normalized frequency
		self.Bn = (2.0*np.pi)**2 * self.mks_length**2 * 1e12*B
		self.Cn = (2.0*np.pi)**2 * self.mks_length**2 * 1e12*C
	def Dxk(self,dens):
		terms = self.Bn.shape[0]
		cl_code = 'const double w2 = k.s0*k.s0;\n'
		cl_code += 'const double B['+str(terms)+'] = {' + str(self.Bn[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.Bn[i+1])
		cl_code += '};\n'
		cl_code += 'const double C['+str(terms)+'] = {' + str(self.Cn[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.Cn[i+1])
		cl_code += '};\n'
		cl_code += 'const double chi = B[0]/(C[0]-w2)'
		for i in range(terms-1):
			cl_code += ' + B['+str(i+1)+']/(C['+str(i+1)+']-w2)'
		cl_code += ';\n'
		cl_code += 'return -dot4(k,k)-w2*chi*' + dens + ';\n'
		return cl_code
	def chi(self,w):
		opacity = super().chi(w)
		terms = self.Bn.shape[0]
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		numerator = np.einsum('...,k',np.ones(freqs),self.Bn)
		denominator = np.einsum('...,k',np.ones(freqs),self.Cn) - np.einsum('...,k',w**2,np.ones(terms))
		mat = numerator / denominator
		return opacity + np.sum(mat,axis=-1)
	def dchidw(self,w):
		terms = self.Bn.shape[0]
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		numerator = np.einsum('...,k',2*w,self.Bn)
		denominator = (np.einsum('...,k',np.ones(freqs),self.Cn) - np.einsum('...,k',w**2,np.ones(terms)))**2
		mat = numerator / denominator
		return np.sum(mat,axis=-1)

class sellmeier_medium_alt2(isotropic_medium):
	r'''Dispersion object supporting susceptibility in the form:

	:math:`\chi = 2\sum_r \frac{N_r}{N_c} \frac{A_r\lambda_r^2}{\lambda^2-\lambda_r^2}`.

	This is a generalization of Eq. 4 from A.A. Voronin & A.M. Zheltikov, Nat. Sci. Rep. DOI:10.1038/srep46111.
	N.b. factor of 2 converting from index perturbation to susceptibility is appropriate only for tenuous media.'''
	def __init__(self,mks_length,Nr,Ar,lr,rel_linewidth):
		'''Construct the dispersion object.

		:param float mks_length: the simulation unit of length in meters
		:param numpy.array Nr: terms in sum with dimension of cm^-3
		:param numpy.array Ar: terms in sum, dimensionless
		:param numpy.array lr: terms in sum with dimensions of nm
		:param numpy.array rel_linewidth: relative linewidth of effective transitions, can be single value or an array.'''
		# normalized coefficients, in terms of normalized frequency
		# In these units N_r/N_c goes to 1/w**2.
		# With all wavelengths reduced by 2*pi we can use w*lambda = 1.
		# For sake of efficiency gather factors in numerator into single variable
		super().__init__(mks_length)
		ncrit = C.m_e * C.epsilon_0 * (C.c/self.mks_length)**2 / C.e**2
		Nn = Nr*1e6/ncrit
		An = Ar
		self.ln2 = (1e-9*lr / (2.0*np.pi*self.mks_length))**2
		self.Neff = 2*Nn*An*self.ln2
		self.wres = 1/np.sqrt(self.ln2)
		if type(rel_linewidth)==np.float or type(rel_linewidth)==np.double:
			self.linewidth = np.ones(self.Neff.shape[0])*rel_linewidth
		else:
			self.linewidth = rel_linewidth
	def Dxk(self,dens):
		terms = self.Neff.shape[0]
		cl_code = 'const double w2 = k.s0*k.s0;\n'
		cl_code += 'const double N['+str(terms)+'] = {' + str(self.Neff[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.Neff[i+1])
		cl_code += '};\n'
		cl_code += 'const double l2['+str(terms)+'] = {' + str(self.ln2[0])
		for i in range(terms-1):
			cl_code += ',' + str(self.ln2[i+1])
		cl_code += '};\n'
		cl_code += 'const double chi = N[0]/(1.0-w2*l2[0])'
		for i in range(terms-1):
			s = str(i+1)
			cl_code += ' + N['+s+']/(1.0-w2*l2['+s+'])'
		cl_code += ';\n'
		cl_code += 'return -dot4(k,k)-w2*chi*' + dens + ';\n'
		return cl_code
	def chi(self,w):
		opacity = super().chi(w)
		terms = self.Neff.shape[0]
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		# Transparency regions with hybrid Gauss-Lorentz linewidths
		numerator = np.einsum('...,k',np.ones(freqs),self.Neff)
		trans = 1 - np.einsum('...,k',w**2,self.ln2)
		atten = 2j*np.einsum('...,k',np.ones(freqs),self.linewidth)
		dw = np.einsum('...,k',w,np.ones(terms)) - np.einsum('...,k',np.ones(freqs),self.wres)
		dw /= self.linewidth*self.wres
		denominator = trans + np.exp(-dw**2)*atten
		mat = numerator / denominator
		return opacity + np.sum(mat,axis=-1)
	def dchidw(self,w):
		terms = self.Neff.shape[0]
		try:
			freqs = w.shape
		except AttributeError:
			freqs = 1
		numerator = np.einsum('...,k',2*w,self.Neff*self.ln2)
		denominator = (1 - np.einsum('...,k',w**2,self.ln2))**2
		mat = numerator / denominator
		return np.sum(mat,axis=-1)

class SimpleAir(sellmeier_medium_alt1):
	# Sellmeier formula from Eq. 2, A.A. Voronin & A.M. Zheltikov, Nat. Sci. Rep. DOI:10.1038/srep46111
	# THIS IS THE SIMPLIFIED ONE valid from 0.2 to 1.0 microns
	def __init__(self,mks_length):
		super().__init__(mks_length,
			B=2*np.array([0.05792105,0.00167917]),
			C=np.array([238.0185,57.362]))

class DryAir(sellmeier_medium_alt2):
	# GSE, A.A. Voronin & A.M. Zheltikov, Nat. Sci. Rep. DOI:10.1038/srep46111
	# Using contributions from r=1,2,3,4,12,13,14 in table 1.
	def __init__(self,mks_length,rel_linewidth=0.0):
		N = 1e15*np.array([9.4136,9.4136,9.4136,9.4136,19870,5329.1,237.63])
		A1 = np.array([4.051e-6,2.897e-5,8.573e-7,1.55e-8,1.2029482,0.26507582,0.93132145])
		l1 = np.array([15131,4290.9,2684.9,2011.3,85,127,87])
		A2 = np.array([1.01e-6,2.728e-5,6.62e-7,5.532e-9,5.796725,7.734925,7.217322])
		l2 = np.array([14218,4223.1,2769.1,1964.6,24.546,29.469,22.645])
		Nd = np.concatenate((N,N))
		Ad = np.concatenate((A1,A2))
		ld = np.concatenate((l1,l2))
		super().__init__(mks_length,Nr=Nd,Ar=Ad,lr=ld,rel_linewidth=rel_linewidth)

class HumidAir(sellmeier_medium_alt2):
	# GSE, A.A. Voronin & A.M. Zheltikov, Nat. Sci. Rep. DOI:10.1038/srep46111
	# Using all contributions from table 1.
	def __init__(self,mks_length,rel_hum,rel_linewidth=0.0):
		# Dry air contribution
		N = 1e15*np.array([9.4136,9.4136,9.4136,9.4136,19870,5329.1,237.63])
		A1 = np.array([4.051e-6,2.897e-5,8.573e-7,1.55e-8,1.2029482,0.26507582,0.93132145])
		l1 = np.array([15131,4290.9,2684.9,2011.3,85,127,87])
		A2 = np.array([1.01e-6,2.728e-5,6.62e-7,5.532e-9,5.796725,7.734925,7.217322])
		l2 = np.array([14218,4223.1,2769.1,1964.6,24.546,29.469,22.645])
		Nd = np.concatenate((N,N))
		Ad = np.concatenate((A1,A2))
		ld = np.concatenate((l1,l2))
		# Compute density of water vapor
		T = 296.0
		Tc = 647.096
		a1 = -7.85951783
		a2 = 1.84408259
		a3 = -11.7866497
		a4 = 22.6807411
		a5 = -15.9618719
		a6 = 1.80122502
		t = 1 - T/Tc
		ps = 22.064e6*np.exp((Tc/T)*(a1*t + a2*t**1.5 + a3*t**3 + a4*t**3.5 + a5*t**4 + a6*t**7.5))
		self.NH2O_mks = rel_hum*ps/(C.k*T)
		# Add in water vapor contribution
		N = np.ones(8)*self.NH2O_mks*1e-6
		A1 = np.array([2.945e-5,3.273e-6,1.862e-6,2.544e-7,1.126e-7,6.856e-9,1.985e-9,0.25787285])
		l1 = np.array([47862,6719,2775.6,1835.6,1417.6,1145.3,947.73,128])
		A2 = np.array([6.583e-8,3.094e-6,2.788e-6,2.181e-7,2.336e-7,9.479e-9,2.882e-9,4.742131])
		l2 = np.array([16603,5729.9,2598.5,1904.8,1364.7,1123.2,935.09,34.924])
		Nh = np.concatenate((Nd,N,N))
		Ah = np.concatenate((Ad,A1,A2))
		lh = np.concatenate((ld,l1,l2))
		super().__init__(mks_length,Nr=Nh,Ar=Ah,lr=lh,rel_linewidth=rel_linewidth)
	def WaterVaporDensity(self):
		return self.NH2O_mks

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

class NaCl(sellmeier_medium):
	def __init__(self,mks_length):
		super().__init__(mks_length,
			A=np.array([0.00055,0.198,0.48398,0.38696,0.25998,0.08796,3.17064,0.30038]),
			L=np.array([0.0,0.05,0.1,0.128,0.158,40.5,60.98,120.34])**2)

class ZnSe(sellmeier_medium):
	# Sellmeier formula from https://refractiveindex.info/?shelf=main&book=ZnSe&page=Connolly
	def __init__(self,mks_length):
		super().__init__(mks_length,
			A=np.array([4.45813734,0.467216334,2.89566290]),
			L=np.array([0.200859853,0.391371166,47.1362108])**2)

class Ge(sellmeier_medium):
	# Sellmeier formula from https://refractiveindex.info/?shelf=main&book=Ge&page=Burnett
	def __init__(self,mks_length):
		super().__init__(mks_length,
			A=np.array([0.4886331,14.5142535,0.0091224]),
			L=np.array([1.393959,0.1626427,752.190]))
