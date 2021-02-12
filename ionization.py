'''
Module :samp:`ionization`
====================================

This is a high level wrapper for the ``photoionization_base`` module.
The premise is to use the rates from ``photoionization_base`` to work out coefficients or curve fits
that can be used for accelerated computations during a simulation.
This also insulates the lower level objects from any need to know about simulation units.
'''

import numpy as np
from scipy import special as spsf
from scipy import constants as C
import scipy.integrate
import photoionization_base as izbase

def GetPlasmaDensity(rate,ngas,dt):
	'''Get plasma density from rate, accounting for gas depletion.
	This is also provided as a method of ``Ionizer``.

	:param numpy.array rate: ionization rate with shape (Nt,Nx,Ny), any units consistent with dt.
	:param numpy.array ngas: gas density with shape (Nx,Ny), any units.
	:param double dt: time step, any units consistent with rate
	:returns: electron density in same units as ngas.'''
	ne_ng = scipy.integrate.cumtrapz(rate[::-1],dx=dt,axis=0,initial=0.0)[::-1]
	return ngas[np.newaxis,...]*(1.0 - np.exp(-ne_ng))
def GetPlasmaDensityCL(cl,shp,ne,ngas,dt):
	'''Similar to GetPlasmaDensity, but runs on OpenCL compute device.
	This is also provided as a method of ``Ionizer``.

	:param tuple shp: shape in the form (Nt,Nx,Ny)
	:param cl_data ne: the data member of the pyopencl array for ne, which on input has the rate
	:param cl_data ngas: the data member of the pyopencl array for ngas, which has the gas density with shape (Nx,Ny)
	:returns: no return value, but plasma density is loaded into ne on output'''
	cl.program('ionization').ComputePlasmaDensity(cl.q,shp[1:3],None,ne,ngas,np.double(dt),np.int32(shp[0]))

class UnitTranslation(izbase.AtomicUnits):
	def __init__(self,mks_length):
		# superclass provides atomic units in mks units
		super().__init__()
		# simulation units in mks units
		self.xs = mks_length
		self.ts = mks_length/C.c
		self.us = C.m_e*C.c**2
		self.es = self.us/self.xs/C.e
		# unit of density in mks
		w0 = C.c/mks_length
		self.ncrit = w0**2 * C.epsilon_0 * C.m_e / C.e**2
	def rate_sim_to_au(self,rate):
		return rate*self.ta/self.ts
	def rate_au_to_sim(self,rate):
		return rate*self.ts/self.ta
	def field_sim_to_au(self,field):
		return field*self.es/self.ea
	def field_au_to_sim(self,field):
		return field*self.ea/self.es
	def energy_sim_to_au(self,energy):
		return energy*self.us/self.ua
	def energy_au_to_sim(self,energy):
		return energy*self.us/self.ua

# The following wrapper classes allow the rate generator to be created using
# simulation units.  N.b. the rate is still returned in atomic units.
# Within SeaRay the wrappers are not called directly, instead they are plugins for the Ionizer class.

class rate_wrapper:
	def __init__(self,mks_length,averaging,Uion,Z,lstar=0,l=0,m=0,w0=0,terms=1):
		self.mks_length = mks_length
		uc = UnitTranslation(mks_length)
		super().__init__(averaging,uc.energy_sim_to_au(Uion),Z,lstar,l,m,uc.rate_sim_to_au(w0),terms)
	def get_mks_length(self):
		return self.mks_length

class ADK(rate_wrapper,izbase.ADK):
	'''Wrapper for base ADK ionization class'''

class PPT_Tunneling(rate_wrapper,izbase.PPT_Tunneling):
	'''Wrapper for base ADK ionization class'''

class PPT(rate_wrapper,izbase.PPT):
	'''Wrapper for base PPT ionization class'''

class PMPB(rate_wrapper,izbase.PMPB):
	'''Wrapper for base PMPB ionization class'''

class StitchedPPT(rate_wrapper,izbase.StitchedPPT):
	'''Wrapper for base StitchedPPT ionization class'''

class Ionizer(UnitTranslation):
	def __init__(self,rate_generator,pts=256,mks_length=0.0):
		'''Create an ionization object.
		If the rate_generator is from the base module, then the mks_length should be provided.
		If the rate_generator is one of the wrappers from this module, mks_length can be omitted.

		:param double mks_length: normalizing length in mks units
		:param double rate_generator: ionization rate object
		:param int pts: number of points to use for curve fitting'''
		if mks_length==0.0:
			super().__init__(rate_generator.get_mks_length())
		else:
			super().__init__(mks_length)
		# Underlying coefficients are kept in atomic units
		self.cutoff_field = 1e-3
		self.rate_gen = rate_generator
		self.C1,self.C2,self.C3 = self.rate_gen.GetCoeff()
		if self.C1==0.0:
			model = lambda x,c0,c1,c2,c3,c4,c5 : c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5
			E_table = np.logspace(np.log10(self.cutoff_field*2),np.log10(1),pts)
			rate_table = self.rate_gen.Rate(E_table)
			self.Clog,var=scipy.optimize.curve_fit(model,np.log(E_table),np.log(rate_table))
	def GetPlasmaDensity(self,rate,ngas,dt):
		'''Wraps the module function of the same name.'''
		return GetPlasmaDensity(rate,ngas,dt)
	def GetPlasmaDensityCL(self,cl,shp,ne,ngas,dt):
		'''Wraps the module function of the same name.'''
		return GetPlasmaDensityCL(cl,shp,ne,ngas,dt)
	def Rate(self,Es):
		'''Get the ionization rate over all space, returned value is in simulation units.

		:param np.array Es: Time domain electric field in simulation units, any shape.
		Dynamics-aware rate generators expect time to be the first axis.
		If using a complex envelope, |Es| should give the crest.'''
		if self.C1>0.0:
			E = self.field_sim_to_au(np.abs(Es)) + self.cutoff_field
			return self.rate_au_to_sim(self.C1*E**self.C2*np.exp(self.C3/E))
		else:
			ans = np.zeros(Es.shape)
			logE = np.log(self.field_sim_to_au(np.abs(Es)) + self.cutoff_field)
			for i in range(6):
				ans += self.Clog[i]*logE**i
			return self.rate_au_to_sim(np.exp(ans))
	def RateCL(self,cl,shp,rate,Es,envelope):
		field_conv = self.field_sim_to_au(1.0)
		rate_conv = self.rate_au_to_sim(1.0)
		if self.C1>0.0:
			if envelope:
				k = cl.program('ionization').EnvelopeRate
			else:
				k = cl.program('ionization').ExplicitRate
			k(cl.q,shp,None,rate,Es,
				np.double(field_conv),np.double(rate_conv),np.double(self.cutoff_field),
				np.double(self.C1),np.double(self.C2),np.double(self.C3))
		else:
			cn = np.array(self.Clog).astype(np.double)
			if envelope:
				k = cl.program('ionization').EnvelopeRateSeries
			else:
				k = cl.program('ionization').ExplicitRateSeries
			k(cl.q,shp,None,rate,Es,
				np.double(field_conv),np.double(rate_conv),np.double(self.cutoff_field),
				cn[0],cn[1],cn[2],cn[3],cn[4],cn[5])
