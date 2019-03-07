'''
Module :samp:`ionization`
-------------------------

Ionization objects describe an ionizing medium inside a ``volume`` object.

Ionization Classes
,,,,,,,,,,,,,,,,,,,
'''
import numpy as np
from scipy import constants as C
import scipy.integrate

class AtomicUnits:
	def __init__(self,mks_length):
		# simulation units in mks units
		self.xs = mks_length
		self.ts = mks_length/C.c
		self.es = C.m_e*C.c/(self.ts*C.e)
		# atomic units in mks units
		self.xa = C.hbar / (C.m_e*C.c*C.alpha)
		self.ta = C.hbar / (C.m_e*C.c**2*C.alpha**2)
		self.ua = C.m_e*C.c**2*C.alpha**2
		self.ea = self.ua/self.xa/C.e
		# unit of density in mks
		w0 = C.c/mks_length
		self.ncrit = w0**2 * C.epsilon_0 * C.m_e / C.e**2
	def rate_au_to_sim(self,rate):
		return rate*self.ts/self.ta
	def field_sim_to_au(self,field):
		return field*self.es/self.ea

class Ionization(AtomicUnits):
	def __init__(self,Uion,Z,mks_density,mks_length):
		'''Create an ionization object.  Base class returns zero rate.

		:param double Uion: ionization potential in atomic units
		:param double Z: residual charge in atomic units
		:param double mks_density: volume object's reference density in mks units
		:param double mks_length: normalizing length in mks units'''
		super().__init__(mks_length)
		self.ref_dens = mks_density
		self.Uion = Uion
		self.Z = Z
	def GetPlasmaDensity(self,ngas,rate,dt):
		'''Get plasma density from rate, accounting for gas depletion.

		:param numpy.array ngas: gas density normalized to the volume's reference with shape (Nx,Ny).
		:param numpy.array rate: ionization rate in simulation units with shape (Nt,Nx,Ny).
		:param double dt: time step in simulation units.
		:returns: electron density in simulation units.'''
		ne = scipy.integrate.cumtrapz(rate[::-1],dx=dt,axis=0,initial=0.0)[::-1]
		ne = ngas[np.newaxis,...]*(1.0 - np.exp(-ne))*self.ref_dens/self.ncrit
		ne[:5] = 0
		return ne
	def InstantaneousRate(self,Es):
		'''Instantaneous tunneling rate

		:param np.array Es: electric field in simulation units, any shape.
		:returns: rate in simulation units.'''
		return np.zeros(Es.shape)
	def AverageRate(self,Es):
		'''Cycle averaged tunneling rate

		:param np.array Es: peak of the electric field in simulation units, any shape.
		:returns: rate in simulation units.'''
		return np.zeros(Es.shape)

class ADK(Ionization):
	def InstantaneousRate(self,Es):
		E = self.field_sim_to_au(Es)
		Uion = self.Uion
		Z = self.Z
		nstar = Z/np.sqrt(2*Uion)
		D = ((4.0*np.exp(1.0)*Z**3)/(E*nstar**4))**nstar
		ans = (E*D*D/(8.0*np.pi*Z))*np.exp(-2*Z**3/(3.0*nstar**3*E))
		return self.rate_au_to_sim(ans)
