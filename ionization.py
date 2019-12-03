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
import scipy.special as spsf

def wfunc(x):
	"""Exponential integral from PPT theory"""
	return np.real(0.5 * np.exp(-x**2) * np.sqrt(np.pi) * spsf.erf(x*1j) / 1j)

def Cycle_Averaging_Factor(Uion,E):
	"""Returns the PPT cycle averaging factor
	Multiply static rate by this factor to get average rate
	inputs (a.u.):
	Uion = ionization potential
	E = electric field"""
	return np.sqrt(3.0/np.pi) * np.sqrt(E) / (2*Uion)**0.75

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
	def rate_sim_to_au(self,rate):
		return rate*self.ta/self.ts
	def rate_au_to_sim(self,rate):
		return rate*self.ts/self.ta
	def field_sim_to_au(self,field):
		return field*self.es/self.ea

class Ionization(AtomicUnits):
	def __init__(self,Uion,Z,mks_density,mks_length,dt=1.0,terms=1):
		'''Create an ionization object.  Base class returns zero rate.

		:param double Uion: ionization potential in atomic units
		:param double Z: residual charge in atomic units
		:param double mks_density: volume object's reference density in mks units
		:param double mks_length: normalizing length in mks units
		:param numpy.array dt: time step in simulation units
		:param int terms: terms to keep in PPT expansion'''
		super().__init__(mks_length)
		self.ref_dens = mks_density
		self.Uion = Uion
		self.Z = Z
		self.dt = dt
		self.terms = terms
		self.cutoff_field = 1e-3
	def ResetParameters(self,timestep=None,terms=None):
		if timestep!=None:
			self.dt = timestep
		if terms!=None:
			self.terms = terms
	def GetPlasmaDensity(self,ngas,rate):
		'''Get plasma density from rate, accounting for gas depletion.

		:param numpy.array ngas: gas density normalized to the volume's reference with shape (Nx,Ny).
		:param numpy.array rate: ionization rate in simulation units with shape (Nt,Nx,Ny).
		:returns: electron density in simulation units.'''
		ne = scipy.integrate.cumtrapz(rate[::-1],dx=self.dt,axis=0,initial=0.0)[::-1]
		ne = ngas[np.newaxis,...]*(1.0 - np.exp(-ne))*self.ref_dens/self.ncrit
		return ne
	def GetPlasmaDensityCL(self,ngas,ne,q,k):
		k(q,ngas.shape,None,ne.data,ngas.data,np.double(self.ref_dens/self.ncrit),np.double(self.dt),np.int32(ne.shape[0]))
	def ExtractEikonalForm(self,E,w00=0.0,bandwidth=1.0):
		"""Extract amplitude, phase, and center frequency from a carrier resolved field E.
		The assumed form is E = Re(amp*exp(i*phase)).

		:param numpy.array E: Electric field, any shape, axis 0 is time.
		:param double w00: Center frequency, if zero deduce from the data (default).
		:param double bandwidth: relative bandwidth to keep, if unity no filtering (default)."""
		Nt = E.shape[0]
		ndims = len(E.shape)
		dw = 2*np.pi/(Nt*self.dt)
		# Get the complexified spectrum and carrier frequency
		Ew = np.fft.fft(E,axis=0)
		Ew[np.int(Nt/2):,...] = 0.0 # eliminate negative frequencies before extracting carrier
		if w00==0.0:
			# Compute a suitable carrier frequency based on intensity weighting
			if ndims==1:
				indices = np.arange(0,Nt)
			else:
				indices = np.einsum('i,jk',np.arange(0,Nt),np.ones(Ew.shape[1:]))
			carrier_idx = np.int(np.average(indices,weights=np.abs(Ew)**2))
			w0 = carrier_idx*dw
			print(w0)
		else:
			# Carrier frequency is set by caller
			w0 = w00
			carrier_idx = np.int(w0/dw)
		# Bandpass filtering
		if bandwidth!=1.0:
			low_idx = np.int((w0-0.5*bandwidth*w0)/dw)
			high_idx = np.int((w0+0.5*bandwidth*w0)/dw)
			Ew[:low_idx] = 0.0
			Ew[high_idx:] = 0.0
		# Form the complex envelope in time
		Ew = np.roll(Ew,-carrier_idx,axis=0)
		Et = 2*np.fft.ifft(Ew,axis=0)
		# Get amplitude and phase
		amp = np.abs(Et)
		phase = (np.angle(Et).swapaxes(0,ndims-1) + w0*np.linspace(0,(Nt-1)*self.dt,Nt)).swapaxes(0,ndims-1)
		return amp,phase,w0
	def InstantaneousRate(self,Es):
		'''Instantaneous tunneling rate

		:param np.array Es: Carrier resolved electric field in simulation units, any shape.  Some subclasses demand that time be the first axis.
		:returns: rate in simulation units.'''
		return np.zeros(Es.shape)
	def AverageRate(self,Es,w):
		'''Cycle averaged tunneling rate

		:param np.array Es: Envelope of the electric field in simulation units, any shape.  The magnitude of the envelope should be equal to the peak value of the underlying field.
		:param double w0: carrier frequency in simulation units.
		:returns: rate in simulation units.'''
		return np.zeros(Es.shape)

class ADK(Ionization):
	def InstantaneousRate(self,Es):
		Uion = self.Uion
		Z = self.Z
		nstar = Z/np.sqrt(2*Uion)
		E = self.field_sim_to_au(np.abs(Es)) + self.cutoff_field
		D = ((4.0*np.exp(1.0)*Z**3)/(E*nstar**4))**nstar
		ans = (E*D*D/(8.0*np.pi*Z))*np.exp(-2*Z**3/(3.0*nstar**3*E))
		return self.rate_au_to_sim(ans)
	def InstantaneousRateCL(self,rate,Es,q,k):
		Uion = self.Uion
		Z = self.Z
		nstar = Z/np.sqrt(2*Uion)
		field_conv = self.field_sim_to_au(1.0)
		rate_conv = self.rate_au_to_sim(1.0)
		D = ((4.0*np.exp(1.0)*Z**3)/nstar**4)**nstar
		C_pre = D*D/(8.0*np.pi*Z)
		C_pow = 1.0-nstar
		C_exp = -2*Z**3/(3.0*nstar**3)
		k(q,rate.shape,None,rate.data,Es.data,
			np.double(field_conv),np.double(rate_conv),np.double(self.cutoff_field),
			np.double(C_pre),np.double(C_pow),np.double(C_exp))


class PPT(Ionization):
	def AverageRate(self,Es,w0):
		F0 = np.sqrt(2*self.Uion)**3
		nstar = self.Z/np.sqrt(2*self.Uion)
		lstar = nstar - 1
		C2 = 2**(2*nstar) / (nstar*spsf.gamma(nstar+lstar+1)*spsf.gamma(nstar-lstar))
		E = self.field_sim_to_au(np.abs(Es)) + self.cutoff_field
		w = self.rate_sim_to_au(w0)
		gam = np.sqrt(2.0*self.Uion)*w/E
		alpha = 2 * (np.arcsinh(gam)-gam/np.sqrt(1+gam**2))
		beta = 2*gam/np.sqrt(1+gam**2)
		g = (3/(2*gam))*((1+1/(2*gam**2))*np.arcsinh(gam)-np.sqrt(1+gam**2)/(2*gam))
		nu = (self.Uion/w) * (1 + 1/(2*gam**2))
		A0 = np.zeros(Es.shape)
		dnu = np.ceil(nu) - nu
		for n in range(self.terms):
			A0 += np.exp(-alpha*(n+dnu))*wfunc(np.sqrt(beta*(n+dnu)))
		A0 *= (4/np.sqrt(3*np.pi)) * (gam**2/(1+gam**2))
		ans = A0*(E*np.sqrt(1+gam**2)/(2*F0))**1.5
		ans *= (2*F0/E)**(2*nstar) # coulomb correction
		ans *= self.Uion*C2*np.sqrt(6/np.pi) * np.exp(-2.0*F0*g/(3*E))
		return self.rate_au_to_sim(ans)
	def InstantaneousRate(self,Es):
		'''Evaluate cycle averaged rate in the tunneling limit (gam=0) and unwind cycle averaging.
		Put ADK here for now, get real PPT tunneling later.'''
		E = self.field_sim_to_au(np.abs(Es)) + self.cutoff_field
		Uion = self.Uion
		Z = self.Z
		nstar = Z/np.sqrt(2*Uion)
		D = ((4.0*np.exp(1.0)*Z**3)/(E*nstar**4))**nstar
		ans = (E*D*D/(8.0*np.pi*Z))*np.exp(-2*Z**3/(3.0*nstar**3*E))
		return self.rate_au_to_sim(ans)

class YI(Ionization):
	"""Yudin-Ivanov phase dependent ionization rate.
	Class cannot be used for single point evaluation.
	The phase is automatically extraced from the carrier resolved field."""
	def InstantaneousRate(self,Es):
		""":param numpy.array Es: Carrier resolved electric field in simulation units, any shape, axis 0 is time."""
		amp,phase,w = self.ExtractEikonalForm(Es)
		amp = self.field_sim_to_au(amp) + self.cutoff_field
		w = self.rate_sim_to_au(w)
		theta = (phase - 0.5*np.pi)%np.pi - 0.5*np.pi
		nstar = self.Z/np.sqrt(2*self.Uion)
		lstar = nstar - 1
		Anl = 2**(2*nstar) / (nstar*spsf.gamma(nstar+lstar+1)*spsf.gamma(nstar-lstar))
		gam = np.sqrt(2.0*self.Uion)*w/amp
		a = 1+gam*gam-np.sin(theta)**2
		b = np.sqrt(a*a+4*gam*gam*np.sin(theta)**2)
		c = np.sqrt((np.sqrt((b+a)/2)+gam)**2 + (np.sqrt((b-a)/2)+np.sin(np.abs(theta)))**2)
		Phi = (gam**2 + np.sin(theta)**2 + 0.5)*np.log(c)
		Phi -= 3*(np.sqrt(b-a)/(2*np.sqrt(2)))*np.sin(np.abs(theta))
		Phi -= (np.sqrt(b+a)/(2*np.sqrt(2)))*gam
		kappa = np.log(gam+np.sqrt(gam**2+1)) - gam/np.sqrt(1+gam**2)
		alpha = 2 * (np.arcsinh(gam)-gam/np.sqrt(1+gam**2))
		beta = 2*gam/np.sqrt(1+gam**2)
		nu = (self.Uion/w) * (1 + 1/(2*gam**2))
		A0 = np.zeros(Es.shape)
		dnu = np.ceil(nu) - nu
		for n in range(self.terms):
			A0 += np.exp(-alpha*(n+dnu))*wfunc(np.sqrt(beta*(n+dnu)))
		A0 *= (4/np.sqrt(3*np.pi)) * (gam**2/(1+gam**2))
		pre = Anl * np.sqrt(3*kappa/gam**3)*(1+gam**2)**0.75 * A0 * self.Uion
		pre *= (2*(2*self.Uion)**1.5 / amp)**(2*nstar-1)
		return self.rate_au_to_sim(pre * np.exp(-amp**2 * Phi / w**3))

class AdHocTwoColor(PPT):
	'''For gam>1 take the sum of PPT for the two colors.
	For gam<1 use ADK.  Assume the two colors are w=1 and w=2.'''
	def InstantaneousRate(self,Es):
		w1 = 1.0
		w2 = 2.0
		gam1 = np.sqrt(2.0*self.Uion)*self.rate_sim_to_au(w1)/self.field_sim_to_au(np.max(Es))
		if gam1<1.0:
			return super().InstantaneousRate(Es)
		else:
			amp,phase,w = self.ExtractEikonalForm(Es,w00=w1,bandwidth=0.5)
			ans = super().AverageRate(amp,w)
			amp,phase,w = self.ExtractEikonalForm(Es,w00=w2,bandwidth=0.5)
			ans += super().AverageRate(amp,w)
			return ans
