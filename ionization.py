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

def CycleAveragingFactor(Uion,E):
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
	def field_au_to_sim(self,field):
		return field*self.ea/self.es

class Ionization(AtomicUnits):
	def __init__(self,mks_length,w0,Uion,Z,density,terms=1,generate_fit=True):
		'''Create an ionization object.  Base class should be treated as abstract.

		:param double mks_length: normalizing length in mks units
		:param double w0: radiation angular frequency
		:param double Uion: ionization potential
		:param double Z: residual charge in atomic units
		:param double density: volume object's reference density
		:param numpy.array dt: time step
		:param int terms: terms to keep in PPT expansion'''
		super().__init__(mks_length)
		self.w0 = self.rate_sim_to_au(w0)
		self.ref_dens = density # proper dimensional conversion would make this relative to self.ncrit
		self.Uion = Uion/C.alpha**2
		self.Z = Z
		self.terms = terms
		self.cutoff_field = 1e-3
		if generate_fit:
			self.GenerateFit(256)
	def GetPlasmaDensity(self,rate,ngas,dt):
		'''Get plasma density from rate, accounting for gas depletion.

		:param numpy.array ngas: gas density normalized to the volume's reference with shape (Nx,Ny).
		:param numpy.array rate: ionization rate in simulation units with shape (Nt,Nx,Ny).
		:returns: electron density in simulation units.'''
		ne = scipy.integrate.cumtrapz(rate[::-1],dx=dt,axis=0,initial=0.0)[::-1]
		ne = ngas[np.newaxis,...]*(1.0 - np.exp(-ne))*self.ref_dens
		return ne
	def GetPlasmaDensityCL(self,cl,shp,ne,ngas,dt):
		'''Compute the plasma density due to ionization on the device.

		:param tuple shp: shape in the form (Nt,Nx,Ny)
		:param cl_data ne: the data member of the pyopencl array for ne, which on input has the rate
		:param cl_data ngas: the data member of the pyopencl array for ngas, which has the gas density with shape (Nx,Ny)
		:returns: no return value, but plasma density is loaded into ne on output'''
		cl.program('ionization').ComputePlasmaDensity(cl.q,shp[1:3],None,
			ne,ngas,np.double(self.ref_dens),np.double(dt),np.int32(shp[0]))
	def ExtractEikonalForm(self,E,dt,w00=0.0,bandwidth=1.0):
		"""Extract amplitude, phase, and center frequency from a carrier resolved field E.
		The assumed form is E = Re(amp*exp(i*phase)).

		:param numpy.array E: Electric field, any shape, axis 0 is time.
		:param double w00: Center frequency, if zero deduce from the data (default).
		:param double bandwidth: relative bandwidth to keep, if unity no filtering (default)."""
		Nt = E.shape[0]
		ndims = len(E.shape)
		dw = 2*np.pi/(Nt*dt)
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
		phase = (np.angle(Et).swapaxes(0,ndims-1) + w0*np.linspace(0,(Nt-1)*dt,Nt)).swapaxes(0,ndims-1)
		return amp,phase,w0
	def Coeff(self,averaging):
		'''Precompute coefficients for ionization formulas'''
		return 0.0,1.0,-1.0
	def Rate(self,Es,averaging):
		'''Get the ionization rate over all space, returned value is in simulation units.

		:param np.array Es: Time domain electric field in simulation units, any shape.  Some subclasses demand that time be the first axis.
		:param bool averaging: Interpret Es a complex envelope and get cycle averaged rate.  In this case |Es| is expected to give the crest.'''
		E = self.field_sim_to_au(np.abs(Es)) + self.cutoff_field
		C1,C2,C3 = self.Coeff(averaging)
		return self.rate_au_to_sim(C1*E**C2*np.exp(C3/E))
	def RateCL(self,cl,shp,rate,Es,averaging):
		field_conv = self.field_sim_to_au(1.0)
		rate_conv = self.rate_au_to_sim(1.0)
		C_pre,C_pow,C_exp = self.Coeff(averaging)
		if averaging:
			k = cl.program('ionization').EnvelopeRate
		else:
			k = cl.program('ionization').ExplicitRate
		k(cl.q,shp,None,rate,Es,
			np.double(field_conv),np.double(rate_conv),np.double(self.cutoff_field),
			np.double(C_pre),np.double(C_pow),np.double(C_exp))
	def GenerateFit(self,pts):
		'''Use a log-log power series to fit the rate; not compatible with YI.

		:param tuple E_bounds: electric field range in simulation units
		:param int pts: points to use in generating the fit'''
		model = lambda x,c0,c1,c2,c3,c4,c5 : c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5
		E_table = self.field_au_to_sim(np.logspace(np.log10(self.cutoff_field*2),np.log10(1),pts))
		rate_table = self.Rate(E_table,True)
		self.Cav,var=scipy.optimize.curve_fit(model,np.log(E_table),np.log(rate_table))
		rate_table = self.Rate(E_table,False)
		self.Cex,var=scipy.optimize.curve_fit(model,np.log(E_table),np.log(rate_table))
	def FittedRate(self,Es,averaging):
		ans = np.zeros(Es.shape)
		logE = np.log(self.field_au_to_sim(self.field_sim_to_au(np.abs(Es)) + self.cutoff_field))
		if averaging:
			cn = self.Cav
		else:
			cn = self.Cex
		for i in range(6):
			ans += cn[i]*logE**i
		return np.exp(ans)
	def FittedRateCL(self,cl,shp,rate,Es,averaging):
		field_conv = self.field_sim_to_au(1.0)
		rate_conv = self.rate_au_to_sim(1.0)
		if averaging:
			k = cl.program('ionization').EnvelopeRateSeries
			cn = np.array(self.Cav).astype(np.double)
		else:
			k = cl.program('ionization').ExplicitRateSeries
			cn = np.array(self.Cex).astype(np.double)
		k(cl.q,shp,None,rate,Es,np.double(field_conv),np.double(self.cutoff_field),cn[0],cn[1],cn[2],cn[3],cn[4],cn[5])

class ADK(Ionization):
	def Coeff(self,averaging):
		Uion = self.Uion
		Z = self.Z
		nstar = Z/np.sqrt(2*Uion)
		field_conv = self.field_sim_to_au(1.0)
		rate_conv = self.rate_au_to_sim(1.0)
		D = ((4.0*np.exp(1.0)*Z**3)/nstar**4)**nstar
		C_pre = D*D/(8.0*np.pi*Z)
		C_pow = 1.0-2*nstar
		C_exp = -2*Z**3/(3.0*nstar**3)
		if averaging:
			C_pre *= np.sqrt(3.0/np.pi) / (2*Uion)**0.75
			C_pow += 0.5
		return C_pre,C_pow,C_exp

class PPT_Tunneling(ADK):
	def Coeff(self,averaging):
		'''Derive from ADK so that in the tunneling limit,
		we can simply use Popov's note to undo application of the Stirling formula'''
		nstar = self.Z/np.sqrt(2*self.Uion)
		NPPT = (2**(2*nstar-1)/spsf.gamma(nstar+1))**2
		NADK = (1/(8*np.pi*nstar))*(4*np.exp(1)/nstar)**(2*nstar)
		C1,C2,C3 = super().Coeff(averaging)
		C1 *= NPPT/NADK
		return C1,C2,C3

class PPT(PPT_Tunneling):
	'''Full PPT theory accounting for multiphoton and tunneling limits.
	If instantaneous rate is requested, the cycle averaging factor is divided out.'''
	def Rate(self,Es,averaging):
		F0 = np.sqrt(2*self.Uion)**3
		nstar = self.Z/np.sqrt(2*self.Uion)
		lstar = nstar - 1
		C2 = 2**(2*nstar) / (nstar*spsf.gamma(nstar+lstar+1)*spsf.gamma(nstar-lstar))
		E = self.field_sim_to_au(np.abs(Es)) + self.cutoff_field
		w = self.w0
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
		if averaging:
			return self.rate_au_to_sim(ans)
		else:
			# If explicit, simply unwind cycle averaging.
			# This does not account for changes in sub-cycle structure with adiabaticity, but gives right average behavior.
			return self.rate_au_to_sim(ans/CycleAveragingFactor(self.Uion,E))
	def RateCL(self,cl,shp,rate,Es,averaging):
		raise ValueError("PPT.RateCL not implemented, use PPT.FittedRateCL.")

class StitchedPPT(PPT):
	'''Switch to tunneling limit for keldysh parameter < 1.
	This is used to deal with slow convergence of PPT expansion.'''
	def Rate(self,Es,averaging):
		gam_cutoff = 0.5
		E = self.field_sim_to_au(np.abs(Es)) + self.cutoff_field
		gam = np.sqrt(2.0*self.Uion)*self.w0/E
		tunneling_selection = np.where(gam<gam_cutoff)
		mpi_selection = np.where(gam>=gam_cutoff)
		rate = np.zeros(E.shape)
		# The following counts on PPT *not* implementing Coeff while PPT_Tunneling does.
		# The fact that we have to do this may indicate a sub-optimal design.
		rate[tunneling_selection] = Ionization.Rate(self,Es[tunneling_selection],averaging)
		rate[mpi_selection] = super().Rate(Es[mpi_selection],averaging)
		return rate

class YI(Ionization):
	"""Yudin-Ivanov phase dependent ionization rate.
	Class cannot be used for single point evaluation.
	The phase is automatically extracted from the carrier resolved field."""
	def Rate(self,Es,averaging,dt):
		""":param numpy.array Es: Carrier resolved electric field in simulation units, any shape, axis 0 is time."""
		amp,phase,w = self.ExtractEikonalForm(Es,dt)
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
