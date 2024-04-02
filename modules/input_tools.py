import numpy as np
from scipy import constants as C
import modules.base as base

class InputHelper:
	def __init__(self,mks_length):
		self.x1 = mks_length
		self.t1 = mks_length/C.c
		self.w1 = 1/self.t1
		self.E1 = C.m_e*C.c*self.w1/C.e
		self.n1 = C.epsilon_0*C.m_e*self.w1**2/C.e**2
		self.P1 = self.n1*C.e*mks_length
		self.u1 = C.m_e*C.c**2
		self.N1 = self.n1*self.x1**3
		self.curr_pos = [None,0.0,0.0,0.0]

	def dnum(self,arg):
		if type(arg)==str:
			val = np.double(arg.split(' ')[0])
			units = arg.split(' ')[1]
			mult = 0.0
			if units=='fs':
				mult = 1e-15/self.t1
			if units=='ps':
				mult = 1e-12/self.t1
			if units=='ns':
				mult = 1e-9/self.t1
			if units=='us':
				mult = 1e-6/self.t1
			if units=='ms':
				mult = 1e-3/self.t1
			if units=='s':
				mult = 1/self.t1
			if units=='um':
				mult = 1e-6/self.x1
			if units=='mm':
				mult = 1e-3/self.x1
			if units=='cm':
				mult = 1e-2/self.x1
			if units=='m':
				mult = 1/self.x1
			if units=='W/m2':
				mult = self.t1*self.x1**2/self.N1/self.u1
			if units=='W/cm2':
				mult = 1e4*self.t1*self.x1**2/self.N1/self.u1
			if units=='m2/W':
				mult = self.N1*self.u1/self.x1**2/self.t1
			if units=='m2/V2':
				mult = C.epsilon_0 * self.E1**3 / self.P1
			if units=='mJ':
				mult = 1e-3/self.u1
			if units=='eV':
				mult = C.e/self.u1
			if units=='cm-3':
				mult = 1e6/self.n1
			if mult==0.0:
				raise ValueError('Unrecognized units in input.')
			val *= mult
		else:
			val = arg
		return val

	def ParaxialFocusMessage(self,w00,focused_a0,f_lens,f_num):
		'''return paraxial theory focusing parameters in message string'''
		f_lens = self.dnum(f_lens)
		paraxial_e_size = 4.0*f_num/w00
		paraxial_zR = 0.5*w00*paraxial_e_size**2
		a00 = 8*f_num**2*focused_a0 / (w00*f_lens)
		mess = '  f/# = {:.2f}\n'.format(f_num)
		mess = mess + '  Theoretical paraxial spot size (um) = {:.3f}\n'.format(1e6*self.x1*paraxial_e_size)
		mess = mess + '  Theoretical paraxial Rayleigh length (um) = {:.2f}\n'.format(1e6*self.x1*paraxial_zR)
		mess = mess + '  Initial intensity (a^2) = {:.2g}\n'.format(a00**2)
		mess = mess + '  Focused paraxial intensity (a^2) = {:.2g}\n'.format(focused_a0**2)
		return mess

	def ParaxialParameters(self,w00,focused_a0,f_lens,f_num):
		'''Get paraxial focusing parameters'''
		f_lens = self.dnum(f_lens)
		paraxial_e_size = 4.0*f_num/w00
		paraxial_zR = 0.5*w00*paraxial_e_size**2
		a00 = 8*f_num**2*focused_a0 / (w00*f_lens)
		return a00,paraxial_e_size,paraxial_zR

	def InitialVectorPotential(self,w00,focused_a0,f_lens,f_num):
		'''return initial vector potential given focused vector potential'''
		f_lens = self.dnum(f_lens)
		return 8*f_num**2*focused_a0 / (w00*f_lens)

	def TransformLimitedBandwidth(self,w00,t00,sigmas):
		t00 = self.dnum(t00)
		sigma_w = 2/t00
		band = (w00 - sigmas*sigma_w , w00 + sigmas*sigma_w)
		if band[0]<0.0:
			raise ValueError('Bandwidth calculation led to negative frequency.')
		return t00,band

	def a0(self,energy,t0,r0,w0):
		'''Get normalized vector potential given pulse parameters

		:param double t0: Gaussian pulse duration, from peak to 1/e of amplitude
		:param double r0: Gaussian spot size, from peak to 1/e of amplitude'''
		energy = self.dnum(energy)
		t0 = self.dnum(t0)
		r0 = self.dnum(r0)
		w0 = self.dnum(w0)
		P0 = np.sqrt(2/np.pi)*energy/t0
		I0 = 2*P0/(np.pi*r0**2)/(self.n1*self.x1**3)
		return np.sqrt(2*I0)/w0

	def chi3(self,n0,n2):
		return (4/3)*n0**2*self.dnum(n2)

	def rot_zx(self,rad):
		'''Returns Euler angles producing a counter-clockwise rotation in zx plane'''
		return (np.pi/2,-rad,-np.pi/2)

	def rot_zx_deg(self,deg):
		'''Returns Euler angles producing a counter-clockwise rotation in zx plane'''
		return (np.pi/2,-deg*np.pi/180,-np.pi/2)

	def set_pos(self,pos):
		base.check_vol_tuple(pos)
		self.curr_pos = pos
		return pos

	def move(self,dx,dy,dz):
		x = self.curr_pos[1] + dx
		y = self.curr_pos[2] + dy
		z = self.curr_pos[3] + dz
		self.curr_pos = [None,x,y,z]
		return [None,x,y,z]

	def polar_move_zx(self,dist,deg):
		'''angle of 0 is in +x direction, 90 degrees is in +z direction, etc..'''
		x = self.curr_pos[1] + dist*np.cos(deg*np.pi/180)
		y = self.curr_pos[2]
		z = self.curr_pos[3] + dist*np.sin(deg*np.pi/180)
		self.curr_pos = [None,x,y,z]
		return [None,x,y,z]
