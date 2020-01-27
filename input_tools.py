import numpy as np
from scipy import constants as C

class InputHelper:
	def __init__(self,mks_length):
		self.x1 = mks_length
		self.t1 = mks_length/C.c
		self.w1 = 1/self.t1
		self.E1 = C.m_e*C.c*self.w1/C.e
		self.n1 = C.epsilon_0*C.m_e*self.w1**2/C.e**2
		self.P1 = self.n1*C.e*mks_length
		self.curr_pos = [0.0,0.0,0.0]

	def ProcessArg(self,arg):
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
			if units=='m2/V2':
				mult = C.epsilon_0 * self.E1**3 / self.P1
			if mult==0.0:
				raise ValueError('Unrecognized units in input.')
			val *= mult
		else:
			val = arg
		return val

	def ParaxialFocusMessage(self,w00,focused_a0,f_lens,f_num):
		'''return paraxial theory focusing parameters in message string'''
		f_lens = self.ProcessArg(f_lens)
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
		f_lens = self.ProcessArg(f_lens)
		paraxial_e_size = 4.0*f_num/w00
		paraxial_zR = 0.5*w00*paraxial_e_size**2
		a00 = 8*f_num**2*focused_a0 / (w00*f_lens)
		return a00,paraxial_e_size,paraxial_zR

	def InitialVectorPotential(self,w00,focused_a0,f_lens,f_num):
		'''return initial vector potential given focused vector potential'''
		f_lens = self.ProcessArg(f_lens)
		return 8*f_num**2*focused_a0 / (w00*f_lens)

	def TransformLimitedBandwidth(self,w00,t00,sigmas):
		t00 = self.ProcessArg(t00)
		sigma_w = 2/t00
		band = (w00 - sigmas*sigma_w , w00 + sigmas*sigma_w)
		if band[0]<0.0:
			raise ValueError('Bandwidth calculation led to negative frequency.')
		return t00,band

	def Wcm2_to_a0(self,intensity,lambda_mks):
		eta0 = np.sqrt(C.mu_0/C.epsilon_0)
		Epeak = np.sqrt(intensity*1e4*2*eta0)
		Enormalized = Epeak/self.E1
		return Enormalized * lambda_mks/(2*np.pi*self.x1)

	def chi3(self,chi3):
		return self.ProcessArg(chi3)

	def mks_n2_to_chi3(self,n0,n2_mks):
		eta0 = np.sqrt(C.mu_0/C.epsilon_0)
		chi3_mks = (2.0/3.0)*n0*n2_mks/eta0
		return self.ProcessArg(str(chi3_mks)+' m2/V2')

	def rot_zx(self,rad):
		'''Returns Euler angles producing a counter-clockwise rotation in zx plane'''
		return (np.pi/2,-rad,-np.pi/2)

	def rot_zx_deg(self,deg):
		'''Returns Euler angles producing a counter-clockwise rotation in zx plane'''
		return (np.pi/2,-deg*np.pi/180,-np.pi/2)

	def set_pos(self,pos):
		self.curr_pos = pos
		return pos

	def move(self,dx,dy,dz):
		x = self.curr_pos[0] + dx
		y = self.curr_pos[1] + dy
		z = self.curr_pos[2] + dz
		self.curr_pos = [x,y,z]
		return [x,y,z]

	def polar_move_zx(self,dist,deg):
		'''angle of 0 is in +x direction, 90 degrees is in +z direction, etc..'''
		x = self.curr_pos[0] + dist*np.cos(deg*np.pi/180)
		y = self.curr_pos[1]
		z = self.curr_pos[2] + dist*np.sin(deg*np.pi/180)
		self.curr_pos = [x,y,z]
		return [x,y,z]
