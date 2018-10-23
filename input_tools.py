import numpy as np
from scipy import constants as C

class InputHelper:
	def __init__(self,mks_length):
		self.x1 = mks_length

	def ProcessArg(self,arg):
		if type(arg)==str:
			val = np.double(arg.split(' ')[0])
			units = arg.split(' ')[1]
			mult = 0.0
			if units=='fs':
				mult = 1e-15*C.c/self.x1
			if units=='ps':
				mult = 1e-12*C.c/self.x1
			if units=='ns':
				mult = 1e-9*C.c/self.x1
			if units=='us':
				mult = 1e-6*C.c/self.x1
			if units=='ms':
				mult = 1e-3*C.c/self.x1
			if units=='s':
				mult = C.c/self.x1
			if units=='um':
				mult = 1e-6/self.x1
			if units=='mm':
				mult = 1e-3/self.x1
			if units=='cm':
				mult = 1e-2/self.x1
			if units=='m':
				mult = 1/self.x1
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
		return t00,band
