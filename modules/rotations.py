'''
Module :samp:`rotations`
====================================

This module computes the nonlinear susceptibility induced by molecular rotations.
'''

import numpy as np
import pyopencl
import logging
import init

class Rotator:
	'''This class solves a many-state density matrix equation for the molecular orientation.'''
	def __init__(self,B: np.double,dalpha: np.double,kT: np.double,dnuc: np.ndarray,damp: np.ndarray,hbar: np.double):
		'''Create a molecular rotator object.

		:param double B: spectroscopic coefficient (inverse meters * mks length)
		:param double dalpha: polarizability difference (volume)
		:param double kT: temperature as an energy
		:param numpy.array dnuc: nuclear degeneracy factor array (determines state count)
		:param numpy.array damp: imaginary part of each level's energy
		:param double hbar: reduced Plank constant, normalized'''
		n = dnuc.shape[0]
		logging.log(logging.INFO,'init '+str(n)+' state rotator')
		jstates = np.arange(0,n,1)
		self.dalpha = dalpha
		self.hbar = hbar
		self.wj = 2*np.pi * B * jstates * (jstates+1)
		X = dnuc*np.exp(-hbar*self.wj/kT)
		Z = np.sum((2*jstates+1)*X)
		self.rho0jm = X/Z if Z>0 else np.zeros(n)
		self.Wj = np.zeros(n).astype(np.cdouble)
		self.Tj = np.zeros(n)
		for j in range(2,n):
			self.Wj[j] = self.wj[j] - self.wj[j-2] + 1j*damp[j]
			self.Tj[j] = self.rho0jm[j] - self.rho0jm[j-2]
		self.Tjj_dev = None
		self.wdt_dev = None
	def AddChi(self,
		cl: init.cl_refs,
		shp: tuple,
		chi,
		E,
		ngas,
		dt: np.double,
		uppe: bool):
		'''Add rotational contribution to susceptibility

		:param cl_refs cl: OpenCL reference bundle
		:param tuple shp: shape in the form (Nt,Nx,Ny)
		:param cl_data chi: the data member of the pyopencl array for susceptibility
		:param cl_data E: the data member of the pyopencl array for the electric field envelope
		:param cl_data ngas: the data member of the pyopencl array for the gas density
		:param double dt: time step'''
		if self.wdt_dev is None:
			self.wdt_dev = pyopencl.array.empty(cl.q,self.Wj.shape,np.cdouble)
			self.wdt_dev.set(self.Wj*dt,queue=cl.q)
		if self.Tjj_dev is None:
			self.Tjj_dev = pyopencl.array.empty(cl.q,self.Wj.shape,np.double)
			self.Tjj_dev.set(self.Tj,queue=cl.q)
		if uppe:
			cl.program('rotations').AddChiUPPE(cl.q,shp[1:3],None,
				chi,
				E,
				ngas,
				self.Tjj_dev.data,
				self.wdt_dev.data,
				np.double(dt),
				np.double(self.dalpha),
				np.double(self.hbar),
				np.int32(shp[0]),
				np.int32(self.Wj.shape[0]))
		else:
			cl.program('rotations').AddChiParaxial(cl.q,shp[1:3],None,
				chi,
				E,
				ngas,
				self.Tjj_dev.data,
				self.wdt_dev.data,
				np.double(dt),
				np.double(self.dalpha),
				np.double(self.hbar),
				np.int32(shp[0]),
				np.int32(self.Wj.shape[0]))
