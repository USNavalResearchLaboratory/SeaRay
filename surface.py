'''
Module :samp:`surface`
----------------------

SeaRay surfaces are the fundamental object governing ray propagation.  Surfaces can reflect or refract rays, and collect field distributions.  They also serve as a boundary between different propagation models.  An optical medium is localized by enclosing it in a set of surfaces to form a :samp:`volume`.

All input dictionaries for surfaces have the following in common:

#. :samp:`dispersion beneath` means dispersion relation on negative z-side of surface.
#. :samp:`dispersion above` means dispersion relation on positive z-side of surface.
#. :samp:`origin` is the location of a reference point which differs by type of surface.
#. :samp:`euler angles` rotates the surface from the default orientation.

Applicable to all surfaces:

#. The default orientation is always such that a typical surface normal is in the z-direction.
#. Below the surface means negative z-side.
#. Above the surface means positive z-side.
#. Typical member functions assume arguments are given in local frame, i.e., the frame of the default orientation and position.
'''
import warnings
import numpy as np
import scipy.spatial
import vec3 as v3
import pyopencl
import pyopencl.array as cl_array
import init
import dispersion
import ray_kernel
import caustic_tools


class base_surface:
	'''Base class for deriving surfaces.'''
	def __init__(self,name):
		self.name = name
		# orientation vector w is the surface normal
		self.orientation = v3.basis()
		# a reference point somewhere on the surface
		self.P_ref = np.array([0,0,0]).astype(np.double)
		# dispersion data on either side of surface
		# the surface normal points toward disp2
		self.disp1 = dispersion.Vacuum()
		self.disp2 = dispersion.Vacuum()
		self.reflective = False
	def OrbitPoints(self):
		return 1
	def Translate(self,r):
		self.P_ref[0] += r[0]
		self.P_ref[1] += r[1]
		self.P_ref[2] += r[2]
	def EulerRotate(self,q):
		self.orientation.EulerRotate(q[0],q[1],q[2])
	def Initialize(self,input_dict):
		try:
			self.disp1 = input_dict['dispersion beneath']
		except KeyError:
			print('INFO: defaulting to vacuum beneath')
		try:
			self.disp2 = input_dict['dispersion above']
		except KeyError:
			print('INFO: defaulting to vacuum above')
		try:
			self.Translate(input_dict['origin'])
		except KeyError:
			print('INFO: defaulting to origin=0')
		try:
			self.EulerRotate(input_dict['euler angles'])
		except KeyError:
			print('INFO: defaulting to euler angles=0')
		try:
			self.reflective = input_dict['reflective']
		except KeyError:
			print('INFO: defaulting to transmissive')
	def RaysGlobalToLocal(self,xp,eikonal):
		'''Transform the ray coordinates from the global system to the local system associated with this surface.

		:param numpy.array xp: phase space data
		:param numpy.array eikonal: eikonal amplitude and phase'''
		xp[...,1:4] -= self.P_ref
		self.orientation.ExpressRaysInBasis(xp,eikonal)
	def RaysLocalToGlobal(self,xp,eikonal):
		'''Transform the ray coordinates from the system associated with this surface to the global system.

		:param numpy.array xp: phase space data
		:param numpy.array eikonal: eikonal amplitude and phase'''
		self.orientation.ExpressRaysInStdBasis(xp,eikonal)
		xp[...,1:4] += self.P_ref
	def UpdateOrbits(self,xp,eikonal,orb):
		'''Append a time level to the orbits data and advance the index.'''
		if orb['idx']!=0:
			orb['data'][orb['idx'],:,:8] = xp[orb['xpsel']]
			orb['data'][orb['idx'],:,8:] = eikonal[orb['eiksel']]
			orb['idx'] += 1
	def GetRgn(self,xp):
		''':returns: rays below the surface , rays above the surface
		:rtype: index array , index array'''
		return np.where(xp[:,0,3]<0) , np.where(xp[:,0,3]>0)
	def DispersionData(self,xp):
		''':returns: ray local susceptibility , ray velocity
		:rtype: numpy.array , numpy.array'''
		inRgn1,inRgn2 = self.GetRgn(xp)
		# Compute vg on current side and susceptibility on opposite side
		chi1 = self.disp1.chi(xp[...,4])
		chi2 = self.disp2.chi(xp[...,4])
		chi1[inRgn1] *= 0
		chi2[inRgn2] *= 0
		chi = chi1 + chi2
		vg1 = self.disp1.vg(xp)
		vg2 = self.disp2.vg(xp)
		vg2[inRgn1] *= 0
		vg1[inRgn2] *= 0
		vg = vg1 + vg2
		return chi,vg
	def PropagateTo(self,xp,eikonal,vg,t,impact):
		xpsub = xp[impact,...]
		eiksub = eikonal[impact,...]
		ray_kernel.FullStep(t[impact],xpsub,eiksub,vg[impact,...])
		xp[impact,...] = xpsub
		eikonal[impact,...] = eiksub
	def SafetyNudge(self,xp,eikonal,vg,impact):
		xpsub = xp[impact,...]
		eiksub = eikonal[impact,...]
		t = np.ones(impact.shape[0])
		ray_kernel.FullStep(t,xpsub,eiksub,vg[impact,...])
		xp[impact,...] = xpsub
		eikonal[impact,...] = eiksub
	def Deflect(self,xp,eikonal,impact,normals,chi):
		# Reflect or Refract and update polarization
		# Only the index on the outgoing side is needed because xp implicitly contains the incidence dispersion.
		u0 = xp[impact,0,5:8]
		u0 /= np.sqrt(np.einsum('...i,...i',u0,u0))[...,np.newaxis]
		kdotn = np.einsum('...i,...i',xp[impact,:,5:8],normals)
		if self.reflective:
			dkmag = -2*kdotn
		else:
			k2diff = (1+chi[impact,:])*xp[impact,:,4]**2 - np.einsum('...i,...i',xp[impact,:,5:8],xp[impact,:,5:8])
			dkmag = np.sign(kdotn)*np.sqrt(kdotn**2+k2diff)-kdotn
		xp[impact,:,5:8] += np.einsum('ij,ijk->ijk',dkmag,normals)
		# Remaining code updates the polarization
		u1 = xp[impact,0,5:8]
		u1 /= np.sqrt(np.einsum('...i,...i',u1,u1))[...,np.newaxis]
		sigma = 0.5*np.cross(u0,u1)
		sigma2 = np.einsum('...i,...i',sigma,sigma)[...,np.newaxis]
		e0 = eikonal[impact,1:4]
		eikonal[impact,1:4] = e0*(1-sigma2)
		eikonal[impact,1:4] += 2*np.cross(sigma,e0)
		eikonal[impact,1:4] += 2*sigma*np.einsum('...i,...i',sigma,e0)[...,np.newaxis]
		eikonal[impact,1:4] /= 1 + sigma2
	def GetNormals(self,xp,impact):
		normals = np.zeros(xp[impact,:,1:4].shape)
		normals[...,2] = 1
		return normals
	def Detect(self,xp,vg):
		# Returns 1D arrays over primary rays
		# time_to_surface = time of impact with the surface
		# impact_filter = indices of primary rays intersecting the surface
		time_to_surface = -xp[:,0,3]/vg[:,0,3]
		time_to_surface[np.where(np.isinf(time_to_surface))] = -1.0
		time_to_surface[np.where(np.isnan(time_to_surface))] = -1.0
		impact_filter = np.where(time_to_surface>0)[0]
		return time_to_surface,impact_filter
	def FullDetect(self,xp,eikonal,disp):
		# Detect in the enclosing coordinate system
		self.RaysGlobalToLocal(xp,eikonal)
		vg = disp.vg(xp)
		time_to_surface,impact = self.Detect(xp,vg)
		self.RaysLocalToGlobal(xp,eikonal)
		return time_to_surface
	def Propagate(self,xp,eikonal,orb={'idx':0},vol_obj=0):
		#print(self.name,'propagating...')
		self.RaysGlobalToLocal(xp,eikonal)
		chi,vg = self.DispersionData(xp)
		time_to_surface,impact = self.Detect(xp,vg)
		if impact.shape[0]>0:
			self.PropagateTo(xp,eikonal,vg,time_to_surface,impact)
			self.RaysLocalToGlobal(xp,eikonal)
			try:
				dens = vol_obj.GetDensity(xp)
			except AttributeError:
				dens = 1.0
			self.RaysGlobalToLocal(xp,eikonal)
			self.Deflect(xp,eikonal,impact,self.GetNormals(xp,impact),dens*chi)
			chi,vg = self.DispersionData(xp)
			self.SafetyNudge(xp,eikonal,vg,impact)
		self.RaysLocalToGlobal(xp,eikonal)
		self.UpdateOrbits(xp,eikonal,orb)
		return impact.shape[0]
	def GetPackedMesh(self):
		return np.zeros(1)
	def Report(self,basename,mks_length):
		print(self.name,': write surface mesh...')
		packed_data = self.GetPackedMesh()
		if packed_data.shape[0]>1:
			np.save(basename+'_'+self.name+'_mesh',packed_data)

class rectangle(base_surface):
	def Initialize(self,input_dict):
		base_surface.Initialize(self,input_dict)
		self.Lx = input_dict['size'][0]
		self.Ly = input_dict['size'][1]
	def Detect(self,xp0,vg):
		xp = np.copy(xp0)
		time_to_surface = -xp[:,0,3]/vg[:,0,3]
		time_to_surface[np.where(np.isinf(time_to_surface))] = -1.0
		time_to_surface[np.where(np.isnan(time_to_surface))] = -1.0
		ray_kernel.TestStep(time_to_surface,xp,vg)
		cond_table = np.logical_or(xp[:,0,1]**2>(self.Lx/2)**2,xp[:,0,2]**2>(self.Ly/2)**2)
		time_to_surface[np.where(cond_table)] = -1.0
		impact_filter = np.where(time_to_surface>0)[0]
		return time_to_surface,impact_filter
	def GetPackedMesh(self):
		# Component 0 is the "color"
		packed_data = np.zeros((2,2,4))
		x = np.array([-self.Lx/2,self.Lx/2])
		y = np.array([-self.Ly/2,self.Ly/2])
		packed_data[...,0] = 1.0
		packed_data[...,1] = np.outer(x,np.ones(2))
		packed_data[...,2] = np.outer(np.ones(2),y)
		self.orientation.ExpressInStdBasis(packed_data[...,1:])
		packed_data[...,1:] += self.P_ref
		return packed_data

class disc(base_surface):
	def Initialize(self,input_dict):
		base_surface.Initialize(self,input_dict)
		self.Rd = input_dict['radius']
	def Detect(self,xp0,vg):
		xp = np.copy(xp0)
		time_to_surface = -xp[:,0,3]/vg[:,0,3]
		time_to_surface[np.where(np.isinf(time_to_surface))] = -1.0
		time_to_surface[np.where(np.isnan(time_to_surface))] = -1.0
		ray_kernel.TestStep(time_to_surface,xp,vg)
		cond_table = xp[:,0,1]**2 + xp[:,0,2]**2 > self.Rd**2
		time_to_surface[np.where(cond_table)] = -1.0
		impact_filter = np.where(time_to_surface>0)[0]
		return time_to_surface,impact_filter
	def GetPackedMesh(self):
		# Component 0 is the "color"
		res = 64
		packed_data = np.zeros((2,res,4))
		rho = np.array([0.0,self.Rd])
		phi = np.linspace(0.0,2*np.pi,res)
		packed_data[...,0] = 1.0
		packed_data[...,1] = np.outer(rho,np.ones(res))*np.cos(np.outer(np.ones(2),phi))
		packed_data[...,2] = np.outer(rho,np.ones(res))*np.sin(np.outer(np.ones(2),phi))
		self.orientation.ExpressInStdBasis(packed_data[...,1:])
		packed_data[...,1:] += self.P_ref
		return packed_data

class triangle(base_surface):
	# Building block class, not accessible from input file
	def __init__(self,a,b,c,n0,n1,n2,disp1,disp2):
		base_surface.__init__(self,'tri')
		self.disp1 = disp1
		self.disp2 = disp2
		# a,b,c,n0,n1,n2 are 3-vectors
		# a,b,c are the vertex coordinates, the order of which defines the local coordinates
		# n0,n1,n2 are normals attached to the vertices, used for interpolation
		# The vertex normals should be close to cross(b-a,c-b)
		self.n0 = np.copy(n0)
		self.n1 = np.copy(n1)
		self.n2 = np.copy(n2)
		normal = np.cross(b-a,c-b)
		self.orientation.Create(b-a,normal)
		# the reference point is one vertex
		self.P_ref = np.copy(a)
		# v1 and v2 point from the reference point to the 2 other vertices
		self.v1 = b-a
		self.v2 = c-a
		# put v1 and v2 in the local coordinate system
		# in this system v1 = (+,0,0) and v2 = (*,+,0)
		self.orientation.ExpressInBasis(self.v1)
		self.orientation.ExpressInBasis(self.v2)
		self.orientation.ExpressInBasis(self.n0)
		self.orientation.ExpressInBasis(self.n1)
		self.orientation.ExpressInBasis(self.n2)
	def EulerRotate(self,alpha,beta,gamma):
		# Rotate the triangle and create basis aligned with v1.
		# Typically this is never needed.
		self.orientation.ExpressInStdBasis(self.v1)
		self.orientation.ExpressInStdBasis(self.v2)
		v3.EulerRotate(self.v1,alpha,beta,gamma)
		v3.EulerRotate(self.v2,alpha,beta,gamma)
		normal = np.cross(self.v1,self.v2-self.v1)
		self.orientation.Create(self.v1,normal)
		self.orientation.ExpressInBasis(self.v1)
		self.orientation.ExpressInBasis(self.v2)
	def GetNormals(self,xp,impact):
		# Get barycentric coordinates
		x = xp[impact,:,1]
		y = xp[impact,:,2]
		x1 = self.v1[0]
		x2 = self.v2[0]
		y2 = self.v2[1]
		detT = y2*x2 - (x2-x1)*y2
		w0 = (-y2*(x-x2) + (x2-x1)*(y-y2))/detT
		w1 = (y2*(x-x2) - x2*(y-y2))/detT
		w2 = 1 - w0 - w1
		# Weight the normals by the barycentric coordinates
		normals = np.einsum('...i,j',w0,self.n0)
		normals += np.einsum('...i,j',w1,self.n1)
		normals += np.einsum('...i,j',w2,self.n2)
		normals /= np.sqrt(np.einsum('...i,...i',normals,normals))[...,np.newaxis]
		return normals
	def Detect(self,xp0,vg):
		xp = np.copy(xp0)
		time_to_surface = -xp[:,0,3]/vg[:,0,3]
		time_to_surface[np.where(np.isinf(time_to_surface))] = -1.0
		time_to_surface[np.where(np.isnan(time_to_surface))] = -1.0
		ray_kernel.TestStep(time_to_surface,xp,vg)
		safety = 1e-3*self.v1[0]
		lb = lambda y : self.v2[0]*y/self.v2[1] - safety
		ub = lambda y : self.v1[0] + (self.v2[0]-self.v1[0])*y/self.v2[1] + safety
		cond_table1 = np.logical_and(xp[:,0,1]>lb(xp[:,0,2]) , xp[:,0,1]<ub(xp[:,0,2]))
		cond_table2 = np.logical_and(xp[:,0,2]>-safety , xp[:,0,2]<self.v2[1]+safety)
		cond_table = np.logical_and(cond_table1,cond_table2)
		time_to_surface[np.where(np.logical_not(cond_table))] = -1.0
		impact_filter = np.where(time_to_surface>0)[0]
		return time_to_surface,impact_filter

class LevelMap(rectangle):
	def LoadMap(self,input_dict):
		# z is expected in C-order, i.e., vertical axis is packed.
		self.z = input_dict['level multiplier']*np.genfromtxt(input_dict['file'],skip_header=input_dict['header lines'])
		#self.z = self.z[::16,::16]
		Nx = self.z.shape[0]
		Ny = self.z.shape[1]
		Lx = input_dict['size'][0]
		Ly = input_dict['size'][1]
		self.x = np.outer( np.linspace(-Lx/2,Lx/2,Nx) , np.ones(Ny) )
		self.y = np.outer( np.ones(Nx) , np.linspace(-Ly/2,Ly/2,Ny) )
		# Set boundaries which exclude the edge cells.
		# Data is known at the corners of the cells.
		# There are Nx*Ny corners and (Nx-1)*(Ny-1) cells.
		self.Lx = Lx - 2*Lx/(Nx-1)
		self.Ly = Ly - 2*Ly/(Ny-1)
		return Nx,Ny
	def Initialize(self,input_dict):
		base_surface.Initialize(self,input_dict)
		Nx,Ny = self.LoadMap(input_dict)
		# Work out surface normals at each mesh point.
		# Average over the 4 possible ways to form the normal.
		self.normals = np.zeros((Nx,Ny,3))
		v1 = np.zeros((Nx,Ny,3))
		v2 = np.zeros((Nx,Ny,3))
		for di in [-1,1]:
			for dj in [-1,1]:
				v1[:,:,0] = np.roll(self.x,di,axis=0) - self.x
				v1[:,:,1] = 0
				v1[:,:,2] = np.roll(self.z,di,axis=0) - self.z
				v2[:,:,0] = 0
				v2[:,:,1] = np.roll(self.y,dj,axis=1) - self.y
				v2[:,:,2] = np.roll(self.z,dj,axis=1) - self.z
				self.normals += di*dj*np.cross(v1,v2)
		self.normals = (self.normals/4).reshape((Nx*Ny,3))
		self.normals /= np.sqrt(np.einsum('...i,...i',self.normals,self.normals))[...,np.newaxis]
		# Create the triangular mesh
		self.pts = np.zeros((Nx*Ny,2))
		self.pts[:,0] = self.x.flatten()
		self.pts[:,1] = self.y.flatten()
		self.pts_z = self.z.flatten()
		self.tri = scipy.spatial.Delaunay(self.pts)
	def ExpandSearch(self,searched_set,last_set):
		# searched_set contains all indices that have been searched previously.
		# last_set contains indices of the most recent search.
		# last_set is a subset of searched_set.
		idx_list = np.zeros(len(last_set)*3).astype(int)
		idx_list = self.tri.neighbors[last_set].flatten()
		idx_list = np.unique(idx_list[np.where(idx_list>-1)])
		return idx_list[np.isin(idx_list,searched_set,invert=True)]
	def Propagate(self,xp,eikonal,orb={'idx':0}):
		self.RaysGlobalToLocal(xp,eikonal)
		chi,vg = self.DispersionData(xp)
		# Compute ray intersection with flattened surface
		t,impact = self.Detect(xp,vg)
		xpsub = xp[impact,...]
		eiksub = eikonal[impact,...]
		ray_kernel.TestStep(t[impact],xpsub,vg[impact,...])
		# Associate each primary ray with a triangle projected onto flattened surface.
		# This will serve as the starting point for the local search.
		tri_list = self.tri.find_simplex(xpsub[:,0,1:3])
		ray_kernel.TestStep(-t[impact],xpsub,vg[impact,...])
		for ray in range(xpsub.shape[0]):
			if tri_list[ray]!=-1:
				xp1 = xpsub[([ray],)]
				eik1 = eiksub[([ray],)]

				new_set = np.array([tri_list[ray]]).astype(np.int)
				searched_set = np.array([]).astype(np.int)
				num_deflected = 0
				while len(new_set)>0:
					i = 0
					while num_deflected==0 and i<len(new_set):
						idx = new_set[i]
						ia = self.tri.simplices[idx,0]
						ib = self.tri.simplices[idx,1]
						ic = self.tri.simplices[idx,2]
						a = np.array([self.pts[ia,0],self.pts[ia,1],self.pts_z[ia]])
						b = np.array([self.pts[ib,0],self.pts[ib,1],self.pts_z[ib]])
						c = np.array([self.pts[ic,0],self.pts[ic,1],self.pts_z[ic]])
						n0 = self.normals[ia]
						n1 = self.normals[ib]
						n2 = self.normals[ic]
						tri = triangle(a,b,c,n0,n1,n2,self.disp1,self.disp2)
						num_deflected = tri.Propagate(xp1,eik1)
						i += 1
					if num_deflected==0:
						searched_set = np.union1d(searched_set,new_set)
						new_set = self.ExpandSearch(searched_set,new_set)
					else:
						new_set = np.array([]).astype(np.int)

				xpsub[([ray],)] = xp1
				eiksub[([ray],)] = eik1
		xp[impact,...] = xpsub
		eikonal[impact,...] = eiksub
		self.RaysLocalToGlobal(xp,eikonal)
		self.UpdateOrbits(xp,eikonal,orb)
	def Report(self,basename,mks_length):
		print(self.name,': writing mesh data...')
		# Repack so we can transform to standard basis and save in single file
		# The z-coordinate in the local basis is saved in the 0-component as the "color"
		packed_data = np.zeros((self.x.shape[0],self.x.shape[1],4))
		packed_data[:,:,0] = self.z
		packed_data[:,:,1] = self.x
		packed_data[:,:,2] = self.y
		packed_data[:,:,3] = self.z
		self.orientation.ExpressInStdBasis(packed_data[...,1:])
		packed_data[...,1:] += self.P_ref
		np.save(basename+'_'+self.name+'_mesh',packed_data)
		np.save(basename+'_'+self.name+'_simplices',self.tri.simplices)

class TestMap(LevelMap):
	def LoadMap(self,input_dict):
		# Make a lens out of triangles for testing purposes.
		Nx = input_dict['mesh points'][0]
		Ny = input_dict['mesh points'][1]
		Lx = input_dict['size'][0]
		Ly = input_dict['size'][1]
		R = input_dict['curvature radius']
		self.x = np.outer( np.linspace(-Lx/2,Lx/2,Nx) , np.ones(Ny) )
		self.y = np.outer( np.ones(Nx) , np.linspace(-Ly/2,Ly/2,Ny) )
		self.z = -R+R*np.sqrt(1 - self.x**2/R**2 - self.y**2/R**2)
		self.Lx = Lx - 2*Lx/(Nx-1)
		self.Ly = Ly - 2*Ly/(Ny-1)
		return Nx,Ny

class quadratic(base_surface):
	def QuadraticTime(self,A,B,C):
		dsc = B**2 - 4*A*C
		q = -0.5*(B+np.sign(B)*np.sqrt(np.abs(dsc)))
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			t1 = q/A
			t2 = C/q
		# Encode no impact as being at an arbitrary time in the past
		t1[np.where(np.isinf(t1))] = -1.0
		t1[np.where(np.isnan(t1))] = -1.0
		t1[np.where(dsc<0.0)] = -1.0
		t2[np.where(np.isinf(t2))] = -1.0
		t2[np.where(np.isnan(t2))] = -1.0
		t2[np.where(dsc<0.0)] = -1.0
		return t1,t2
	def Detect(self,xp0,vg):
		A,B,C = self.GetQuadraticEqCoefficients(xp0,vg)
		t1,t2 = self.QuadraticTime(A,B,C)
		xp = np.copy(xp0)
		ray_kernel.TestStep(t1,xp,vg)
		t1_good = self.ClipImpact(xp)
		xp = np.copy(xp0)
		ray_kernel.TestStep(t2,xp,vg)
		t2_good = self.ClipImpact(xp)
		# Impacts outside clipped region encoded as in the past
		t1[np.where(np.logical_not(t1_good))] = -1.0
		t2[np.where(np.logical_not(t2_good))] = -1.0
		# Choose the earlier event provided it is in the future
		# Magnitude of negative times is incorrect, but we only need the fact that they are negative.
		t1[np.where( np.logical_and(t2<t1,t2>0.0) )] = 0.0
		t2[np.where( np.logical_and(t1<t2,t1>0.0) )] = 0.0
		t1[np.where( np.logical_and(t1<0.0,t2>0.0) )] = 0.0
		t2[np.where( np.logical_and(t2<0.0,t1>0.0) )] = 0.0
		time_to_surface = t1 + t2
		impact_filter = np.where(time_to_surface>0.0)[0]
		return time_to_surface,impact_filter

class cylindrical_shell(quadratic):
	'''Default position : cylinder centered at origin
	Default orientation : cylinder axis is z-axis
	Dispersion beneath means inside the cylinder, above is outside'''
	def Initialize(self,input_dict):
		base_surface.Initialize(self,input_dict)
		self.Rc = input_dict['radius']
		self.dz = input_dict['length']
	def GetRgn(self,xp):
		rho2 = np.einsum('...i,...i',xp[:,0,1:3],xp[:,0,1:3])
		beneath_test = rho2<self.Rc
		inRgn1 = np.where(beneath_test)
		inRgn2 = np.where(np.logical_not(beneath_test))
		return inRgn1,inRgn2
	def GetNormals(self,xp,impact):
		normals = xp[impact,:,1:4]
		normals[...,2] = 0.0
		normals /= np.sqrt(np.einsum('...i,...i',normals,normals))[...,np.newaxis]
		return normals
	def GetQuadraticEqCoefficients(self,xp0,vg):
		vrho2 = np.einsum('...i,...i',vg[:,0,1:3],vg[:,0,1:3])
		rho2 = np.einsum('...i,...i',xp0[:,0,1:3],xp0[:,0,1:3])
		A = vrho2
		B = 2*(vg[:,0,1]*xp0[:,0,1]+vg[:,0,2]*xp0[:,0,2])
		C = rho2-self.Rc**2
		return A,B,C
	def ClipImpact(self,xp):
		return xp[:,0,3]**2<self.dz**2/4
	def GetPackedMesh(self):
		# Component 0 is the "color"
		res = 64
		packed_data = np.zeros((res,2,4))
		phi = np.linspace(0.0,2*np.pi,res)
		z = np.array([-self.dz/2,self.dz/2])
		packed_data[...,0] = 1.0
		packed_data[...,1] = self.Rc*np.cos(np.outer(phi,np.ones(2)))
		packed_data[...,2] = self.Rc*np.sin(np.outer(phi,np.ones(2)))
		packed_data[...,3] = np.outer(np.ones(res),z)
		self.orientation.ExpressInStdBasis(packed_data[...,1:])
		packed_data[...,1:] += self.P_ref
		return packed_data

class SphericalCap(quadratic):
	'''Default position: center of sphere is at +Rs
	Default orientation: cap centroid is at origin'''
	def Initialize(self,input_dict):
		base_surface.Initialize(self,input_dict)
		self.Rs = input_dict['radius of sphere']
		self.Re = input_dict['radius of edge']
	def GetRgn(self,xp):
		xc = xp[:,0,1:4] - np.array([0.,0.,self.Rs])
		rc = np.sqrt(np.einsum('...i,...i',xc,xc))
		cap_angle = np.arcsin(self.Re/self.Rs)
		cap_thickness = self.Rs*(1-np.cos(cap_angle))
		beneath_test = np.logical_and(rc>self.Rs,xc[:,2]<cap_thickness-self.Rs)
		inRgn1 = np.where(beneath_test)
		inRgn2 = np.where(np.logical_not(beneath_test))
		return inRgn1,inRgn2
	def GetNormals(self,xp,impact):
		normals = xp[impact,:,1:4] - np.array([0.,0.,self.Rs])
		normals /= np.sqrt(np.einsum('...i,...i',normals,normals))[...,np.newaxis]
		return -normals
	def GetQuadraticEqCoefficients(self,xp0,vg):
		vc = vg[:,0,1:4]
		xc = xp0[:,0,1:4] - np.array([0.,0.,self.Rs])
		A = np.einsum('...i,...i',vc,vc)
		r2 = np.einsum('...i,...i',xc,xc)
		B = 2*np.einsum('...i,...i',xc,vc)
		C = r2 - self.Rs**2
		return A,B,C
	def ClipImpact(self,xp):
		cap_angle = np.arcsin(self.Re/self.Rs)
		cap_thickness = self.Rs*(1-np.cos(cap_angle))
		return xp[:,0,3]<cap_thickness
	def GetPackedMesh(self):
		# Component 0 is the "color"
		res = 64
		cap_angle = np.arcsin(self.Re/self.Rs)
		packed_data = np.zeros((res,4,4))
		phi = np.outer(np.linspace(0.0,2*np.pi,res),np.ones(4))
		theta = np.outer(np.ones(res),np.linspace(0.0,cap_angle,4))
		packed_data[...,0] = 1.0
		packed_data[...,1] = self.Rs*np.cos(phi)*np.sin(theta)
		packed_data[...,2] = self.Rs*np.sin(phi)*np.sin(theta)
		packed_data[...,3] = self.Rs - self.Rs*np.cos(theta)
		self.orientation.ExpressInStdBasis(packed_data[...,1:])
		packed_data[...,1:] += self.P_ref
		return packed_data

class Paraboloid(quadratic):
	'''Default position: paraboloid focus is at origin
	Default orientation: paraboloid vertex is at -f
	Acceptance angle defines angle between focused marginal ray and z-axis'''
	def Initialize(self,input_dict):
		base_surface.Initialize(self,input_dict)
		self.f = input_dict['focal length']
		self.theta0 = input_dict['off axis angle']
		self.acc = input_dict['acceptance angle']
	def GetRgn(self,xp):
		rho2 = np.einsum('...i,...i',xp[:,0,1:3],xp[:,0,1:3])
		zpar = rho2/(4*self.f) - self.f
		beneath_test = xp[:,0,3]<zpar
		inRgn1 = np.where(beneath_test)
		inRgn2 = np.where(np.logical_not(beneath_test))
		return inRgn1,inRgn2
	def GetNormals(self,xp,impact):
		normals = xp[impact,:,1:4]
		normals /= np.sqrt(np.einsum('...i,...i',normals,normals))[...,np.newaxis]
		normals -= np.array([0.,0.,1.])
		normals /= np.sqrt(np.einsum('...i,...i',normals,normals))[...,np.newaxis]
		return -normals
	def GetQuadraticEqCoefficients(self,xp0,vg):
		vrho2 = np.einsum('...i,...i',vg[:,0,1:3],vg[:,0,1:3])
		rho2 = np.einsum('...i,...i',xp0[:,0,1:3],xp0[:,0,1:3])
		A = vrho2
		B = 2*(vg[:,0,1]*xp0[:,0,1]+vg[:,0,2]*xp0[:,0,2]-2*self.f*vg[:,0,3])
		C = rho2-4*self.f*(self.f+xp0[:,0,3])
		return A,B,C
	def ClipImpact(self,xp):
		hmax = 2*self.f*(1-np.cos(self.acc))/np.sin(self.acc)**2 # hypotenuse
		zmax = -hmax*np.cos(self.acc)
		return np.logical_and(xp[:,0,3]>=-self.f , xp[:,0,3]<zmax)
	def GetPackedMesh(self):
		# Component 0 is the "color"
		res = 64
		packed_data = np.zeros((res,res,4))
		zfunc = lambda r : r**2/(4*self.f) - self.f
		hmax = 2*self.f*(1-np.cos(self.acc))/np.sin(self.acc)**2 # hypotenuse
		rhomax = hmax*np.sin(self.acc)
		rho = np.outer(np.linspace(0.0,rhomax,res),np.ones(res))
		phi = np.outer(np.ones(res),np.linspace(0.0,2*np.pi,res))
		packed_data[...,0] = 1.0
		packed_data[...,1] = rho*np.cos(phi)
		packed_data[...,2] = rho*np.sin(phi)
		packed_data[...,3] = zfunc(rho)
		self.orientation.ExpressInStdBasis(packed_data[...,1:])
		packed_data[...,1:] += self.P_ref
		return packed_data

class BeamProfiler(rectangle):
	def Report(self,basename,mks_length):
		base_surface.Report(self,basename,mks_length)
		l1 = 1000*mks_length
		xc = np.mean(self.xps[...,1])
		yc = np.mean(self.xps[...,2])
		zc = np.mean(self.xps[...,3])
		xrms = np.sqrt(np.mean((self.xps[...,1]-xc)**2))
		yrms = np.sqrt(np.mean((self.xps[...,2]-yc)**2))
		zrms = np.sqrt(np.mean((self.xps[...,3]-zc)**2))
		print('    ray centroid (mm) = {:.2f},{:.2f},{:.2f}'.format(l1*xc,l1*yc,l1*zc))
		print('    ray rms spot (mm) = {:.2f},{:.2f},{:.2f}'.format(l1*xrms,l1*yrms,l1*zrms))
		vgroup = self.disp2.vg(self.xps)
		zf = caustic_tools.ParaxialFocus(self.xps,vgroup)
		print('    Relative paraxial ray focal position (mm) =',zf*l1)
		return xc,yc,zc,xrms,yrms,zrms
	def Propagate(self,xp,eikonal,orb={'idx':0}):
		#print(self.name,'propagating...')
		self.RaysGlobalToLocal(xp,eikonal)
		chi,vg = self.DispersionData(xp)
		time_to_surface,impact = self.Detect(xp,vg)
		self.PropagateTo(xp,eikonal,vg,time_to_surface,impact)
		self.xps = np.copy(xp[impact,0,:])
		self.eiks = np.copy(eikonal[impact,...])
		self.RaysLocalToGlobal(xp,eikonal)
		self.UpdateOrbits(xp,eikonal,orb)
	def PackVector(self,Ax,Ay,Az):
		Axyz = np.zeros(Ax.shape+(3,)).astype(np.complex)
		Axyz[...,0] = Ax
		Axyz[...,1] = Ay
		Axyz[...,2] = Az
		return Axyz

class EikonalProfiler(BeamProfiler):
	def Report(self,basename,mks_length):
		BeamProfiler.Report(self,basename,mks_length)
		print('    saving ray data on surface...')
		np.save(basename+'_'+self.name+'_xps',self.xps)
		np.save(basename+'_'+self.name+'_eiks',self.eiks)

class FullWaveProfiler(BeamProfiler):
	def Initialize(self,input_dict):
		rectangle.Initialize(self,input_dict)
		self.dz = input_dict['distance to caustic']
		self.Lz = input_dict['size'][2]
		self.N = input_dict['grid points']
	def Report(self,basename,mks_length):
		xc,yc,zc,xrms,yrms,zrms = BeamProfiler.Report(self,basename,mks_length)
		field_tool = caustic_tools.FourierTool(self.N,(self.Lx,self.Ly,self.Lz))
		print('    constructing fields in eikonal plane...')
		dom,Ax = field_tool.GetBoundaryFields(np.array([xc,yc,zc]),self.xps,self.eiks,1)
		dom,Ay = field_tool.GetBoundaryFields(np.array([xc,yc,zc]),self.xps,self.eiks,2)
		dom,Az = field_tool.GetBoundaryFields(np.array([xc,yc,zc]),self.xps,self.eiks,3)
		np.save(basename+'_'+self.name+'_plane_eik',self.PackVector(Ax,Ay,Az))
		print('    constructing fields in wave zone...')
		# Form k00 using the first ray; assume monochromatic rays.
		k00 = np.sqrt(self.xps[0,5]**2 + self.xps[0,6]**2 + self.xps[0,7]**2)
		dom3d,Ax = field_tool.GetFields(k00,self.dz,Ax)
		dom3d,Ay = field_tool.GetFields(k00,self.dz,Ay)
		dom3d,Az = field_tool.GetFields(k00,self.dz,Az)
		np.save(basename+'_'+self.name+'_plane_wave',self.PackVector(Ax,Ay,Az))
		np.save(basename+'_'+self.name+'_plane_plot_ext',dom3d)

class CylindricalProfiler(BeamProfiler):
	def Initialize(self,input_dict):
		rectangle.Initialize(self,input_dict)
		self.dz = input_dict['distance to caustic']
		self.Lz = input_dict['size'][2]
		self.N = input_dict['grid points']
	def InitializeCL(self,cl,input_dict):
		plugin_str = ''
		program = init.setup_cl_program(cl,'caustic.cl',plugin_str)
		kernel_dict = { 'transform' : program.transform }
		self.kernel = kernel_dict[input_dict['integrator']]
		self.queue = cl.queue()
	def Report(self,basename,mks_length):
		xc,yc,zc,xrms,yrms,zrms = BeamProfiler.Report(self,basename,mks_length)
		print('    diagonalizing matrix...')
		field_tool = caustic_tools.BesselBeamTool(self.N,(self.Lx/2,2*np.pi,self.Lz),self.queue,self.kernel)
		print('    constructing fields in eikonal plane...')
		dom,Ax = field_tool.GetBoundaryFields(np.array([xc,yc,zc]),self.xps,self.eiks,1)
		dom,Ay = field_tool.GetBoundaryFields(np.array([xc,yc,zc]),self.xps,self.eiks,2)
		dom,Az = field_tool.GetBoundaryFields(np.array([xc,yc,zc]),self.xps,self.eiks,3)
		np.save(basename+'_'+self.name+'_bess_eik',self.PackVector(Ax,Ay,Az))
		print('    constructing fields in wave zone...')
		# Form k00 using the first ray; assume monochromatic rays.
		k00 = np.sqrt(self.xps[0,5]**2 + self.xps[0,6]**2 + self.xps[0,7]**2)
		dom3d,Ax = field_tool.GetFields(k00,self.dz,Ax)
		dom3d,Ay = field_tool.GetFields(k00,self.dz,Ay)
		dom3d,Az = field_tool.GetFields(k00,self.dz,Az)
		np.save(basename+'_'+self.name+'_bess_wave',self.PackVector(Ax,Ay,Az))
		np.save(basename+'_'+self.name+'_bess_plot_ext',dom3d)
