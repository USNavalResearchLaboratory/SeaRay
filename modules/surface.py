'''
Module :samp:`surface`
----------------------

SeaRay surfaces are the fundamental object governing ray propagation.
Surfaces can reflect or refract rays, and collect field distributions.
They also serve as a boundary between different propagation models.
An optical medium is localized by enclosing it in a set of surfaces to form a :samp:`volume`.

All input dictionaries for surfaces have the following in common:

#. :samp:`dispersion beneath` means dispersion relation on negative z-side of surface.
#. :samp:`dispersion above` means dispersion relation on positive z-side of surface.
#. :samp:`origin` is the location of a reference point which differs by type of surface.
#. :samp:`euler angles` rotates the surface from the default orientation.

Applicable to all surfaces:

#. The default orientation is always such that a typical surface normal is in the positive z-direction.
#. Beneath the surface means negative z-side.
#. Above the surface means positive z-side.
#. If the surface bounds a volume, it should be oriented so that beneath resolves to inside and above resolves to outside.
#. A corollary to the above is that the normals point outward with respect to a volume.
#. Typical member functions assume arguments are given in local frame, i.e., the frame of the default orientation and position.
'''
import warnings
import numpy as np
import scipy.spatial
import vec3 as v3
import dispersion
import ray_kernel
import grid_tools
import caustic_tools
import logging
from base import base_volume_interface as base_volume
from base import check_four_tuple,check_surf_tuple,check_vol_tuple

class base_surface:
	'''Base class for deriving surfaces.'''
	def __init__(self,name):
		self.name = name
		self.clip = -0.1
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
		check_vol_tuple(r)
		self.P_ref[0] += r[1]
		self.P_ref[1] += r[2]
		self.P_ref[2] += r[3]
	def EulerRotate(self,q):
		if len(q)!=3:
			raise ValueError("expected 3 Euler angles")
		self.orientation.EulerRotate(q[0],q[1],q[2])
	def Initialize(self,input_dict):
		try:
			self.disp1 = input_dict['dispersion beneath']
		except KeyError:
			logging.info('defaulting to vacuum beneath')
		try:
			self.disp2 = input_dict['dispersion above']
		except KeyError:
			logging.info('defaulting to vacuum above')
		try:
			self.Translate(input_dict['origin'])
		except KeyError:
			logging.info('defaulting to origin=0')
		try:
			self.EulerRotate(input_dict['euler angles'])
		except KeyError:
			logging.info('defaulting to euler angles=0')
		try:
			self.reflective = input_dict['reflective']
		except KeyError:
			logging.info('defaulting to transmissive')
		try:
			self.bandpass = input_dict['frequency band']
		except KeyError:
			self.bandpass = None
	def InitializeCL(self,cl,input_dict):
		self.cl = cl
	def PositionGlobalToLocal(self,xp):
		'''Transform position vectors only'''
		xp[...,1:4] -= self.P_ref
		self.orientation.ExpressInBasis(xp[...,1:4])
	def xvGlobalToLocal(self,xp,vg):
		xp[...,1:4] -= self.P_ref
		self.orientation.ExpressInBasis(xp[...,1:4])
		self.orientation.ExpressInBasis(vg[...,1:4])
	def RaysGlobalToLocal(self,xp,eikonal,vg):
		'''Transform all ray data from the global system to the local system associated with this surface.

		:param numpy.array xp: phase space data
		:param numpy.array eikonal: eikonal amplitude and phase'''
		xp[...,1:4] -= self.P_ref
		self.orientation.ExpressRaysInBasis(xp,eikonal,vg)
	def PositionLocalToGlobal(self,xp):
		'''Transform position vectors only'''
		self.orientation.ExpressInStdBasis(xp[...,1:4])
		xp[...,1:4] += self.P_ref
	def xvLocalToGlobal(self,xp,vg):
		self.orientation.ExpressInStdBasis(xp[...,1:4])
		self.orientation.ExpressInStdBasis(vg[...,1:4])
		xp[...,1:4] += self.P_ref
	def RaysLocalToGlobal(self,xp,eikonal,vg):
		'''Transform all ray data from the system associated with this surface to the global system.

		:param numpy.array xp: phase space data
		:param numpy.array eikonal: eikonal amplitude and phase'''
		self.orientation.ExpressRaysInStdBasis(xp,eikonal,vg)
		xp[...,1:4] += self.P_ref
	def UpdateOrbits(self,xp,eikonal,orb):
		'''Append a time level to the orbits data and advance the index.'''
		if orb['idx']!=0:
			orb['data'][orb['idx'],:,:8] = xp[orb['xpsel']]
			orb['data'][orb['idx'],:,8:] = eikonal[orb['eiksel']]
			orb['idx'] += 1
	def GetNormals(self,xp):
		normals = np.zeros(xp[...,1:4].shape)
		normals[...,2] = 1
		return normals
	def GetDownstreamSusceptibility(self,xp,kdotn):
		upward = np.where(kdotn>0)
		downward = np.where(kdotn<0)
		chi_beneath = self.disp1.chi(xp[...,4])
		chi_above = self.disp2.chi(xp[...,4])
		chi_beneath[upward] *= 0
		chi_above[downward] *= 0
		return chi_beneath + chi_above
	def GetDownstreamVelocity(self,xp,kdotn):
		upward = np.where(kdotn>0)
		downward = np.where(kdotn<0)
		vg_beneath = self.disp1.vg(xp)
		vg_above = self.disp2.vg(xp)
		vg_beneath[upward] *= 0
		vg_above[downward] *= 0
		return vg_beneath + vg_above
	def GetRelDensity(self,xp,vol: base_volume | None):
		'''Get the density at the ray position from the enclosed volume object'''
		if vol==None:
			return 1.0
		else:
			self.PositionLocalToGlobal(xp)
			dens = vol.GetRelDensity(xp)
			self.PositionGlobalToLocal(xp)
			return dens
	def SafetyNudge(self,xp,vg):
		xp[...,:4] += vg
	def SelectBand(self,xp):
		'''Get an array of bundle indices in the passband'''
		if self.bandpass!=None:
			return np.where(np.logical_or(xp[:,0,4]>self.bandpass[0],xp[:,0,4]<self.bandpass[1]))[0]
		else:
			return np.arange(xp.shape[0]).astype(int)
	def Deflect(self,xp,eikonal,vg,vol: base_volume | None):
		'''The main SeaRay function handling reflection and refraction.
		Called from within Propagate().
		Updates both the momentum and polarization.'''
		# Save the starting ray direction for use in polarization update
		u0 = np.copy(xp[...,0,5:8])
		u0 /= np.sqrt(np.einsum('...i,...i',u0,u0))[...,np.newaxis]
		# Turn the momentum vector.
		# Only the downstream susceptibility is needed because xp implicitly contains the incidence dispersion.
		# Rays that are cutoff in refraction get reflected
		normals = self.GetNormals(xp)
		kdotn = np.einsum('...i,...i',xp[...,5:8],normals)
		if self.reflective:
			dkmag = -2*kdotn
		else:
			chi = self.GetDownstreamSusceptibility(xp,kdotn)
			chi *= self.GetRelDensity(xp,vol)
			k2diff = (1+chi)*xp[...,4]**2 - np.einsum('...i,...i',xp[...,5:8],xp[...,5:8])
			dkmag_complex = np.sign(kdotn)*np.sqrt(0j+kdotn**2+k2diff)-kdotn
			cutoff_rays = np.where(np.imag(dkmag_complex!=0.0))
			dkmag = np.real(dkmag_complex)
			dkmag[cutoff_rays] = -2*kdotn[cutoff_rays]
		if self.bandpass!=None:
			filtered_rays = np.where(np.logical_or(xp[...,4]<self.bandpass[0],xp[...,4]>self.bandpass[1]))
			dkmag[filtered_rays] = 0.0
		xp[...,5:8] += np.einsum('ij,ijk->ijk',dkmag,normals)
		kdotn = np.einsum('...i,...i',xp[...,5:8],normals)
		vg[...] = self.GetDownstreamVelocity(xp,kdotn)
		# Remaining code updates the polarization
		u1 = np.copy(xp[...,0,5:8])
		u1 /= np.sqrt(np.einsum('...i,...i',u1,u1))[...,np.newaxis]
		w = np.cross(u0,u1)
		q = np.sqrt(np.einsum('...i,...i',w,w))
		cosq = np.cos(q)
		sinq = np.sin(q)
		M = np.zeros((eikonal.shape[0],3,3))
		M[...,0,0] = w[...,0]**2 + cosq*(w[...,1]**2 + w[...,2]**2)
		M[...,1,1] = w[...,1]**2 + cosq*(w[...,0]**2 + w[...,2]**2)
		M[...,2,2] = w[...,2]**2 + cosq*(w[...,0]**2 + w[...,1]**2)
		M[...,0,1] = w[...,0]*w[...,1]*(1-cosq) - w[...,2]*q*sinq
		M[...,0,2] = w[...,0]*w[...,2]*(1-cosq) + w[...,1]*q*sinq
		M[...,1,2] = w[...,1]*w[...,2]*(1-cosq) - w[...,0]*q*sinq
		M[...,1,0] = w[...,0]*w[...,1]*(1-cosq) + w[...,2]*q*sinq
		M[...,2,0] = w[...,0]*w[...,2]*(1-cosq) - w[...,1]*q*sinq
		M[...,2,1] = w[...,1]*w[...,2]*(1-cosq) + w[...,0]*q*sinq
		filter = np.where(q**2>1e-20)[0]
		eikonal[filter,1:4] = np.einsum('...ij,...j',M[filter,:]/q[filter,np.newaxis,np.newaxis]**2,eikonal[filter,1:4])
	def GetRawImpactTimes(self,xp,vg):
		return -xp[...,3]/vg[...,3]
	def ClipImpact(self,xp):
		''':returns: booleans of shape (bundles,rays) where true indicates a clipped ray
		:rtype: numpy.array'''
		return np.zeros(xp[...,0].shape).astype(bool)
	def Detect(self,xp,vg):
		'''Determine which bundles should interact with the surface.

		:returns: time of ray impact, indices of impacting bundles
		:rtype: numpy.array, numpy.array, numpy.array'''
		# Encode no impact with a negative time
		time_to_surface = self.GetRawImpactTimes(xp,vg)
		time_to_surface[np.where(np.isinf(time_to_surface))] = self.clip
		time_to_surface[np.where(np.isnan(time_to_surface))] = self.clip
		# Include the effect of non-analytical clipping functions
		xp1 = np.copy(xp)
		ray_kernel.TestStep(time_to_surface,xp1,vg)
		cond_table = self.ClipImpact(xp1)
		time_to_surface[np.where(cond_table)] = self.clip
		# If a satellite misses, but the primary hits, deflect the satellite
		# based on extrapolation of surface data and approximate synchronism.
		ray_kernel.RepairSatellites(time_to_surface)
		impact_filter = np.where(time_to_surface[:,0]>0)[0]
		return time_to_surface,impact_filter
	def GlobalDetect(self,xp,eikonal,vg):
		# Detect in the enclosing coordinate system, return primary times
		self.RaysGlobalToLocal(xp,eikonal,vg)
		time_to_surface,impact = self.Detect(xp,vg)
		self.RaysLocalToGlobal(xp,eikonal,vg)
		return time_to_surface[...,0]
	def Propagate(self,xp,eikonal,vg,orb={'idx':0},vol: base_volume | None = None):
		'''The main function to propagate ray bundles through the surface.
		Rays are left slightly downstream of the surface.'''
		self.RaysGlobalToLocal(xp,eikonal,vg)
		dt,impact = self.Detect(xp,vg)
		if impact.shape[0]>0:
			dts = dt[impact,...]
			xps,eiks,vgs = ray_kernel.ExtractRays(impact,xp,eikonal,vg)
			ray_kernel.FullStep(dts,xps,eiks,vgs)
			self.Deflect(xps,eiks,vgs,vol)
			self.SafetyNudge(xps,vgs)
			ray_kernel.UpdateRays(impact,xp,eikonal,vg,xps,eiks,vgs)
		self.RaysLocalToGlobal(xp,eikonal,vg)
		self.UpdateOrbits(xp,eikonal,orb)
		return impact.shape[0]
	def GetPackedMesh(self):
		return np.zeros(1)
	def GetSimplices(self):
		return np.zeros(1)
	def Report(self,basename,mks_length):
		print(self.name,': write surface mesh...')
		packed_data = self.GetPackedMesh()
		if packed_data.shape[0]>1:
			np.save(basename+'_'+self.name+'_mesh',packed_data)
		simplices = self.GetSimplices()
		if simplices.shape[0]>1:
			np.save(basename+'_'+self.name+'_simplices',simplices)

class rectangle(base_surface):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.Lx = input_dict['size'][0]
		self.Ly = input_dict['size'][1]
	def ClipImpact(self,xp):
		return np.logical_or(xp[...,1]**2>(self.Lx/2)**2,xp[...,2]**2>(self.Ly/2)**2)
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
		super().Initialize(input_dict)
		self.Rd = input_dict['radius']
	def ClipImpact(self,xp):
		return xp[...,1]**2 + xp[...,2]**2 > self.Rd**2
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
	'''Building block class, does not respond to input file dictionaries.
	Meant to be used as an element in surface_mesh class.'''
	def __init__(self,a,b,c,n0,n1,n2,disp1,disp2,bandpass=(0.,100.)):
		'''Arguments are given in the owner's coordinate system.
		The reference point and orientation matrix for the triangle are derived from these.
		In the local system the triangle is in xy plane. '''
		base_surface.__init__(self,'tri')
		self.disp1 = disp1
		self.disp2 = disp2
		self.bandpass = bandpass
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
		# put triangle and normals the local coordinate system
		# in this system v1 = (+,0,0) and v2 = (*,+,0)
		self.orientation.ExpressInBasis(self.v1)
		self.orientation.ExpressInBasis(self.v2)
		self.orientation.ExpressInBasis(self.n0)
		self.orientation.ExpressInBasis(self.n1)
		self.orientation.ExpressInBasis(self.n2)
	def GetNormals(self,xp):
		# Get barycentric coordinates
		x = xp[...,1]
		y = xp[...,2]
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
	def ClipImpact(self,xp):
		safety = 1e-3*self.v1[0]
		lb = lambda y : self.v2[0]*y/self.v2[1] - safety
		ub = lambda y : self.v1[0] + (self.v2[0]-self.v1[0])*y/self.v2[1] + safety
		cond_table1 = np.logical_and(xp[...,1]>lb(xp[...,2]) , xp[...,1]<ub(xp[...,2]))
		cond_table2 = np.logical_and(xp[...,2]>-safety , xp[...,2]<self.v2[1]+safety)
		return np.logical_not(np.logical_and(cond_table1,cond_table2))
	def GetPackedMesh(self):
		# Component 0 is the "color"
		# Triangles have a degenerate point in this scheme
		packed_data = np.zeros((2,2,4))
		packed_data[...,0] = 1.0
		packed_data[0,0,1] = 0.0
		packed_data[0,1,1] = self.v1[0]
		packed_data[1,0,1] = self.v2[0]
		packed_data[1,1,1] = self.v2[0]
		packed_data[0,0,2] = 0.0
		packed_data[0,1,2] = self.v1[1]
		packed_data[1,0,2] = self.v2[1]
		packed_data[1,1,2] = self.v2[1]
		self.orientation.ExpressInStdBasis(packed_data[...,1:])
		packed_data[...,1:] += self.P_ref
		return packed_data

class surface_mesh(base_surface):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		Nx,Ny = self.CreateMeshPointsAndNormals(input_dict)
		# Create the triangular mesh
		self.pts = np.zeros((Nx*Ny,2))
		self.pts[:,0] = self.x.flatten()
		self.pts[:,1] = self.y.flatten()
		self.pts_z = self.z.flatten()
		self.tri = scipy.spatial.Delaunay(self.pts)
	def ComputeNormalsFromOrderedMesh(self,Nx,Ny):
		# Default is to estimate normals using the mesh itself.
		# This should not be used if the mesh has degenerate nodes.
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
	def ExpandSearch(self,searched_set,last_set):
		# searched_set contains all indices that have been searched previously.
		# last_set contains indices of the most recent search.
		# last_set is a subset of searched_set.
		idx_list = np.zeros(len(last_set)*3).astype(int)
		idx_list = self.tri.neighbors[last_set].flatten()
		idx_list = np.unique(idx_list[np.where(idx_list>-1)])
		return idx_list[np.isin(idx_list,searched_set,invert=True)]
	def GetStartingSimplices(self,dt,xp,vg):
		ray_kernel.TestStep(dt,xp,vg)
		tri_list = self.tri.find_simplex(xp[...,0,1:3])
		ray_kernel.TestStep(-dt,xp,vg)
		return tri_list
	def GetTriangle(self,idx):
		ia = self.tri.simplices[idx,0]
		ib = self.tri.simplices[idx,1]
		ic = self.tri.simplices[idx,2]
		a = np.array([self.pts[ia,0],self.pts[ia,1],self.pts_z[ia]])
		b = np.array([self.pts[ib,0],self.pts[ib,1],self.pts_z[ib]])
		c = np.array([self.pts[ic,0],self.pts[ic,1],self.pts_z[ic]])
		n0 = self.normals[ia]
		n1 = self.normals[ib]
		n2 = self.normals[ic]
		return triangle(a,b,c,n0,n1,n2,self.disp1,self.disp2)
	def Detect(self,xp,vg):
		'''Performs an efficient search for the target simplex.
		Assumes linear trajectory cannot cross more than one simplex.'''
		# Compute ray intersection with flattened surface
		self.simplices = np.ones(xp.shape[0]).astype(int)
		dt,impact = super().Detect(xp,vg)
		dts = dt[impact,...]
		xps,vgs = ray_kernel.extract_rays(impact,xp,vg)
		# Search for the simplex that goes with each primary ray
		simps = self.GetStartingSimplices(dts,xps,vgs)
		for bundle in range(xps.shape[0]):
			idx0 = simps[bundle]
			if idx0!=-1:
				xp1,vg1 = ray_kernel.extract_rays([bundle],xps,vgs)
				new_set = np.array([idx0]).astype(int)
				searched_set = np.array([]).astype(int)
				search_complete = False
				while not search_complete:
					for idx in new_set:
						simplex = self.GetTriangle(idx)
						simplex.xvGlobalToLocal(xp1,vg1)
						dt1,impact1 = simplex.Detect(xp1,vg1)
						simplex.xvLocalToGlobal(xp1,vg1)
						if dt1[0,0]!=self.clip:
							# if dt1>0 we found the interaction.
							# if dt1<0 we proved there is no interaction.
							# either way the search should stop.
							simps[bundle] = idx
							dts[bundle,...] = dt1[0]
							search_complete = True
							break
					else: # yes, it lines up with for
						searched_set = np.union1d(searched_set,new_set)
						new_set = self.ExpandSearch(searched_set,new_set)
						if new_set.shape[0]==0:
							dts[bundle,...] = self.clip
							search_complete = True
		ray_kernel.RepairSatellites(dts)
		# Update the simplices and times with the restriction estimate
		self.simplices[impact] = simps
		dt[impact,...] = dts
		# Restrict the impact further
		impact = np.where(dt[...,0]>0)[0]
		return dt,impact
	def Propagate(self,xp,eikonal,vg,orb={'idx':0},vol: base_volume | None = None):
		'''Propagate rays through a surface mesh.  The medium on either side must be uniform.'''
		self.RaysGlobalToLocal(xp,eikonal,vg)
		dt,impact = self.Detect(xp,vg)
		dts = dt[impact,...]
		simps = self.simplices[impact]
		xps,eiks,vgs = ray_kernel.ExtractRays(impact,xp,eikonal,vg)
		# propagate bundles one at a time
		for bundle in range(xps.shape[0]):
			dt1 = dts[[bundle],...]
			xp1,eik1,vg1 = ray_kernel.ExtractRays([bundle],xps,eiks,vgs)
			ray_kernel.FullStep(dt1,xp1,eik1,vg1)
			simplex = self.GetTriangle(simps[bundle])
			simplex.RaysGlobalToLocal(xp1,eik1,vg1)
			simplex.Deflect(xp1,eik1,vg1,vol)
			simplex.SafetyNudge(xp1,vg1)
			simplex.RaysLocalToGlobal(xp1,eik1,vg1)
			ray_kernel.UpdateRays([bundle],xps,eiks,vgs,xp1,eik1,vg1)
		ray_kernel.UpdateRays(impact,xp,eikonal,vg,xps,eiks,vgs)
		self.RaysLocalToGlobal(xp,eikonal,vg)
		self.UpdateOrbits(xp,eikonal,orb)
	def GetPackedMesh(self):
		# Component 0 is the "color"
		packed_data = np.zeros((self.x.shape[0],self.x.shape[1],4))
		packed_data[:,:,0] = self.z
		packed_data[:,:,1] = self.x
		packed_data[:,:,2] = self.y
		packed_data[:,:,3] = self.z
		self.orientation.ExpressInStdBasis(packed_data[...,1:])
		packed_data[...,1:] += self.P_ref
		return packed_data
	def GetSimplices(self):
		return self.tri.simplices

class LevelMap(surface_mesh,rectangle):
	def CreateMeshPointsAndNormals(self,input_dict):
		# z is expected in C-order, i.e., vertical axis is packed.
		self.z = input_dict['level multiplier']*np.genfromtxt(input_dict['file'],skip_header=input_dict['header lines'])
		#self.z = self.z[::16,::16]
		Nx = self.z.shape[0]
		Ny = self.z.shape[1]
		Lx = input_dict['size'][0]
		Ly = input_dict['size'][1]
		self.x = np.outer( np.linspace(-Lx/2,Lx/2,Nx) , np.ones(Ny) )
		self.y = np.outer( np.ones(Nx) , np.linspace(-Ly/2,Ly/2,Ny) )
		# Create an inset to insure all simplices within the clipping boundary
		# Data is known at the corners of the cells.
		# There are Nx*Ny corners and (Nx-1)*(Ny-1) cells.
		self.Lx = Lx - 2*Lx/(Nx-1)
		self.Ly = Ly - 2*Ly/(Ny-1)
		self.ComputeNormalsFromOrderedMesh(Nx,Ny)
		return Nx,Ny

class Grating(rectangle):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.dord = np.double(input_dict['diffracted order'])
		self.gdens = np.double(input_dict['groove density'])
		self.reflective = True
	def GetNormals(self,xp):
		'''Premise is to select an effective normal based on ray frequency.
		Then the base_surface Deflect function can be re-used.'''
		normals = np.zeros(xp[...,1:4].shape)
		normals[...,2] = 1.0
		thetai = -np.arctan2(xp[...,5],xp[...,7])
		thetar = np.arcsin(np.sin(thetai) - 2*np.pi*self.dord*self.gdens/xp[...,4])
		yrot = 0.5*(thetar-thetai)
		v3.BundleRotateY(normals,yrot)
		return normals

class AsphericCap(surface_mesh,disc):
	'''Positive radius has concavity in +z direction, parallel to normals'''
	def CreateMeshPointsAndNormals(self,input_dict):
		# Make an aspheric surface out of triangles
		check_surf_tuple(input_dict['mesh points'])
		Nx = input_dict['mesh points'][1]
		Ny = input_dict['mesh points'][2]
		C = 1.0/input_dict['radius of sphere']
		k = input_dict['conic constant']
		A = input_dict['aspheric coefficients']
		chi = lambda rho2 : np.sqrt(1-(1+k)*C**2*rho2)
		sag = lambda rho2 : C*rho2/(1+chi(rho2)) + A[0]*rho2**2 + A[1]*rho2**3 + A[2]*rho2**4 + A[3]*rho2**5 + A[4]*rho2**6
		dzdr = lambda rho : C*rho/chi(rho**2) + 4*A[0]*rho**3 + 6*A[1]*rho**5 + 8*A[2]*rho**7 + 10*A[3]*rho**9 + 12*A[4]*rho**11
		rho = np.linspace(0.0,self.Rd,Nx)
		phi = np.linspace(0.0,2*np.pi,Ny)
		self.x = np.outer(rho,np.cos(phi))
		self.y = np.outer(rho,np.sin(phi))
		self.z = sag(self.x**2 + self.y**2)
		# Create an inset to insure all simplices within the clipping boundary
		self.Rd = self.Rd - self.Rd/(Nx-1)
		self.cap_thickness = sag(self.Rd**2)
		# Compute the normals
		self.normals = np.zeros((Nx,Ny,3))
		self.normals[:,:,0] = -np.outer(dzdr(rho),np.cos(phi))
		self.normals[:,:,1] = -np.outer(dzdr(rho),np.sin(phi))
		self.normals[:,:,2] = 1.0
		self.normals = self.normals.reshape((Nx*Ny,3))
		self.normals /= np.sqrt(np.einsum('...i,...i',self.normals,self.normals))[...,np.newaxis]
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
		t1[np.where(np.isinf(t1))] = self.clip
		t1[np.where(np.isnan(t1))] = self.clip
		t1[np.where(dsc<0.0)] = self.clip
		t2[np.where(np.isinf(t2))] = self.clip
		t2[np.where(np.isnan(t2))] = self.clip
		t2[np.where(dsc<0.0)] = self.clip
		return t1,t2
	def Detect(self,xp0,vg):
		A,B,C = self.GetQuadraticEqCoefficients(xp0,vg)
		t1,t2 = self.QuadraticTime(A,B,C)
		xp = np.copy(xp0)
		ray_kernel.TestStep(t1,xp,vg)
		t1_bad = self.ClipImpact(xp)
		t1[np.where(t1_bad)] = self.clip
		xp = np.copy(xp0)
		ray_kernel.TestStep(t2,xp,vg)
		t2_bad = self.ClipImpact(xp)
		t2[np.where(t2_bad)] = self.clip
		# Choose the earlier event provided it is in the future
		# Magnitude of negative times is incorrect, but we only need the fact that they are negative.
		t1[np.where( np.logical_and(t2<t1,t2>0.0) )] = 0.0
		t2[np.where( np.logical_and(t1<t2,t1>0.0) )] = 0.0
		t1[np.where( np.logical_and(t1<0.0,t2>0.0) )] = 0.0
		t2[np.where( np.logical_and(t2<0.0,t1>0.0) )] = 0.0
		time_to_surface = t1 + t2
		# If a satellite misses but the primary hits, still deflect the satellite
		ray_kernel.RepairSatellites(time_to_surface)
		#time_to_surface[np.where(time_to_surface<0)] = time_to_surface[...,0:1]
		impact_filter = np.where(time_to_surface[...,0]>0)[0]
		return time_to_surface,impact_filter

class cylindrical_shell(quadratic):
	'''Default position : cylinder centered at origin
	Default orientation : cylinder axis is z-axis
	Dispersion beneath means inside the cylinder, above is outside'''
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.Rc = input_dict['radius']
		self.dz = input_dict['length']
	def GetNormals(self,xp):
		normals = xp[...,1:4]
		normals[...,2] = 0.0
		normals /= np.sqrt(np.einsum('...i,...i',normals,normals))[...,np.newaxis]
		return normals
	def GetQuadraticEqCoefficients(self,xp0,vg):
		vrho2 = np.einsum('...i,...i',vg[...,1:3],vg[...,1:3])
		rho2 = np.einsum('...i,...i',xp0[...,1:3],xp0[...,1:3])
		A = vrho2
		B = 2*(vg[...,1]*xp0[...,1]+vg[...,2]*xp0[...,2])
		C = rho2-self.Rc**2
		return A,B,C
	def ClipImpact(self,xp):
		return xp[...,3]**2>self.dz**2/4
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
		super().Initialize(input_dict)
		self.Rs = input_dict['radius of sphere']
		self.Re = input_dict['radius of edge']
	def GetNormals(self,xp):
		normals = xp[...,1:4] - np.array([0.,0.,self.Rs])
		normals /= np.sqrt(np.einsum('...i,...i',normals,normals))[...,np.newaxis]
		return -np.sign(self.Rs)*normals
	def GetQuadraticEqCoefficients(self,xp0,vg):
		vc = vg[...,1:4]
		xc = xp0[...,1:4] - np.array([0.,0.,self.Rs])
		A = np.einsum('...i,...i',vc,vc)
		r2 = np.einsum('...i,...i',xc,xc)
		B = 2*np.einsum('...i,...i',xc,vc)
		C = r2 - self.Rs**2
		return A,B,C
	def ClipImpact(self,xp):
		cap_angle = np.arcsin(self.Re/self.Rs)
		cap_thickness = self.Rs*(1-np.cos(cap_angle))
		return np.sign(cap_thickness)*xp[...,3]>np.abs(cap_thickness)
	def GetPackedMesh(self):
		# Component 0 is the "color"
		res = (32,8)
		cap_angle = np.arcsin(self.Re/self.Rs)
		packed_data = np.zeros((res[0],res[1],4))
		phi = np.outer(np.linspace(0.0,2*np.pi,res[0]),np.ones(res[1]))
		theta = np.outer(np.ones(res[0]),np.linspace(0.0,cap_angle,res[1]))
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
		super().Initialize(input_dict)
		self.f = input_dict['focal length']
		self.theta0 = input_dict['off axis angle']
		self.acc = input_dict['acceptance angle']
	def GetNormals(self,xp):
		normals = np.copy(xp[...,1:4])
		normals /= np.sqrt(np.einsum('...i,...i',normals,normals))[...,np.newaxis]
		normals -= np.array([0.,0.,1.])
		normals /= np.sqrt(np.einsum('...i,...i',normals,normals))[...,np.newaxis]
		return -normals
	def GetQuadraticEqCoefficients(self,xp0,vg):
		vrho2 = np.einsum('...i,...i',vg[...,1:3],vg[...,1:3])
		rho2 = np.einsum('...i,...i',xp0[...,1:3],xp0[...,1:3])
		A = vrho2
		B = 2*(vg[...,1]*xp0[...,1]+vg[...,2]*xp0[...,2]-2*self.f*vg[...,3])
		C = rho2-4*self.f*(self.f+xp0[...,3])
		return A,B,C
	def ClipImpact(self,xp):
		hmax = 2*self.f*(1-np.cos(self.acc))/np.sin(self.acc)**2 # hypotenuse
		zmax = -hmax*np.cos(self.acc)
		return np.logical_or(xp[...,3]<-self.f , xp[...,3]>zmax)
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

class NoiseMask(rectangle):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		km = 2*np.pi/input_dict['inner scale']
		k0 = 2*np.pi/input_dict['outer scale']
		Ns = input_dict['grid']
		ds = (self.Lx/(Ns[0]-1),self.Ly/(Ns[1]-1)) # make it so outer nodes lie on boundary
		kc = (np.pi/ds[0],np.pi/ds[1])
		kx = np.fft.ifftshift(grid_tools.cyclic_nodes(-kc[0],kc[0],Ns[0]))
		ky = np.fft.ifftshift(grid_tools.cyclic_nodes(-kc[1],kc[1],Ns[1]))
		k2 = np.outer(kx**2,np.ones(Ns[1])) + np.outer(np.ones(Ns[0]),ky**2)
		screen = np.exp(-k2/km**2)*(0j+k2+k0**2)**(-11/6)
		pos = lambda i : slice(1,int(Ns[i]/2))
		neg = lambda i : slice(int(Ns[i]/2)+1,Ns[i])
		crit = lambda i : int(Ns[i]/2)
		screen *= np.random.random(Ns)-0.5 + 1j*np.random.random(Ns)-0.5j
		screen[neg(0),neg(1)] = np.conj(screen[pos(0),pos(1)][::-1,::-1])
		screen[neg(0),pos(1)] = np.conj(screen[pos(0),neg(1)][::-1,::-1])
		screen[0,:] = 0
		screen[:,0] = 0
		screen[crit(0),:] = np.real(screen[crit(0),:])
		screen[:,crit(1)] = np.real(screen[:,crit(1)])
		screen = np.real(np.fft.ifft(np.fft.ifft(screen,axis=0),axis=1))
		self.screen = 1 + input_dict['amplitude']*screen/np.max(np.abs(screen))
		self.xn = np.linspace(-self.Lx/2,self.Lx/2,Ns[0])
		self.yn = np.linspace(-self.Ly/2,self.Ly/2,Ns[1])
	def Deflect(self,xp,eikonal,vg,vol: base_volume | None):
		# Frequency filtering
		sel = self.SelectBand(xp)
		# Set up interpolation
		obj = scipy.interpolate.RegularGridInterpolator((self.xn,self.yn),self.screen)
		pts = np.stack((xp[sel,0,1],xp[sel,0,2]),axis=-1)
		attenuations = obj(pts)
		eikonal[sel,1:4] *= attenuations[:,np.newaxis]

class IdealCompressor(rectangle):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.gdd = input_dict['group delay dispersion']
		self.w0 = input_dict['center frequency']
	def Deflect(self,xp,eikonal,vg,vol: base_volume | None):
		# Frequency filtering
		sel = self.SelectBand(xp)
		# Shift the ray time
		xp[sel,:,0] += self.gdd*(xp[sel,:,4]-self.w0)
		# Phase shift; gdd>0 has red leading blue
		eikonal[sel,0] += 0.5*self.gdd*(xp[sel,0,4]-self.w0)**2
		kdotn = np.ones(xp.shape[:-1]) # getting vg only requires the sign of k.n
		vg[...] = self.GetDownstreamVelocity(xp,kdotn)

class IdealLens(disc):
	'''The ideal lens will focus collimated rays to a perfect point.
	It will also adjust the phase to preserve pulse duration,
	just as a parabolic mirror would do.'''
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.f = input_dict['focal length']
	def Deflect(self,xp,eikonal,vg,vol: base_volume | None):
		# Save the starting ray direction for use in polarization update
		u0 = np.copy(xp[...,0,5:8])
		u0 /= np.sqrt(np.einsum('...i,...i',u0,u0))[...,np.newaxis]
		# Use image plane to deduce new momentum
		t = 2*self.f/vg[...,3] # correct for either + or - lens
		img_pts = xp[...,:4]-vg*t[...,np.newaxis] # start with object pts
		img_pts[...,1:3] *= -1.0 # inversion
		img_pts[...,3] += 4*self.f # translate to image plane
		k2 = np.einsum('...i,...i',xp[:,:,5:8],xp[:,:,5:8])[...,np.newaxis]
		R = np.sign(self.f)*(img_pts[...,1:4] - xp[...,1:4])
		R2 = np.einsum('...i,...i',R,R)[...,np.newaxis]
		xp[:,:,5:8] = np.sqrt(k2)*R/np.sqrt(R2)
		# Advance ray time and phase
		R = np.array([0.0,0.0,self.f]) - xp[...,1:4]
		R2 = np.einsum('...i,...i',R,R)[...,np.newaxis]
		tadv = np.sign(self.f)*(np.sqrt(R2)-np.abs(self.f))
		eikonal[:,0] -= tadv[:,0,0]*xp[:,0,4]
		xp[...,0] -= tadv[...,0]
		kdotn = np.ones(xp.shape[:-1]) # getting vg only requires the sign of k.n
		vg[...] = self.GetDownstreamVelocity(xp,kdotn)
		# Remaining code updates the polarization
		u1 = np.copy(xp[...,0,5:8])
		u1 /= np.sqrt(np.einsum('...i,...i',u1,u1))[...,np.newaxis]
		w = np.cross(u0,u1)
		q = np.sqrt(np.einsum('...i,...i',w,w))
		cosq = np.cos(q)
		sinq = np.sin(q)
		M = np.zeros((eikonal.shape[0],3,3))
		M[...,0,0] = w[...,0]**2 + cosq*(w[...,1]**2 + w[...,2]**2)
		M[...,1,1] = w[...,1]**2 + cosq*(w[...,0]**2 + w[...,2]**2)
		M[...,2,2] = w[...,2]**2 + cosq*(w[...,0]**2 + w[...,1]**2)
		M[...,0,1] = w[...,0]*w[...,1]*(1-cosq) - w[...,2]*q*sinq
		M[...,0,2] = w[...,0]*w[...,2]*(1-cosq) + w[...,1]*q*sinq
		M[...,1,2] = w[...,1]*w[...,2]*(1-cosq) - w[...,0]*q*sinq
		M[...,1,0] = w[...,0]*w[...,1]*(1-cosq) + w[...,2]*q*sinq
		M[...,2,0] = w[...,0]*w[...,2]*(1-cosq) - w[...,1]*q*sinq
		M[...,2,1] = w[...,1]*w[...,2]*(1-cosq) + w[...,0]*q*sinq
		filter = np.where(q**2>1e-20)[0]
		eikonal[filter,1:4] = np.einsum('...ij,...j',M[filter,:]/q[filter,np.newaxis,np.newaxis]**2,eikonal[filter,1:4])

class FlyingFocus(disc):
	'''Create an arbitrary flying focus.'''
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.rmax = input_dict['max radius']
		self.zbeg = input_dict['start pos']
		self.zend = input_dict['end pos']
		self.vfront = input_dict['front velocity']
	def Deflect(self,xp,eikonal,vg,vol: base_volume | None):
		# Save the starting ray direction for use in polarization update
		u0 = np.copy(xp[...,0,5:8])
		u0 /= np.sqrt(np.einsum('...i,...i',u0,u0))[...,np.newaxis]
		# direct ray toward flying focus for each radius
		r2 = np.einsum('...i,...i',xp[:,:,1:4],xp[:,:,1:4])
		r = np.sqrt(r2)
		alpha = (self.zend-self.zbeg)/self.rmax
		img_pts = np.zeros(xp.shape)
		img_pts[...,3] = self.zbeg + alpha*r
		k2 = np.einsum('...i,...i',xp[:,:,5:8],xp[:,:,5:8])[...,np.newaxis]
		R = img_pts[...,1:4] - xp[...,1:4]
		R2 = np.einsum('...i,...i',R,R)[...,np.newaxis]
		xp[:,:,5:8] = np.sqrt(k2)*R/np.sqrt(R2)
		# Advance ray time and phase
		tadv = -(alpha*r[...,np.newaxis]/self.vfront - np.sqrt(R2))
		eikonal[:,0] -= tadv[:,0,0]*xp[:,0,4]
		xp[...,0] -= tadv[...,0]
		kdotn = np.ones(xp.shape[:-1]) # getting vg only requires the sign of k.n
		vg[...] = self.GetDownstreamVelocity(xp,kdotn)
		# Remaining code updates the polarization
		u1 = np.copy(xp[...,0,5:8])
		u1 /= np.sqrt(np.einsum('...i,...i',u1,u1))[...,np.newaxis]
		w = np.cross(u0,u1)
		q = np.sqrt(np.einsum('...i,...i',w,w))
		cosq = np.cos(q)
		sinq = np.sin(q)
		M = np.zeros((eikonal.shape[0],3,3))
		M[...,0,0] = w[...,0]**2 + cosq*(w[...,1]**2 + w[...,2]**2)
		M[...,1,1] = w[...,1]**2 + cosq*(w[...,0]**2 + w[...,2]**2)
		M[...,2,2] = w[...,2]**2 + cosq*(w[...,0]**2 + w[...,1]**2)
		M[...,0,1] = w[...,0]*w[...,1]*(1-cosq) - w[...,2]*q*sinq
		M[...,0,2] = w[...,0]*w[...,2]*(1-cosq) + w[...,1]*q*sinq
		M[...,1,2] = w[...,1]*w[...,2]*(1-cosq) - w[...,0]*q*sinq
		M[...,1,0] = w[...,0]*w[...,1]*(1-cosq) + w[...,2]*q*sinq
		M[...,2,0] = w[...,0]*w[...,2]*(1-cosq) - w[...,1]*q*sinq
		M[...,2,1] = w[...,1]*w[...,2]*(1-cosq) + w[...,0]*q*sinq
		filter = np.where(q**2>1e-20)[0]
		eikonal[filter,1:4] = np.einsum('...ij,...j',M[filter,:]/q[filter,np.newaxis,np.newaxis]**2,eikonal[filter,1:4])

class IdealHarmonicGenerator(disc):
	'''Move energy from w to 2w.  Amplitude is exchanged between existing ray frequencies.
	We are following a protocol of never altering any ray frequency.
	The band refers to the band defining the pulse centered at the fundamental.'''
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.delay = input_dict['harmonic delay']
		self.n = input_dict['harmonic number']
		self.eff = input_dict['efficiency']
	def Deflect(self,xp,eikonal,vg,vol: base_volume | None):
		# Define the two coupled bands
		sel1 = np.where(np.logical_and(xp[:,0,4]>=self.bandpass[0],xp[:,0,4]<=self.bandpass[1]))[0]
		sel2 = np.where(np.logical_and(xp[:,0,4]>self.n*self.bandpass[0],xp[:,0,4]<self.n*self.bandpass[1]))[0]
		# Generate reference harmonics that will be interpolated
		xph = np.copy(xp[sel1,...])
		eikh = np.copy(eikonal[sel1,...])
		xph[...,4:8] *= self.n
		eikh[...,0] *= self.n
		eikh[...,0] += self.delay*xph[:,0,4]
		eikh[...,1:4] *= np.sqrt(self.eff)/self.n**1.5
		eikonal[sel1,1:4] *= np.sqrt(1-self.eff)
		# Interpolate to modify existing rays
		momentum_fill = [0.0,0.0,-1.0]
		for i in range(4):
			for j in range(5,8):
				xp[sel2,i,j] = scipy.interpolate.griddata((xph[:,i,1],xph[:,i,4]),xph[:,i,j],(xp[sel2,i,1],xp[sel2,i,4]),fill_value=momentum_fill[j-5])
		for j in range(0,4):
			eikonal[sel2,j] = scipy.interpolate.griddata((xph[:,0,1],xph[:,0,4]),eikh[:,j],(xp[sel2,0,1],xp[sel2,0,4]),fill_value=1e-20)
		xp[sel2,:,0] += self.delay
		kdotn = np.ones(xp.shape[:-1]) # getting vg only requires the sign of k.n
		vg[...] = self.GetDownstreamVelocity(xp,kdotn)

class Filter(disc):
	'''Apply a spectral transfer function.'''
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.H = input_dict['transfer function']
	def Deflect(self,xp,eikonal,vg,vol: base_volume | None):
		H_i = self.H(xp[:,0,4])
		G = np.abs(H_i)
		eikonal[:,0] += np.angle(H_i)
		eikonal[:,1] *= G
		eikonal[:,2] *= G
		eikonal[:,3] *= G
		kdotn = np.ones(xp.shape[:-1]) # getting vg only requires the sign of k.n
		vg[...] = self.GetDownstreamVelocity(xp,kdotn)

class AnalyticalMask(disc):
	'''Apply a spatial transfer function.'''
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.H = input_dict['transfer function']
	def Deflect(self,xp,eikonal,vg,vol: base_volume | None):
		H_i = self.H(xp[:,0,1],xp[:,0,2])
		G = np.abs(H_i)
		eikonal[:,0] += np.angle(H_i)
		eikonal[:,1] *= G
		eikonal[:,2] *= G
		eikonal[:,3] *= G
		kdotn = np.ones(xp.shape[:-1]) # getting vg only requires the sign of k.n
		vg[...] = self.GetDownstreamVelocity(xp,kdotn)

class BeamProfiler(rectangle):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
	def Report(self,basename,mks_length):
		base_surface.Report(self,basename,mks_length)
		l1 = 1000*mks_length
		wc = np.mean(self.xps[...,4])
		xc = np.mean(self.xps[...,1])
		yc = np.mean(self.xps[...,2])
		zc = np.mean(self.xps[...,3])
		wrms = np.sqrt(np.mean((self.xps[...,4]-wc)**2))
		xrms = np.sqrt(np.mean((self.xps[...,1]-xc)**2))
		yrms = np.sqrt(np.mean((self.xps[...,2]-yc)**2))
		zrms = np.sqrt(np.mean((self.xps[...,3]-zc)**2))
		print('    ray centroid (mm) = {:.2g},{:.2g},{:.2g}'.format(l1*xc,l1*yc,l1*zc))
		print('    ray rms spot (mm) = {:.2g},{:.2g},{:.2g}'.format(l1*xrms,l1*yrms,l1*zrms))
		vgroup = self.disp2.vg(self.xps)
		zf = caustic_tools.ParaxialFocus(self.xps,vgroup)
		print('    Relative paraxial ray focal position (mm) =',zf*l1)
		print('    Ray bundle count =',self.hits,self.filtered_hits)
		print('    Conserved micro-action = {:.3g}'.format(self.micro_action))
		print('    Transversality = {:.3g}'.format(self.transversality))
		return wc,xc,yc,zc,wrms,xrms,yrms,zrms
	def Propagate(self,xp,eikonal,vg,orb={'idx':0}):
		self.RaysGlobalToLocal(xp,eikonal,vg)
		dt,impact = self.Detect(xp,vg)
		xps,eiks,vgs = ray_kernel.ExtractRays(impact,xp,eikonal,vg)
		ray_kernel.FullStep(dt[impact,...],xps,eiks,vgs)
		ray_kernel.UpdateRays(impact,xp,eikonal,vg,xps,eiks,vgs)
		# Frequency filtering
		sel = np.intersect1d(self.SelectBand(xp),impact)
		self.xps = np.copy(xp[sel,0,:])
		self.eiks = np.copy(eikonal[sel,...])
		self.micro_action = ray_kernel.GetMicroAction(xp,eikonal,vg)
		self.transversality = ray_kernel.GetTransversality(xp,eikonal)
		self.hits = impact.shape[0]
		self.filtered_hits = sel.shape[0]
		self.RaysLocalToGlobal(xp,eikonal,vg)
		self.UpdateOrbits(xp,eikonal,orb)
		return impact.shape[0]

class EikonalProfiler(BeamProfiler):
	def Report(self,basename,mks_length):
		BeamProfiler.Report(self,basename,mks_length)
		print('    saving ray data on surface...')
		np.save(basename+'_'+self.name+'_xps',self.xps)
		np.save(basename+'_'+self.name+'_eiks',self.eiks)

class FullWaveProfiler(BeamProfiler):
	'''Place surface in an eikonal plane.  Set "distance to caustic" to the distance from the
	eikonal plane to the center of the wave zone'''
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.dz = input_dict['distance to caustic']
		self.Lz = input_dict['size'][2]
		self.N = input_dict['wave grid']
		check_four_tuple(self.N)
	def wave_band(self,wc):
		if self.N[0]==1:
			return (wc - 1.0 , wc + 1.0)
		else:
			return self.bandpass
	def Report(self,basename,mks_length):
		wc,xc,yc,zc,wrms,xrms,yrms,zrms = BeamProfiler.Report(self,basename,mks_length)
		field_tool = caustic_tools.FourierTool(self.N,self.wave_band(wc),(xc,yc,zc),(self.Lx,self.Ly,self.Lz),self.cl)
		print('    constructing fields in eikonal plane...')
		E = np.zeros(self.N[:3]+(3,)).astype(np.cdouble)
		E[...,0] = field_tool.GetBoundaryFields(self.xps,self.eiks,1)
		E[...,1] = field_tool.GetBoundaryFields(self.xps,self.eiks,2)
		E[...,2] = field_tool.GetBoundaryFields(self.xps,self.eiks,3)
		np.save(basename+'_'+self.name+'_plane_eik',E)
		print('    constructing fields in wave zone...')
		A = np.zeros(self.N+(3,)).astype(np.cdouble)
		A[...,0],dom4d = field_tool.GetFields(self.dz,E[...,0])
		A[...,1],dom4d = field_tool.GetFields(self.dz,E[...,1])
		A[...,2],dom4d = field_tool.GetFields(self.dz,E[...,2])
		np.save(basename+'_'+self.name+'_plane_wave',A)
		np.save(basename+'_'+self.name+'_plane_plot_ext',dom4d)

class CylindricalProfiler(FullWaveProfiler):
	'''Place surface in an eikonal plane.  Set "distance to caustic" to the distance from the
	eikonal plane to the center of the wave zone'''
	def Report(self,basename,mks_length):
		wc,xc,yc,zc,wrms,xrms,yrms,zrms = BeamProfiler.Report(self,basename,mks_length)
		print('    diagonalizing matrix...')
		field_tool = caustic_tools.BesselBeamTool(self.N,self.wave_band(wc),(xc,yc,zc),(self.Lx,self.Ly,self.Lz),self.cl)
		print('    constructing fields in eikonal plane...')
		E = np.zeros(self.N[:3]+(3,)).astype(np.cdouble)
		E[...,0] = field_tool.GetBoundaryFields(self.xps,self.eiks,1)
		E[...,1] = field_tool.GetBoundaryFields(self.xps,self.eiks,2)
		E[...,2] = field_tool.GetBoundaryFields(self.xps,self.eiks,3)
		np.save(basename+'_'+self.name+'_bess_eik',E)
		print('    constructing fields in wave zone...')
		A = np.zeros(self.N+(3,)).astype(np.cdouble)
		A[...,0],dom4d = field_tool.GetFields(self.dz,E[...,0])
		A[...,1],dom4d = field_tool.GetFields(self.dz,E[...,1])
		A[...,2],dom4d = field_tool.GetFields(self.dz,E[...,2])
		np.save(basename+'_'+self.name+'_bess_wave',A)
		np.save(basename+'_'+self.name+'_bess_plot_ext',dom4d)
