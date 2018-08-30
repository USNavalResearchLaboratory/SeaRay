'''
Module :samp:`volume`
---------------------

All input dictionaries for volumes have the following in common:

#. :samp:`dispersion inside` means dispersion inside the volume.
#. :samp:`dispersion outside` means dispersion outside the volume.
#. :samp:`origin` is the location of a reference point which differs by type of volume.
#. :samp:`euler angles` rotates the volume from the default orientation.

This module uses multiple inheritance resulting in a diamond structure.  All volumes inherit from ``base_volume``.  This branches into some objects that describe a region, and others that describe a nonuniformity inside the region.  Nonuniform volumes are derived from both, hence the diamond structure.  It is important to use the ``super()`` function in the initialization chain.
'''
import numpy as np
import scipy.spatial
import vec3 as v3
import pyopencl
import pyopencl.array as cl_array
import init
import dispersion
import ray_kernel
import grid_tools
import caustic_tools
import surface

class base_volume:
	'''
	The base_volume is an abstract class which takes care of rays entering and exiting an arbitrary collection of surfaces that form a closed region.  Volumes filled uniformly can be derived simply by creating the surface list in :samp:`self.Initialize`.
	'''
	def __init__(self,name):
		self.name = name
		self.orientation = v3.basis()
		self.P_ref = np.array([0,0,0]).astype(np.double)
		self.disp_in = dispersion.Vacuum()
		self.disp_out = dispersion.Vacuum()
		self.surfaces = []
	def OrbitPoints(self):
		return 2
	def Translate(self,r):
		self.P_ref[0] += r[0]
		self.P_ref[1] += r[1]
		self.P_ref[2] += r[2]
	def EulerRotate(self,q):
		self.orientation.EulerRotate(q[0],q[1],q[2])
	def Initialize(self,input_dict):
		self.disp_in = input_dict['dispersion inside']
		self.disp_out = input_dict['dispersion outside']
		self.Translate(input_dict['origin'])
		self.EulerRotate(input_dict['euler angles'])
	def RaysGlobalToLocal(self,xp,eikonal):
		xp[...,1:4] -= self.P_ref
		self.orientation.ExpressRaysInBasis(xp,eikonal)
	def RaysLocalToGlobal(self,xp,eikonal):
		self.orientation.ExpressRaysInStdBasis(xp,eikonal)
		xp[...,1:4] += self.P_ref
	def SelectRaysForSurface(self,t_list,idx):
		# select valid times of impact for surface idx
		# valid times are positive, and less than positive times on all other surfaces
		ts = t_list[idx]
		cond = ts>0
		for ti in t_list:
			cond = np.logical_and(cond,np.logical_or(ts<=ti,ti<0))
		return np.where(cond)[0]
	def UpdateOrbits(self,xp,eikonal,orb):
		if orb['idx']!=0:
			orb['data'][orb['idx'],:,:8] = xp[orb['xpsel']]
			orb['data'][orb['idx'],:,8:] = eikonal[orb['eiksel']]
			orb['idx'] += 1
	def Transition(self,xp,eikonal,orb,disp):
		self.RaysGlobalToLocal(xp,eikonal)
		t = []
		for surf in self.surfaces:
			t.append(surf.FullDetect(xp,eikonal,disp))
		for idx,surf in enumerate(self.surfaces):
			impact = self.SelectRaysForSurface(t,idx)
			if impact.shape[0]>0:
				xp1 = xp[impact,...]
				eik1 = eikonal[impact,...]
				surf.Propagate(xp1,eik1,vol_obj=self)
				xp[impact,...] = xp1
				eikonal[impact,...] = eik1
		self.RaysLocalToGlobal(xp,eikonal)
		self.UpdateOrbits(xp,eikonal,orb)
	def Propagate(self,xp,eikonal,orb):
		self.Transition(xp,eikonal,orb,self.disp_out)
		self.Transition(xp,eikonal,orb,self.disp_in)
	def Report(self,basename,mks_length):
		print(self.name,': write surface meshes...')
		for idx,surf in enumerate(self.surfaces):
			packed_data = surf.GetPackedMesh()
			if packed_data.shape[0]>1:
				packed_data[...,0] = 0.3
				packed_data[...,1:] += self.P_ref
				np.save(basename+'_'+self.name+str(idx)+'_mesh',packed_data)

class SphericalLens(base_volume):
	'''rcurv beneath and above are signed radii of curvature.
	Positive sign is convex, negative is concave.
	The thickness is measured along central axis of lens.
	The extremities on axis are equidistant from the origin'''
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.R1 = input_dict['rcurv beneath']
		self.R2 = input_dict['rcurv above']
		self.Lz = input_dict['thickness']
		self.Re = input_dict['aperture radius']
		self.surfaces.append(surface.SphericalCap('s1'))
		self.surfaces.append(surface.SphericalCap('s2'))
		self.surfaces.append(surface.cylindrical_shell('shell'))
		surf_dict = { 'dispersion beneath' : self.disp_out,
			'dispersion above' : self.disp_in,
			'origin' : (0.,0.,-self.Lz/2),
			'euler angles' : (0.,0.,0.),
			'radius of sphere' : abs(self.R1),
			'radius of edge' : self.Re }
		if self.R1<0.0:
			surf_dict['euler angles'] = (0.,np.pi,0.)
			surf_dict['dispersion beneath'] = self.disp_in
			surf_dict['dispersion above'] = self.disp_out
		self.surfaces[0].Initialize(surf_dict)
		surf_dict = { 'dispersion beneath' : self.disp_out,
			'dispersion above' : self.disp_in,
			'origin' : (0.,0.,self.Lz/2),
			'euler angles' : (0.,np.pi,0.),
			'radius of sphere' : abs(self.R2),
			'radius of edge' : self.Re }
		if self.R2<0.0:
			surf_dict['euler angles'] = (0.,0.,0.)
			surf_dict['dispersion beneath'] = self.disp_in
			surf_dict['dispersion above'] = self.disp_out
		self.surfaces[1].Initialize(surf_dict)
		cap_angle1 = np.arcsin(self.Re/self.R1)
		cap_thickness1 = self.R1*(1-np.cos(cap_angle1))
		cap_angle2 = np.arcsin(self.Re/self.R2)
		cap_thickness2 = self.R2*(1-np.cos(cap_angle2))
		surf_dict = { 'dispersion beneath' : self.disp_in,
			'dispersion above' : self.disp_out,
			'origin' : (0.,0.,0.5*(cap_thickness1-cap_thickness2)),
			'euler angles' : (0.,0.,0.),
			'radius' : self.Re,
			'length' : self.Lz-cap_thickness1-cap_thickness2 }
		self.surfaces[2].Initialize(surf_dict)

class Box(base_volume):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.size = input_dict['size']
		self.surfaces.append(surface.rectangle('f1'))
		surf_dict = { 'dispersion beneath' : self.disp_out,
			'dispersion above' : self.disp_in,
			'origin' : (-self.size[0]/2,0,0),
			'euler angles' : (np.pi/2,np.pi/2,0),
			'size' : (self.size[1],self.size[2]) }
		self.surfaces[-1].Initialize(surf_dict)
		self.surfaces.append(surface.rectangle('f2'))
		surf_dict = { 'dispersion beneath' : self.disp_out,
			'dispersion above' : self.disp_in,
			'origin' : (self.size[0]/2,0,0),
			'euler angles' : (-np.pi/2,np.pi/2,0),
			'size' : (self.size[1],self.size[2]) }
		self.surfaces[-1].Initialize(surf_dict)
		self.surfaces.append(surface.rectangle('f3'))
		surf_dict = { 'dispersion beneath' : self.disp_out,
			'dispersion above' : self.disp_in,
			'origin' : (0,-self.size[1]/2,0),
			'euler angles' : (0,-np.pi/2,0),
			'size' : (self.size[0],self.size[2]) }
		self.surfaces[-1].Initialize(surf_dict)
		self.surfaces.append(surface.rectangle('f4'))
		surf_dict = { 'dispersion beneath' : self.disp_out,
			'dispersion above' : self.disp_in,
			'origin' : (0,self.size[1]/2,0),
			'euler angles' : (0,np.pi/2,0),
			'size' : (self.size[0],self.size[2]) }
		self.surfaces[-1].Initialize(surf_dict)
		self.surfaces.append(surface.rectangle('f5'))
		surf_dict = { 'dispersion beneath' : self.disp_out,
			'dispersion above' : self.disp_in,
			'origin' : (0,0,-self.size[2]/2),
			'euler angles' : (0,0,0),
			'size' : (self.size[0],self.size[1]) }
		self.surfaces[-1].Initialize(surf_dict)
		self.surfaces.append(surface.rectangle('f6'))
		surf_dict = { 'dispersion beneath' : self.disp_out,
			'dispersion above' : self.disp_in,
			'origin' : (0,0,self.size[2]/2),
			'euler angles' : (0,0,0),
			'size' : (self.size[0],self.size[1]) }
		self.surfaces[-1].Initialize(surf_dict)

class Cylinder(base_volume):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.Rd = input_dict['radius']
		self.Lz = input_dict['length']
		self.surfaces.append(surface.disc('d1'))
		self.surfaces.append(surface.disc('d2'))
		self.surfaces.append(surface.cylindrical_shell('shell'))
		surf_dict = { 'dispersion beneath' : self.disp_out,
			'dispersion above' : self.disp_in,
			'origin' : (0.,0.,-self.Lz/2),
			'euler angles' : (0.,0.,0.),
			'radius' : self.Rd }
		self.surfaces[0].Initialize(surf_dict)
		surf_dict = { 'dispersion beneath' : self.disp_in,
			'dispersion above' : self.disp_out,
			'origin' : (0.,0.,self.Lz/2),
			'euler angles' : (0.,0.,0.),
			'radius' : self.Rd }
		self.surfaces[1].Initialize(surf_dict)
		surf_dict = { 'dispersion beneath' : self.disp_in,
			'dispersion above' : self.disp_out,
			'origin' : (0.,0.,0.),
			'euler angles' : (0.,0.,0.),
			'radius' : self.Rd,
			'length' : self.Lz }
		self.surfaces[2].Initialize(surf_dict)

class nonuniform_volume(base_volume):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.vol_dict = input_dict
	def OrbitPoints(self):
		return 2+np.int(self.vol_dict['steps']/self.vol_dict['subcycles'])
	def Propagate(self,xp,eikonal,orb):
		self.Transition(xp,eikonal,orb,self.disp_out)
		self.RaysGlobalToLocal(xp,eikonal)
		ray_kernel.track(self.queue,self.kernel,xp,eikonal,self.vol_dict,orb)
		# need to transform orbits
		self.RaysLocalToGlobal(xp,eikonal)
		self.Transition(xp,eikonal,orb,self.disp_in)

class grid_volume(base_volume):
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.vol_dict = input_dict
		self.ne = np.zeros(1)
	def OrbitPoints(self):
		return 2+np.int(self.vol_dict['steps']/self.vol_dict['subcycles'])
	def Propagate(self,xp,eikonal,orb):
		self.Transition(xp,eikonal,orb,self.disp_out)
		self.RaysGlobalToLocal(xp,eikonal)
		ray_kernel.track_RIC(self.queue,self.kernel,xp,eikonal,self.ne,self.vol_dict,orb)
		# need to transform orbits
		self.RaysLocalToGlobal(xp,eikonal)
		self.Transition(xp,eikonal,orb,self.disp_in)

class PlasmaChannel(nonuniform_volume,Cylinder):
	def InitializeCL(self,cl,input_dict):
		# Set up the dispersion function in OpenCL kernel
		plugin_str = '\ninline double dot4(const double4 x,const double4 y);'

		plugin_str += '\ninline double wp2(const double4 x)\n'
		plugin_str += '{\n'
		plugin_str += 'const double c0 = ' + str(input_dict['radial coefficients'][0]) + ';\n'
		plugin_str += 'const double c2 = ' + str(input_dict['radial coefficients'][1]) + ';\n'
		plugin_str += 'const double c4 = ' + str(input_dict['radial coefficients'][2]) + ';\n'
		plugin_str += 'const double c6 = ' + str(input_dict['radial coefficients'][3]) + ';\n'
		plugin_str += '''const double r2 = x.s1*x.s1 + x.s2*x.s2;
						return c0 + c2*r2 + c4*r2*r2 + c6*r2*r2*r2;
						}\n'''

		plugin_str += '\ninline double outside(const double4 x)\n'
		plugin_str += '{\n'
		plugin_str += 'const double Rd = ' + str(self.Rd) + ';\n'
		plugin_str += 'const double dz = ' + str(self.Lz) + ';\n'
		plugin_str += '''const double r2 = x.s1*x.s1 + x.s2*x.s2;
						return (double)(r2>Rd*Rd || x.s3*x.s3>0.25*dz*dz);
						}\n'''

		plugin_str += '\ninline double D_alpha(const double4 x,const double4 k)\n'
		plugin_str += '{\n'
		plugin_str += self.disp_in.Dxk('wp2(x)')
		plugin_str += '}\n\n'

		program = init.setup_cl_program(cl,'ray_integrator.cl',plugin_str)
		kernel_dict = { 'symplectic' : program.Symplectic }
		self.kernel = kernel_dict[input_dict['integrator']]
		self.queue = cl.queue()
	def GetDensity(self,xp):
		coeff = self.vol_dict['radial coefficients']
		r2 = np.einsum('...i,...i',xp[...,1:3],xp[...,1:3])
		return coeff[0] + coeff[1]*r2 + coeff[2]*r2**2 + coeff[3]*r2**3

class Grid(grid_volume,Box):
	def LoadMap(self,input_dict):
		temp = input_dict['density multiplier']*np.load(input_dict['file'])
		self.ne = np.zeros((temp.shape[0]+4,temp.shape[1]+4,temp.shape[2]+4))
		self.ne[2:-2,2:-2,2:-2] = temp

		self.ne[0,:,:] = self.ne[2,:,:]
		self.ne[1,:,:] = self.ne[2,:,:]
		self.ne[-1,:,:] = self.ne[-3,:,:]
		self.ne[-2,:,:] = self.ne[-3,:,:]

		self.ne[:,0,:] = self.ne[:,2,:]
		self.ne[:,1,:] = self.ne[:,2,:]
		self.ne[:,-1,:] = self.ne[:,-3,:]
		self.ne[:,-2,:] = self.ne[:,-3,:]

		self.ne[:,:,0] = self.ne[:,:,2]
		self.ne[:,:,1] = self.ne[:,:,2]
		self.ne[:,:,-1] = self.ne[:,:,-3]
		self.ne[:,:,-2] = self.ne[:,:,-3]

		self.dx = self.size[0]/(self.ne.shape[0]-4)
		self.dy = self.size[1]/(self.ne.shape[1]-4)
		self.dz = self.size[2]/(self.ne.shape[2]-4)
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.LoadMap(input_dict)
	def InitializeCL(self,cl,input_dict):
		# Set up the dispersion function in OpenCL kernel
		plugin_str = '\n#define MAC_CART 1.0;\n'
		plugin_str += '\n#define MAC_CYL 0.0;\n'
		plugin_str += '\n#define MAC_DX4 (double4)(0.0,'+str(self.dx)+','+str(self.dy)+','+str(self.dz)+');\n'
		plugin_str += '\n#define MAC_NUM4 (int4)(1,'+str(self.ne.shape[0])+','+str(self.ne.shape[1])+','+str(self.ne.shape[2])+');\n'

		plugin_str += '\ninline double dot4(const double4 x,const double4 y);'
		plugin_str += '\ninline double Gather(__global double *dens,const double4 x);'

		plugin_str += '\ninline double outside(const double4 x)\n'
		plugin_str += '{\n'
		plugin_str += 'const double Lx = ' + str(self.size[0]) + ';\n'
		plugin_str += 'const double Ly = ' + str(self.size[1]) + ';\n'
		plugin_str += 'const double Lz = ' + str(self.size[2]) + ';\n'
		plugin_str += 'return (double)(x.s1*x.s1>0.25*Lx*Lx || x.s2*x.s2>0.25*Ly*Ly || x.s3*x.s3>0.25*Lz*Lz);\n}\n'

		plugin_str += '\ninline double D_alpha(__global double *dens,const double4 x,const double4 k)\n'
		plugin_str += '{\n'
		plugin_str += self.disp_in.Dxk('Gather(dens,x)')
		plugin_str += '}\n\n'

		program = init.setup_cl_program(cl,'ray_in_cell.cl',plugin_str)
		kernel_dict = { 'symplectic' : program.Symplectic }
		self.kernel = kernel_dict[input_dict['integrator']]
		self.get_density_k = program.GetDensity
		self.queue = cl.queue()
	def GetDensity(self,xp):
		return ray_kernel.gather(self.queue,self.get_density_k,xp,self.ne)

class TestGrid(Grid):
	def LoadMap(self,input_dict):
		N = input_dict['mesh points']
		N = (N[0]+4,N[1]+4,N[2]+4)
		self.dx = self.size[0]/(N[0]-4)
		self.dy = self.size[1]/(N[1]-4)
		self.dz = self.size[2]/(N[2]-4)
		x = grid_tools.cell_centers(-2*self.dx-self.size[0]/2,2*self.dx+self.size[0]/2,N[0])
		y = grid_tools.cell_centers(-2*self.dy-self.size[1]/2,2*self.dy+self.size[1]/2,N[1])
		z = grid_tools.cell_centers(-2*self.dz-self.size[2]/2,2*self.dz+self.size[2]/2,N[2])
		rho2 = np.outer(x**2,np.ones(N[1])) + np.outer(np.ones(N[0]),y**2)
		coeff = self.vol_dict['radial coefficients']
		fr = coeff[0] + coeff[1]*rho2 + coeff[2]*rho2**2 + coeff[3]*rho2**4
		self.ne = input_dict['density multiplier']*np.einsum('ij,k->ijk',fr,np.ones(N[2]))

class AxisymmetricGrid(grid_volume,Cylinder):
	def LoadMap(self,input_dict):
		temp = input_dict['density multiplier']*np.load(input_dict['file'])
		self.ne = np.zeros((temp.shape[0]+4,temp.shape[1]+4))
		self.ne[2:-2,2:-2] = temp
		self.ne[0,2:-2] = temp[1,:]
		self.ne[1,2:-2] = temp[0,:]
		self.ne[:,0] = self.ne[:,2]
		self.ne[:,1] = self.ne[:,2]
		self.ne[:,-1] = self.ne[:,-3]
		self.ne[:,-2] = self.ne[:,-3]
		self.dr = self.Rd/(self.ne.shape[0]-4)
		self.dz = self.Lz/(self.ne.shape[1]-4)
	def Initialize(self,input_dict):
		super().Initialize(input_dict)
		self.LoadMap(input_dict)
	def InitializeCL(self,cl,input_dict):
		# Set up the dispersion function in OpenCL kernel
		plugin_str = '\n#define MAC_CART 0.0;\n'
		plugin_str += '\n#define MAC_CYL 1.0;\n'
		plugin_str += '\n#define MAC_DX4 (double4)(0.0,'+str(self.dr)+',1.0,'+str(self.dz)+');\n'
		plugin_str += '\n#define MAC_NUM4 (int4)(1,'+str(self.ne.shape[0])+',1,'+str(self.ne.shape[1])+');\n'

		plugin_str += '\ninline double dot4(const double4 x,const double4 y);'
		plugin_str += '\ninline double Gather(__global double *dens,const double4 x);'

		plugin_str += '\ninline double outside(const double4 x)\n'
		plugin_str += '{\n'
		plugin_str += 'const double Rd = ' + str(self.Rd) + ';\n'
		plugin_str += 'const double Lz = ' + str(self.Lz) + ';\n'
		plugin_str += '''const double r2 = x.s1*x.s1 + x.s2*x.s2;
						return (double)(r2>Rd*Rd || x.s3*x.s3>0.25*Lz*Lz);
						}\n'''

		plugin_str += '\ninline double D_alpha(__global double *dens,const double4 x,const double4 k)\n'
		plugin_str += '{\n'
		plugin_str += self.disp_in.Dxk('Gather(dens,x)')
		plugin_str += '}\n\n'

		program = init.setup_cl_program(cl,'ray_in_cell.cl',plugin_str)
		kernel_dict = { 'symplectic' : program.Symplectic }
		self.kernel = kernel_dict[input_dict['integrator']]
		self.get_density_k = program.GetDensity
		self.queue = cl.queue()
	def GetDensity(self,xp):
		return ray_kernel.gather(self.queue,self.get_density_k,xp,self.ne)

class AxisymmetricTestGrid(AxisymmetricGrid):
	def LoadMap(self,input_dict):
		N = input_dict['mesh points']
		N = (N[0]+4,N[1]+4)
		self.dr = self.Rd/(N[0]-4)
		self.dz = self.Lz/(N[1]-4)
		rho = grid_tools.cell_centers(-2*self.dr,self.Rd+2*self.dr,N[0])
		z = grid_tools.cell_centers(-2*self.dz-self.Lz/2,2*self.dz+self.Lz/2,N[1])
		coeff = self.vol_dict['radial coefficients']
		fr = coeff[0] + coeff[1]*rho**2 + coeff[2]*rho**4 + coeff[3]*rho**6
		fz = np.ones(N[1])
		#fz = np.exp(-z**2/self.Lz**2)
		self.ne = input_dict['density multiplier']*np.outer(fr,fz)
