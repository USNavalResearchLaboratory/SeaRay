'''
Module :samp:`volume`
---------------------

All input dictionaries for volumes have the following in common:

#. :samp:`dispersion inside` means dispersion inside the volume.
#. :samp:`dispersion outside` means dispersion outside the volume.
#. :samp:`origin` is the location of a reference point which differs by type of volume.
#. :samp:`euler angles` rotates the volume from the default orientation.

This module uses multiple inheritance resulting in a diamond structure.  All volumes inherit from ``base_volume``.
This branches into some objects that describe a region, and others that describe a nonuniformity inside the region. 
Nonuniform volumes are derived from both, hence the diamond structure.
It is important to use the ``super()`` function in the initialization chain.

There is a concept of "reference density" and "relative density" herein.
The reference density is a density whereat optical parameters are specified, e.g., dispersion parameters.
The relative density is the ratio of density to reference density.
A possible source of confusion is that the reference density is itself normalized.
'''
import copy
import numpy as np
import logging
import vec3 as v3
import dispersion
import ray_kernel as ray_kernel
import paraxial_kernel
import uppe_kernel as uppe_kernel
import grid_tools
import surface
from base import check_rz_tuple,check_vol_tuple

class base_volume:
    '''
    The base_volume is an abstract class which takes care of rays entering and exiting an arbitrary collection of surfaces
    that form a closed region.  Volumes filled uniformly can be derived simply by creating the surface list in :samp:`self.Initialize`.
    '''
    def __init__(self,name):
        self.name = name
        self.orientation = v3.basis()
        self.P_ref = np.array([0,0,0]).astype(np.double)
        self.n_ref = 1.0
        self.disp_in = dispersion.Vacuum()
        self.disp_out = dispersion.Vacuum()
        self.surfaces: list[surface.base_surface] = []
        self.propagator = 'eikonal'
    def OrbitPoints(self):
        return 2
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
        self.disp_in = input_dict['dispersion inside']
        self.disp_out = input_dict['dispersion outside']
        self.Translate(input_dict['origin'])
        self.EulerRotate(input_dict['euler angles'])
        try:
            self.propagator = input_dict['propagator']
        except KeyError:
            self.propagator = 'eikonal'
        try:
            self.n_ref = input_dict['density reference']
        except KeyError:
            self.n_ref = 1.0
        if self.propagator not in ['eikonal','paraxial','uppe']:
            raise ValueError('Unrecognized propagator type.')
    def InitializeCL(self,cl,input_dict):
        self.cl = cl
    def RaysGlobalToLocal(self,xp,eikonal,vg):
        xp[...,1:4] -= self.P_ref
        self.orientation.ExpressRaysInBasis(xp,eikonal,vg)
    def RaysLocalToGlobal(self,xp,eikonal,vg):
        self.orientation.ExpressRaysInStdBasis(xp,eikonal,vg)
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
    def OrbitsLocalToGlobal(self,orb):
        pts = self.OrbitPoints()
        i1 = orb['idx'] - pts
        i2 = orb['idx']
        x = orb['data'][i1:i2,:,1:4]
        p = orb['data'][i1:i2,:,5:8]
        a = orb['data'][i1:i2,:,9:12]
        self.orientation.ExpressInStdBasis(x)
        self.orientation.ExpressInStdBasis(p)
        self.orientation.ExpressInStdBasis(a)
        x += self.P_ref
    def Transition(self,xp,eikonal,vg,orb):
        t = []
        for surf in self.surfaces:
            t.append(surf.GlobalDetect(xp,eikonal,vg))
        for idx,surf in enumerate(self.surfaces):
            impact = self.SelectRaysForSurface(t,idx)
            if impact.shape[0]>0:
                xps,eiks,vgs = ray_kernel.ExtractRays(impact,xp,eikonal,vg)
                surf.Propagate(xps,eiks,vgs,vol=self)
                ray_kernel.UpdateRays(impact,xp,eikonal,vg,xps,eiks,vgs)
        self.UpdateOrbits(xp,eikonal,orb)
    def Propagate(self,xp,eikonal,vg,orb):
        self.RaysGlobalToLocal(xp,eikonal,vg)
        self.Transition(xp,eikonal,vg,orb)
        self.Transition(xp,eikonal,vg,orb)
        self.RaysLocalToGlobal(xp,eikonal,vg)
        self.OrbitsLocalToGlobal(orb)
    def GetRelDensity(self,xp):
        '''Density relative to reference density'''
        return 1.0
    def Report(self,basename,mks_length):
        print(self.name,': write surface meshes...')
        for idx,surf in enumerate(self.surfaces):
            packed_data = surf.GetPackedMesh()
            self.orientation.ExpressInStdBasis(packed_data[...,1:])
            if packed_data.shape[0]>1:
                packed_data[...,0] = 0.3
                packed_data[...,1:] += self.P_ref
                np.save(basename+'_'+self.name+str(idx)+'_mesh',packed_data)
            simplices = surf.GetSimplices()
            if simplices.shape[0]>1:
                np.save(basename+'_'+self.name+str(idx)+'_simplices',simplices)

class SphericalLens(base_volume):
    '''rcurv beneath and above are signed radii of curvature.
    Positive radius means concavity faces +z.
    The thickness is measured along central axis of lens.
    The extremities on axis are equidistant from the origin.
    Surfaces need to be carefully oriented so that normals are outward.'''
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        self.R1 = input_dict['rcurv beneath']
        self.R2 = input_dict['rcurv above']
        self.Lz = input_dict['thickness']
        self.Re = input_dict['aperture radius']
        self.surfaces.append(surface.SphericalCap('s1'))
        self.surfaces.append(surface.SphericalCap('s2'))
        self.surfaces.append(surface.cylindrical_shell('shell'))
        surf_dict = { 'origin' : (None,0.,0.,-self.Lz/2),
            'radius of sphere' : -self.R1,
            'radius of edge' : self.Re,
            'euler angles' : (0.,np.pi,0.),
            'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out }
        self.surfaces[0].Initialize(surf_dict)
        surf_dict = { 'origin' : (None,0.,0.,self.Lz/2),
            'radius of sphere' : self.R2,
            'radius of edge' : self.Re,
            'euler angles' : (0.,0.,0.),
            'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out }
        self.surfaces[1].Initialize(surf_dict)
        cap_angle1 = np.arcsin(self.Re/self.R1)
        cap_thickness1 = self.R1*(1-np.cos(cap_angle1))
        cap_angle2 = np.arcsin(self.Re/self.R2)
        cap_thickness2 = self.R2*(1-np.cos(cap_angle2))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0.,0.,0.5*(cap_thickness1+cap_thickness2)),
            'euler angles' : (0.,0.,0.),
            'radius' : self.Re,
            'length' : self.Lz-cap_thickness1+cap_thickness2 }
        self.surfaces[2].Initialize(surf_dict)

class AsphericLens(base_volume):
    '''Curved beneath, planar above.  curvature radius is signed.
    Positive sign is convex, negative is concave.
    The thickness is measured along central axis of lens.
    The extremities on axis are equidistant from the origin'''
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        Lz = input_dict['thickness']
        R1 = input_dict['rcurv beneath']
        Re = input_dict['aperture radius']
        self.surfaces.append(surface.AsphericCap('s1'))
        self.surfaces.append(surface.disc('s2'))
        self.surfaces.append(surface.cylindrical_shell('shell'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0.,0.,-Lz/2),
            'euler angles' : (0.,np.pi,0.),
            'radius' : Re,
            'radius of sphere' : -R1,
            'mesh points' : input_dict['mesh points'],
            'conic constant' : input_dict['conic constant'],
            'aspheric coefficients' : tuple(-np.array(input_dict['aspheric coefficients'])) }
        self.surfaces[0].Initialize(surf_dict)
        cap_thickness = -self.surfaces[0].cap_thickness
        Rd = self.surfaces[0].Rd # reduced relative to Re
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0.,0.,Lz/2),
            'euler angles' : (0.,0.,0.),
            'radius' : Rd }
        self.surfaces[1].Initialize(surf_dict)
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0.,0.,0.5*cap_thickness),
            'euler angles' : (0.,0.,0.),
            'radius' : Rd,
            'length' : Lz-cap_thickness }
        self.surfaces[2].Initialize(surf_dict)

class Box(base_volume):
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        self.size = input_dict['size']
        self.surfaces.append(surface.rectangle('f1'))
        yrot = lambda q : (-np.pi/2,-q,np.pi/2)
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,-self.size[0]/2,0,0),
            'euler angles' : yrot(-np.pi/2),
            'size' : (self.size[2],self.size[1]) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('f2'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,self.size[0]/2,0,0),
            'euler angles' : yrot(np.pi/2),
            'size' : (self.size[2],self.size[1]) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('f3'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0,-self.size[1]/2,0),
            'euler angles' : (0,np.pi/2,0),
            'size' : (self.size[0],self.size[2]) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('f4'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0,self.size[1]/2,0),
            'euler angles' : (0,-np.pi/2,0),
            'size' : (self.size[0],self.size[2]) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('f5'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0,0,-self.size[2]/2),
            'euler angles' : (0,np.pi,0),
            'size' : (self.size[0],self.size[1]) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('f6'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0,0,self.size[2]/2),
            'euler angles' : (0,0,0),
            'size' : (self.size[0],self.size[1]) }
        self.surfaces[-1].Initialize(surf_dict)

class Prism(base_volume):
    '''Isoceles prism.  Dispersion is in local xz plane.  Base is in the plane x = -size[0]/2.'''
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        yrot = lambda q : (-np.pi/2,-q,np.pi/2)
        self.size = input_dict['size']
        halfangle = np.arctan(0.5*self.size[2]/self.size[0])
        hyp = self.size[0]/np.cos(halfangle)
        self.surfaces.append(surface.rectangle('base'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,-self.size[0]/2,0,0),
            'euler angles' : yrot(-np.pi/2),
            'size' : (self.size[2],self.size[1]) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('in'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0,0,-np.sin(halfangle)*hyp/2),
            'euler angles' : yrot(np.pi-halfangle),
            'size' : (hyp,self.size[1]) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('out'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0,0,np.sin(halfangle)*hyp/2),
            'euler angles' : yrot(halfangle),
            'size' : (hyp,self.size[1]) }
        self.surfaces[-1].Initialize(surf_dict)
        a = np.array([self.size[0]/2,self.size[1]/2,0.0])
        b = np.array([-self.size[0]/2,self.size[1]/2,self.size[2]/2])
        c = np.array([-self.size[0]/2,self.size[1]/2,-self.size[2]/2])
        n = np.array([0.0,1.0,0.0])
        self.surfaces.append(surface.triangle(a,b,c,n,n,n,self.disp_in,self.disp_out))
        a[1] *= -1
        b[1] *= -1
        c[1] *= -1
        n = np.array([0.0,-1.0,0.0])
        self.surfaces.append(surface.triangle(c,b,a,n,n,n,self.disp_in,self.disp_out))

class PellinBroca(base_volume):
    '''Pellin-Broca Prism. Local origin is the pivot point, which is on side C.
    The ``size`` tuple is given as (A,h,B) where h is the height,
    A is the input side length, and B is the output side length.  Sides C and D are inferred
    from the ``angle`` parameter (refraction angle through A, also angle AD), and the requirement that input
    and output beams make a 90 degree angle.  Side C is the reflector::

                B
           ------------
           |          |         x
           |         |          ^
         A |         *  C       |
           |        |           |
           |       |            -------> z
           |   ----
           ----
                D
        '''
    # The following formulas come from sympy, see extras
    def SideLengths(self,A,B,qr):
        a = np.pi/4 + 2*qr
        b = np.pi/4 + qr
        C = (np.sqrt(2)*A*np.sin(a) + A + np.sqrt(2)*B*np.cos(a) - B)/(2*np.sin(b))
        D = np.sqrt(2)*(-A*np.cos(b) + B*np.sin(b))
        return A,B,C,D
    def MidplaneVertices(self,A,B,C,D,qr):
        cot = lambda q : 1/np.tan(q)
        a = np.pi/4 + 2*qr
        b = np.pi/4 + qr
        # Work out CD referenced to AD.
        CDx = -np.sqrt(2)*A*np.sin(a)/2 + A/2 - np.sqrt(2)*B*np.cos(a)/2 + B/2
        CDz = B - (np.sqrt(2)*A*np.sin(a) + A + np.sqrt(2)*B*np.cos(a) - B)*np.cos(b)/(2*np.sin(b))
        # Pivot point referenced to AD.
        Px = (A*cot(b) + A - B)/(cot(b) + 1)
        Pz = A - (A*cot(b) + A - B)/(cot(b) + 1)
        # Return vertices DA,AB,BC,CD
        return (-Px,0.0,-Pz) , (A-Px,0.0,-Pz) , (A-Px,0.0,B-Pz) , (CDx-Px,0.0,CDz-Pz)
    def SideCenters(self,A,B,C,D,qr):
        DA,AB,BC,CD = self.MidplaneVertices(A,B,C,D,qr)
        Ac = 0.5*(np.array(DA)+np.array(AB))
        Bc = 0.5*(np.array(AB)+np.array(BC))
        Cc = 0.5*(np.array(BC)+np.array(CD))
        Dc = 0.5*(np.array(CD)+np.array(DA))
        return Ac,Bc,Cc,Dc
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        yrot = lambda q : (-np.pi/2,-q,np.pi/2)
        sin = np.sin
        cos = np.cos
        sqrt = np.sqrt
        pi = np.pi
        cot = lambda q : 1/np.tan(q)
        qr = input_dict['angle'] # design refraction angle, conventionally pi/6
        A = input_dict['size'][0] # input plane length
        h = input_dict['size'][1] # height
        B = input_dict['size'][2] # output plane length

        A,B,C,D = self.SideLengths(A,B,qr)
        Ac,Bc,Cc,Dc = self.SideCenters(A,B,C,D,qr)
        DA,AB,BC,CD = self.MidplaneVertices(A,B,C,D,qr)

        self.surfaces.append(surface.rectangle('in'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,) + tuple(Ac),
            'euler angles' : yrot(np.pi),
            'size' : (A,h) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('out'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,) + tuple(Bc),
            'euler angles' : yrot(np.pi/2),
            'size' : (B,h) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('back'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'reflective' : True,
            'origin' : (None,) + tuple(Cc),
            'euler angles' : yrot(-(np.pi/4-qr)),
            'size' : (C,h) }
        self.surfaces[-1].Initialize(surf_dict)
        self.surfaces.append(surface.rectangle('extra'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,) + tuple(Dc),
            'euler angles' : yrot(-(np.pi/2-qr)),
            'size' : (D,h) }
        self.surfaces[-1].Initialize(surf_dict)

        # top and bottom surfaces of the PB prism formed by two triangles
        dy = np.array([0.0,h/2,0.0])
        a = np.array(DA) + dy
        b = np.array(BC) + dy
        c = np.array(AB) + dy
        n = np.array([0.0,1.0,0.0])
        self.surfaces.append(surface.triangle(a,b,c,n,n,n,self.disp_in,self.disp_out))
        a[1] *= -1
        b[1] *= -1
        c[1] *= -1
        n = np.array([0.0,-1.0,0.0])
        self.surfaces.append(surface.triangle(c,b,a,n,n,n,self.disp_in,self.disp_out))
        a = np.array(DA) + dy
        b = np.array(CD) + dy
        c = np.array(BC) + dy
        n = np.array([0.0,1.0,0.0])
        self.surfaces.append(surface.triangle(a,b,c,n,n,n,self.disp_in,self.disp_out))
        a[1] *= -1
        b[1] *= -1
        c[1] *= -1
        n = np.array([0.0,-1.0,0.0])
        self.surfaces.append(surface.triangle(c,b,a,n,n,n,self.disp_in,self.disp_out))

    def OrbitPoints(self):
        return 3
    def Propagate(self,xp,eikonal,vg,orb):
        self.RaysGlobalToLocal(xp,eikonal,vg)
        self.Transition(xp,eikonal,vg,orb)
        self.Transition(xp,eikonal,vg,orb)
        self.Transition(xp,eikonal,vg,orb)
        self.RaysLocalToGlobal(xp,eikonal,vg)
        self.OrbitsLocalToGlobal(orb)

class PellinBroca2(base_volume):
    '''Alternate Pellin-Broca Prism. The pivot point, on side D, is placed at the origin.
    The ``size`` tuple is given as (A,h,B) where h is the height,
    A is the input side length, and B is the output side length.  Sides C and D are inferred
    from the ``angle`` parameter (refraction angle through A, also angle AD-pi/4), and the
    requirement that input and output beams make a 90 degree angle.  Side D is the reflector::

                  B
          ---------------
          |             |          x
          |             | C        ^
        A |        ------          |
          |   ---*-                |
          ----    D                ------> z
        '''

    def SideLengths(self,A,B,qr):
        a = np.pi/4+qr
        return A,B,A-B/np.tan(a),B/np.sin(a)
    def MidplaneVertices(self,A,B,C,D,qr):
        a = np.pi/4+qr
        # Pivot point referenced to vertex AD
        Px = A/(1+np.tan(a))
        Pz = A/(1+1/np.tan(a))
        # Return vertices DA,AB,BC,CD
        return (-Px,0.0,-Pz) , (A-Px,0.0,-Pz) , (A-Px,0.0,B-Pz) , (A-C-Px,0.0,B-Pz)
    def SideCenters(self,A,B,C,D,qr):
        DA,AB,BC,CD = self.MidplaneVertices(A,B,C,D,qr)
        Ac = 0.5*(np.array(DA)+np.array(AB))
        Bc = 0.5*(np.array(AB)+np.array(BC))
        Cc = 0.5*(np.array(BC)+np.array(CD))
        Dc = 0.5*(np.array(CD)+np.array(DA))
        return Ac,Bc,Cc,Dc
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        yrot = lambda q : (-np.pi/2,-q,np.pi/2)
        qr = input_dict['angle'] # design refraction angle, conventionally pi/6
        A = input_dict['size'][0] # input plane length
        h = input_dict['size'][1] # height
        B = input_dict['size'][2] # output plane length

        A,B,C,D = self.SideLengths(A,B,qr)
        Ac,Bc,Cc,Dc = self.SideCenters(A,B,C,D,qr)
        DA,AB,BC,CD = self.MidplaneVertices(A,B,C,D,qr)

        # B surface
        self.surfaces.append(surface.rectangle('out'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,) + tuple(Bc),
            'euler angles' : yrot(np.pi/2),
            'size' : (B,h) }
        self.surfaces[-1].Initialize(surf_dict)
        # A surface
        self.surfaces.append(surface.rectangle('in'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,) + tuple(Ac),
            'euler angles' : yrot(np.pi),
            'size' : (A,h) }
        self.surfaces[-1].Initialize(surf_dict)
        # C surface
        self.surfaces.append(surface.rectangle('side'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,) + tuple(Cc),
            'euler angles' : yrot(0.0),
            'size' : (C,h) }
        self.surfaces[-1].Initialize(surf_dict)
        # D surface
        self.surfaces.append(surface.rectangle('reflective'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'reflective' : True,
            'origin' : (None,) + tuple(Dc),
            'euler angles' : yrot(-np.pi/4-qr),
            'size' : (D,h) }
        self.surfaces[-1].Initialize(surf_dict)

        # top and bottom surfaces of the PB prism formed by two triangles
        dy = np.array([0.0,h/2,0.0])
        a = np.array(DA) + dy
        b = np.array(BC) + dy
        c = np.array(AB) + dy
        n = np.array([0.0,1.0,0.0])
        self.surfaces.append(surface.triangle(a,b,c,n,n,n,self.disp_in,self.disp_out))
        a[1] *= -1
        b[1] *= -1
        c[1] *= -1
        n = np.array([0.0,-1.0,0.0])
        self.surfaces.append(surface.triangle(c,b,a,n,n,n,self.disp_in,self.disp_out))
        a = np.array(DA) + dy
        b = np.array(CD) + dy
        c = np.array(BC) + dy
        n = np.array([0.0,1.0,0.0])
        self.surfaces.append(surface.triangle(a,b,c,n,n,n,self.disp_in,self.disp_out))
        a[1] *= -1
        b[1] *= -1
        c[1] *= -1
        n = np.array([0.0,-1.0,0.0])
        self.surfaces.append(surface.triangle(c,b,a,n,n,n,self.disp_in,self.disp_out))

    def OrbitPoints(self):
        return 3
    def Propagate(self,xp,eikonal,vg,orb):
        self.RaysGlobalToLocal(xp,eikonal,vg)
        self.Transition(xp,eikonal,vg,orb)
        self.Transition(xp,eikonal,vg,orb)
        self.Transition(xp,eikonal,vg,orb)
        self.RaysLocalToGlobal(xp,eikonal,vg)
        self.OrbitsLocalToGlobal(orb)

class RetroPrism(Prism):
    '''Same as Prism but assumes TIR on one surface.'''
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        self.surfaces[0].reflective = True
    def OrbitPoints(self):
        return 3
    def Propagate(self,xp,eikonal,vg,orb):
        self.RaysGlobalToLocal(xp,eikonal,vg)
        self.Transition(xp,eikonal,vg,orb)
        self.Transition(xp,eikonal,vg,orb)
        self.Transition(xp,eikonal,vg,orb)
        self.RaysLocalToGlobal(xp,eikonal,vg)
        self.OrbitsLocalToGlobal(orb)

class Cylinder(base_volume):
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        self.Rd = input_dict['radius']
        self.Lz = input_dict['length']
        self.surfaces.append(surface.disc('d1'))
        self.surfaces.append(surface.disc('d2'))
        self.surfaces.append(surface.cylindrical_shell('shell'))
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0.,0.,-self.Lz/2),
            'euler angles' : (0.,np.pi,0.),
            'radius' : self.Rd }
        self.surfaces[0].Initialize(surf_dict)
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0.,0.,self.Lz/2),
            'euler angles' : (0.,0.,0.),
            'radius' : self.Rd }
        self.surfaces[1].Initialize(surf_dict)
        surf_dict = { 'dispersion beneath' : self.disp_in,
            'dispersion above' : self.disp_out,
            'origin' : (None,0.,0.,0.),
            'euler angles' : (0.,0.,0.),
            'radius' : self.Rd,
            'length' : self.Lz }
        self.surfaces[2].Initialize(surf_dict)

class nonuniform_volume(base_volume):
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        self.vol_dict = input_dict
    def OrbitPoints(self):
        try:
            return 2+int(self.vol_dict['steps']/self.vol_dict['subcycles'])
        except KeyError:
            return 2
    def Propagate(self,xp,eikonal,vg,orb,dens=None):
        self.RaysGlobalToLocal(xp,eikonal,vg)
        self.Transition(xp,eikonal,vg,orb)
        if self.propagator=='paraxial':
            ret = paraxial_kernel.track(self.cl,xp,eikonal,vg,self.vol_dict)
            self.paraxial_wave = ret[0]
            self.paraxial_j = ret[1]
            self.paraxial_chi = ret[2]
            self.paraxial_plasma = ret[3]
            self.dom4d = ret[4]
            self.UpdateOrbits(xp,eikonal,orb)
        if self.propagator=='uppe':
            ret = uppe_kernel.track(self.cl,xp,eikonal,vg,self.vol_dict)
            self.uppe_wave = ret[0]
            self.uppe_j = ret[1]
            self.uppe_chi = ret[2]
            self.uppe_plasma = ret[3]
            self.dom4d = ret[4]
            self.UpdateOrbits(xp,eikonal,orb)
        if self.propagator=='eikonal':
            logging.info("start eikonal propagation")
            ray_kernel.SyncSatellites(xp,vg)
            logging.info("finished sync")
            if np.any(dens==None):
                ray_kernel.track(self.cl,xp,eikonal,self.vol_dict,orb)
            else:
                ray_kernel.track_RIC(self.cl,xp,eikonal,dens,self.vol_dict,orb)
            vg[...] = self.disp_in.vg(xp)
            self.Transition(xp,eikonal,vg,orb)
        self.RaysLocalToGlobal(xp,eikonal,vg)
        self.OrbitsLocalToGlobal(orb)
    def Report(self,basename,mks_length):
        super().Report(basename,mks_length)
        if self.propagator=='paraxial':
            print('    Write paraxial wave data...')
            np.save(basename+'_'+self.name+'_paraxial_wave',self.paraxial_wave)
            np.save(basename+'_'+self.name+'_paraxial_j',self.paraxial_j)
            np.save(basename+'_'+self.name+'_paraxial_chi',self.paraxial_chi)
            np.save(basename+'_'+self.name+'_paraxial_plasma',self.paraxial_plasma)
            np.save(basename+'_'+self.name+'_paraxial_plot_ext',self.dom4d)
        if self.propagator=='uppe':
            print('    Write UPPE wave data...')
            np.save(basename+'_'+self.name+'_uppe_wave',self.uppe_wave)
            np.save(basename+'_'+self.name+'_uppe_j',self.uppe_j)
            np.save(basename+'_'+self.name+'_uppe_chi',self.uppe_chi)
            np.save(basename+'_'+self.name+'_uppe_plasma',self.uppe_plasma)
            np.save(basename+'_'+self.name+'_uppe_plot_ext',self.dom4d)

class grid_volume(nonuniform_volume):
    def Initialize(self,input_dict):
        super().Initialize(input_dict)
        self.ne = np.zeros(1)
    def Propagate(self,xp,eikonal,vg,orb):
        super().Propagate(xp,eikonal,vg,orb,dens=self.ne)

class AnalyticDensity(nonuniform_volume):
    '''Base class used to create an analytical density profile.
    Input dictionary key "density function" is a string containing a CL expression which may depend on x and r2.
    Here x is a double4, x.s1 = x, x.s2 = y, x.s3 = z, and r2 = x**2 + y**2
    Input dictionary key "density lambda" is a Python lambda function of (x,y,z,r2).
    Here the arguments are numpy arrays with shape (bundles,rays), and the return value **must have the same shape**.'''
    def get_density_plugin(self,input_dict):
        # Set up the dispersion function in OpenCL kernel
        plugin_str = '\ninline double dot4(const double4 x,const double4 y);'
        plugin_str += '\ninline double wp2(const double4 x)\n'
        plugin_str += '{\n'
        plugin_str += 'const double r2 = x.s1*x.s1 + x.s2*x.s2;\n'
        plugin_str += 'return ' + input_dict['density function'] + ';'
        plugin_str += '}\n'
        plugin_str += '\ninline double D_alpha(const double4 x,const double4 k)\n'
        plugin_str += '{\n'
        plugin_str += self.disp_in.Dxk('wp2(x)')
        plugin_str += '}\n\n'
        return plugin_str
    def GetRelDensity(self,xp):
        x = xp[...,1]
        y = xp[...,2]
        z = xp[...,3]
        r2 = np.einsum('...i,...i',xp[...,1:3],xp[...,1:3])
        return self.vol_dict['density lambda'](x,y,z,r2)

class AnalyticCylinder(AnalyticDensity,Cylinder):
    '''Fill a cylinder with an analytical density profile.  See also AnalyticDensity class.'''
    def InitializeCL(self,cl,input_dict):
        # Here and below we are writing a unique OpenCL program for this object.
        # Therefore do not assign to self.cl by reference, instead use copy.
        self.cl = copy.copy(cl)
        self.cl.setup_workgroups(input_dict)
        plugin_str = super().get_density_plugin(input_dict)
        plugin_str += '\ninline double outside(const double4 x)\n'
        plugin_str += '{\n'
        plugin_str += 'const double Rd = ' + str(self.Rd) + ';\n'
        plugin_str += 'const double dz = ' + str(self.Lz) + ';\n'
        plugin_str += '''const double r2 = x.s1*x.s1 + x.s2*x.s2;
                        return (double)(r2>Rd*Rd || x.s3*x.s3>0.25*dz*dz);
                        }\n'''
        self.cl.add_program('kernels','ray_integrator',plugin=plugin_str)

class AnalyticBox(AnalyticDensity,Box):
    '''Fill a box with an analytical density profile.  See also AnalyticDensity class.'''
    def InitializeCL(self,cl,input_dict):
        # Here and below we are writing a unique OpenCL program for this object.
        # Therefore do not assign to self.cl by reference, instead use copy.
        self.cl = copy.copy(cl)
        self.cl.setup_workgroups(input_dict)
        plugin_str = super().get_density_plugin(input_dict)
        plugin_str += '\ninline double outside(const double4 x)\n'
        plugin_str += '{\n'
        plugin_str += 'const double Lx = ' + str(self.size[0]) + ';\n'
        plugin_str += 'const double Ly = ' + str(self.size[1]) + ';\n'
        plugin_str += 'const double Lz = ' + str(self.size[2]) + ';\n'
        plugin_str += 'return (double)(x.s1*x.s1>0.25*Lx*Lx || x.s2*x.s2>0.25*Ly*Ly || x.s3*x.s3>0.25*Lz*Lz);\n}\n'
        self.cl.add_program('kernels','ray_integrator',plugin=plugin_str)

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
        self.cl = copy.copy(cl)
        self.cl.setup_workgroups(input_dict)
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

        self.cl.add_program('kernels','ray_in_cell',plugin=plugin_str)
    def GetRelDensity(self,xp):
        return ray_kernel.gather(self.cl,xp,self.ne)

class TestGrid(Grid):
    def LoadMap(self,input_dict):
        check_vol_tuple(input_dict['mesh points'])
        N = input_dict['mesh points']
        N = (N[1]+4,N[2]+4,N[3]+4)
        self.dx = self.size[0]/(N[1]-4)
        self.dy = self.size[1]/(N[2]-4)
        self.dz = self.size[2]/(N[3]-4)
        x = grid_tools.cell_centers(-2*self.dx-self.size[0]/2,2*self.dx+self.size[0]/2,N[1])
        y = grid_tools.cell_centers(-2*self.dy-self.size[1]/2,2*self.dy+self.size[1]/2,N[2])
        z = grid_tools.cell_centers(-2*self.dz-self.size[2]/2,2*self.dz+self.size[2]/2,N[3])
        rho2 = np.outer(x**2,np.ones(N[2])) + np.outer(np.ones(N[1]),y**2)
        coeff = self.vol_dict['radial coefficients']
        fr = coeff[0] + coeff[1]*rho2 + coeff[2]*rho2**2 + coeff[3]*rho2**4
        self.ne = input_dict['density multiplier']*np.einsum('ij,k->ijk',fr,np.ones(N[3]))

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
        self.cl = copy.copy(cl)
        self.cl.setup_workgroups(input_dict)
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

        self.cl.add_program('kernels','ray_in_cell',plugin_str)
    def GetRelDensity(self,xp):
        return ray_kernel.gather(self.cl,xp,self.ne)

class AxisymmetricTestGrid(AxisymmetricGrid):
    def LoadMap(self,input_dict):
        check_rz_tuple(input_dict['mesh points'])
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
