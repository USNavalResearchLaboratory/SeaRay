'''
Module :samp:`base`
----------------------

This is a place for root level interfaces or implementations and other type checking related components.
'''
import numpy as np

class base_volume_interface:
    '''
    The base_volume is an abstract class which takes care of rays entering and exiting an arbitrary collection of surfaces
    that form a closed region.  Volumes filled uniformly can be derived simply by creating the surface list in :samp:`self.Initialize`.
    '''
    def OrbitPoints(self) -> int:
        pass
    def Translate(self,r):
        pass
    def EulerRotate(self,q):
        pass
    def Initialize(self,input_dict):
        pass
    def InitializeCL(self,cl,input_dict):
        pass
    def RaysGlobalToLocal(self,xp,eikonal,vg):
        pass
    def RaysLocalToGlobal(self,xp,eikonal,vg):
        pass
    def SelectRaysForSurface(self,t_list,idx) -> np.ndarray:
        pass
    def UpdateOrbits(self,xp,eikonal,orb):
        pass
    def OrbitsLocalToGlobal(self,orb):
        pass
    def Transition(self,xp,eikonal,vg,orb):
        pass
    def Propagate(self,xp,eikonal,vg,orb):
        pass
    def GetRelDensity(self,xp) -> np.double | np.ndarray:
        pass
    def Report(self,basename,mks_length):
        pass

def wrong_tuple(nm,num,idxSome,idxNone,tup,bounds):
    if len(tup)!=num or any([tup[i]==None for i in idxSome]) or any([tup[i]!=None for i in idxNone]):
        raise ValueError('Expected ' + nm + (' boundary' if bounds else '') + ' tuple, got ' + str(tup))

def check_line_tuple(tup,bounds=False):
    if bounds:
        wrong_tuple('line-like',8,[0,1,6,7],[2,3,4,5],tup,bounds)
    else:
        wrong_tuple('line-like',4,[0,3],[1,2],tup,bounds)

def check_ray_tuple(tup,bounds=False):
    if bounds:
        wrong_tuple('ray-like',8,[0,1,2,3,4,5],[6,7],tup,bounds)
    else:
        wrong_tuple('ray-like',4,[0,1,2],[3],tup,bounds)

def check_surf_tuple(tup,bounds=False):
    if bounds:
        wrong_tuple('surface-like',8,[2,3,4,5],[0,1,6,7],tup,bounds)
    else:
        wrong_tuple('surface-like',4,[1,2],[0,3],tup,bounds)

def check_vol_tuple(tup,bounds=False):
    if bounds:
        wrong_tuple('volume-like',8,[2,3,4,5,6,7],[0,1],tup,bounds)
    else:
        wrong_tuple('volume-like',4,[1,2,3],[0],tup,bounds)

def check_four_tuple(tup,bounds=False):
    if bounds:
        wrong_tuple('full 4d',8,[0,1,2,3,4,5,6,7],[],tup,bounds)
    else:
        wrong_tuple('full 4d',4,[0,1,2,3],[],tup,bounds)

def check_rz_tuple(tup):
    wrong_tuple('RZ',2,[0,1],[],tup,False)

def check_list_of_dict(obj):
    if not isinstance(obj,list):
        raise TypeError("Expected list, got " + str(obj))
    if not all([isinstance(item,dict) for item in obj]):
        raise TypeError("Expected list of dictionaries")