'''
Module :samp:`base`
----------------------

This is a place for root level interfaces or implementations.
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
