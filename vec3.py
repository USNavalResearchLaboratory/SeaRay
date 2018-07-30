import numpy as np

def RotateZ(v,q):
	st = np.sin(q);
	ct = np.cos(q);
	xOld = v[0];
	v[0] = xOld*ct - v[1]*st;
	v[1] = xOld*st + v[1]*ct;

def RotateY(v,q):
	st = np.sin(q);
	ct = np.cos(q);
	zOld =  v[2];
	v[2] = zOld*ct - v[0]*st;
	v[0] = zOld*st + v[0]*ct;

def RotateX(v,q):
	st = np.sin(q);
	ct = np.cos(q);
	yOld = v[1];
	v[1] = yOld*ct - v[2]*st;
	v[2] = yOld*st + v[2]*ct;

def EulerRotate(v,alpha,beta,gamma):
	RotateZ(v,gamma)
	RotateX(v,beta)
	RotateZ(v,alpha)

class basis:
	def __init__(self):
		self.M = np.eye(3)
	def Create(self,U,W):
		V = np.cross(W,U)
		self.M[0,:] = U / np.sqrt(np.sum(U*U))
		self.M[1,:] = V / np.sqrt(np.sum(V*V))
		self.M[2,:] = W / np.sqrt(np.sum(W*W))
	def Print(self):
		print(self.M)
	def VectorTransformation(self,vec,sumkey):
		vec[...] = np.einsum(sumkey,self.M,vec)
	def ExpressInBasis(self,vec):
		self.VectorTransformation(vec,'ij,...j')
	def ExpressInStdBasis(self,vec):
		self.VectorTransformation(vec,'ij,...i')
	def RayTransformation(self,xp,eik,sumkey):
		xp[...,1:4] = np.einsum(sumkey,self.M,xp[...,1:4])
		xp[...,5:8] = np.einsum(sumkey,self.M,xp[...,5:8])
		eik[...,1:4] = np.einsum(sumkey,self.M,eik[...,1:4])
	def ExpressRaysInBasis(self,xp,eik):
		self.RayTransformation(xp,eik,'ij,...j')
	def ExpressRaysInStdBasis(self,xp,eik):
		self.RayTransformation(xp,eik,'ij,...i')
	def AboutX(self,q):
		RotateX(self.M[0,:],q)
		RotateX(self.M[1,:],q)
		RotateX(self.M[2,:],q)
	def AboutY(self,q):
		RotateY(self.M[0,:],q)
		RotateY(self.M[1,:],q)
		RotateY(self.M[2,:],q)
	def AboutZ(self,q):
		RotateZ(self.M[0,:],q)
		RotateZ(self.M[1,:],q)
		RotateZ(self.M[2,:],q)
	def EulerRotate(self,alpha,beta,gamma):
		EulerRotate(self.M[0,:],alpha,beta,gamma)
		EulerRotate(self.M[1,:],alpha,beta,gamma)
		EulerRotate(self.M[2,:],alpha,beta,gamma)
