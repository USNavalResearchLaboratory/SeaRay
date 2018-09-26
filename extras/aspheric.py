import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Program to work out the aspheric coefficients for a given focal length and refractive index

# The refractive index of the lens material
n = np.double(1.531)
# The back focal length
f = np.double(0.002)
# Thickness of lens measured between on axis extrema
thickness = f/2

def fitting_func(rho2,R,A4,A6,A8,A10,A12):
	'''The standard aspheric sag formula'''
	k = np.double(0.0)
	C = 1/R
	chi = lambda rho2 : np.sqrt(1-(1+k)*C**2*rho2)
	return C*rho2/(1+chi(rho2)) + A4*rho2**2 + A6*rho2**3 + A8*rho2**4 + A10*rho2**5 + A12*rho2**6

def rhs(q,z):
	'''Right hand side of the equation dz/dq = rhs(q,z).
	Here q is the angle of a focused ray after exiting the lens.
	The lens surface (rho,z) is parameterized as (rho(q),z(q)).'''
	chi = lambda q : np.sin(q)/np.sqrt(n**2-np.sin(q)**2)
	chi1 = lambda q : np.sin(q)/(np.sqrt(n**2-np.sin(q)**2)-1)
	dchi = lambda q : np.cos(q)*(1+chi(q)**2)/np.sqrt(n**2-np.sin(q)**2)
	return (f/np.cos(q)**2 + dchi(q)*(thickness-z))*chi1(q)/(1+chi(q)*chi1(q))

def rhof(q,z):
	'''Equation for rho(q); it is assumed z(q) has already been computed.'''
	return f*np.tan(q) + (thickness-z)*np.sin(q)/np.sqrt(n**2-np.sin(q)**2)

sol = solve_ivp(rhs,[np.double(0.0),np.double(np.pi/6)],[np.double(0.0)],t_eval=np.linspace(0.0,np.pi/6,1024).astype(np.double),rtol=1e-11,atol=1e-11)
q = sol.t
z = sol.y[0]
rho = rhof(q,z)

params,cov = curve_fit(fitting_func,rho**2,z,[f,1/f**3,1/f**5,1/f**7,1/f**9,1/f**11])
print('Radius of curvature =',params[0])
print('Aspheric coefficients normalized to f^(n-1) =',params[1:]*[f**3,f**5,f**7,f**9,f**11])

rho_max = np.max(rho)
rho_fit = np.linspace(0.0,rho_max,100)
z_fit = fitting_func(rho_fit**2,params[0],params[1],params[2],params[3],params[4],params[5])
rho_fit = np.concatenate((rho_fit,[rho_max,0.0]))
z_fit = np.concatenate((z_fit,[thickness,thickness]))
plt.plot(z_fit,rho_fit,'r--')
plt.plot(z,rho,'k-')

plt.show()
