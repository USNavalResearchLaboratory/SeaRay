#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef double tw_Float;
typedef double2 tw_Complex;
// Operations with tw_Float and tw_Complex:
// a*b : one or both operands must be tw_Float
// a+b : both operands must be the same type
// a/b : b must be tw_Float
// a-b : both operands must be the same type

tw_Complex cmul(tw_Complex a,tw_Complex b)
{
	return (tw_Complex)(a.s0*b.s0-a.s1*b.s1,a.s0*b.s1+a.s1*b.s0);
}

tw_Complex cexp(tw_Complex x)
{
	return (tw_Complex)(cos(x.s1)*exp(-x.s0),sin(x.s1)*exp(-x.s0));
}

__kernel void PropagateLinear(__global tw_Complex * q,__global tw_Complex * kz,__global tw_Float * kg,const tw_Float z)
{
	// POINT PROTOCOL
	// q = [w][x][y]
	// Forming A = exp(i*(kz+kg)*z)*q
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Complex Q = q[idx];
	const tw_Float Kr = kz[idx].s0;
	const tw_Float Ki = kz[idx].s1;
	const tw_Complex iKz = (tw_Complex)(-Ki*z,(Kr+kg[i0])*z);
	q[idx] = cmul(cexp(iKz),Q);
}

__kernel void CurrentToODERHS(__global tw_Complex * J,__global tw_Complex * kz,__global tw_Float * kg,const tw_Float z)
{
	// POINT PROTOCOL
	// J = [w][x][y]
	// Forming S = 0.5i*np.exp(-i*kz*z)*J/real(kz)
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Complex j = J[idx];
	const tw_Float Kr = kz[idx].s0;
	const tw_Float Ki = kz[idx].s1;
	const tw_Complex iKz = (tw_Complex)(-Ki*z,(Kr+kg[i0])*z);
	const tw_Complex i2 = (tw_Complex)(0.0,0.5);
	J[idx] = cmul(i2,cmul(cexp(-iKz),j))/Kr;
}

__kernel void ComputePlasmaPolarization(
	__global tw_Float * P,
	__global tw_Float * E,
	__global tw_Float * ne,
	const tw_Float Omega,
	const tw_Float gamma,
	const tw_Float dt,
	const int steps)
{
	// STRIP PROTOCOL
	// P,F are 3d arrays with [t][x][y]
	// P is expected to be zeros
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int offset = j0 + i0*Ny;
	const int s = Nx*Ny;

	for (int k=steps-8;k>=0;--k)
	{
		P[offset+k*s] = dt*dt*ne[offset+k*s+s]*E[offset+k*s+s] + (2.0+dt*gamma)*P[offset+k*s+s] - P[offset+k*s+2*s];
		P[offset+k*s] /= 1.0 + dt*gamma + dt*dt*Omega*Omega;
	}
}

__kernel void AddKerrPolarization(__global tw_Float * P, __global tw_Float * E,const tw_Float chi3)
{
	// TEMPORAL POINT PROTOCOL
	// P = [t][x][y]
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Float Ei = E[idx];
	P[idx] += chi3*Ei*Ei*Ei;
}
