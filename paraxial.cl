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
	return (tw_Complex)(cos(x.s1)*exp(x.s0),sin(x.s1)*exp(x.s0));
}

tw_Float cmag(tw_Complex x)
{
	return sqrt(x.s0*x.s0 + x.s1*x.s1);
}

__kernel void LoadModulus(__global tw_Complex * q,__global tw_Float * modulus)
{
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Complex Q = q[idx];
	modulus[idx] = cmag(Q);
}

__kernel void LoadStepSize(__global tw_Complex * q,__global tw_Complex * s,__global tw_Float *dz,const tw_Float dz0,const tw_Float dphi,const tw_Float amin)
{
	// alpha = amplitude rate = Re(s*conj(q))/|q|**2
	// keff = phase rate = Im(s*conj(q))/|q|**2
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Float minRate = 0.5*dphi/dz0;
	const tw_Complex Q = q[idx];
	const tw_Complex S = s[idx];
	const tw_Float alpha = (Q.s0*S.s0 + Q.s1*S.s1) / (Q.s0*Q.s0 + Q.s1*Q.s1 + amin*amin);
	const tw_Float keff = (Q.s0*S.s1 - Q.s1*S.s0) / (Q.s0*Q.s0 + Q.s1*Q.s1 + amin*amin);
	const tw_Float K = minRate + fabs(keff) - alpha*(tw_Float)(alpha<0.0); // ignore creation terms
	dz[idx] = dphi/K;
}

__kernel void PropagateLinear(__global tw_Complex * q,__global tw_Complex * kz,const tw_Float z)
{
	// SPECTRAL POINT PROTOCOL
	// q = [w][x][y]
	// Forming A = exp(i*kz*z)*q
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
	const tw_Complex iKz = (tw_Complex)(-Ki*z,Kr*z);
	q[idx] = cmul(cexp(iKz),Q);
}

__kernel void CurrentToODERHS(__global tw_Complex * J,__global tw_Complex * kz,__global tw_Float * k00,const tw_Float z)
{
	// SPECTRAL POINT PROTOCOL
	// J = [w][x][y]
	// Forming S = (i/2)*np.exp(-i*kz*z)*J/k00
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2); // watch collision with wavenumber k0
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Complex j = J[idx];
	const tw_Float Kr = kz[idx].s0;
	const tw_Float Ki = kz[idx].s1;
	const tw_Complex iKz = (tw_Complex)(-Ki*z,Kr*z);
	const tw_Complex i2 = (tw_Complex)(0.0,0.5);
	J[idx] = cmul(i2,cmul(cexp(-iKz),j))/k00[i0]; // k00 has np.newaxis involved
}

__kernel void AddPlasmaCurrent(__global tw_Complex * J,__global tw_Complex * A,__global tw_Float * ne)
{
	// TEMPORAL POINT PROTOCOL
	// J,A = [t][x][y]
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Float a2 = A[idx].s0*A[idx].s0 + A[idx].s1*A[idx].s1;
	J[idx] += ne[idx]*(0.25*a2-1.0)*A[idx];
}

__kernel void SetKerrPolarization(__global tw_Complex * P, __global tw_Complex * E,const tw_Float chi3)
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
	const tw_Float e2 = E[idx].s0*E[idx].s0 + E[idx].s1*E[idx].s1;
	P[idx] = 0.5*chi3*e2*E[idx];
}

__kernel void AddNonuniformChi(__global tw_Complex * P, __global tw_Complex * E,__global tw_Complex * dchi)
{
	// SPECTRAL POINT PROTOCOL
	// P = [w][x][y]
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	P[idx] += cmul(dchi[idx],E[idx]);
}
