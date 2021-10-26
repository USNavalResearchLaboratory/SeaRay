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

__kernel void RKStage(__global tw_Complex *qw,
	__global tw_Complex *qi,
	__global tw_Complex *qf,
	__global tw_Complex *kn,
	const tw_Float Csol,
	const tw_Float Ceval)
{
	// POINT PROTOCOL
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	qw[idx] = qi[idx] + Ceval*kn[idx]; // the estimate for the next source evaluation
	qf[idx] += Csol*kn[idx]; // accumulating the final solution
}

__kernel void RKFinish(__global tw_Complex *qw,__global tw_Complex *qf,__global tw_Complex *kn,const tw_Float Csol)
{
	// POINT PROTOCOL
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	qw[idx] = qf[idx] + Csol*kn[idx]; // last stage, all we need is the final solution
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

__kernel void CurrentToODERHS(__global tw_Complex * J,__global tw_Complex * kz,__global tw_Float *kg,const tw_Float z)
{
	// POINT PROTOCOL
	// J = [w][x][y]
	// Forming S = 0.5i*exp(-i*(kz+kg)*z)*J/real(kz)
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


__kernel void VectorPotentialToElectricField(__global tw_Complex *A,__global tw_Complex *kz)
{
	// SPECTRAL-WK POINT PROTOCOL, cylindrical
	// A = [w][kr][m]
	// For a special set of conditions E = i*w*A - grad(phi) = (i*w-0.5*i*kperp2/w)A ~ i*kz*A
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const tw_Complex iKz = cmul((tw_Complex)(0.0,1.0),kz[idx]);
	A[idx] = cmul(iKz,A[idx]);
}

__kernel void SetKerrPolarization(__global tw_Float * P, __global tw_Float * E,const tw_Float chi3)
{
	// TEMPORAL POINT PROTOCOL
	// P,E = [t][x][y]
	// On input P has <E.E>, on output the polarization
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	P[idx] *= 1.5*chi3*E[idx];
}
