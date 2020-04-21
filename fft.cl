#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef int tw_Int;
typedef double tw_Float;
typedef double2 tw_Complex;
// Operations with tw_Float and tw_Complex:
// a*b : one or both operands must be tw_Float
// a+b : both operands must be the same type
// a/b : b must be tw_Float
// a-b : both operands must be the same type

__constant tw_Float pi = 3.141592653589793;

tw_Complex cmul(tw_Complex a,tw_Complex b)
{
	return (tw_Complex)(a.s0*b.s0-a.s1*b.s1,a.s0*b.s1+a.s1*b.s0);
}

void Swap(__global tw_Float* a,__global tw_Float* b)
{
	tw_Float temp = *a;
	*a = *b;
	*b = temp;
}

int CountTrueBits(tw_Int theBits)
{
	tw_Int i,numTrue = 0;
	tw_Int numBits = sizeof(tw_Int)*8;
	tw_Int test = 1;

	for (i=0;i<numBits;i++)
	{
		if (test & theBits)
			numTrue++;
		theBits >>= 1;
	}
	return numTrue;
}

void ReverseBits(tw_Int* theBits,tw_Int numBits)
{
	tw_Int i;
	tw_Int ans = 0;
	tw_Int test = 1;
	tw_Int currVal = 1;
	currVal <<= numBits - 1;

	for (i=0;i<numBits;i++)
	{
		if (test & *theBits)
			ans += currVal;
		currVal >>= 1;
		test <<= 1;
	}
	*theBits = ans;
}

void ComplexFFT(__global tw_Float *realPart,__global tw_Float *imagPart,
	tw_Int numPoints,tw_Int interval,tw_Float inversion,tw_Float expsign)
{
	tw_Int i,j;
	tw_Int m,n,numBits;

	/*** ARRANGE DATA IN BIT REVERSED ORDER ***/

	n = numPoints;
	numBits = 0;
	while (n>1)
	{
		n >>= 1;
		numBits++;
	}
	for (n=0;n<numPoints;n++)
	{
		m = n;
		ReverseBits(&m,numBits);
		if (m>n)
		{
			Swap(&realPart[m*interval],&realPart[n*interval]);
			Swap(&imagPart[m*interval],&imagPart[n*interval]);
		}
	}

	/*** CONSTRUCT TRANSFORM ***/

	tw_Int transforms,points,a,b;
	tw_Complex Fa,Fb,W,FW,recurrenceFactor;

	transforms = numPoints;
	points = 1;

	while (transforms > 1)
	{
		// The convention for the sign of the exponential is determined by the sign of recurrenceFactor.s1.
		// The negative sign corresponds to numpy's default convention.
		recurrenceFactor.s0 = cos(pi*inversion/(tw_Float)(points));
		recurrenceFactor.s1 = expsign*sin(pi*inversion/(tw_Float)(points));
		W.s0 = 1.0;
		W.s1 = 0.0;
		for (i=0;i<points;i++)
		{
			for (j=0;j<transforms;j+=2)
			{
				a = interval*(j*points + i);
				b = interval*((j+1)*points + i);
				Fa = (tw_Complex)(realPart[a],imagPart[a]);
				Fb = (tw_Complex)(realPart[b],imagPart[b]);
				FW = cmul(Fb,W);
				Fb = Fa - FW;
				Fa = Fa + FW;
				realPart[a] = Fa.s0; imagPart[a] = Fa.s1;
				realPart[b] = Fb.s0; imagPart[b] = Fb.s1;
			}
			W = cmul(W,recurrenceFactor);
		}
		points *= 2;
		transforms /= 2;
	}

	if (inversion==-1.0)
	{
		for (i=0;i<numPoints;i++)
		{
			realPart[i*interval] /= (tw_Float)(numPoints);
			imagPart[i*interval] /= (tw_Float)(numPoints);
		}
	}
}

void RealFFT(__global tw_Float *array,tw_Int num,tw_Int interval,bool invert)
{
	// Computes N/2-1 complex numbers and 2 real numbers
	// The two reals are the lowest and highest frequencies
	// the lowest freq is stored in array[0] and the highest in array[1]
	// Then complex(array[2],array[3]) is the second frequency, etc.
	// The negative frequencies are not calculated but can be obtained from symmetry

	tw_Int i,i1,i2,i3,i4;
	tw_Float c1,c2,theta;
	tw_Complex h1,h2;
	tw_Complex recurrenceFactor,W;

	theta = pi/(tw_Float)(num/2);
	if (!invert)
	{
		c1 = 0.5;
		c2 = -0.5;
		ComplexFFT(array,&array[interval],num/2,2*interval,1.0,1.0);
	}
	else
	{
		c1 = 0.5;
		c2 = 0.5;
		theta = -theta;
	}
	// Conjugating the recurrence Factor is not sufficient for altering the
	// exponential sign convention.  However one can conjugate the result.
	recurrenceFactor.s0 = cos(theta);
	recurrenceFactor.s1 = sin(theta);
	W = recurrenceFactor;
	for (i=1;i<(num/4);i++)
	{
		i1 = interval*(2*i);
		i2 = interval*(2*i + 1);
		i3 = interval*(num - 2*i);
		i4 = interval*(num + 1 - 2*i);
		h1 = (tw_Complex)(c1*(array[i1] + array[i3]),c1*(array[i2] - array[i4]));
		h2 = (tw_Complex)(-c2*(array[i2] + array[i4]),c2*(array[i1] - array[i3]));
		h2 = cmul(W,h2);
		h2 += h1;
		array[i1] = h2.s0;
		array[i2] = h2.s1;
		h2 -= 2*h1;
		array[i3] = -h2.s0;
		array[i4] = h2.s1;
		W = cmul(W,recurrenceFactor);
	}
	if (!invert)
	{
		array[0] = (h1.s0=array[0])+array[interval];
		array[interval] = h1.s0 - array[interval];
	}
	else
	{
		array[0] = c1*((h1.s0=array[0])+array[interval]);
		array[interval] = c1*(h1.s0-array[interval]);
		ComplexFFT(array,&array[interval],num/2,2*interval,-1.0,1.0);
	}
}

__kernel void FFT(__global tw_Float *data,tw_Int steps)
{
	// Wrapper designed to be consistent with numpy.fft functions.
	// STRIP PROTOCOL
	// data = [t][x][y][2], with shape (steps,...)
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int offset = 2*(j0 + i0*Ny);
	const int s = 2*Nx*Ny;

	ComplexFFT(&data[offset],&data[offset+1],steps,s,1.0,-1.0);
}

__kernel void IFFT(__global tw_Float *data,tw_Int steps)
{
	// Wrapper designed to be consistent with numpy.fft functions.
	// STRIP PROTOCOL
	// data = [t][x][y][2], with shape (steps,...)
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int offset = 2*(j0 + i0*Ny);
	const int s = 2*Nx*Ny;

	ComplexFFT(&data[offset],&data[offset+1],steps,s,-1.0,-1.0);
}

__kernel void RFFT(__global tw_Float *in,__global tw_Float *out,tw_Int steps)
{
	// Wrapper designed to be consistent with numpy.fft functions.
	// STRIP PROTOCOL
	// in = [t][x][y], with shape (steps,...)
	// out = [w][x][y][2], with shape (steps/2+1,...), last dimension is real and imaginary part
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int offset = j0 + i0*Ny;
	const int s = Nx*Ny;

	RealFFT(&in[offset],steps,s,false);
	// Now we have to pack the complex data numpy's way
	out[2*offset] = in[offset];
	out[2*offset+1] = 0.0;
	for (int i=1;i<steps/2;i++)
	{
		out[2*offset+2*s*i] = in[offset+2*i*s];
		out[2*offset+2*s*i+1] = -in[offset+(2*i+1)*s];
	}
	out[2*offset+steps*s] = in[offset+s];
	out[2*offset+steps*s+1] = 0.0;
	// Restore the input array
	RealFFT(&in[offset],steps,s,true);
}

__kernel void IRFFT(__global tw_Float *in,__global tw_Float *out,tw_Int modes)
{
	// Wrapper designed to be consistent with numpy.fft functions.
	// STRIP PROTOCOL
	// in = [w][x][y][2], with shape (modes,...), last dimension is real and imaginary part
	// out = [t][x][y], with shape (2*modes-2,...)
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int Nx = get_global_size(0);
	const int Ny = get_global_size(1);
	const int offset = j0 + i0*Ny;
	const int s = Nx*Ny;
	const int steps = 2*modes-2;

	// First pack numpy's complex array for RealFFT processing
	out[offset] = in[2*offset];
	out[offset+s] = in[2*offset+2*s*(modes-1)];
	for (int i=1;i<modes-1;i++)
	{
		out[offset+2*i*s] = in[2*offset+2*s*i];
		out[offset+(2*i+1)*s] = -in[2*offset+2*s*i+1];
	}
	RealFFT(&out[offset],steps,s,true);
}

__kernel void DtSpectral(__global tw_Complex * A,__global double * w,const double sgn)
{
	// SPECTRAL POINT PROTOCOL
	// A = [w][x][y]
	// A --> -sgn*iw*A
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const double Ar = A[idx].s0;
	const double Ai = A[idx].s1;
	const double w0 = w[i0];
	A[idx].s0 = sgn*w0*Ai;
	A[idx].s1 = -sgn*w0*Ar;
}

__kernel void iDtSpectral(__global tw_Complex * A,__global double * w,const double sgn)
{
	// SPECTRAL POINT PROTOCOL
	// A = [w][x][y]
	// A --> i*sgn*A/w
	const int i0 = get_global_id(0);
	const int j0 = get_global_id(1);
	const int k0 = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nj = get_global_size(1);
	const int Nk = get_global_size(2);
	const int idx = i0*Nj*Nk + j0*Nk + k0;
	const double Ar = A[idx].s0;
	const double Ai = A[idx].s1;
	const double w0 = w[i0];
	A[idx].s0 = -sgn*Ai/w0;
	A[idx].s1 = sgn*Ar/w0;
}

/////////////////////////////////
//                             //
//  Hankel Transform Support   //
//                             //
/////////////////////////////////

__kernel void RootVolumeMultiply(__global tw_Complex *data,__global tw_Float *w)
{
	const int f = get_global_id(0);
	const int i = get_global_id(1);
	const int m = get_global_id(2);
	const int Nf = get_global_size(0);
	const int Ni = get_global_size(1);
	const int Nm = get_global_size(2);

	data[f*Ni*Nm + i*Nm + m] *= w[i];
}

__kernel void RootVolumeDivide(__global tw_Complex *data,__global tw_Float *w)
{
	const int f = get_global_id(0);
	const int i = get_global_id(1);
	const int m = get_global_id(2);
	const int Nf = get_global_size(0);
	const int Ni = get_global_size(1);
	const int Nm = get_global_size(2);

	data[f*Ni*Nm + i*Nm + m] /= w[i];
}

__kernel void FFT_axis1(__global tw_Float *data,tw_Int N1)
{
	// Wrapper designed to be consistent with numpy.fft functions.
	// STRIP PROTOCOL
	// data = [w][r/x][phi/y][2]
	const int i0 = get_global_id(0);
	const int i2 = get_global_id(1);
	const int N2 = get_global_size(1);
	const int offset = 2*(i0*N1*N2 + i2);
	const int s = 2*N2;

	ComplexFFT(&data[offset],&data[offset+1],N1,s,1.0,-1.0);
}

__kernel void FFT_axis2(__global tw_Float *data,tw_Int N2)
{
	// Wrapper designed to be consistent with numpy.fft functions.
	// STRIP PROTOCOL
	// data = [w][r/x][phi/y][2]
	const int i0 = get_global_id(0);
	const int i1 = get_global_id(1);
	const int N1 = get_global_size(1);
	const int offset = 2*(i0*N1*N2 + i1*N2);
	const int s = 2;

	ComplexFFT(&data[offset],&data[offset+1],N2,s,1.0,-1.0);
}

__kernel void IFFT_axis1(__global tw_Float *data,tw_Int N1)
{
	// Wrapper designed to be consistent with numpy.fft functions.
	// STRIP PROTOCOL
	// data = [w][r/x][phi/y][2]
	const int i0 = get_global_id(0);
	const int i2 = get_global_id(1);
	const int N2 = get_global_size(1);
	const int offset = 2*(i0*N1*N2 + i2);
	const int s = 2*N2;

	ComplexFFT(&data[offset],&data[offset+1],N1,s,-1.0,-1.0);
}

__kernel void IFFT_axis2(__global tw_Float *data,tw_Int N2)
{
	// Wrapper designed to be consistent with numpy.fft functions.
	// STRIP PROTOCOL
	// data = [w][r/x][phi/y][2]
	const int i0 = get_global_id(0);
	const int i1 = get_global_id(1);
	const int N1 = get_global_size(1);
	const int offset = 2*(i0*N1*N2 + i1*N2);
	const int s = 2;

	ComplexFFT(&data[offset],&data[offset+1],N2,s,-1.0,-1.0);
}

__kernel void RadialTransform(__global tw_Float *T,__global tw_Complex *vin,__global tw_Complex *vout)
{
	// Each work item is computing one element of vout
	// Equivalent of vout = numpy.einsum('ijm,fjm->fim',T,vin)
	// Assumes ij is a square matrix
	const int f = get_global_id(0);
	const int i = get_global_id(1);
	const int m = get_global_id(2);
	const int Nf = get_global_size(0);
	const int Ni = get_global_size(1);
	const int Nm = get_global_size(2);
	tw_Complex ans = (tw_Complex)(0.0,0.0);

	for (int j=0;j<Ni;j++)
		ans += T[i*Ni*Nm + j*Nm + m]*vin[f*Ni*Nm + j*Nm + m];
	vout[f*Ni*Nm + i*Nm + m] = ans;
}

__kernel void InverseRadialTransform(__global tw_Float *T,__global tw_Complex *vin,__global tw_Complex *vout)
{
	// Each work item is computing one element of vout
	// Equivalent of vout = numpy.einsum('jim,fjm->fim',T,vin)
	// Assumes ij is a square matrix
	const int f = get_global_id(0);
	const int i = get_global_id(1);
	const int m = get_global_id(2);
	const int Nf = get_global_size(0);
	const int Ni = get_global_size(1);
	const int Nm = get_global_size(2);
	tw_Complex ans = (tw_Complex)(0.0,0.0);

	for (int j=0;j<Ni;j++)
		ans += T[j*Ni*Nm + i*Nm + m]*vin[f*Ni*Nm + j*Nm + m];
	vout[f*Ni*Nm + i*Nm + m] = ans;
}
