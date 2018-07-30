__kernel void transform(	__global double *T,
							__global double2 *vin,
							__global double2 *vout)
{
	// Equivalent of vout = numpy.einsum('ijk,jkl->ikl',T,vin)
	// Assumes ij is a square matrix
	const int i = get_global_id(0);
	const int k = get_global_id(1);
	const int l = get_global_id(2);
	const int Ni = get_global_size(0);
	const int Nk = get_global_size(1);
	const int Nl = get_global_size(2);
	double2 ans = (double2)(0.0,0.0);

	for (int j=0;j<Ni;j++)
		ans += T[i*Ni*Nk + j*Nk + k]*vin[j*Nk*Nl + k*Nl + l];
	vout[i*Nk*Nl + k*Nl + l] = ans;
}

__kernel void divergence(__global double2 *Ax,
							__global double2 *Az,
							__global double2 *div,
							const double dr,
							const double dz)
{
	// Mixed rep. (Cartesian components) cylindrical divergence
	// Assumes linear polarization in x
	// div(A) = cos(phi)dAx/dr - i*mx*sin(phi)*Ax/r +
	//    sin(phi)*dAy/dr + i*my*cos(phi)*Ay/r + dAz/dz
	// Assume evaluation at phi=0
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int zDim = get_global_size(1)+2;
	int n = j + i*zDim;
	div[n] = 0.5*(Az[n+1] - Az[n-1])/dz + 0.5*(Ax[n+zDim] - Ax[n-zDim])/dr;
}

__kernel void laplacian(__global double2 *A,
							__global double2 *del2,
							const double dr,
							const double dz,
							const double m)
{
	// del2(A) = -m^2*A/r2 + dr(A)/r + dr2(A) + dz2(A)
	// assuming A(phi) = A*exp(i*m*phi), and A is any Cartesian component
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int zDim = get_global_size(1)+2;
	const double r = i*dr+0.5*dr;
	int n = j + i*zDim;
	del2[n] = (A[n+1] - 2*A[n] + A[n-1])/(dz*dz);
	del2[n] += (A[n+zDim] - 2*A[n] + A[n-zDim])/(dr*dr);
	del2[n] += 0.5*(A[n+zDim]-A[n-zDim])/(r*dr);
	del2[n] -= m*m*A[n]/(r*r);
}
