SeaRay Description
------------------

SeaRay is a beam propagation code written in Python.  A SeaRay simulation is a set of physical regions with a propagation structure, i.e., a propagation equation which holds in that region.  The glue that connects these regions is ray tracing.  The propagation law that holds in a region is not fundamentally limited by the code structure, e.g., it can be paraxial, eikonal, full Maxwell, etc..
SeaRay uses PyOpenCL to take advantage of parallel hardware such as multi-core, many-core, and GPU devices.  You do not need to compile SeaRay, but there are a number of Python packages that have to be installed.  The top level packages are

#. :samp:`numpy`---Numerical python (fast manipulation of multi-dimensional arrays)
#. :samp:`scipy`---Numerical algorithms (often ports of legacy FORTRAN libraries)
#. :samp:`matplotlib`---Plotting package, used in the SeaRay plotter
#. :samp:`mayavi`---An optional package for advanced 3D plotting
#. :samp:`pyopencl`---Hardware acceleration for Python
