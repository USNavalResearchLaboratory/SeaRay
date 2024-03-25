Changes for v1.0
================


Breaking Changes (Outward Facing)
---------------------------------

* Input file readers are more strict, but also provide more helpful error messages.
* Outer lists are no longer accepted, for batch jobs import ``rays`` and call the ``run`` method.
* The nonlinear susceptibility is scaled with density. The value that is entered into the volume is the value at the reference density.  Linear susceptibility has always been handled this way.
* The nonlinear filter works differently. The filter applies to the current density after all nonlinear contributions have been added.  Hence the filter will generally be extended to higher frequencies, e.g., to keep everything up to fourth harmonic, assuming usual normalization, you would have ``(0.0,4.0)``.
* Four tuples are used more aggressively in place of three tuples.  The goal is to be more consistent.
* Four tuples should use ``None`` for any unused components.  This creates more opportunities for runtime checks.
    - Ray-like tuples should end with None, e.g., ``(w,x,y,None)``
    - Space-like tuples should start with None, e.g., ``(None,x,y,z)``
    - Bounds should be handled similarly, e.g., ``(None,None,x0,x1,y0,y1,z0,z1)``
* Explorer notebooks are in ``extras/explore``

Breaking Changes (Internal)
---------------------------

* The project tree has changed.  The following are in sub-directories.
    - OpenCL programs (``kernels``)
    - Python modules (``modules``)
* The ray bundle now forms a tetrahedron rather than a hexahedron

New Physics
------------

* Multi-level density matrix treatment of molecular rotations

Improved Behaviors
------------------

* More runtime checks of the input file
* The main program ``rays.py`` works as either a module or a command line program
    - See ``examples/uppe/batch`` for an example that varies parameters across runs
* Conflicts between ``matplotlib`` and ``maya`` are better handled when using the plotter.