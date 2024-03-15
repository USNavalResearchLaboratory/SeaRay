Changes for v1.0
================


Breaking Changes
----------------

* The project has been factored.  The following are in sub-directories.
    - OpenCL programs (``kernels``)
    - Python modules (``modules``)
    - explorer notebooks (``explore``)
* The nonlinear susceptibility is scaled with density. The value that is entered into the volume is the value at the reference density.  Linear susceptibility has always been handled this way.
* The nonlinear filter works differently. The filter applies to the current density after all nonlinear contributions have been added.  Hence the filter will generally be extended to higher frequencies, e.g., to keep everything up to fourth harmonic, assuming usual normalization, you would have ``(0.0,4.0)``.

New Features
------------

* Multi-level density matrix treatment of molecular rotations

Improved Behaviors
------------------

* Conflicts between ``matplotlib`` and ``maya`` are better handled when using the plotter.