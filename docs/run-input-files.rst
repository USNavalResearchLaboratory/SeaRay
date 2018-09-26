Input Files
===========

SeaRay Example Files
--------------------

You will learn the most by studying the examples.  They can be found in :samp:`{raysroot}/examples`.

General Principles
-------------------

The SeaRay input file is strictly speaking, a Python module.  However you do not need to know much Python to create one, and you need practically no knowledge of Python to edit one.

The input file can be simple or complicated.  In its simplest form, it is a set of lists of Python dictionaries with hard coded values.  More sophisticated input files can be created by making the dictionary values variables.  The full power of the Python language can be employed to compute these variables inside the input file, if desired.

The reason for the lists is that the input file can describe a set of simulations.  The lists are lists of simulations.  The five lists that must appear in the input file are as follows.

	.. glossary::

		``sim``
			Contains dictionaries describing general parameters of the simulations, such as units.

		``wave``
			Contains dictionaries describing the initial electromagnetic wave configurations

		``ray``
			Contains dictionaries describing how the initial rays should be loaded

		``optics``
			Contains another list, containing dictionaries describing optical elements

		``diagnostics``
			Contains dictionaries describing how to write out the data

	.. tip::

		A useful perspective is that SeaRay ignores the whole input file except for the five lists named above.  It makes no difference how the lists are created within.  Best practice in post-processing is also to look only at the five lists.

The first elements of ``sim``, ``wave``, ``ray``, ``optics``, and ``diagnostics`` describe the first simulation, the second elements describe the second simulation, etc..  It is possible to set up sophisticated batch jobs using this system.

Minimum Python
--------------

The two Python concepts you need are that of a list and a dictionary.

A Python list is a set of elements enclosed in square brackets, separated by commas.  For example, a list of integers is create like this::

	my_integers = [0,1,2]

The elements of the list can be more complicated things, including other lists, or dictionaries.

A Python dictionary is a set of key-value pairs.  For example::

	favorites = { 'favorite color' : 'red' ,
	              'favorite integer' : 0 ,
	              'favorite floating point number' : 6.1 }

creates a dictionary of my favorites.  The keys are the strings on the left of the colon, the values are the elements on the right, which can be any Python object.  In Python everything is an object, including simple types like strings, integers, and floating point numbers.

Geometry
---------------

The simulation defines a global three dimensional Cartesian coordinate system.  Each object also has its own coordinate system, called the local system.  Rays are initially loaded into the global space.  Internally they are always represented by Cartesian vectors, although the initial configuration can be chosen to have n-fold radial symmetry.  As the rays enter or exit various optical elements, they are generally transformed into and out of the local system of the object.

The euler angles :math:`(\alpha,\beta,\gamma)` are used to set the orientation of an object.  In the active view of the tranformation, the object is first rotated about the z axis by :math:`\gamma`, then about the x axis by :math:`\beta`, and finally about the z axis again by :math:`\alpha`.  These are all right handed rotations.
Most optical elements have their default orientation such that the z axis is coincident with the center ray of a beam perfectly aligned with the object.  Typically it is most convenient, therefore, to orient the optical system to lie in z-y plane, so that the single euler angle :math:`\beta` can be used to direct the beam anywhere in the plane.
When visualizing the z-y plane, it is most convenient to picture positive z as toward the top of the screen, positive y as toward the right of the screen, and positive x as out of the screen.

Four Dimensional Object Properties
----------------------------------

Many input file elements are in the form of four-vectors, represented as Python tuples, e.g., ``(t,x,y,z)``.  This examples represents the typical arrangement, where time or frequency is the first element, and spatial coordinates are the next three elements.  This pattern is used for specifying time + position, energy + momentum, frequency grid + spatial grid, etc..
