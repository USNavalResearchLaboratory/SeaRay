Input Files
===========

SeaRay Example Files
--------------------

You will learn the most by studying the examples.  They can be found in :samp:`{raysroot}/examples`.

General Principles
-------------------

The SeaRay input file is strictly speaking, a Python module.  However you do not need to know much Python to create one, and you need practically no knowledge of Python to edit one.

The input file can be simple or complicated.  In its simplest form, it is a set of Python lists and dictionaries with hard coded values.  More sophisticated input files can be created by making the dictionary values variables.  The full power of the Python language can be employed to compute these variables inside the input file, if desired.

The input file is required to have the following four objects:

.. glossary::

	``sim``
		Dictionary of general parameters of the simulations, such as units.  This object must be serializable.

	``sources``
		List of dictionaries, with each dictionary describing a set of rays and waves.  The waves are used to set the eikonal data carried with the rays.  This object must be serializable.

	``optics``
		List of dictionaries, with each dictionary describing an optical element, detection surface, or wave propagation region.  This object does *not* need to be serializable.

	``diagnostics``
		Dictionary of general parameters pertaining to how diagnostics are written.  This object must be serializable.

.. tip::

	A useful perspective is that SeaRay ignores the whole input file except for the four objects named above.  It makes no difference how the objects are created within.  Best practice in post-processing is also to look only at the four objects, some of which are serialized and saved with the output data.

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

There is another kind of list called a tuple.  Tuples are denoted using parenthesis::

	my_vector = (1.0,0.0,0.0)

A tuple is basically a list that cannot be modified after it is created.

In Python there is a special value called ``None``.  SeaRay sometimes requires a ``None`` value rather than simply omitting it or making it zero.

Geometry
---------------

The simulation defines a global three dimensional Cartesian coordinate system.  Each object also has its own coordinate system, called the local system.  Rays are initially loaded into the global space.  Internally they are always represented by Cartesian vectors, although the initial configuration can be chosen to have n-fold radial symmetry.  As the rays enter or exit various optical elements, they are generally transformed into and out of the local system of the object.

The euler angles :math:`(\alpha,\beta,\gamma)` are used to set the orientation of an object.  In the active view of the transformation, the object is first rotated about the z axis by :math:`\gamma`, then about the x axis by :math:`\beta`, and finally about the z axis again by :math:`\alpha`.  These are all right handed rotations.

Most optical elements have their default orientation such that the z axis is coincident with the center ray of a beam perfectly aligned with the object. Objects with transverse asymmetry are typically oriented such that beams are deflected in the xz plane.

The position of an object is given in the input file as a global Cartesian coordinate.  Each object has a reference point, typically the centroid of the object, or some other signficant point, which coincides with the global coordinate of the object.

When positioning and orienting an object, the rotation about the reference point happens first, then the translation to the global position.

Helper Class
,,,,,,,,,,,,

There is a helper class that can be imported into an input file to assist with positioning and orienting objects, among other things.

Four Dimensional Object Properties
----------------------------------

Many input file elements are in the form of four-vectors, represented as Python tuples, e.g., ``(t,x,y,z)``.  This example represents the typical arrangement, where time or frequency is the first element, and spatial coordinates are the next three elements.  This pattern is used for specifying time + position, energy + momentum, frequency grid + spatial grid, etc..  When some component of a four-tuple is not needed, it is set to ``None``.  For example, a point on a ray surface would be given as ``(w,x,y,None)``.
