Input Files
===========

Random Comments
---------------

The euler angles :math:`(\alpha,\beta,\gamma)` are used to set the orientation of an object.  In the active view of the tranformation, the object is first rotated about the z axis by :math:`\gamma`, then about the x axis by :math:`\beta`, and finally about the z axis again by :math:`\alpha`.  These are all right handed rotations.
Most optical elements have their default orientation such that the z axis is coincident with the center ray of a beam perfectly aligned with the object.  Typically it is most convenient, therefore, to orient the optical system to lie in z-y plane, so that the single euler angle :math:`\beta` can be used to direct the beam anywhere in the plane.
When visualizing the z-y plane, it is most convenient to picture positive z as toward the top of the screen, positive y as toward the right of the screen, and positive x as out of the screen.

SeaRay Example Files
--------------------

You will learn the most by studying the examples.  They can be found in :samp:`{raysroot}/examples`.
