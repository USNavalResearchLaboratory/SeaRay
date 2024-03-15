How To Guide
============

Multiple Pulses
---------------

As of this writing there is no way to create multiple pulses at a single frequency from
the input file.  This is a limitation of the wave object.

You can, however, create an arbitrary number of pulses at various frequencies.
To do this simply create multiple wave objects at distinct frequencies.
Rays of a given frequency will have their initial condition affected only
by the wave that has an appreciable amplitude at that frequency.

There is still only one ray box, the rays are created and launched consistent with the
given set of waves.

Delay Lines
-----------

You can delay a pulse at one frequency with respect to a pulse at a different frequency with the `Filter` object,
remembering that in frequency space a delay takes the form of multiplication by

:math:`\exp(i\omega \Delta t)`

For example, to delay a pulse at frequency ``wprobe`` with respect to a pulse at a lower frequency ``w00`` you might have::

    optics.append({})
    optics[-1]['object'] = surface.Filter('delay')
    optics[-1]['origin'] = (0.0,0.0,dnum('-1 mm'))
    optics[-1]['radius'] = dnum('1 cm')
    optics[-1]['transfer function'] = lambda w: np.exp(1j*w*np.heaviside(w-0.5*(wprobe+w00),0.5)*dnum('1 ps'))

Here we use a simple ``lambda``, but in general there is no limit to the complexity of the transfer function.