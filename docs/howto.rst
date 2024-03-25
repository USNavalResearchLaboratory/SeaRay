How To Guide
============

Keep in Mind
------------

Rays represent solutions of the eikonal wave equation at a particular frequency.
Superpositions are used to build complex waveforms.

Rays are always initialized on a single surface. The amplitude and direction of the rays are determined by
a superposition of wave modes.  The phase can only be adjusted using downstream optics (see below).  Equivalently,
all pulses are synchronized upon initialization.

Generally, the initial set of waves should have no overlap in frequency space.
If they do, the superposition field needs to be eikonal.

Coupling to Paraxial Regions
----------------------------

Ray bundles can be coupled into volumes with paraxial wave propagators.

* set the volume's ``propagator`` to ``paraxial``
* the ray bundle must have a lower frequency bound greater than zero
* the rays should be paraxial
* frequency node count should be a power of 2

Coupling to Unidirectional Regions
----------------------------------

Ray bundles can be coupled into volumes with UPPE wave propagators.

* set the volume's ``propagator`` to ``uppe``
* the ray bundle must have a lower frequency bound of *exactly zero*
* frequency node count should be a power of 2 *plus one*

Multicolor Pulses
-----------------

To create multiple pulses at *distinct* frequencies, create multiple wave objects at distinct frequencies.
Rays of a given frequency will have their initial condition affected only
by the wave that has an appreciable amplitude at that frequency.

Delay Lines
-----------

The ``Filter`` object can be used to produce any waveform in principle.
An important filter is the delay operator

:math:`\exp(i\omega \Delta t)`

For example, to delay all pulses by one picosecond::

    optics.append({})
    optics[-1]['object'] = surface.Filter('delay')
    optics[-1]['origin'] = (0.0,0.0,0.0)
    optics[-1]['radius'] = dnum('1 cm')
    optics[-1]['transfer function'] = lambda w: np.exp(1j*w*dnum('1 ps'))

Here we use a simple ``lambda``, but in general there is no limit to the complexity of the transfer function.
One caveat is the transfer function must be able to handle array inputs.

Double Pulse
------------

To create a double pulse out of a single pulse, use a superposition of the identity and delay operators:

:math:`2^{-1/2}(1 + \exp(i\omega \Delta t))`

For the python transfer function::

    optics[-1]['transfer function'] = lambda w: (1 + np.exp(1j*w*dnum('1 ps'))) / np.sqrt(2)

Delayed Multicolor Pulses
-------------------------

To delay one color with respect to another, the delay operator has to be multiplied by a frequency selection
window.  For example::

    def delayColor(w,w1,w2):
        rect = np.heaviside(w-w1,0.5) * np.heaviside(w2-w,0.5)
        return 1 + rect * (np.exp(1j*w*dnum('1 ps'))-1)
    
    optics[-1]['transfer function'] = lambda w: delayColor(w,1.9,2.1)
