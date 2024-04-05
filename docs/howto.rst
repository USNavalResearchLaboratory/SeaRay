How To Guide
============

Keep in Mind
------------

Rays represent solutions of the eikonal wave equation at a particular frequency.
The eikonal fields on a surface act as a boundary condition for wave propagation.
The eikonal condition places limits on surface curvature and tangential derivatives,
but not on pulse format.

Initializing Eikonal Waves
--------------------------

In experiments with lasers, one almost always has an eikonal wave to start with.
Wave regions are usually the outcome of an eikonal optical system.
This motivates eikonal waves as a boundary condition.

Rays are initialized on a single surface. Initial amplitudes are determined by
a set of wave modes. Each mode should occupy a distinct frequency band.
More complex superpositions are created using downstream optics.

Coupling to Paraxial Regions
----------------------------

Ray bundles can be coupled into volumes with paraxial wave propagators. SeaRay will work out the waveform associated with the rays.

* set the volume's ``propagator`` to ``paraxial``
* make sure incoming wave is x-polarized
* the ray bundle must have a lower frequency bound greater than zero
* the rays should be paraxial
* frequency node count should be a power of 2

Coupling to Unidirectional Regions
----------------------------------

Ray bundles can be coupled into volumes with UPPE wave propagators. SeaRay will work out the waveform associated with the rays.

* set the volume's ``propagator`` to ``uppe``
* make sure incoming wave is x-polarized
* the ray bundle must have a lower frequency bound of *exactly zero*
* frequency node count should be a power of 2 *plus one*

Full Wave Simulations
---------------------

If you need to use a full wave description from start to finish, you can setup the ``incoming wave`` function in any propagation volume.
If this key exists, the incoming ray data will be ignored, and the user's own callback function will be used to fill the required array.
The callback function takes the mesh nodes as arguments and returns an array, e.g.::

    def incoming_gaussian(w_nodes,x_nodes,y_nodes):
        f = lambda x,x0,dx: np.exp(-(x-x0)**2/dx**2)
        return np.einsum('i,j,k',f(w_nodes,1,0.1),f(x_nodes,0,10),f(y_nodes,0,10))

This function becomes the value of the ``incoming wave`` key::

    optics[-1]['incoming wave'] = incoming_gaussian

If you want to make the callback depend on parameters of your own, use lambda, e.g., to have an adjustable delay::

    def incoming_wave(w,x,y,delay):
        # some code to compute A
        return A * np.exp(1j*w*delay)[:,np.newaxis,np.newaxis]

    optics[-1]['incoming wave'] = lambda w,x,y: incoming_wave(w,x,y,1.0)

Multicolor Pulses
-----------------

To create multiple pulses at *distinct* frequencies, create multiple wave objects at distinct frequencies.
Rays at a point (w,x,y) will have their initial condition affected only
by the wave that has the highest amplitude at that point.

Delay Lines
-----------

The ``Filter`` object can be used to produce any waveform in principle.
An important filter is the delay operator

:math:`\exp(i\omega \Delta t)`

For example, to delay all pulses by one picosecond::

    optics.append({})
    optics[-1]['object'] = surface.Filter('delay')
    optics[-1]['origin'] = (None,0,0)
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

Setup a Batch Job
-----------------

One way to setup a batch job is to write a script to import ``rays`` and then iteratively call ``rays.run``.
The arguments to ``rays.run`` can be retrieved by importing any SeaRay input file.
To vary parameters, you can adjust the input file objects after importing them.
N.b. in general you will need to reload the input file after each run using ``importlib.reload``.
This is because, as of this writing, there are no guarantees that objects from the input file are not
modified during a simulation.