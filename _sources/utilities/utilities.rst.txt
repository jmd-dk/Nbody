Utilities
=========
Besides just *running* simulations, several related features are built into
CO\ *N*\ CEPT, like initial condition generation and power spectrum
computation. While these features are utilized automatically during a
simulation, it is sometimes useful to be able to utilize them on their own,
e.g. to produce a power spectrum from a snapshot file. To this end,
CO\ *N*\ CEPT provides various *utilities* which are able to hook into the
main code and utilize a given feature, without running any simulation.
Also, some utilities exists on their own merit, and introduce features that
is not used during simulation.

The available utilities are as follows:

.. toctree::
   :maxdepth: 2

   class
   convert
   gadget
   info
   play
   powerspec
   render2D
   render3D
   update
   watch



