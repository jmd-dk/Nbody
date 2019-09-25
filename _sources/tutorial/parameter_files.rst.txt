Working with parameter files
----------------------------
Specifying the many parameters to the ``concept`` script via the ``-c`` option
quickly becomes tiresome. A better solution is to write all parameters in a
file, say ``params/tutorial`` (the ``params`` directory already exists).
Copy the content below to such a file:

.. code-block:: python3

   # Non-parameter variable used to control the size of the simulation
   _size = 64

   # Input/output
   initial_conditions = {
       'name'   : 'matter component',
       'species': 'matter particles',
       'N'      : _size**3,
   }
   output_dirs = {
       'powerspec': paths['output_dir'] + '/' + basename(paths['params'])
   }
   output_times = {
       'powerspec': [a_begin, 0.3, 1],
   }

   # Numerical parameters
   boxsize = 256*Mpc/h
   φ_gridsize = 2*_size

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.02

   # Physics
   select_forces = {'all': {'gravity': 'p3m'}}

To run CO\ *N*\ CEPT using these parameters, do

.. code-block:: bash

   ./concept -p params/tutorial -n 4

The simulation specified by the above parameters is quite similar to the
simulations of the previous section, though we've now been more explicit about
the cosmology and also refined the output specifications. Doing so, we've made
use of several helpful tricks:

* The comments / section headings are there for organisational purposes only.
  These may be omitted, and the order in which the parameters are specified
  does not matter.

* Parameter files are glorified Python scripts, and so we may utilize the full
  Python (3) language when defining the parameters. We can also define
  helper variables that are *not* parameters, like ``_size`` above, from which
  we can simultaneously adjust ``'N'`` and ``φ_gridsize``.

  .. caution::
     Though not stricly necessary, the leading underscore in ``_size`` signals
     to CO\ *N*\ CEPT that this is not an actual parameter. If we were to use
     just ``size`` (or any other unrecognized parameter name without a leading
     underscore), a warning would be emitted.

* We have explicitly specified the directory for power spectra output
  (``output_dirs``). The value is constructed using the ``paths`` variable,
  which holds absolute paths specified in the ``.paths`` file.
  If you want to take a look at this ``.paths`` file, use e.g.

  .. code-block:: bash

     less ../.paths

  Looking up ``paths['params']`` is somewhat different, as ``'params'`` is not
  in the ``.paths`` file. Instead, this dynamically maps to the full path of
  the parameter file itself. To get the filename only (i.e. ``'tutorial'``),
  we use the ``basename`` function. In total, the power spectrum output
  directory is set to
  ``'/path/to/concept_installation/concept/output/tutorial'``. We could have
  gotten away with just writing ``'output/tutorial'`` out statically. When
  specifying relative paths, these are always with respect to the
  ``concept`` directory.

* We've specified multiple times at which to dump power spectra
  (``output_times``). The exact data type used is of no importance, i.e.
  using ``()`` or ``{}`` rather than ``[]`` is fine. Also, note that we use
  the parameter ``a_begin`` (creating a power spectrum of the initial
  conditions), the value of which isn't set before furhter down. Generally,
  the order of variable definitions inside parameter files is of no importance.

* The simulation takes place in a cubic box, the side length of which is set
  by ``boxsize``. No default set of units are ever assumed by CO\ *N*\ CEPT,
  and so it's critical that you explicitly tack on units on all parameters that
  are not unitless. For the ``boxsize``, the extra fancy unit of
  :math:`\mathrm{Mpc}/h` is used, with
  :math:`h \equiv H_0/(100\,\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1})`
  determined dynamically from the Hubble constant ``H0`` set further down.

* The parameters ``Ωb`` and ``Ωcdm`` of course set the amount of baryons and
  cold dark matter, respectively. Together, these otherwise distinct species
  are collectively referred to as just *matter*. Thus, declaring the species
  to be ``'matter particles'`` in the ``initial_conditions`` implies that
  these particles will represent both the cold dark and the bayonic matter.
  Though rarely preferred, we *can* choose to treat the two matter species as
  seperate components;

  .. code-block:: Python3

     initial_conditions = [
         {
             'name'   : 'dark matter component',
             'species': 'dark matter particles',
             'N'      : _size**3,
         },
         {
             'name'   : 'baryonic component',
             'species': 'baryons',
             'N'      : _size**3,
         },
     ]

  doubling the total number of particles.

  In such a case we of course would like gravity to apply to both components,
  and so both component names need to be registered with ``select_forces``.
  A shortcut is used in the paramater file above, where we simply declare that
  *all* components interact under gravity, and uses the same method
  (here ``'p3m'``). Note that you're free to mix and match, so that e.g. the
  ``'baryon component'`` uses ``'pm'`` while the ``'dark matter component'``
  uses ``'p3m'``.

  .. note::
     While you have completely free choice in picking the value to supply to
     the ``'name'`` field when specifying components in ``initial_conditions``,
     the legal values to supply to ``'species'`` are particular. For *particle
     components*, the difference between species lie purely in their initial
     conditions, while *fluid components* (e.g. neutrinos) may have distinct
     traits like specific time-varying equation of state. We shall delve into
     fluid components later, when discussing
     :doc:`beyond matter-only simulations<beyond_matter_only>`.


The ``-p`` parameter file option can be mixed with the ``-c`` command line
parameter option. As an example, consider leaving out ``_size`` from the
parameter file, and instead supplying it when running the code:

.. code-block:: bash

   ./concept -p params/tutorial -c '_size = 64' -n 4

If you forget to specify ``_size`` --- or any other variable used in the
parameter file --- CO\ *N*\ CEPT will exit with an error, letting you know.


.. topic:: Checking previously used parameters

   Among the first lines of output of any CO\ *N*\ CEPT run is the path to the
   parameter file in use. As this is included in the log file, you can always go
   back and check the parameters used by a given run. However, this information
   isn't reliable, as we may have modified the parameter file since its original
   use. To this end, a complete copy of the parameter file is made upon every
   invocation of CO\ *N*\ CEPT, and stored in the ``params`` directory. The
   name of this copy is written together with the original parameter file when
   CO\ *N*\ CEPT starts.

   When mixing ``-p`` and ``-c``, the combined parameters are what's stored in
   the copied parameter file.


So far we've introduced only the most essential parameters. The remaining
sections of this tutorial will introduce more as needed, but the full parameter
scope will not be shown. If you're on the lookout for something particular or
just curious about what's possible, study the ``params/example_params``
parameter file.

