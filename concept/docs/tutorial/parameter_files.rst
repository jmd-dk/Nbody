Working with parameter files
----------------------------
Specifying the many parameters to the ``concept`` script via the ``-c`` option
quickly becomes tiresome. A better solution is to write all parameters in a
text file, say ``params/tutorial`` (the ``params`` directory already exists).
Copy the content below to such a file:

.. code-block:: python3
   :caption: params/tutorial
   :name: params-parameter-files

   # Non-parameter helper variable used to control the size of the simulation
   _size = 64

   # Input/output
   initial_conditions = {
       'species': 'matter',
       'N'      : _size**3,
   }
   output_dirs = {
       'powerspec': paths['output_dir'] + '/' + basename(paths['params'])
   }
   output_times = {
       'powerspec': [a_begin, 0.3, 1],
   }

   # Numerical parameters
   boxsize = 4*_size*Mpc/h
   potential_options = 2*_size

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.02

To run CO\ *N*\ CEPT using this parameter file, do

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -n 4

The simulation specified by the above parameters is quite similar to the
simulations of the previous section, though we've now been more explicit about
the cosmology and the size of the simulation box, and also refined the output
specifications. In doing so, we've made use of several helpful tricks:

* The comments / section headings are there for organisational purposes only.
  These may be omitted, and the order in which the parameters are specified
  does not matter.

* Parameter files are glorified Python scripts, and so we may utilize the full
  Python (3) language when defining the parameters. We can also define
  helper variables that are not themselves parameters, like ``_size`` above,
  which here is used to simultaneously adjust the number of particles ``'N'``,
  the side length of the simulation box ``boxsize`` and the size of the
  potential grid. Though not strictly necessary, it is preferable for such
  helper variables to be named with a leading underscore ``_``, to separate
  them from actual parameters.

* We have explicitly specified the directory for power spectra output in the
  ``output_dirs`` parameter. The value is constructed using the ``paths``
  variable, which holds absolute paths specified in the ``.paths`` file.
  If you want to take a look at this ``.paths`` file, do e.g.

  .. code-block:: bash

     cat ../.paths

  Writing ``paths['output_dir']`` then maps to whatever value is specified for
  ``output_dir`` in the ``.paths`` file, which will be
  ``'/path/to/concept_installation/concept/output'``. Looking up
  ``paths['params']`` is somewhat different, as ``params`` is not in the
  ``.paths`` file. Instead, this dynamically maps to the full path of the
  parameter file itself. To get the file name only (i.e. ``'tutorial'``),
  we use the ``basename`` function. In total, the power spectrum output
  directory gets set to
  ``'/path/to/concept_installation/concept/output/tutorial'``. We could have
  gotten away with just writing ``'output/tutorial'`` out statically. When
  specifying relative paths, these are always with respect to the
  ``concept`` directory.

* We've specified multiple times at which to dump power spectra in
  ``output_times``. Note that we use the parameter ``a_begin`` (creating a
  power spectrum of the initial conditions), the value of which is set further
  down. Generally, the order of variable definitions --- even when one depends
  on another --- is of no importance inside parameter files.

* The simulation takes place in a cubic box, the side length of which is set
  by ``boxsize``. In the parameter specification above, we've let ``boxsize``
  be proportional to ``_size``, which has the effect of keeping the particle
  density constant when varying ``_size``: If e.g. ``_size`` is increased by
  a factor of 2, ``boxsize`` increases by a factor of 2 as well, meaning that
  the simulation volume goes up by a factor of 8, which is compensated by a
  similar increase in the number of particles by a factor :math:`2^3 = 8`.

  No default units are ever assumed by CO\ *N*\ CEPT when it reads parameters,
  and so it's critical that you explicitly tack on units on all dimensional
  parameters. For ``boxsize``, the extra fancy unit of :math:`\text{Mpc}/h` is
  used above, with
  :math:`h \equiv H_0/(100\, \text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1})`
  inferred dynamically from the Hubble constant ``H0`` set further down.

* The parameters ``Ωb`` and ``Ωcdm`` of course set the present amount of
  baryons and cold dark matter, respectively. Together, these otherwise
  distinct species are collectively referred to as just *matter*. Thus,
  declaring the species to be ``'matter'`` in the ``initial_conditions``
  implies that the particles will represent both the baryonic
  and cold dark matter.

The ``-p`` (parameter file) option can be mixed with the ``-c`` (command-line
parameter) option. As an example, consider leaving out the definition of
``_size`` from the parameter file and instead supplying it when running the
code:

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_size = 64" \
       -n 4

If you forget to specify ``_size`` --- or any other variable referenced by the
parameter file --- CO\ *N*\ CEPT will exit with an error, letting you know.



.. raw:: html

   <h3>Log files</h3>

The printed output of all CO\ *N*\ CEPT runs gets logged in the ``logs``
directory. Each run (or *job*) gets a unique integer ID, which is also used as
the file name for the logged output. The log file name of any CO\ *N*\ CEPT
run is stated when the program starts, and the job ID is written again at the
end. Also, the job ID is included in the header of the power spectrum data
files.

.. tip::
   To view the logged output of e.g. CO\ *N*\ CEPT run 1 with proper
   coloring, use ``less -rf logs/1``. Arrow keys to navigate, ``q`` to quit.



.. raw:: html

   <h3>Checking previously used parameters</h3>

Among the first lines of output of any CO\ *N*\ CEPT run is the path to the
parameter file in use. As this is included in the log file, you can always go
back and check the parameters used by a given run. However, this information
isn't reliable, as we may have modified the parameter file since its original
use. To this end, a complete copy of the parameter file is made upon every
invocation of CO\ *N*\ CEPT, and stored in the ``params`` directory. The file
name of this copy is written together with the name of the original parameter
file when CO\ *N*\ CEPT starts. The name is generated from the current time
and has the format ``.YYYYMMDDhhmmssSSS`` (year, month, day, hour, minute,
second, millisecond).

When mixing ``-p`` and ``-c``, the combined parameters are what's stored in the
copied parameter file.

As an exercise, get the job ID of the latest simulation from the header of one
of the power spectrum data files (e.g. ``output/tutorial/powerspec_a=1.00``),
find the file name of the parameter file copy from the corresponding log file,
and check that the parameters specified are as you expect. Any command-line
parameters will be placed at the bottom.

So far we've introduced only the most essential parameters. The remaining
sections of this tutorial will introduce further parameters --- and expand on
already encountered ones --- as needed. For full documentation on all
available parameters, consult :doc:`Parameters </parameters/parameters>`.

