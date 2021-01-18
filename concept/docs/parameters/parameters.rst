Parameters
==========
The following sections list all the various parameters accepted by
CO\ *N*\ CEPT, along with explanations, default and example values. Each
section deals with a specific category of parameters.

To learn how to *use* parameters and parameter files, see the
:doc:`tutorial </tutorial/parameter_files>`.

Parameters are specified as live Python 3 variables. As such, it is helpful to
be familiar with basic Python syntax and knowledge of data types, such as
``str``\ ings, ``list``\ s and ``dict``\ ionaries.

Below you will find the parameter categories, corresponding to the sections
ahead. Besides bringing some organization to the large set of parameters,
these categories have no meaning.

.. toctree::
   :maxdepth: 1

   input_output
   numerical_parameters
   cosmology
   physics
   simulation_options
   graphics
   system_of_units
   debugging_options

The rest of this page contains helpful information regarding parameter
specification generally, independent of the specific parameters.



.. raw:: html

   <h3>Units</h3>

For dimensional parameters, it is important to explicitly tack the unit onto
the value, as no default units are ever assumed by CO\ *N*\ CEPT. For example,
you may define the ``boxsize`` as

.. code-block:: python3

   boxsize = 512*Mpc

but you should never leave out the unit,

.. code-block:: python3

   boxsize = 512  # Wrong!

A vast set of units are understood, like ``km``, ``pc``, ``kpc``, ``Mpc``,
``Gpc``, ``s``, ``yr``, to name a few.

.. note::
   The actual numerical value resulting from e.g. ``512*Mpc`` depends on the
   internal unit system employed by CO\ *N*\ CEPT. The unit system itself is
   not statically defined, but specified on a per-simulation basis via
   :doc:`parameters <system_of_units>`. Though the vast majority of users
   have no need to manually specify the internal unit system, it is useful
   for e.g. testing against other simulation codes that use some other,
   fixed system of units.



.. raw:: html

   <h3>Parameter files as Python scripts</h3>

You may think of parameter files as Python scripts, and use any valid Python 3
statements to formulate or compute your parameters, including ``print()`` for
debugging. Even without any explicit ``import`` statement present in the
parameter file, all of ``numpy`` is readily available. As an example, consider
the following specification of 7 power spectra output placed logarithmically
equidistant in scale factor :math:`a`, from :math:`a = 0.01` to :math:`a = 1`:

.. code-block:: python3

   output_times = {'powerspec': logspace(log10(0.01), log10(1), 7)}

Also readily available for use in parameter files is the special ``paths``
variable. This is a ``dict`` holding the absolute paths to various useful
directories and files. In fact, the content of the ``paths`` ``dict`` is
supplied by the ``.paths`` file. In addition to all of these paths, the
``paths`` ``dict`` also store the absolute path to the parameter file in which
it is used. A nice way to ensure that the output is dumped into a proper but
unique location is then to use

.. code-block:: python3

   output_dirs = {'powerspec': paths['output_dir'] + '/' + basename(paths['params'])}

which directs CO\ *N*\ CEPT to dump power spectra in a subdirectory with the
same name as the parameter file, within the dedicated ``output`` directory as
specified in the ``.paths`` file. Here, ``basename`` is really
``os.path.basename``, which again is made available without the need for an
explicit ``import``.



.. raw:: html

   <h3>Parameter files as <em>glorified</em> Python scripts</h3>

The parameter files of CO\ *N*\ CEPT are in fact even more powerful than
regular Python scripts. The additional super powers are described below.



.. raw:: html

   <h4>Non-linear parsing of parameter file content</h4>

Parameters may be defined in terms of each other. Unlike regular variables,
the definition order does not matter (i.e. you may reference a variable
*before* it is defined). The magic that makes this work is cyclic execution of
the parameter file, so that a variable that is needed on some line but only
defined by some later line, is correctly set at the second execution. This
non-linear variable dependence may continue arbitrarily, requiring many more
execution passes.



.. raw:: html

   <h4>Non-parameter variables</h4>

Using additional, non-parameter variables in parameter files can help provide
better organisation. For example, you may want to define both the ``boxsize``
and the resolution of the gravitational potential grid in terms of a common
variable:

.. code-block:: python3

   # Conveniently placed at the top
   _size = 256

   # Numerical parameters
   boxsize = _size*Mpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'p3m': 2*_size,
           },
       },
   }

Both the ``boxsize`` and the P³M grid size within ``potential_options`` can
now be simultaneously updated through the newly introduced ``_size`` variable,
which itself is not a parameter. When defined in a parameter file,
CO\ *N*\ CEPT treats any variable whose name does not begin with an underscore
'``_``' as a parameter; hence '``_size``' and not '``size``'.



.. raw:: html

   <h4>Dynamic value insertion using ellipses</h4>

Several of the parameters are ``dict``'s with which one often want the
values to be identical for multiple keys. Instead of typing out the same
value multiple times, this may be inferred dynamically from the placement
of ellipses '``...``' like so:

.. code-block:: python3

   output_dirs = {
       'snapshot' : paths['output_dir'] + '/' + basename(paths['params']),
       'powerspec': ...,
       'render2D' : ...,
       'render3D' : ...,
   }

Here, the keys ``'powerspec'``, ``'render2D'`` and ``'render23'`` will map
to the same value as ``'snapshot'``. More generally, an ellipsis is replaced
by the first non-ellipsis value encountered when looking back up the key-value
definitions, wrapping around if necessary. Thus,

.. code-block:: python3

   output_times = {
       'snapshot' : ...,
       'powerspec': 1,
       'render2D' : ...,
       'render3D' : [0.1, 0.3, 1],
   }

results in ``'snapshot'`` being mapped to ``[0.1, 0.3, 1]`` and ``'render2D'``
to ``1``.



.. raw:: html

   <h4>Inferred parameters</h4>

Finally, some parameters are inferred from other parameters. The value of
these may be used to define other parameters, but they should not be
explicitly specified themselves. Currently, the only two such *inferred*
parameters are ``h`` and ``Ων``:

* ``h`` is simply defined through
  :math:`h \equiv H_0/(100\, \text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1})`,
  with the Hubble constant :math:`H_0` being a normal parameter, ``H0``,
  defined as e.g.

  .. code-block:: python3

     H0 = 67*km/(s*Mpc)

  in which case ``h`` is set equal to ``0.67``. Having ``h`` available is
  useful if you want to use the common unit of :math:`\text{Mpc}/h`, e.g. when
  defining the box size:

  .. code-block:: python3

     boxsize = 256*Mpc/h

* ``Ων`` is the total density parameter :math:`\Omega_\nu` for all *massive*
  neutrino species. It is set based on the massive neutrino parameters defined
  by the ``class_params`` :ref:`parameter <class_params>`. As the computation
  of :math:`\Omega_\nu` is non-trivial, this is nice to have available for
  simulations with massive neutrinos where the sum
  :math:`\Omega_{\text{cdm}} + \Omega_\nu` is constrained. With e.g.
  :math:`\Omega_{\text{cdm}} + \Omega_\nu = 0.27`, one would set

  .. code-block:: python3

     Ωcdm = 0.27 - Ων

in the parameter file. As ``Ων`` unlike ``h`` is non-trivial to compute from
existing parameters, its value will be printed at the beginning of the
CO\ *N*\ CEPT run (when running with massive neutrinos).

