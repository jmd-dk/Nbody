Other kinds of output
---------------------
So far, the result of any simulation has been power spectra, though several
other types of output are available, as exemplified by the below parameter
file:

.. code-block:: python3
   :caption: params/tutorial
   :name: params-output-types
   :emphasize-lines: 10, 12-13, 16, 18-19, 21-26, 38-54

   # Non-parameter variable used to control the size of the simulation
   _size = 64

   # Input/output
   initial_conditions = {
       'species': 'matter',
       'N'      : _size**3,
   }
   output_dirs = {
       'snapshot' : paths['output_dir'] + '/' + basename(paths['params']),
       'powerspec': ...,
       'render2D' : ...,
       'render3D' : ...,
   }
   output_times = {
       'snapshot' : 0.1,
       'powerspec': [a_begin, 0.3, 1],
       'render3D' : ...,
       'render2D' : logspace(log10(a_begin), log10(1), 15),
   }
   powerspec_select = {
       'matter': {'data': True, 'linear': True, 'plot': False},
   }
   render2D_select = {
       'matter': {'data': False, 'image': True, 'terminal image': True},
   }

   # Numerical parameters
   boxsize = 128*Mpc
   potential_options = 2*_size

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωcdm    = 0.27
   Ωb      = 0.049
   a_begin = 0.02

   # Graphics
   render2D_options = {
       'gridsize': {
           'matter': _size,
       },
       'terminal resolution': {
           'matter': 80,
       },
       'colormap': {
           'matter': 'inferno',
       },
   }
   render3D_colors = {
       'matter': 'lime',
   }
   render3D_bgcolor    = 'black'
   render3D_resolution = 640

Run a simulation using the :ref:`above <params-output-types>` parameters, e.g.
by saving them to ``params/tutorial`` and executing

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -n 4

This will take a few minutes. You may read along in the meantime.

We see that besides power spectra, we now have *snapshots* and *renders*, the
latter of which comes in a 2D and a 3D version. The ellipses (``...``) used
above in ``output_dirs`` indicate that we want all kinds of output to go to
the same directory.

For the ``output_times``, different values are given for three of the output
types, while ``'render3D'`` is set to use the same times as the output just
above it, i.e. that of ``'powerspec'``. For ``'render2D'``, we've specified 15
outputs spaced logarithmically equidistant between
:math:`a = a_{\text{begin}} = 0.02` and :math:`a = 1`.

Among the new parameters introduced are ``powerspec_select``, in which we have
specified that we only want the data files --- also including the linear theory
spectrum --- as output, not plots of this data.



3D renders
..........
Looking in the output directory, among other things you'll find image files
with names starting with ``render3D``. These are --- unsurprisingly --- the 3D
renders. The colours are controlled through the ``render3D_colors`` and
``render3D_bgcolor`` parameters, while the (square) size (in pixels) is set by
``render3D_resolution``. All particles of a given component gets the same
colour, though different colours may be used for different components when
running such simulations. The brightness of each pixel indicate the local
energy density.

The colours used (here ``'lime'`` and ``'black'``) may be any colour recognised
by `Matplotlib <https://matplotlib.org/>`_. A list of named colours is
available `here <https://matplotlib.org/gallery/color/named_colors.html>`_.
Alternatively, you may pass a 3-tuple of RGB values (e.g.
``render3D_bgcolor = (1, 0, 0)`` makes the background red).



2D renders
..........
The 2D renders show the particle configuration projected along one of the axes
of the box. These can often be prettier than their 3D counterparts, as a
colormap is used to visualise the density field, rather than just a single
colour combined with alpha compositing.

In the ``render2D_select`` parameter we've specified that we want images as
well as terminal images, but no data. Here, *images* refer to the 2D render
image files you see in the output directory. *Terminal images* are rendered
directly in the terminal as part of the printed output, as you have probably
noticed. If you turn on the *data* output, the 2D render data will further be
stored in an HDF5 file.

The options for the 2D renders are collected in the ``render2D_options``
parameter. Here ``gridsize`` sets the resolution of the cubic grid onto which
the particles are interpolated in order to produce the render. The height and
width of the image files (in pixels) are then equal to ``gridsize``. The
terminal image is produced by resizing the interpolation grid to match the
resolution given in ``'terminal resolution'``. The terminal image is then
displayed using one character slot per grid cell, i.e. the terminal render
will be ``'terminal resolution'`` (80) characters wide. Since the terminal
render is constructed from the original 2D render, this does not show more
details even though the resolution is higher (80 vs. 64).

Also available through ``render2D_options`` is the colormap to use. Check out
`this <https://matplotlib.org/gallery/color/colormap_reference.html>`_
for a list of available colormaps.



.. raw:: html

   <h3>The play utility</h3>

For this next trick, the simulation need to have finished, and we need to know
its job ID.

.. tip::
   To grab the job ID from a power spectra data file, you can do e.g.

   .. code-block:: bash

      grep -o "job [^ ]*" output/tutorial/powerspec_a=1.00

With the job ID at hand, try the following:

.. code-block:: bash

   ./concept -u play <ID>  # replace <ID> with job ID number

You should see a nice animation of the evolution of the large-scale structure,
playing out right in the terminal! The animation is produced from the terminal
images stored in the log file ``logs/<ID>``.

The ``-u`` option to the ``concept`` script signals CO\ *N*\ CEPT to start up
a *utility* rather than running a simulation. These utilities are handy (and
sometimes goofy) side programs baked into CO\ *N*\ CEPT. Another such utility
--- the *info utility* --- is encountered just below, and we will encounter
others in later sections of the tutorial. For full documentation on each
available utility, consult :doc:`Utilities </utilities/utilities>`.



Snapshots
.........
Snapshots are raw dumps of the total system, in this case the position and
momenta of all :math:`N = 64^3` particles. CO\ *N*\ CEPT uses its own snapshot
format, which is simply a well-structured HDF5 file.

.. tip::
   For a great graphical tool to explore HDF5 files in general, check out
   `ViTables <https://vitables.org/>`_. If you encounter problems viewing HDF5
   files produced by CO\ *N*\ CEPT, check that you are using ViTables 3.

   You can install ViTables as part of the CO\ *N*\ CEPT installation via

   .. code-block:: bash

      (source concept && $python -m pip install vitables)

   after which you can run it by e.g. doing

   .. code-block:: bash

      (source concept && $python -m vitables &)

Such snapshots are useful if you want to process the raw data using some
external program. You can also initialize a simulation from a snapshot, instead
of generating initial conditions from scratch. To try this, redefine the
initial conditions to simply be the path to the snapshot produced by the
simulation you just ran:

.. code-block:: python3

   initial_conditions = 'output/tutorial/snapshot_a=0.10.hdf5'

Also, you should change ``a_begin`` to be ``0.1`` as to comply with the time at
which the snapshot was dumped. Finally, before rerunning the simulation
starting from the snapshot, you should probably comment out at least the
``'render2D'`` ``output_times``, as to not clutter up the output directory too
heavily.

If you forget to correct ``a_begin``, a warning will be emitted. The same goes
for other obvious inconsistencies between the parameter file and the snapshot,
like if ``boxsize`` or ``Ωcdm`` is wrong. To be able to do this, some meta data
about the cosmology and numerical setup is stored in the snapshot as well.

.. note::
   Warnings are shown in red bold text, easily distinguishable from the main
   output text. As warnings (as opposed to errors) results from non-fatal
   issues, CO\ *N*\ CEPT will continue running. A warning emitted during the
   simulation may hint that something has gone wrong, meaning that the results
   perhaps should not be trusted. To make sure that no warnings go unnoticed,
   CO\ *N*\ CEPT will notify you at the end of the simulation. A separate error
   log, ``logs/<ID>_err``, containing just warning and error messages, will
   also be present.

To generate a warning and error log file, try wrongly specifying e.g.
``Ωb = 0.05``. Once the simulation has completed, check out the error
log file.

If you intend to run many simulations using the same initial conditions, it's
worthwhile to initialize these from a common snapshot, as it saves computation
time in the beginning of the simulation. To produce such an initial snapshot,
simply set ``output_times = {'snapshot': a_begin}``, in which case
CO\ *N*\ CEPT will exit right after the snapshot has been dumped at the
initial time, without doing any simulation. Also, the whole purpose of having
the ``ICs`` directory is to hold such initial condition snapshots. To dump
snapshots to this directory, set the ``'snapshot'`` entry in ``output_dirs``
to ``paths['ics_dir']``. We can achieve both without even altering the
parameter file:

.. code-block:: python3

   ./concept \
       -p params/tutorial \
       -c "output_times = {'snapshot': a_begin}" \
       -c "output_dirs = {'snapshot': paths['ics_dir']}" \
       -n 4

You may also want to use CO\ *N*\ CEPT purely as an initial condition
generator, and perform the actual simulation using some other code. If so, the
standard CO\ *N*\ CEPT snapshot format is of little use. To this end,
CO\ *N*\ CEPT also supports the binary Fortran format of
`GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/>`_, specifically the
*second* type (``SnapFormat = 2`` in GADGET-2), which is understood by several
other simulation codes and tools. To use this snapshot format in place of the
standard one, add ``snapshot_type = 'gadget2'`` to your parameter file.



.. raw:: html

   <h3>The info utility</h3>

We mentioned `ViTables <https://vitables.org/>`_ as a great way to peek inside
the default CO\ *N*\ CEPT (HDF5) snapshots. It would be nice to have a general
tool which worked for the supported GADGET-2 snapshots as well. Luckily,
CO\ *N*\ CEPT comes with just such a tool: the *info utility*. To try it out,
simply do

.. code-block:: bash

   ./concept -u info output/tutorial

The content of all snapshots --- standard (HDF5) or GADGET-2 format --- in the
``output/tutorial`` directory will now be printed to the screen. Should you
want information about just a specific snapshot, simply provide its entire
path.

