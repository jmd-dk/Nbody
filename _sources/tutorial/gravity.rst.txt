Mastering gravity
-----------------
The fact that gravity is extremely important for the particle evolution has
already been demonstrated. We would like our simulation to be able to compute
gravity in a manner that is both accurate and efficient, but in practice one
has to sacrifice one for the other. In CO\ *N*\ CEPT, the details of the
gravitational computation is controlled by a large set of parameters, enabling
us to tune this tradeoff between accuracy and efficiency to our heart's
desire.

In order to learn how to control gravity within CO\ *N*\ CEPT we shall run
some sample simulations. The parameter specifications below are very similar
to the ones encountered in the previous section. Save these in
``params/tutorial``.


.. code-block:: python3

   # Non-parameter helper variable used to control the size of the simulation
   _size = 64

   # Input/output
   initial_conditions = {
       'name'   : 'matter',
       'species': 'matter particles',
       'N'      : _size**3,
   }
   output_dirs = {
       'powerspec': paths['output_dir'] + '/' + basename(paths['params'])
   }
   output_bases = {
       'powerspec': f'powerspec_{_sim}',
   }
   output_times = {
       'powerspec': 1,
   }

   # Numerical parameters
   boxsize = 256*Mpc

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.02

   # Physics
   select_forces = {
       'all': 'gravity',
   }

Now run a simulation using these parameters, by executing

.. code-block:: bash

   ./concept -p params/tutorial -c '_sim = "A"'

and optionally specifying whatever number of CPU cores you'd like with the
``-n`` option.

.. note::

   Note that a choice of e.g. ``-n 3`` is illegal as
   :math:`N^{\frac{1}{3}}=64` is not divisible by :math:`3`. More of such
   restrictions exist. If an illegal number of processes is chosen, this will
   be detected and a helpful error message is shown.

The ``output_bases`` parameter has been used to specify the base of filenames
for power spectrum output. This is set up to depend on a helper variable
``_sim``, which we supply from the command-line. This way we can run multiple
simulations without having their outputs overwrite each other, by specifying a
different value of ``_sim`` for each run.

The result of the simulation will be a data file
``output/tutorial/powerspec_A_a=1.00`` of the matter power spectrum at
:math:`a=1`, along with the corresponding plot
``output/tutorial/powerspec_A_a=1.00.png``.



The PM method
.............
Without further specification, CO\ *N*\ CEPT solves gravity using the
particle-mesh (PM) method. We may also specify this choice explicitly using

.. code-block:: python3

   select_forces = {
       'all': {'gravity': 'pm'},
   }

The PM method works by solving for the gravitational potential on a grid. By
default, the resolution of this grid is set to match the number of particles.
Again, we may specify this choice explicitly using

.. code-block:: python3

   select_forces = {
       'all': {'gravity': ('pm', _size)},
   }

The potential grid is then a cubic
``_size``:math:`\times`\ ``_size``:math:`\times`\ ``_size`` mesh, dividing
the box into smaller cells.

To test whether this really amounts to the default gravitational set-up,
update ``select_forces`` in ``params/tutorial`` according to the above and run
a new simulation, this time using

.. code-block:: bash

   ./concept -p params/tutorial -c '_sim = "B"'

Visually comparing ``powerspec_A_a=1.00.png`` to ``powerspec_B_a=1.00.png``,
we see that the two simulations were indeed exactly the same.

The potential grid introduces a length scale below which gravity cannot be
resolved, namely the width of a mesh cell, here ``boxsize/_size``. We may
expect to achieve a more accurate result by lowering this length scale, e.g.
by doubling the size (resolution) of the potential grid (in each direction):

.. code-block:: python3

   select_forces = {
       'all': {'gravity': ('pm', 2*_size)},
   }

Try this out, using ``_sim = "C"``.

Comparing the plots of simulation B and C, it's clear that the choice of grid
size affects the result. What isn't clear is which result to trust.



The P³M method
..............
A different gravitational method is the particle-particle-particle-mesh (P³M)
method. This method also makes use of a potential grid, but here the grid is
used only to get the gravitational force between particles separated by a
distance much greater than the potential cell size, minimizing discretization
errors. The remaning --- so-called short-range --- component of gravity
between nearby particles is then computed separately, by directly pairing up
particles (no potential grid required).

To switch out PM for P³M, simply replace ``'pm'`` in the ``select_forces``
parameter with ``'p3m'``. Let's run a simulation D using

.. code-block:: python3

   select_forces = {
       'all': {'gravity': ('p3m', _size)},
   }

You will notice that this time, the simulation takes several times longer to
complete. Studying the text printed to the screen, we see that the work within
each time step is split into a short-range and a long-range part. From the
computation times stated in parentheses to the right, it's clear that the
short-range part is by far the most time-consuming.

Once completed, compare the plot from the P³M simulation D to that of the PM
simulations B and C. You will find that we now have three different results.

As before, let's now double the grid size, and perform a final simulation, E:

.. code-block:: python3

   select_forces = {
       'all': {'gravity': ('p3m', 2*_size)},
   }

You will find that this time, the P³M simulation isn't nearly as slow, as
the smaller potential cell size leads to less work to be done for the
short-range part.

Comparing the results of simulation D and E, we see that these are very
similar, implying that the size of the potential grid is largely irrelevant
regarding the accuracy of the P³M method. To better compare the results of
simulation A--E, you should plot the different power spectra together in a
single plot, using the information in the data files. You may do this using
your favourite plotting program, or --- for your convenience --- using the
script below:

.. code-block:: python3

   import glob, os, re
   import numpy as np
   import matplotlib.pyplot as plt

   this_dir = os.path.dirname(os.path.realpath(__file__))
   fig, ax = plt.subplots()
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*')):
       if filename.endswith('.png'):
           continue
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       sim = re.search('powerspec_(.+)_', filename).group(1)
       linetype = '--' if sim in {'B', 'E'} else '-'
       ax.loglog(k, P_sim, linetype, label=f'simulation {sim}')
   ax.loglog(k, P_lin, 'k--', label='linear')
   ax.legend()
   ax.set_xlabel(r'$k$ $[\mathrm{Mpc}^{-1}]$')
   ax.set_ylabel(r'power $[\mathrm{Mpc}^3]$')
   fig.tight_layout()
   fig.savefig(f'{this_dir}/plot.png')

To use the above script, save the code to a file in the ``output/tutorial``
directory, say ``output/tutorial/plot.py``, then do

.. code-block:: bash

   ./concept -m output/tutorial/plot.py

.. note::
   The ``-m`` command-line option redirects ``concept`` to run the specified
   Python script, rather than launching a CO\ *N*\ CEPT simulation. In this
   case, this is almost equivalent to just running the script using Python
   directly, but using ``./concept -m`` we are guarenteed that the environment
   is set up correctly, according to the CO\ *N*\ CEPT installation.

The script will produce the output ``output/tutorial/plot.py``.

Studying the collective plot containing all our simulation power spectra, it
is indeed clear that the P³M simulations agree far better with each other than
do the PM simulations. We should thus set our trust in the P³M power spectra,
not the PM ones. Generally, though much slower, the P³M method is then the
preferable gravitational method to use in CO\ *N*\ CEPT. As we saw, increasing
the size of the potential grid used can dramatically cut down the computation
time. In fact, it may well be worth it to increase it even further than what
was done here.

It is remarkable that not only is the PM force not invariant under a change in
grid size, but the smaller of the two grids actually produced the better
result, as evident from the collective plot (taking the P³M power spectra to
be essentially exact).

Though inadequate for precise particle simulations, the PM method is still a
valuable tool, as sometimes the benefits of rapid simulations outweigh the
drawbacks of loss of precision. Also, as we shall
:doc:`delve into later<beyond_matter_only>`, PM gravity is as accurate as
can be for fluid (i.e. non-particle) components, used to model species
different from matter. Finally --- though sticking to default values are
recommended --- you should be aware that the details about the PM and P³M
methods can be further specified using
:doc:`various parameters</parameters/numerical_parameters>`, affecting the
performance and accuracy.



.. topic:: The PP method

   Finally, let it be known that CO\ *N*\ CEPT also supports the
   bare particle-particle (PP) method, which simply performs the naïve
   pairwise force computation between all particles, effecively computing
   "short"-range forces as in the P³M method, but now across the entire
   simulation box.

   As expected, the PP method may be specified using e.g.

   .. code-block:: python3

      select_forces = {
          'all': {'gravity': 'pp'},
      }

   If you actually try this out, reduce ``_size`` to *at most* 32, as running
   a PP simulation with :math:`64^3` particles literately takes days. Though
   slightly more accurate than the P³M method, the PP method should never be
   used for running simulations and is only really included in CO\ *N*\ CEPT
   as it is used internally when running tests. Running (well, starting) a
   small PP simulation at least once will help any user of *N*-body codes
   appreciate the enourmous performance gains achieved by the more
   sophisticated methods.

