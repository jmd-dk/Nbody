Mastering gravity
-----------------
The fact that gravity is extremely important for the cosmic evolution has
already been demonstrated. We would like our simulation to be able to compute
gravity in a manner that is both accurate and efficient. In CO\ *N*\ CEPT, the
details of the gravitational computation is controlled by a large set of
parameters, enabling us to tune this trade-off between accuracy and efficiency
to our heart's desire.

In order to learn how to control gravity within CO\ *N*\ CEPT we shall run
some sample simulations. The parameter specifications below are very similar
to the ones encountered in the previous section.

.. code-block:: python3
   :caption: params/tutorial
   :name: params-gravity
   :emphasize-lines: 12-14, 29-32

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
   output_bases = {
       'powerspec': f'powerspec_{_sim}',
   }
   output_times = {
       'powerspec': 1,
   }

   # Numerical parameters
   boxsize = 256*Mpc
   potential_options = _size

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.02

   # Physics
   select_forces = {
       'matter': {'gravity': 'pm'},
   }

Save these parameters to e.g. ``params/tutorial``.

.. tip::
   If the file ``params/tutorial`` already exists due to previous tutorial
   work, you may safely overwrite the old content. Similarly, it's best to
   clean up (or remove entirely) the ``output/tutorial`` directory before
   continuing. If you wish to keep the old content, you can do e.g.

   .. code-block:: bash

      mv output/tutorial output/tutorial_3

   If you wish to store the outputs of each tutorial section separately, you
   can also name the parameter file(s) something more imaginative than just
   '``tutorial``'. The output directories are set up to reflect the parameter
   file name.

Now run a simulation using the :ref:`tutorial <params-gravity>` parameter file
by executing

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_sim = 'A'"

and optionally specifying whatever number of processes you'd like with the
``-n`` option.

The ``output_bases`` parameter has been used to specify the base of filenames
for power spectrum output. This is set up to depend on a helper variable
``_sim``, which we supply from the command-line. We can then run multiple
simulations without having their outputs overwrite each other, by supplying
different values for ``_sim``.

The result of the simulation will be a data file
``output/tutorial/powerspec_A_a=1.00`` of the matter power spectrum at
:math:`a = 1`, along with the corresponding plot
``output/tutorial/powerspec_A_a=1.00.png``.



The PM method
.............
A new parameter appearing in the :ref:`tutorial <params-gravity>` parameter
file is ``select_forces``, which is set to map ``'matter'`` to the
gravitational interaction using the particle-mesh (PM) method. This method
works by solving the gravitational potential on a cubic grid, the size of
which is set through ``potential_options``. The potential used for our
first simulation, A, was then a cube of size
``_size``:math:`\times`\ ``_size``:math:`\times`\ ``_size``
(i.e. :math:`64\times 64\times 64`), dividing the box into smaller cells.

The potential grid introduces a length scale below which gravity cannot be
resolved, namely the width of a grid cell, here ``boxsize/_size``. We may
expect to achieve a more accurate result by lowering this length scale, e.g.
by doubling the size (resolution) of the potential grid (in each direction).
To try this out, update ``potential_options`` to

.. code-block:: python3

   potential_options = 2*_size

in ``params/tutorial`` and run a new simulation, this time using

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_sim = 'B'"

Visually comparing ``powerspec_A_a=1.00.png`` to ``powerspec_B_a=1.00.png``,
we see that increasing the potential grid size produces a lager power
spectrum, as we would perhaps expect. To check whether doubling the grid size
was enough to achieve convergence, let's further run a simulation C with
triple the grid size, i.e. switch to using

.. code-block:: python3

   potential_options = 3*_size

in ``params/tutorial`` and execute

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_sim = 'C'"

Properly comparing the three individual output plots is not so easy. To better
compare the results, you should plot the different power spectra together in a
single plot, using the information in the data files. You may do this using
your favourite plotting tool, or --- for your convenience --- using the below
script:

.. code-block:: python3
   :caption: output/tutorial/plot.py
   :name: plot-gravity

   import glob, os, re
   import numpy as np
   import matplotlib.pyplot as plt

   # Read in data
   P_sims = {}
   this_dir = os.path.dirname(os.path.realpath(__file__))
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*'), key=os.path.getmtime):
       match = re.search(r'powerspec_(.+)_', filename)
       if not match or filename.endswith('.png'):
           continue
       sim = match.group(1)
       k, P_sims[sim], P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)

   # Plot
   fig, ax = plt.subplots()
   linestyles = ['-', '--', ':', '-.']
   for sim, P_sim in P_sims.items():
       linestyle = linestyles[
           sum(np.allclose(line.get_ydata(), P_sim, 5e-2) for line in ax.lines)
           %len(linestyles)
       ]
       ax.loglog(k, P_sim, linestyle, label=f'simulation {sim}')
   ylim = ax.get_ylim()
   ax.loglog(k, P_lin, 'k--', label='linear', linewidth=1)
   ax.set_xlim(k[0], k[-1])
   ax.set_ylim(ylim)
   ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   ax.set_ylabel(r'$P\, [\mathrm{Mpc}^3]$')
   ax.legend()
   fig.tight_layout()
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Do not feel bad about using this plotting script --- and others to come later
on in the tutorial --- without studying it in any detail. To run the script,
save the :ref:`above code <plot-gravity>` to a file in the ``output/tutorial``
directory, say ``output/tutorial/plot.py``, then do

.. code-block:: bash

   ./concept -m output/tutorial/plot.py

.. note::
   The ``-m`` command-line option redirects ``concept`` to run the specified
   Python script, rather than launching a CO\ *N*\ CEPT simulation. In this
   case, this is almost equivalent to just running the script using Python
   directly, but using ``./concept -m`` we are guaranteed that the environment
   is set up correctly, according to the CO\ *N*\ CEPT installation.

The script will produce the output ``output/tutorial/plot.png``. Investigating
this comparison plot, we see that in fact simulation B --- the one with the
"in-between" potential grid size --- has the most power. It's then unclear
which of the three power spectra to trust, if any.

Though inadequate for precision simulations, the PM method remains a valuable
tool due to its unprecedented efficiency, as sometimes the benefits of rapid
simulations outweigh the drawbacks from the loss of precision. Also, as we
shall :doc:`delve into later <beyond_matter_only>`, PM gravity is as accurate
as can be for fluid (i.e. non-particle) components, used to model species
different from matter.



The P³M method
..............
A different gravitational method is the particle-particle-particle-mesh (P³M)
method. This method also makes use of a potential grid, but here the grid is
used only to get the gravitational force between particles separated by a
distance much greater than the potential cell size, minimizing discretisation
errors. The remaining --- so-called *short-range* --- component of gravity
between nearby particles is then computed separately, by directly pairing up
particles and computing the force between them (no potential grid required).

To switch out PM for P³M, simply replace ``'pm'`` in the ``select_forces``
parameter with ``'p3m'``. Let's run a simulation D using

.. code-block:: python3

   select_forces = {
       'matter': {'gravity': 'p3m'},
   }

while also resetting ``potential_options`` back to

.. code-block:: python3

   potential_options = _size

You will notice that this time, the simulation takes much longer to complete.
Studying the text printed to the screen, we see that the work within
each time step is split into a short-range and a long-range part. From the
computation times stated in parentheses to the right, it's clear that the
short-range part is by far the most time-consuming.

Once completed, add the P³M (D) result to the comparison plot by rerunning
:ref:`plot.py <plot-gravity>`. You will see that P³M lies somewhere in-between
B and C at intermediary scales, while having more power than any of the PM's
at the smallest scales.

As before, let's now double the grid size, and perform a final simulation, E.
You will find that this time, the P³M simulation isn't nearly as slow, as
the smaller potential cell size means that there is less work to be done for
the short-range part.

Updating the comparison plot with the latest result, we see some long-awaited
stability: The two P³M simulations give very nearly identical results, and
as such these are the ones we should put our trust in.

As changing the size of the potential grid under P³M doesn't really affect
the output, we are free to optimize it for performance. Generally, using a P³M
grid twice that of the "particle grid"
(``potential_options``\ :math:`= 2 \sqrt[3]{N}`) is recommended;

.. code-block:: python3

   potential_options = 2*_size



.. raw:: html

   <h3>Default parameters</h3>

You may wonder what gravitational method was used back for our previous
simulations, where we did not explicitly select a method through
``select_forces``. Here it turns out that CO\ *N*\ CEPT goes with P³M by
default.

For nested parameters such as ``select_forces``, it's generally fine to not
write out everything in explicit detail, but instead rely on defaults. Thus,
specifying gravitational P³M may then be done through either of

.. code-block:: python3

   select_forces = {
      'matter': {'gravity', 'p3m'},
   }

   # or

   select_forces = {
      'matter': 'gravity',
   }

or by leaving out a specification for ``select_forces`` entirely.

Similarly, the ``potential_options`` parameter is really a nested parameter by
which various details of the (PM and P³M)
:doc:`potential can be controlled </parameters/numerical_parameters>`.



.. raw:: html

   <h3>The PP method</h3>

Finally, let it be known that CO\ *N*\ CEPT also supports the bare
particle-particle (PP) method, which simply performs the naïve pairwise
force computation between all particles, effectively computing "short"-range
forces as in the P³M method, but now across the entire simulation box.

As expected, the PP method may be specified using

.. code-block:: python3

   select_forces = {
       'matter': {'gravity': 'pp'},
   }

If you try this out, reduce ``_size`` to *at most* 32, as running a PP
simulation with :math:`N = 64^3` particles literately takes days. Though
slightly more accurate than the P³M method, the PP method should never be
used for running production simulations and is only included in CO\ *N*\ CEPT
for use with internal testing. Running (well, starting) a small PP simulation
at least once will help any user of *N*-body codes appreciate the enormous
performance gains achieved by the more sophisticated methods.

