Comparison with GADGET-2
------------------------
In this section we shall perform equivalent simulations with CO\ *N*\ CEPT and
the well-known `GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/>`_ code.
Due to the two codes utilising different numerical schemes, as well as the
fact that CO\ *N*\ CEPT --- unlike GADGET-2 --- is designed to be consistent
with general relativistic perturbation theory, running simulations with
CO\ *N*\ CEPT that are truly equivalent to those of GADGET-2 requires some
tricks.

If you are unfamiliar with GADGET-2 or have no interest in running GADGET-2
equivalent simulations with CO\ *N*\ CEPT, you may skip this section. However,
some of the tricks to come are not directly related to GADGET-2.



.. raw:: html

   <h3>CONCEPT</h3>

As always, we start with a CO\ *N*\ CEPT parameter file, which you should save
as e.g. ``params/tutorial``:

.. code-block:: python3
   :caption: params/tutorial
   :name: params-gadget
   :emphasize-lines: 12, 40

   # Input/output
   if _gen:
       initial_conditions = {
           'species': 'matter',
           'N'      : _size**3,
       }
   else:
       initial_conditions = (
           paths['ics_dir'] + '/'
           + basename(paths['params']) + f'_a={a_begin}'
       )
   snapshot_type = 'gadget2'
   output_dirs = {
       'snapshot': (
           paths['ics_dir'] if _gen
           else paths['output_dir'] + '/' + basename(paths['params'])
       ),
       'powerspec': ...,
   }
   output_bases = {
      'snapshot': basename(paths['params']) if _gen else 'snapshot',
   }
   output_times = {
       'snapshot' : a_begin if _gen else '',
       'powerspec': '' if _gen else [0.1, 0.3, 1.0],
   }

   # Numerical parameters
   boxsize = 512*Mpc/h
   potential_options = 2*_size

   # Cosmology
   H0 = 67*km/(s*Mpc)
   Ωcdm = 0.27
   Ωb = 0.049
   a_begin = 0.02

   # Non-parameter variables
   _size = 64
   _gen = False

To perform the same simulation with both CO\ *N*\ CEPT and GADGET-2, we shall
want to start from a common initial snapshot. The
:ref:`parameter file <params-gadget>` is set up to produce this initial
snapshot when ``_gen`` is ``True``. To create the snapshot then, do

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_gen = True"

(as always, feel free to tack on the number of processes you want
with ``-n``). The snapshot will be dumped to the ``ICs`` directory.

Starting from this snapshot we can now perform the CO\ *N*\ CEPT simulation by
running with ``-c "_gen = False"``. As this is also the value set in the
:ref:`parameter file <params-gadget>`, we may in fact start the simulation
simply by

.. code-block:: bash

   ./concept -p params/tutorial

Note that the ``initial_conditions`` is set to the path of the generated
snapshot when ``_gen`` is ``False``.

The power spectra outputs of the CO\ *N*\ CEPT simulation will be dumped to
``output/tutorial`` as usual.



.. raw:: html

   <h3>GADGET-2</h3>

For GADGET-2, the first thing we need is the source code itself. If you have
installed CO\ *N*\ CEPT the easy way, you already have it! Let's make a copy
just for use with this tutorial:

.. code-block:: bash

   (source concept && cp -r "${Gadget2_dir}" output/tutorial/)

You now have the complete GADGET-2 code in ``output/tutorial/Gadget2``.

The ``Makefile`` of GADGET-2 needs to be set up with correct path information
for its dependencies. Furthermore, various options needs to be set in order
for the GADGET-2 simulation to come to be equivalent to the CO\ *N*\ CEPT
simulation. Last but not least we need a GADGET-2 parameter file equivalent to
the :ref:`CONCEPT parameter file <params-gadget>`. All of this can be
conveniently achieved using the *gadget utility* included with CO\ *N*\ CEPT:

.. code-block:: bash

   ./concept \
       -u gadget output/tutorial/Gadget2 \
       -p params/tutorial

The ``output/tutorial/Gadget2`` directory now has a properly set up
``Makefile`` and a parameter file called ``params``. The output times of the
:ref:`original parameter file <params-gadget>` have also been copied to an
``outputlist``, similarly placed in ``output/tutorial/Gadget2``.

.. note::
   The parameters specified in the GADGET-2 ``Makefile`` and ``params`` file
   include the cosmology, the grid size of the PM grid, the particle softening
   lengths and more, all set to match those used by the specific CO\ *N*\ CEPT
   simulation as specified by the :ref:`parameter file <params-gadget>`
   together with default CO\ *N*\ CEPT values.

   The GADGET-2 TreePM gravitational method is used, which is quite similar in
   essence to the P³M method used by the CO\ *N*\ CEPT simulation, the
   difference being that the tree approximates the gravitational short-range
   force by grouping particles together.

   Lastly, the values of physical constants employed by CO\ *N*\ CEPT and
   GADGET-2 differ slightly, which will make a noticeable difference for the
   comparison. If the GADGET-2 source code you are using is the one that goes
   with the CO\ *N*\ CEPT installation, all such constants of the GADGET-2
   source code have been patched to match the values used by CO\ *N*\ CEPT.

With the various GADGET-2 files in place, let's build the code and launch
the simulation:

.. code-block:: bash

   (source concept \
       && cd output/tutorial/Gadget2 \
       && make clean \
       && make \
       && mpiexec -n 4 ./Gadget2 params)

where you may wish to change the specified number of processes deployed.

As GADGET-2 is not able to produce power spectra, the results from the
simulation are raw snapshots, placed along with other files in
``output/tutorial/Gadget2/output``. To investigate these snapshots, we can
make use of the info utility:

.. code-block:: bash

   ./concept -u info output/tutorial/Gadget2/output

You should find that the snapshots indeed have the same values of the scale
factor :math:`a` as specified in the :ref:`parameter file <params-gadget>`
(perhaps with small numerical errors).

To measure the power spectra of the particle distributions contained within
the snapshots, we may make use of the *powerspec utility*:

.. code-block:: bash

   ./concept \
       -u powerspec output/tutorial/Gadget2/output \
       -p params/tutorial

(as always, you may choose to throw more processes at the task using ``-n``).



.. raw:: html

   <h3>Comparison and adjustments</h3>

You should now have CO\ *N*\ CEPT power spectra in ``output/tutorial`` and
corresponding GADGET-2 power spectra in ``/output/tutorial/Gadget2/output``.
We can of course look at each pair of plots, but for a proper comparison we
should plot the power spectra together in a single plot. You may use the
script below:

.. code-block:: python3
   :caption: output/tutorial/plot.py
   :name: plot-gadget

   import glob, os, re
   import numpy as np
   import matplotlib.pyplot as plt

   # Read in CO𝘕CEPT data
   def read(dirname):
       P = {}
       for filename in sorted(glob.glob(f'{dirname}/powerspec*'), key=os.path.getmtime):
           if filename.endswith('.png'):
               continue
           with open(filename) as f:
               header = f.readline()
           a = float(re.search('a = (.+)', header).group(1).rstrip('.'))
           k, P[float(f'{a:.3f}')] = np.loadtxt(filename, usecols=(0, 2), unpack=True)
       return k, P
   this_dir = os.path.dirname(os.path.realpath(__file__))
   k, P_concept = read(this_dir)

   # Read in GADGET-2 data
   k, P_gadget = read(f'{this_dir}/Gadget2/output')

   # Plot
   fig, axes = plt.subplots(2, sharex=True)
   for a, P in P_concept.items():
       axes[0].loglog(k, P, label=f'$a = {a}$ (CO$N$CEPT)')
       if a in P_gadget:
           axes[0].loglog(k, P_gadget[a], 'k--')
           axes[1].semilogx(k, 100*(P/P_gadget[a] - 1))
   axes[0].set_xlim(k[0], k[-1])
   axes[1].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   axes[0].set_ylabel(r'$P\, [\mathrm{Mpc}^3]$')
   axes[1].set_ylabel(r'$P_{\mathrm{CO}N\mathrm{CEPT}}/P_{\mathrm{GADGET}} - 1\, [\%]$')
   axes[0].legend()
   axes[0].tick_params('x', direction='inout', which='both')
   axes[1].set_zorder(-1)
   fig.tight_layout()
   fig.subplots_adjust(hspace=0)
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Store the script as e.g. ``output/tutorial/plot.py`` and run it using

.. code-block:: bash

   ./concept -m output/tutorial/plot.py

The upper subplot --- with absolute power spectra --- of the generated
``output/tutorial/plot.png`` should show a good qualitative match between the
two codes, whereas the lower subplot --- with relative power spectra ---
should reveal a disagreement of several percent. This disagreement stems from
both physical and numerical differences between the codes, which we will now
alleviate by adjusting the CO\ *N*\ CEPT simulation.



.. raw:: html

   <h4>Cell-centred vs. cell-vertex discretisation</h4>

The initial particle distribution, stored within the generated initial
snapshot file ``ICs/tutorial_a=0.02``, should ideally be close to homogeneous
and isotropic. We can check this qualitatively by plotting the distribution
in 3D. For this, let's use the *render3D utility*:

.. code-block:: bash

   ./concept \
       -u render3D ICs/tutorial_a=0.02 \
       -p params/tutorial \
       -c "render3D_resolution = 2000" \
       -c "render3D_colors = ('black', 1.5)" \
       -c "render3D_bgcolor = 'white'"

This produces an image file in the ``ICs`` directory. Zooming in on one of the
corners of the box, it should be clear that while we do have a close to
homogeneous system, it is far from isotropic.

.. note::
   The reason for the ``-c`` command-line options above is to produce a render
   which more clearly shows what we're after. Playing around with the given
   values is encouraged. Their function should quickly become apparent.

The strong anisotropy of the particles is inherited from the grid used for
realisation of the initial conditions. This grid-like structure of the
particle distribution makes the system sensitive towards other grids used
during the simulation, e.g. the potential grid. In CO\ *N*\ CEPT it is chosen
to use "cell-centred" grid discretisation, whereas GADGET-2 uses
"cell-vertex" grid discretisation. In effect this means that the potential
grids of the two codes are shifted by half a grid cell relative to each other,
in all three dimensions. This detail oughtn't matter much, but the grid
structure imprinted on the particle distribution, together with the fact that
our simulations are rather small (:math:`N = 64^3`) simulations, makes for an
exaggerated effect.

We can force CO\ *N*\ CEPT to use cell-vertex discretisation by specifying

.. code-block:: python3

   cell_centered = False

Introduce this new parameter --- either by editing the parameter file or
supplying it with ``-c`` --- and rerun the CO\ *N*\ CEPT simulation. After
also updating the comparison plot, you should find that the relative error is
now much smaller, though still a few percent.

.. note::
   Though we now have better agreement, we should not be tempted to conclude
   that cell-vertex discretisation is superior to its cell-centred cousin.
   The improvement came from using the *same* discretisation.

   The real solution would be to redo the initial conditions without the
   isotropies, e.g. using what is known as *glass* (pre-)initial conditions,
   though this feature is not (yet) available in CO\ *N*\ CEPT.

   The mismatch between results of cell-centred and cell-vertex simulations is
   reduced for larger simulations, as the details of what's going on at the
   scale of half a grid cell become more negligible for the whole. Thus for
   large (say :math:`N \gtrsim 512^3`) simulations, we may ignore this issue
   while keep using standard grid (i.e. non-glass) (pre-)initial conditions.



.. raw:: html

   <h4>Radiation in the background</h4>

A key difference between CO\ *N*\ CEPT and GADGET-2 is that the former
--- unlike the latter --- tries to incorporate all species of a given
cosmology, as demonstrated in the
:doc:`previous section <beyond_matter_only>`. As such, CO\ *N*\ CEPT uses
a full background evolution obtained from the
`CLASS <http://class-code.net/>`_ code. By default, this CLASS background
includes matter, :math:`\Lambda` and radiation, whereas the background of
GADGET-2 only consists of matter and :math:`\Lambda`.

One cannot remove radiation from the CLASS background, as a cosmology without
radiation is not supported by CLASS. Instead we can deactivate the CLASS
background entirely, which will turn on an internal background solver of
CO\ *N*\ CEPT, which like that of GADGET-2 only includes matter and
:math:`\Lambda`. To do so, add

.. code-block:: python3

   enable_class_background = False

to the parameters (while also still keeping ``cell_centered = False``) and
rerun the CO\ *N*\ CEPT simulation.

Updating the comparison plot, the codes should now agree to within a percent.



.. raw:: html

   <h4>Increased short-range precision</h4>

The gravitational force of the P³M method in CO\ *N*\ CEPT and the TreePM
method of GADGET-2 is split into a long-range and a short-range part at a
scale :math:`r_{\text{s}}`, which in both CO\ *N*\ CEPT and GADGET-2 has a
default value of :math:`r_{\text{s}} = 1.25 \Delta x`, with :math:`\Delta x`
the cell size of the potential grid. The short-range force between pairs of
particles separated by a distance somewhat larger than :math:`r_{\text{s}}`
falls of exponentially with the distance. Particle pairs with a separation
:math:`r > r_{\text{r}}` may then be neglected, provided
:math:`r_{\text{r}} \gg r_{\text{s}}`.

In both CO\ *N*\ CEPT and GADGET-2, the range of the short-range force has a
default value of :math:`r_{\text{r}} = 4.5r_{\text{s}}`. Due to the grouping
of particles by the tree in GADGET-2, the short-range forces between some
particles separated by distances somewhat larger than :math:`r_{\text{r}}` are
probably being taken into account as well. We can then hope to obtain still
better agreement with GADGET-2 by increasing the value of :math:`r_{\text{r}}`
in CO\ *N*\ CEPT slightly. Try rerunning CO\ *N*\ CEPT with
:math:`r_{\text{r}} = 5.5r_{\text{s}}` using

.. code-block:: python3

   shortrange_params = {'range': '5.5*scale'}

and update the comparison plot. You should find that the two codes now agree
well within 1%.

For still better agreement, we could continue increasing the accuracy of
*both* the CO\ *N*\ CEPT and the GADGET-2 simulation, though we shall not do
so here.



.. raw:: html

   <h4>Final notes</h4>

Though it does not matter for the comparison, the simulations performed in
this section have been performed in a manner which is slightly inconsistent
--- not which each other, but with physics.

CO\ *N*\ CEPT strives to be consistent with general relativistic perturbation
theory. It does so by using the full background from CLASS, and adding in
gravitational effects from linear perturbations of other species (when
:doc:`enabled <beyond_matter_only>`) and the metric itself, ensuring that the
simulation stays in the so-called *N*-body gauge. All this fanciness is hardly
needed when running without massive neutrinos or other exotic species, which
GADGET-2 does not support.

GADGET-2 only allows for the simulation of matter (though besides the standard
cold dark matter particles, it further supports (smoothed-particle)
hydrodynamical baryons!), and always uses a background containing just matter
and :math:`\Lambda`. Sticking to even the simplest :math:`\Lambda`\ CDM
cosmologies (which *do* include photons), neglecting the gravitational tug
from radiation is unacceptable for precision simulations. To resolve this
problem, the usual trick used with Newtonian *N*-body codes is to generate
initial conditions at :math:`a = a_{\text{begin}}` using the results of
general relativistic perturbations theory (which also accounts for radiation)
at math:`a = 1`, but scaled back in time using the Newtonian growth factor.
Thus only the :math:`a = 1` results of such *back-scaled* simulations are
really physical.

As CO\ *N*\ CEPT works in *N*-body gauge, the initial conditions generated for
this section are in this gauge. As the corresponding *N*-body gauge
perturbations have not been continually applied to the particles during their
evolution, these have drifted away from this gauge. As we further did not
use back-scaling for the initial conditions (CO\ *N*\ CEPT simply realises
the :math:`a = a_{\text{begin}}` *N*-body gauge perturbations directly), the
:math:`a = 1` results deviate slightly from what we would have got,
starting from back-scaled initial conditions. The simulations ran during this
section is then not quite compatible with either of these two standards.

Note that with ``enable_class_background = False`` and without adding in
linear species, CO\ *N*\ CEPT *does* evolve the system just like GADGET-2,
i.e. with the exact same physics. The *N*-body gauge vs. back-scaling issue
then only concerns the initial conditions. To run CO\ *N*\ CEPT with
back-scaled initial conditions, simply provide such an initial snapshot
yourself (as back-scaling is not available as an option when generating
initial conditions with CO\ *N*\ CEPT).

With the above note of warning about the perils of insufficient rigour taken
towards the two techniques of the *N*-body gauge and back-scaling, let it be
said that the results from the two techniques (and indeed the "mixed"
technique used by the simulations in this section) differ only slightly
--- assuming simple :math:`\Lambda`\ CDM cosmologies.

