Beyond matter-only simulations
------------------------------
If you've followed this tutorial so far, you're now fully capable of using
CO\ *N*\ CEPT for running matter-only simulations. More fancy simulations ---
taking other species into account --- are also possible, as this section will
demonstrate.

To assess the effects resulting from including non-matter species, we shall
perform a slew of simulations throughout this section. As most effects are
small, we shall also learn how to crank up the precision, introducing several
new parameters. If you have no interest in any of this, feel free to skip this
section, as it is rather lengthy and more technical than the other sections of
this tutorial.



Radiation
.........

.. raw:: html

   <h3>Introduction</h3>

Though universes of interest may contain several different species, in
cosmological *N*-body simulations we are typically interested in following the
evolution of the matter component only, as this is the only component which is
non-linear at cosmological scales. Even so, we may wish to not neglect the
small gravitational effects on matter from the other, linear species. We can
do this by solving the given cosmology (including the matter) in linear
perturbation theory ahead of time, and then feed the linear gravitational
field from all species but matter to the *N*-body simulation, applying the
otherwise missing gravity as an external force.

CO\ *N*\ CEPT uses the `CLASS <http://class-code.net/>`_ code to solve the
linear perturbation equations. For details on how the linear perturbations
are applied to the *N*-body particles during the simulation, we refer to the
paper on
":doc:`Fully relativistic treatment of light neutrinos in 𝘕-body simulations</publications>`".

We begin our exploration by performing a standard matter-only simulation, as
specified by the below parameter file:

.. code-block:: python3

   # Non-parameter helper variable used to control the size of the simulation
   _size = 192

   # Input/output
   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
   ]
   for _species in _lin.split(','):
       if not _species:
           continue
       initial_conditions.append(
           # Linear fluid component
           {
               'species'        : _species.strip(),
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': paths['output_dir'] + '/' + basename(paths['params'])
   }
   output_bases = {
       'powerspec': f'powerspec_lin={_lin}'.replace(' ', ''),
   }
   output_times = {
       'powerspec': 2*a_begin,
   }

   # Numerical parameters
   boxsize = 4*Gpc

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.01

   # Physics
   select_forces = {
       'particles': {'gravity': ('p3m', 2*_size)},
       'fluid'    : {'gravity': 'pm'},
   }

   # Non-parameter helper variable used to specify linear components.
   # Should be supplied as a command-line argument.
   _lin = ''

As usual, save the parameters in e.g. ``params/tutorial`` and run the
simulation via

.. code-block:: bash

   ./concept -p params/tutorial

possibly with the addition of ``-n 4`` or some other number of processes.

.. note::

   The remainder of this tutorial leaves out explicit mention of the ``-n``
   option to ``concept`` invocations. Feel free to add this yourself.

A relatively large number of particles :math:`N = 192^3` is used in order to
increase the precision of the simulation. Our goal is to investigate the
effects from radiation perturbations, which are most pronounced at very large
scales and early times --- hence the large ``boxsize`` and early output time.
The unfamiliar parameter specifications will be explained in due time.



.. raw:: html

   <h3>The hunt for high precision</h3>

Investigating the resulting ``output/tutorial/powerspec_lin=_a=0.02.png`` you
should see a familiar looking simulation power spectrum: decently looking at
intermediary :math:`k`, inaccurate at small :math:`k` and with obvious
numerical artifacts at large :math:`k`. As we are interested in fine details
at low :math:`k`, we need to improve the precision here. We can do so by
adding

.. code-block:: python3

   # Simulation options
   primordial_amplitude_fixed = True

to the parameter file and rerunning the simulation. This has the effect of
replacing the uncorrelated random amplitudes of the primordial noise used to
generate the initial conditions with amplitudes that are all of the same size.
With this change, ``powerspec_lin=_a=0.02.png`` should look much better at
low :math:`k` (larger :math:`k` are not very affected by this, as here many
more :math:`\vec{k}` with the same magnitude :math:`|\vec{k}|=k` goes into
producing each data point (recorded in the ``modes`` column in the power
spectrum data file), reducing errors arising due to small number statistics).

We expect the evolution of the *N*-body particles to be completely linear at
these large scales and early times, and so we may use the difference between
the simulation and linear power spectrum as a measure for the error in the
simulation. To better see this difference, we shall make use of the below
plotting script:

.. code-block:: python3

   import glob, os, re
   import numpy as np
   import matplotlib.pyplot as plt

   this_dir = os.path.dirname(os.path.realpath(__file__))
   fig, axes = plt.subplots(2, sharex=True)
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*')):
       match = re.search(r'powerspec_lin=(.*)_a=[\d.]+$', filename)
       if not match:
           continue
       lin = match.group(1).replace(',', ', ').replace('+', ' + ')
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       linestyle, zorder = ('--', 1) if '+' in lin else ('-', 0)
       axes[0].loglog(k, P_sim, linestyle, zorder=zorder,
           label=f'simulation: lin = {lin}',
       )
       axes[1].semilogx(k, (P_sim/P_lin - 1)*100, linestyle, zorder=zorder)
   axes[0].loglog(k, P_lin, 'k--', label='linear')
   axes[1].semilogx(k, (P_lin/P_lin - 1)*100, 'k--')
   axes[0].legend()
   k_min, k_max = k[~np.isnan(P_lin)][[0, -1]]
   axes[0].set_xlim(k_min, 0.5*k_max)
   axes[0].set_ylim(0.9*np.nanmin(P_lin), 1.1*np.nanmax(P_lin))
   axes[1].set_ylim(-1, 1)
   axes[1].set_xlabel(r'$k$ $[\mathrm{Mpc}^{-1}]$')
   axes[0].set_ylabel(r'$P\,[\mathrm{Mpc}^3]$')
   axes[1].set_ylabel(r'$P_{\mathrm{sim}}/P_{\mathrm{lin}}-1\,[\%]$')
   axes[0].tick_params('x', direction='inout', which='both')
   axes[0].set_zorder(np.inf)
   fig.tight_layout()
   fig.subplots_adjust(hspace=0)
   fig.savefig(f'{this_dir}/plot.png')

Save the script as e.g. ``output/tutorial/plot.py`` and run it using

.. code-block:: bash

   ./concept -m output/tutorial/plot.py

This will produce ``output/tutorial/plot.png``, where the bottom panel shows
the relative error between the simulated power spectrum and that computed
using purely linear theory. They should agree to within a percent at the
lowest :math:`k`. At higher :math:`k` the agreement is worse. Though this can
be remedied by increasing the resolution of the simulation (i.e. by increasing
``_size``), we shall not do so here, as we focus on the lower :math:`k` only.

The power spectra outputted by the simulation are binned using a constant
linear bin size in :math:`k`. This is usually desirable, though higher
precision at the lowest :math:`k` can be achieved by leaving out this
binning. The bin size is controlled by the ``powerspec_binsize`` parameter.
By setting it to ``0`` we disable binning altogether. Add

.. code-block:: python3

   powerspec_binsize = 0

to the parameter file, then rerun the simulation and the plotting script.
Though the simulation power spectrum generally becomes more jagged, you should
observe better agreement with linear theory at low :math:`k`.



.. raw:: html

   <h3>Including linear species</h3>

With our high-precision setup established, we are ready to start experimenting
with adding in the missing species to the simulation, hopefully leading to
better agreement with linear theory on the largest scales. To keep the clutter
within ``output/tutorial`` to a minimum, go ahead and add

.. code-block:: python3

   powerspec_select = {
       'matter': {'data': True, 'plot': False},
   }

to the parameter file before continuing.

The inhomogeneities in the CMB causes a slight gravitational tug on matter,
perturbing its evolution. To add this effect to the simulation, we need to add
a photon component. This could look like (do not change the parameter file)

.. code-block:: python3

   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
       # Linear photon fluid
       {
           'species'        : 'photons',
           'gridsize'       : _size,
           'boltzmann order': -1,
       },
   ]

We do not want to model the photons using *N*-body particles, but rather as a
collection of spaially fixed grids, storing the energy density, momentum
density, etc. This is referred to as the *fluid representation* --- as opposed
to the *particle representation* --- and is generally preferable for
non-linear components. To represent the photon component as a fluid, we
specify ``'gridsize'`` in place of ``'N'``, where ``'gridsize'`` is the number
of grid cells along each dimension, for the 3D Cartesian fluid grids. Finally,
the number of fluid quantities --- and corresponding grids --- to take into
account is implicitly specified by the *Boltzmann order*. As is
`customary <https://arxiv.org/abs/astro-ph/9506072>`_ in linear perturbation
theory, we transform the Boltzmann equation for a given species into an
infinite hierarchy of multipole moments :math:`\ell \geq 0`. We then partition
this hierarchy in two; a non-linear part :math:`\ell \leq \ell_{\text{nl}}`
and a linear part :math:`\ell > \ell_{\text{nl}}`, where
:math:`\ell_{\text{nl}}` is precisely the Boltzmann order. A few examples
shall illuminate this concept:

- **Bolzmann order 0**: The evolution equation for the lowest moment (i.e. the
  continuity equation for the energy density) is solved non-linearly during
  the simulation, while higher moments like momentum density (on which the
  energy density depends) are solved in pure linear theory.
- **Boltzmann order 1**: The evolution equations for the two lowest moments
  (i.e. the continuity equation for the energy density and the Euler equation
  for the momentum density) are solved non-linearly during the simulation,
  while higher moments like pressure and shear (on which the energy and
  momentum density depend) are solved in pure linear theory.
- **Boltzmann order -1**: None of the moments are treated non-linearly, i.e.
  this results in a purely linear component. Though the evolution of such a
  component is independent of the simulation, a purely linear component may
  still act with a force on other, non-linear components during the simulation.

.. note::

   Though higher Boltzmann orders are well-defined, the largest Boltzmann
   order currently implemnted in CO\ *N*\ CEPT is 1.

We shall look into Boltzmann orders different from -1 in the subsection on
:ref:`non-linear massive neutrinos<nonlinear_massive_neutrinos>`. For now we
shall keep to the purely linear case of Boltzmann order -1.

For two components to have any affect on each other, they must both be
registered with the same force in the ``select_forces`` parameter. Looking at
this parameter in the parameter file, we see that both ``'particles'`` and
``'fluid'`` components have been assigned gravity. For fluids, only the PM
method with a grid size equal to the fluid grid size is sensible.

The parameter file has already been set up to include optional linear fluid
components, using the ``_lin`` command-line parameter. To perform a simulation
with the inclusion of linear photons, run

.. code-block:: bash

   ./concept -p params/tutorial -c '_lin = "photons"'

.. tip::

   Note that the ``_lin`` helper variable is defined at the bottom of the
   parameter file to have an empty value (leading to no linear species being
   included). As this is placed after all actual parameters, this defines a
   default value which is used when ``_lin`` is not given as a command-line
   parameter. As ``_size`` is defined at the top, supplying ``_size`` as a
   command-line parameter will have no effect.

Now redo the plot, and the results of both the matter-only and the matter plus
photon simulation should appear. The plot will show that --- sadly ---
including the photons does *not* lead to better large-scale behavior.

CO\ *N*\ CEPT deligates all linear (and background) computations to the CLASS
code. Though we have specified :math:`H_0`, :math:`\Omega_{\text{b}}` and
:math:`\Omega_{\text{cdm}}` in the parameter file, many cosmological
parameters are still left unspecified. Here the default CLASS parameters are
used, which in addition to baryons, cold dark matter and photons also contain
massless neutrinos. With our hope renewed, let's run a simulation which
include both linear photons and linear massless neutrinos:

.. code-block:: bash

   ./concept -p params/tutorial -c '_lin = "photons, massless neutrinos"'

To our horror, including the neutrinos made the disagreement between the
simulation and linear theory even larger!

The gravity applied to the non-linear matter particles from the linear photon
and neutrino fluids is the *Newtonian* gravity, i.e. that which results from
their energy densities. By contrast, the linear theory computation includes
full general relativistic gravity, meaning that we have still to account for
the gravitational effects due to the momentum density, pressure and shear of
the photons and neutrinos. As this part of gravity amounts to a general
relativistic correction, we shall refer to it as the *metric* contribution.
That is, we invent a new numerical species, the metric, containing the
collective non-Newtonian gravitational effects due to all physical species. As
demonstrated in the paper on
":doc:`Fully relativistic treatment of light neutrinos in 𝘕-body simulations</publications>`",
this metric species might be numerically realized as a (fictitious) linear
energy density field, the *Newtonian* gravity from which implements exactly
the missing general relativistic corrections!

.. note::

   For the metric species to be able to supply the correct force, the entire
   simulation must be performed in a particular gauge; the *N*-body gauge.
   That is, initial conditions for non-linear species as well as linear input
   during the simulation must all be in this gauge. This is the default (and
   only) mode of CO\ *N*\ CEPT. Note that all outputs are similarly in this
   gauge, including linear (CLASS) power spectra. Direct comparison to output
   from other *N*-body codes (which usually do not define a gauge at all) is
   perfectly doable, as the choice of gauge only becomes aparrent at very
   large scales.

To finally run a simulation which include the gravitational effects from
photons and neutrinos in their entirety, run

.. code-block:: bash

   ./concept -p params/tutorial -c '_lin = "photons, massless neutrinos, metric"'

Replotting, you should see a much better behaved simulation power spectrum.



.. raw:: html

   <h3>Combining species</h3>

If you've read along in the terminal output during the simulations, you may
have noticed that the energy density, :math:`\varrho`, of each of the linear
species are realized in turn. We can save some time and memory by treating all
linear species as a single, collective component. To specify this, we would
normally write e.g.

.. code-block:: python3

   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
       # Linear fluid component
       {
           'species'        : 'photons + massless neutrinos + metric',
           'gridsize'       : _size,
           'boltzmann order': -1,
       },
   ]

Using our clever parameter file however, we may of course specify this
directly at the command-line using

.. code-block:: bash

   ./concept -p params/tutorial -c '_lin = "photons + massless neutrinos + metric"'

This idea of combining species is embraced fully by CO\ *N*\ CEPT. As such,
the species ``'photons + massless neutrinos'`` may be collectively referred to
simply as ``'radiation'``. Thus,

.. code-block:: bash

   ./concept -p params/tutorial -c '_lin = "radiation + metric"'

works just as well. You should run one of the above and check that you obtain
the same result as before.

You are in fact already very familiar with the idea of combining species, as
``'matter'`` really means ``'baryons + cold dark matter'``.

.. tip::

   When performing simulaitons in a cosmology without massless neutrinos,
   specifying ``'photons + massless neutrinos'`` as the species of a component
   will produce an error. However, specfying ``'radiation'`` is always safe,
   as this dynamically maps to the set of all radiation species present in the
   current cosmology, whatever this may be. Similarly, ``'matter'`` is safe to
   use even in a cosmology without



.. raw:: html

   <h3>Achieving perfection</h3>

Though the final simulation power spectrum indeed appear well behaved at large
scales, you might not --- after all this effort --- be happy about it
disagreeing with linear theory at the 0.1% level. This remaining disagreement
stems from numerical inaccuracy in the simulation, which we may remedy by
lowering the time step size.

The time step size is limited by a set of conditions/limiters, each classified
as either a 'background' or 'non-linear' condition. The maximum allowed time
step size within each category is scaled by the parameter
``Δt_base_background_factor`` and ``Δt_base_nonlinear_factor``, respectively.
At the very linear times and scales with which we are currently operating,
it's a safe bet that the maximum allowed time step is set by one of the
background limiters.

To make the time steps 10 times smaller than usually, place

.. code-block:: python3

   Δt_base_background_factor = 0.1

in the parameter file and rerun the full simulation. Note that this will not
increase the total number of time steps (and thus the computation time) by a
factor of 10, as the time step is periodically increased (though always in
accordance to ``Δt_base_background_factor`` and ``Δt_base_nonlinear_factor``).

With this last tweak, the simulation power spectrum should agree with linear
theory far better than 0.1%, at the largest scales. Incidentally, you may
increase the output time all the way to :math:`a = 1` while retaining
excellent agreement with linear theory. Keeping the time steps small, such
simulations take a long time however.



Massive neutrinos
.................

The previous subsection demonstrated how simulations of matter can be made to
agree extremely well with linear theory at linear scales, if we include the
gravitational contributions from the otherwise missing species, which were
treated linearly. It did this by comparing the simulated power spectrum
directly to the linear one, for the same cosmology.

With confidence in the strategy of including linear species, let's now look at
the relative difference in matter power between two separate cosmologies, with
and without the inclusion of linear species. As dividing one simulated power
spectrum by another cancels out much of the numerical noise, this time we can
obtain high accuracy without using any of the special tricks from the previous
subsection.



.. raw:: html

   <h3>Adding massive neutrinos to the background cosmology</h3>

We wish to compute the effect on the matter power spectrum caused by
neglecting the fact that neutrinos really do have mass. For this purpose, we
shall make use of the below parameter file:

.. code-block:: python3

   # Non-parameter helper variable used to control the size of the simulation
   _size = 128

   # Input/output
   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
   ]
   for _species in _lin.split(','):
       if not _species:
           continue
       initial_conditions.append(
           # Linear fluid component
           {
               'species'        : _species.strip(),
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': paths['output_dir'] + '/' + basename(paths['params'])
   }
   output_bases = {
       'powerspec': f'powerspec_mass={_mass}eV_lin={_lin}'.replace(' ', ''),
   }
   output_times = {
       'powerspec': 1,
   }
   powerspec_select = {
       'matter': {'data': True, 'plot': False},
   }

   # Numerical parameters
   boxsize = 2*Gpc

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27 - Ων
   a_begin = 0.01
   class_params = {
       # Disable massless neutrinos
       'N_ur': 0,
       # Add 3 massive neutrinos of equal mass
       'N_ncdm'  : 1,
       'deg_ncdm': 3,
       'm_ncdm'  : max(_mass/3, 1e-100),  # Avoid exact value of 0.0
   }

   # Physics
   select_forces = {
       'particles': {'gravity': ('p3m', 2*_size)},
       'fluid'    : {'gravity': 'pm'},
   }

   # Non-parameter helper variables which should
   # be supplied as command-line arguments.
   _mass = 0  # Sum of neutrino masses in eV
   _lin = ''  # Linear species to include

You may want to save this and get a simulation going (without supplying any
command-line parameters, for now) while you read on.

The new elements appearing in the parameter file are:

- The ``class_params`` parameter has been added. Items defined within
  ``class_params`` are passed onto CLASS and are thus used for the background
  and linear computations. That is, ``class_params`` is used to change the
  cosmology used within the CO\ *N*\ CEPT simulation away from the default
  cosmology as defined by CLASS.

  As for CO\ *N*\ CEPT itself, a vast number of CLASS parameters exist. The
  best source for exploring these is probably the
  `explanatory.ini <https://github.com/lesgourg/class_public/blob/master/explanatory.ini>`_
  example CLASS parameter file, which also lists default values.

  .. caution::

     As :math:`H_0` (``H0``), :math:`\Omega_{\text{b}}` (``Ωb``) and
     :math:`\Omega_{\text{cdm}}` (``Ωcdm``) already exist as stand-alone
     CO\ *N*\ CEPT parameters, these should never be supplied explicitly to
     ``class_params``.

  Of interest to us now are ``'N_ur'`` and ``'N_ncdm'``; the number of
  **u**\ n\ **r**\ elativistic species (massless neutrinos) and
  **n**\ on-\ **c**\ old **d**\ ark **m**\ atter species (massive neutrinos).
  In the above parameter specifications we switch out the default use of
  massless neutrinos with one (``'N_ncdm': 1``) 3-times degenerate
  (``'deg_ncdm': 3``) neutrino, which really amounts to three separate
  neutrinos but all of the same mass (``'m_ncdm'``).

- Besides ``_lin``, another command-line parameter ``_mass`` is now in play.
  This is the sum of neutrino masses :math:`\sum m_\nu`, in eV. As we have
  three neutrinos of equal mass, the neutrino mass ``'m_ncdm'`` is set to
  ``_mass/3``.

  We can in fact obtain massless neutrinos in CLASS using the 'ncdm' species
  by setting ``'m_ncdm'`` to zero. However, a switch in CLASS detects if this
  mass is exactly zero and  changes the value noticeably. To avoid this, we
  ensure that a value of ``0.0`` gets replaced by a tiny but non-zero number
  (here :math:`10^{-100}`).

- As you may gather from the name 'non-cold dark matter', the massive
  neutrinos behave like unrelativistic dark matter during most if not all of
  the simulation time span (unless ``_mass`` is set very low). When specifying
  the amount of dark matter in the cosmology, one may then choose to state
  :math:`\Omega_{\text{cdm}} + \Omega_\nu` instead of
  :math:`\Omega_{\text{cdm}}` alone. Since :math:`\Omega_\nu` is implicitly
  fixed by the choice of neutrino masses, this means that
  :math:`\Omega_{\text{cdm}}` can no longer be chosen freely. Rather, if we
  want the total dark matter energy density parameter to equal e.g. 0.27,
  :math:`\Omega_{\text{cdm}} + \Omega_{\nu} = 0.27`, we must specify
  ``Ωcdm = 0.27 - Ων``, as is done in the above parameter file. Just like
  ``h`` is automatically inferred from ``H0``, so is ``Ων`` automatically
  inferred from ``class_params``. As this latter inference is non-trivial, the
  resulting ``Ων`` is written to the terminal at the beginning of the
  simulation.

Once the first simulation --- with a cosmology including three neutrinos of
zero mass --- is done, run a simulation with e.g.
:math:`\sum m_\nu = 0.1\,\text{eV}`. Assuming the parameter file is stored as
the usual ``params/tutorial``:

.. code-block:: bash

    ./concept -p params/tutorial -c '_mass = 0.1'

With both simulations done, we can plot their relative power spectrum. To do
this, you should make use of the following script:

.. code-block:: python3

   import glob, os, re
   import numpy as np
   import matplotlib.pyplot as plt

   this_dir = os.path.dirname(os.path.realpath(__file__))
   P_sims, P_lins = {}, {}
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*')):
       match = re.search(r'powerspec_mass=(.+)eV_lin=(.*)_a=[\d.]+$', filename)
       if not match:
           continue
       mass = float(match.group(1))
       lin = match.group(2).replace(',', ', ').replace('+', ' + ')
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       P_sims[mass, lin] = P_sim[mask]
       P_lins[mass] = P_lin[mask]
   k = k[mask]
   for (mass, lin), P_sim in P_sims.items():
       if not mass:
           continue
       plt.semilogx(k, (P_sim/P_sims[0, lin] - 1)*100,
           '-' if lin else '--',
           label=f'simulation: mass = {mass} eV, lin = {lin}',
       )
       P_lin = P_lins.pop(mass, None)
       if P_lin is None:
           continue
       plt.semilogx(k, (P_lin/P_lins[0] - 1)*100, 'k--',
           label=f'linear: mass = {mass} eV')
   plt.legend()
   plt.xlim(k[0], k[-1])
   plt.xlabel(r'$k$ $[\mathrm{Mpc}^{-1}]$')
   plt.ylabel(r'$P_{\Sigma m_\nu > 0}/P_{\Sigma m_\nu = 0}-1\,[\%]$')
   plt.tight_layout()
   plt.savefig(f'{this_dir}/plot.png')

As usual, to run the script, save it as e.g. ``output/tutorial/plot.py`` and
invoke

.. code-block:: bash

   ./concept -m output/tutorial/plot.py

The resulting ``output/tutorial/plot.png`` should show that letting the
neutrinos have mass results in a few percent suppression of the matter power
spectrum. At intermediary :math:`k` the simulation and linear relative power
spectra agrees, whereas they do not for the smallest and largest :math:`k`.
In the case of large :math:`k`, you should see that the non-linear solution
forms a trough below the linear one, before rising up above it near the
largest :math:`k` shown. This is the well-known non-linear supression dip,
the low-:math:`k` end of which marks the beginning of the truly non-linear
regime. We thus trust the simulated results at the high-:math:`k` end of the
plot, while we trust the linear results at the low-:math:`k` end.



.. raw:: html

   <h3>Adding gravitation from massive neutrinos</h3>

The hope is now to be able to correct the simulated relative power spectrum at
low :math:`k` by including the missing species to the simulation, without this
altering the high-:math:`k` behavior. Besides ``'massive neutrinos'``, we
should not forget about ``'photons'`` and the ``'metric'``. Note that
``'massive neutrinos'`` are not considered part of ``'radiation'``. We can
however just write ``'neutrinos'``, as this refers to all neutrinos (massive
('ncdm') as well as massless ('ur')) present in the cosmology. To rerun both
cosmologies with all linear species included, we might call ``concept`` within
a Bash for-loop:

.. code-block:: bash

   for mass in 0 0.1; do
       ./concept \
           -p params/tutorial \
           -c "_mass = $mass" \
           -c '_lin = "photons + neutrinos + metric"'
   done

Once completed, redo the plot. You should find that including the linear
species did indeed correct the large-scale behavior while leaving the
small-scale behavior intact.



.. raw:: html

   <h3>Tweaking the CLASS computation</h3>

Though better agreement with linear theory is achieved after the inclusion of
the linear species, the plot also shows that this inclusion leads to a less
smooth relative spectrum. The added noise stems from the massive neutrinos,
the evolution of which is not solved perfectly by CLASS. A large set of
general and massive neutrino specifc CLASS precision parameters exist, which
can remedy this problem.

Here we shall look at just one such CLASS parameter; ``'evolver'``. This sets
the ODE solver to be used by CLASS, and may be either ``0`` (Runge-Kutta
Cash-Karp) or ``1`` (ndf15, the default). For high-precision CLASS
computations used for *N*-body simulations, it is generally preferable to
switch to the Runge-Kutta solver. To do this, just add

.. code-block:: python3

   # Use the Runge-Kutta evolver
   'evolver': 0,

to the ``class_params`` in the parameter file.

With this change, rerun the two simulations with linear species included (you
may also rerun all four simulations, but the matter-only ones are hardly
affected by the change to ``'evolver'``). After replotting, the simulated
relative power spectrum should have been smoothed out at low :math:`k`,
showing excellent agreement with the linear prediction.



Dynamical dark energy
.....................
*Under construction!*



Decaying cold dark matter
.........................
*Under construction!*



.. _nonlinear_massive_neutrinos:

Non-linear massive neutrinos
............................
*Under construction!*

