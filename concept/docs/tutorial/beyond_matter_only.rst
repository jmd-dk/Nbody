Beyond matter-only simulations
------------------------------
If you've followed this tutorial so far, you're now fully capable of using
CO\ *N*\ CEPT for running matter-only simulations. More exotic simulations ---
taking additional species into account --- are also possible, as this section
will demonstrate.

To assess the effects resulting from including non-matter species, we shall
perform a slew of simulations throughout this section. As most effects are
small, we shall also learn how to improve the precision in different
circumstances, introducing several new parameters. If you have no interest in
any of this, feel free to skip this section, as it's rather lengthy and more
technical than the previous sections of this tutorial.

Each subsection below tackles a separate species/subject. Though you might be
interested in only one of these, it's highly recommended to go through them
in order, as they build upon each other. Though each subsection deals with
a particular species in isolation, these may be mixed and matched at will when
running simulations.

.. contents::
   :local:
   :depth: 1



.. _radiation:

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
equations of linear perturbation theory. For details on how the linear
perturbations are applied to the *N*-body particles during the simulation, we
refer to the paper on
":doc:`Fully relativistic treatment of light neutrinos in 𝘕-body simulations </publications>`".

We begin our exploration by performing a standard matter-only simulation, as
specified by the below parameter file:

.. code-block:: python3
   :caption: params/tutorial :math:`\,` (radiation)
   :name: params-radiation
   :emphasize-lines: 12-22, 36-43, 53

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
           # Linear fluid component(s)
           {
               'species'        : _species,
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': paths['output_dir'] + '/' + basename(paths['params'])
   }
   output_bases = {
       'powerspec': f'powerspec{_lin=}'
           .replace(' ', '').replace('"', '').replace("'", ''),
   }
   output_times = {
       'powerspec': 2*a_begin,
   }

   # Numerical parameters
   boxsize = 4*Gpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'pm' :   _size,
               'p3m': 2*_size,
           },
       },
   }

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.01

   # Non-parameter helper variable used to specify linear components.
   # Should be supplied as a command-line parameter.
   _lin = ''

As usual, save the parameters as e.g. ``params/tutorial``.

.. note::
   Before running simulations, it's best to ensure that the output directory
   (``output/tutorial``) is empty (or non-existent), so that old output does
   not get mixed in with the new. The various plotting scripts of this
   tutorial may not function correctly if run from a directory containing
   "old" output.

With a clean ``output/tutorial`` directory, run the simulation:

.. code-block:: bash

   ./concept -p params/tutorial

possibly with the addition of ``-n 4`` or some other number of processes.

.. note::
   The remainder of this tutorial leaves out explicit mention of the ``-n``
   option to ``concept`` invocations. Please add whatever number of processes
   you would like yourself.

A relatively large number of particles :math:`N = 192^3` is used in order to
increase the precision of the simulation. Our goal is to investigate the
effects from radiation perturbations, which are most pronounced at very large
scales and early times --- hence the large ``boxsize`` and early output time.
The unfamiliar parameter specifications will be explained in due time.

.. note::
   Do not fret if :ref:`the above parameter file <params-radiation>` --- and
   those to come further down in this section --- looks complicated. They
   *are* more complicated than normal parameter files, because here a single
   parameter file is designed to be used for several different simulations,
   necessitating parameter definitions being dependent on various flags.



.. raw:: html

   <h3>The hunt for high precision</h3>

Investigating the resulting ``output/tutorial/powerspec_lin=_a=0.02.png`` you
should see a familiar looking simulation power spectrum: decently looking at
intermediary :math:`k`, inaccurate at small :math:`k` and with obvious
numerical artefacts at large :math:`k`. As we are interested in fine details
at low :math:`k`, we need to improve the precision here. We can do so by
adding

.. code-block:: python3

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
   :caption: output/tutorial/plot.py :math:`\,` (radiation)
   :name: plot-radiation

   import glob, os, re
   import numpy as np
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   P_sims = {}
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*'), key=os.path.getmtime):
       matches = re.findall(r'(?=_(.*?)=(.*?)_)', filename)
       if not matches or filename.endswith('.png'):
           continue
       for var, val in matches:
           exec(f'{var} = "{val}"')
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       P_sims[lin] = P_sim[mask]
   P_lin = P_lin[mask]
   k = k[mask]

   # Plot
   fig, axes = plt.subplots(2, sharex=True)
   linestyles = ['-', '--', ':', '-.']
   for lin, P_sim in P_sims.items():
       linestyle = linestyles[
           sum(np.allclose(line.get_ydata()[:30], P_sim[:30], 1e-3) for line in axes[0].lines)
           %len(linestyles)
       ]
       axes[0].loglog(k, P_sim, linestyle, label=f'simulation: {lin = }')
       axes[1].semilogx(k, (P_sim/P_lin - 1)*100, linestyle)
   axes[0].loglog(k, P_lin, 'k--', label='linear', linewidth=1)
   axes[1].semilogx(k, (P_lin/P_lin - 1)*100, 'k--', linewidth=1)
   k_max = 0.5*k[-1]
   axes[0].set_xlim(k[0], k_max)
   axes[0].set_ylim(0.95*min(P_lin[k < k_max]), 1.05*max(P_lin[k < k_max]))
   axes[1].set_ylim(-1, 1)
   axes[1].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   axes[0].set_ylabel(r'$P\, [\mathrm{Mpc}^3]$')
   axes[1].set_ylabel(r'$P_{\mathrm{simulation}}/P_{\mathrm{linear}} - 1\, [\%]$')
   axes[0].legend(fontsize=9)
   axes[0].tick_params('x', direction='inout', which='both')
   axes[1].set_zorder(-1)
   fig.tight_layout()
   fig.subplots_adjust(hspace=0)
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Save the script as e.g. ``output/tutorial/plot.py`` and run it using

.. code-block:: bash

   ./concept -m output/tutorial/plot.py

This will produce ``output/tutorial/plot.png``, where the bottom panel shows
the relative error between the simulated power spectrum and that computed
using purely linear theory. They should agree to within a percent at the
lowest :math:`k`. At higher :math:`k` the agreement is worse. Though this can
be remedied by increasing the resolution of the simulation (i.e. by increasing
``_size``), we shall not do so here, as we focus on the lower :math:`k` only.

The power spectra outputted by the simulation are binned using a linear bin
size in :math:`k`. This is usually desirable, though higher precision at the
lowest :math:`k` can be achieved by leaving out this binning. The bin size
--- among many other specifics --- is controlled by the ``powerspec_options``
parameter. Disabling the binning can be achieved by setting the bin size
to ``0``. Add

.. code-block:: python3

   powerspec_options = {
       'binsize': 0,
   }

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
       'matter': {'data': True, 'linear': True, 'plot': False},
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
collection of spatially fixed grids, storing the energy density, momentum
density, etc. This is referred to as the *fluid representation* --- as opposed
to the *particle representation* --- and is generally preferable for
non-linear components. To represent the photon component as a fluid, we
specify ``'gridsize'`` in place of ``'N'``, where ``'gridsize'`` is the number
of grid cells along each dimension, for the cubic fluid grids. Finally,
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
  still act with a force on other, non-linear components during
  the simulation.

.. note::
   Though higher Boltzmann orders are well-defined, the largest Boltzmann
   order currently implemented in CO\ *N*\ CEPT is :math:`1`.

We shall look into Boltzmann orders different from :math:`-1` in the
subsection on
:ref:`non-linear massive neutrinos <nonlinear_massive_neutrinos>`. For now we
shall keep to the purely linear case of Boltzmann order :math:`-1`.

For fluid components (whether linear or not), the only sensible gravitational
method is that of PM. Looking at :ref:`the parameter file <params-radiation>`,
we see that a grid size of ``_size`` is specified for ``'pm'``. This *must*
match the *fluid grid size*, i.e. the size specified for ``'gridsize'`` for
the fluid component in ``initial_conditions``, so that the geometry of the
grid(s) internal to the fluid component matches that of the potential grid.
Now the simulation then have *two* potential grids; one for P³M self-gravity
of the matter particles (of grid size ``2*_size``) and one for PM one-way
gravity from the linear photon fluid to the matter particles (of grid
size ``_size``).

.. note::
   As the PM grid size has to match the fluid grid size, you do in fact not
   need to specify the ``'pm'`` item of ``potential_options`` in
   :ref:`the parameter file <params-radiation>`. You still need to have the
   explicit specifications of ``'gridsize'``, ``'gravity'`` and ``'p3m'``,
   though. Specifying simply ``potential_options = 2*_size`` sets the P³M
   *and* PM grid size to ``2*_size``, which fails for our fluid with grid
   size ``_size``.

The parameter file has already been set up to include optional linear fluid
components, using the ``_lin`` command-line parameter. To perform a simulation
with the inclusion of linear photons, run

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_lin = 'photons'"

.. tip::
   Note that the ``_lin`` helper variable is defined at the bottom of the
   parameter file to have an empty value (leading to no linear species being
   included). As this is placed after all actual parameters, this defines a
   default value which is used when ``_lin`` is not given as a command-line
   parameter. As ``_size`` is defined at the top, supplying ``_size`` as a
   command-line parameter will have no effect.

Now redo the plot, and the results of both the matter-only and the matter plus
photon simulation should appear. The plot will show that --- sadly ---
including the photons does *not* lead to better large-scale behaviour.

CO\ *N*\ CEPT delegates all linear (and background) computations to the
`CLASS <http://class-code.net/>`_ code. Though we have specified :math:`H_0`,
:math:`\Omega_{\text{b}}` and :math:`\Omega_{\text{cdm}}` in the parameter file,
many cosmological parameters are still left unspecified. Here the default CLASS
parameters are used, which in addition to baryons, cold dark matter and photons
also contain massless neutrinos. With our hope renewed, let's run a simulation
which includes both linear photons and linear massless neutrinos:

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_lin = 'photons, massless neutrinos'"

Redoing the plot, we discover that including the neutrinos made the
disagreement between the simulation and linear theory even larger!

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
":doc:`Fully relativistic treatment of light neutrinos in 𝘕-body simulations </publications>`",
this metric species might be numerically realized as a (fictitious) linear
energy density field, the *Newtonian* gravity from which implements exactly
the missing general relativistic corrections.

.. note::
   For the metric species to be able to supply the correct force, the entire
   simulation must be performed in a particular gauge; the *N*-body gauge.
   That is, initial conditions for non-linear species as well as linear input
   during the simulation must all be in this gauge. This is the default (and
   only) mode of CO\ *N*\ CEPT. Note that all outputs are similarly in this
   gauge, including non-linear (CO\ *N*\ CEPT) and linear (CLASS) power
   spectra. Direct comparison to output from other *N*-body codes (which
   usually do not define a gauge at all) is perfectly doable, as the choice of
   gauge only becomes aparrent at very large scales.

To finally run a simulation which includes the gravitational effects from
photons and neutrinos in their entirety, run

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_lin = 'photons, massless neutrinos, metric'"

Re-plotting, you should see a much better behaved simulation power spectrum at
large scales, agreeing with linear theory to well within 0.1%.



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

Using :ref:`our clever parameter file <params-radiation>` however, we may
specify this directly at the command-line using

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_lin = 'photons + massless neutrinos + metric'"

This idea of combining species is embraced fully by CO\ *N*\ CEPT. As such,
the species ``'photons + massless neutrinos'`` may be collectively referred to
simply as ``'radiation'``. Thus,

.. code-block:: bash

   ./concept -p params/tutorial -c "_lin = 'radiation + metric'"

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
   use even in a cosmology without e.g. cold dark matter.



Massive neutrinos
.................
The previous subsection demonstrated how simulations of matter can be made to
agree extremely well with linear theory at linear scales, if we include the
gravitational contributions from the otherwise missing species, which were
treated linearly. We did this by comparing the simulated power spectrum
directly to the linear one, for the same cosmology.

With confidence in the strategy of including linear species, let's now look at
the relative difference in matter power between two separate cosmologies, with
and without the inclusion of linear species. As dividing one simulated power
spectrum by another cancels out much of the numerical noise, this time we can
obtain high accuracy without using any of the tricks from the previous
subsection.

Our aim shall be to compute the effect on the matter power spectrum caused by
neglecting the fact that neutrinos really do have mass, albeit small. If you
wish to study the underlying theory as well as the implementation in
CO\ *N*\ CEPT, we refer to the paper on
":doc:`Fully relativistic treatment of light neutrinos in 𝘕-body simulations </publications>`".



.. raw:: html

   <h3>Adding massive neutrinos to the background cosmology</h3>

To compute the effect on the matter power spectrum caused by neglecting the
fact that neutrinos do have some mass, we shall make use of the below
parameter file:

.. code-block:: python3
   :caption: params/tutorial :math:`\,` (massive neutrinos)
   :name: params-massive-neutrinos
   :emphasize-lines: 50, 52-59, 63

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
           # Linear fluid component(s)
           {
               'species'        : _species,
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': paths['output_dir'] + '/' + basename(paths['params'])
   }
   output_bases = {
       'powerspec': f'powerspec{_mass=}{_lin=}'
           .replace(' ', '').replace('"', '').replace("'", ''),
   }
   output_times = {
       'powerspec': 1,
   }
   powerspec_select = {
       'matter': {'data': True, 'linear': True, 'plot': False},
   }

   # Numerical parameters
   boxsize = 2*Gpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'p3m': 2*_size,
           },
       },
   }

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

   # Non-parameter helper variables which should
   # be supplied as command-line parameters.
   _mass = 0   # Sum of neutrino masses in eV
   _lin  = ''  # Linear species to include

You may want to save this as e.g. ``params/tutorial`` and get a simulation
going --- of course using

.. code-block:: bash

    ./concept -p params/tutorial

--- while you read on.

The new elements appearing in the parameter file are:

- The ``class_params`` parameter has been added. Items defined within
  ``class_params`` are passed onto CLASS and are thus used for the background
  and linear computations. That is, ``class_params`` is used to change the
  cosmology used within the CO\ *N*\ CEPT simulation away from the default
  cosmology as defined by CLASS.

  As with CO\ *N*\ CEPT itself, a vast number of CLASS parameters exist. The
  best source for exploring these is probably the
  `explanatory.ini <https://github.com/lesgourg/class_public/blob/master/explanatory.ini>`_
  example CLASS parameter file, which also lists default values.

  .. caution::
     As :math:`H_0` (``H0``), :math:`\Omega_{\text{b}}` (``Ωb``) and
     :math:`\Omega_{\text{cdm}}` (``Ωcdm``) already exist as CO\ *N*\ CEPT
     parameters, these should never be specified explicitly within
     ``class_params``.

  Of interest to us now are ``'N_ur'`` and ``'N_ncdm'``; the number of
  **u**\ ltra-\ **r**\ elativistic species (massless neutrinos) and
  **n**\ on-\ **c**\ old **d**\ ark **m**\ atter species (massive neutrinos).
  In the above parameter specifications we switch out the default use of
  massless neutrinos with one (``'N_ncdm': 1``) 3-times degenerate
  (``'deg_ncdm': 3``) massive neutrino, which really amounts to three separate
  neutrinos but all of the same mass (``'m_ncdm'``).

- Besides ``_lin``, another command-line parameter ``_mass`` is now in play.
  This is the sum of neutrino masses :math:`\sum m_\nu`, in eV. As we have
  three neutrinos of equal mass, the neutrino mass ``'m_ncdm'`` is set to
  ``_mass/3``.

  We can in fact obtain massless neutrinos in CLASS using the 'ncdm' species
  by setting ``'m_ncdm'`` to zero. To avoid potential safeguards employed by
  CLASS, we ensure that a specified value of exactly ``0.0`` gets replaced by
  a tiny but non-zero number (here :math:`10^{-100}`).

- As you may gather from the name 'non-cold dark matter', the massive
  neutrinos behave like unrelativistic dark matter during most if not all of
  the simulation time span (unless ``_mass`` is set very low). When specifying
  the amount of dark matter in the cosmology, one may then choose to state
  :math:`\Omega_{\text{cdm}} + \Omega_\nu` instead of
  :math:`\Omega_{\text{cdm}}` alone. Since :math:`\Omega_\nu` is implicitly
  fixed by the choice of neutrino masses (and a few other parameters), this
  means that :math:`\Omega_{\text{cdm}}` can no longer be chosen freely.
  Rather, if we want the total dark matter energy density parameter to equal
  e.g. 0.27, :math:`\Omega_{\text{cdm}} + \Omega_{\nu} = 0.27`, we must
  specify ``Ωcdm = 0.27 - Ων``, as is done in the
  :ref:`above parameter file <params-massive-neutrinos>`. Just like ``h`` is
  automatically inferred from ``H0``, so is ``Ων`` automatically inferred
  from ``class_params``. As this latter inference is non-trivial, the
  resulting ``Ων`` is written to the terminal at the beginning of the
  simulation.

Once the first simulation --- with a cosmology including three "massive"
neutrinos of zero mass --- is done, run a simulation with e.g.
:math:`\sum m_\nu = 0.1\, \text{eV}`:

.. code-block:: bash

    ./concept \
       -p params/tutorial \
       -c "_mass = 0.1"

With both simulations done, we can plot their relative power spectrum. To do
this, you should make use of the following script:

.. code-block:: python3
   :caption: output/tutorial/plot.py :math:`\,` (massive neutrinos)
   :name: plot-massive-neutrinos

   import glob, os, re
   import numpy as np
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   P_sims, P_lins = {}, {}
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*'), key=os.path.getmtime):
       matches = re.findall(r'(?=_(.*?)=(.*?)_)', filename)
       if not matches or filename.endswith('.png'):
           continue
       for var, val in matches:
           try:
               exec(f'{var} = {val}')
           except:
               exec(f'{var} = "{val}"')
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       P_sims[mass, lin] = P_sim[mask]
       P_lins[mass     ] = P_lin[mask]
   k = k[mask]

   # Plot
   fig, ax = plt.subplots()
   for (mass, lin), P_sim in P_sims.items():
       if not mass:
           continue
       mass_nonzero = mass
       P_sim_ref = P_sims.get((0, lin))
       if P_sim_ref is not None:
           ax.semilogx(k, (P_sim/P_sim_ref - 1)*100,
               label=f'simulation: {mass = } eV, {lin = }',
           )
   ax.semilogx(k, (P_lins[mass_nonzero]/P_lins[0] - 1)*100, 'k--',
       label='linear', linewidth=1,
   )
   ax.set_xlim(k[0], k[-1])
   ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   ax.set_ylabel(
       rf'$P_{{\Sigma m_\nu = {mass_nonzero}\, \mathrm{{eV}}}}'
       rf'/P_{{\Sigma m_\nu = 0}} - 1\, [\%]$'
   )
   ax.legend(fontsize=9)
   fig.tight_layout()
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

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
largest :math:`k` shown. This is the well-known non-linear suppression dip,
the low-:math:`k` end of which marks the beginning of the truly non-linear
regime. We thus trust the simulated results at the high-:math:`k` end of the
plot, while we trust the linear results at the low-:math:`k` end.



.. raw:: html

   <h3>Adding gravitation from massive neutrinos</h3>

The hope is now to be able to correct the simulated relative power spectrum at
low :math:`k` by including the missing species to the simulation, without this
altering the high-:math:`k` behaviour. Besides ``'massive neutrinos'``, we
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
           -c "_lin = 'photons + neutrinos + metric'"
   done

Once completed, redo the plot. You should find that including the linear
species did indeed correct the large-scale behaviour while leaving the
small-scale behaviour intact.



.. raw:: html

   <h3>Tweaking the CLASS computation</h3>

Though better agreement with linear theory is achieved after the inclusion of
the linear species, the plot also shows that this inclusion leads to a less
smooth relative spectrum. The added noise stems from the massive neutrinos,
the evolution of which is not solved perfectly by CLASS. A large set of
general and massive neutrino specific CLASS precision parameters exist, which
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
affected by the change to ``'evolver'``). After re-plotting, the simulated
relative power spectrum should have been smoothed out at low :math:`k`,
showing excellent agreement with the linear prediction.



Dynamical dark energy
.....................
This subsection investigates how to perform simulations where dark energy is
dynamic, specifically using the equation of state
:math:`w(a) = w_0 + (1 - a)w_a`. Beyond just changing the background
evolution, having :math:`w \neq -1` also causes perturbations within the dark
energy, leading to an additional gravitational tug. If you're interested in
the physics of dark energy perturbations as well as their implementation in
CO\ *N*\ CEPT, we refer to the paper on
":doc:`Dark energy perturbations in 𝘕-body simulations </publications>`".



.. raw:: html

   <h3>Cosmological constant <span class="math notranslate nohighlight">\(\Lambda\)</span></h3>

So far this tutorial has mentioned nothing about dark energy, but really it
has been there all along, as a cosmological constant :math:`\Lambda` affecting
the background evolution.

CO\ *N*\ CEPT always assumes the universe to be flat;
:math:`\sum_\alpha \Omega_\alpha = 1`, :math:`\alpha` running over all
species. Written out in the standard cosmologies we have looked at thus far,
this looks like

.. math::

     \Omega_{\text{b}}
   + \Omega_{\text{cdm}}
   + \Omega_{\gamma}
   + \Omega_{\nu}
   + \Omega_{\Lambda}
   = 1\, ,

where :math:`\Omega_{\text{b}}` and :math:`\Omega_{\text{cdm}}` are defined
through the ``Ωb`` and ``Ωcdm`` CO\ *N*\ CEPT parameters, while
:math:`\Omega_{\gamma}` (photons) and :math:`\Omega_{\nu}` (massless and
massive neutrinos) are defined through CLASS parameters (typically,
:math:`\Omega_{\gamma}` is defined implicitly through the CMB temperature
``class_params['T_cmb']`` while :math:`\Omega_{\nu}` is defined implicitly
through the effective number of massless neutrino species
``class_params['N_ur']``, the number of massive neutrino species
``class_params['N_ncdm']`` and ``class_params['deg_ncdm']``, their masses
``class_params['m_ncdm']`` and temperatures ``class_params['T_ncdm']``). The
remaining dark energy density :math:`\Omega_{\Lambda}` is simply chosen as to
ensure a flat universe.

Though :math:`\Lambda` is present, it does not tug on the matter (or anything
else) as it remains completely homogeneous throughout time, which is why we
never need to include it as a linear species.



.. raw:: html

   <h3>Dynamical dark energy</h3>

We can set :math:`w_0` and :math:`w_a` through the CLASS parameters
``'w0_fld'`` and ``'wa_fld'``. In CLASS, the cosmological constant is
implemented as a separate species, rather than as the special case
:math:`w_0 = -1`, :math:`w_a = 0` of the dynamical dark energy species (in
CLASS called 'fld' for dark energy **fl**\ ui\ **d**). To disable the
cosmological constant, set the ``'Omega_Lambda'`` CLASS parameter to ``0``. In
total, specifying dynamical dark energy could then look like

.. code-block:: python3

   class_params = {
       # Disable cosmological constant
       'Omega_Lambda': 0,
       # Dark energy fluid parameters
       'w0_fld': -0.7,
       'wa_fld': 0,
   }

To test the effect on the matter from switching from :math:`\Lambda` to
dynamical dark energy (here :math:`w_0 = -1\, \rightarrow\, w_0 = -0.7`), we
shall make use of the following parameter file, which you should save as e.g.
``params/tutorial``:

.. code-block:: python3
   :caption: params/tutorial :math:`\,` (dynamical dark energy)
   :name: params-dynamical-dark-energy
   :emphasize-lines: 52-60, 62-65, 69

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
           # Linear fluid component(s)
           {
               'species'        : _species,
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': paths['output_dir'] + '/' + basename(paths['params'])
   }
   output_bases = {
       'powerspec': f'powerspec{_de=}{_lin=}'
           .replace(' ', '').replace('"', '').replace("'", ''),
   }
   output_times = {
       'powerspec': 1,
   }
   powerspec_select = {
       'matter': {'data': True, 'linear': True, 'plot': False},
   }

   # Numerical parameters
   boxsize = 3*Gpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'p3m': 2*_size,
           },
       },
   }

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.1
   if _de != 'Lambda':
       class_params = {
           # Disable cosmological constant
           'Omega_Lambda': 0,
           # Dark energy fluid parameters
           'w0_fld' : -0.7,
           'wa_fld' : 0,
           'cs2_fld': 0.01,
       }

   # Simulation options
   Δt_base_background_factor = 2
   Δt_base_nonlinear_factor  = 2
   N_rungs                   = 1

   # Non-parameter helper variables which should
   # be supplied as command-line parameters.
   _de  = 'Lambda'  # Type of dark energy
   _lin = ''        # Linear species to include

The parameter file is set up to use :math:`\Lambda` by default, while
dynamical dark energy is enabled by supplying ``-c "_de = 'dynamical'"``. One
can also supply ``-c "_de = 'Lambda'"`` to explicitly select :math:`\Lambda`.
Perform a simulation using both types of dark energy using

.. code-block:: bash

   for de in Lambda dynamical; do
       ./concept \
           -p params/tutorial \
           -c "_de = '$de'"
   done

The parameter specifications ``Δt_base_background_factor = 2`` and
``Δt_base_nonlinear_factor = 2`` double the allowable time step size, while
``N_rungs = 1`` effectively disables the adaptive time stepping. In addition,
we start the simulation rather late at ``a_begin = 0.1``, as the effects from
dark energy show up only at late times. All of this is just to speed up the
simulations, as we do not require excellent precision.

.. note::
   The (global) time step size during the simulation is limited by a set of
   conditions/limiters, each classified as either a 'background' or a
   'non-linear' condition. The maximum allowed time step size within each
   category is scaled by the ``Δt_base_background_factor`` and
   ``Δt_base_nonlinear_factor`` parameter, respectively.

.. note::
   The adaptive particle time stepping is a feature enabled by default when
   using the P³M method, which assigns separate time step sizes to the
   different particles, allowing for small time steps in dense regions and
   large time steps in less dense regions, achieving both accuracy and
   numerical efficiency. The possible particle time step sizes are exactly the
   base time step size divided by :math:`2^n`, where
   :math:`n \in \{0, 1, 2, \dots\}` is referred to as the *rung*. The number
   of available rungs (and thus the minimum allowed particle time step) is
   determined through the ``N_rungs`` parameter. With ``N_rungs = 1``, all
   particles are kept fixed at rung 0, i.e. the base time step, and so no
   adaptive time stepping takes place. The distribution of particles across
   all rungs is printed at the start of each time step, for ``N_rungs > 1``.

To make the usual plot of the relative power spectrum --- this time comparing
the matter spectrum within a cosmology with a cosmological constant
(:math:`w = -1`) to one with dynamical dark energy (here :math:`w = -0.7`) ---
we shall make use of the following plotting script:

.. code-block:: python3
   :caption: output/tutorial/plot.py :math:`\,` (dynamical dark energy)
   :name: plot-dynamical-dark-energy

   import glob, os, re
   import numpy as np
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   P_sims, P_lins = {}, {}
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*'), key=os.path.getmtime):
       matches = re.findall(r'(?=_(.*?)=(.*?)_)', filename)
       if not matches or filename.endswith('.png'):
           continue
       for var, val in matches:
           exec(f'{var} = "{val}"')
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       P_sims[de, lin] = P_sim[mask]
       P_lins[de     ] = P_lin[mask]
   k = k[mask]

   # Plot
   fig, ax = plt.subplots()
   linestyles = ['-', '--', ':', '-.']
   for (de, lin), P_sim in P_sims.items():
       if de == 'Lambda':
           continue
       lin_ref = lin
       for lin_ignore in ('darkenergy', 'fld'):
           lin_ref = (lin_ref.replace(lin_ignore, '')
               .replace('++', '+').replace(',,', ',')
               .replace('+,', ',').replace(',+', ',')
               .strip('+,')
           )
       P_sim_ref = P_sims.get(('Lambda', lin_ref))
       label = f'simulation: {lin = }'
       if P_sim_ref is None:
           P_sim_ref = P_sims['Lambda', '']
           label += ' (dynamical sim only)'
       P_rel = (P_sim/P_sim_ref - 1)*100
       linestyle = linestyles[
           sum(np.allclose(line.get_ydata(), P_rel, 5e-3) for line in ax.lines)
           %len(linestyles)
       ]
       ax.semilogx(k, P_rel, linestyle, label=label)
   ax.semilogx(k, (P_lins['dynamical']/P_lins['Lambda'] - 1)*100, 'k--',
       label='linear', linewidth=1,
   )
   ax.set_xlim(k[0], k[-1])
   ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   ax.set_ylabel(r'$P_{\mathrm{dynamical}}/P_{\Lambda} - 1\, [\%]$')
   ax.legend(fontsize=10)
   fig.tight_layout()
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Save this to e.g. ``output/tutorial/plot.py`` and run it using

.. code-block:: bash

   ./concept -m output/tutorial/plot.py

The generated plot should show that the matter power is reduced quite a bit
when switching to using the dynamical dark energy. At large :math:`k`, we see
the usual non-linear suppression dip. At low/linear :math:`k`, the power
suppression is larger in the simulation power spectrum than in the linear one.
This is due to inhomogeneities forming in the dark energy species itself, the
tug on matter we have not incorporated into the simulation. This effect is
enlarged as we :ref:`have specified <params-dynamical-dark-energy>` a low dark
energy sound speed ``'cs2_fld'`` (given in units of the speed of light
squared, :math:`c^2`).



.. raw:: html

   <h3>Adding gravitation from dark energy perturbations</h3>

As usual, the missing gravity can be incorporated into the simulation by
including the missing species in the simulation as a linear component. The
parameter file has once again been set up to be able to do this via the
``_lin`` command-line parameter. To run both cosmologies again, this time
including all linear species, do e.g.

.. code-block:: bash

   for de in Lambda dynamical; do
       lin="radiation + metric"
       [ $de == dynamical ] && lin+=" + dark energy"
       ./concept \
           -p params/tutorial \
           -c "_de = '$de'" \
           -c "_lin = '$lin'"
   done

where 'radiation' includes the photons and massless neutrinos supplied by
CLASS by default.

Notice that we do not include 'dark energy' when running with :math:`\Lambda`,
as here there are no dark energy perturbations.

After re-plotting, you should see that the simulation spectrum now matches
the linear prediction at low :math:`k`.

Though including all species --- i.e. also photons and neutrinos --- is what
should be done for production runs, it can be educational to run with fewer
linear species, to separate out their individual effects on the matter
spectrum. Besides being small, the effects from radiation perturbations should
be very close to identical between the two cosmologies. We thus expect effects
from radiation to be completely negligible for the relative power
spectrum. To test this, perform a simulation with dynamical dark energy,
including only dark energy as a linear species:

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_de = 'dynamical'" \
       -c "_lin = 'dark energy'"

If you now redo the plot, a relative spectrum between the newly run simulation
and the :math:`\Lambda` simulation without any linear species will be added.
You will see that though it's close, it has a bit too much power suppression
at low :math:`k`.

The missing power is not due to the missing radiation, but rather the missing
gravity from the rather large pressure perturbations in the dark energy, which
is accounted for by the fictitious metric species. Performing a simulation
including the metric as well,

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_de = 'dynamical'" \
       -c "_lin = 'dark energy + metric'"

and re-plotting, we see that we indeed achieve the same result as when running
with radiation.

.. note::
   As the metric always contains the total gravitational contribution from
   momentum, pressure and shear perturbations of all species, it is not
   possible to completely separate out the gravitational effects from each
   species using this set-up. For example, the last simulation above *does*
   include some photon and neutrino gravity, since the metric still contains
   contributions from their momentum, pressure and shear perturbations. The
   :math:`\Lambda` simulation with which it is paired up for the plot does
   not, however, as here the simulation is performed without including
   the metric.



Decaying cold dark matter
.........................
This subsection deals with the case of :bolditalic:`d`\ *ecaying* **c**\ old
**d**\ ark **m**\ atter ('dcdm'), in particular the scenario where it decays
to some new form of massless non-interacting radiation; what we might call
**d**\ ark **r**\ adiation or **d**\ ecay **r**\ adiation ('dr'). While
the decay radiation is handled in exactly the same way as was done for other
linear species in the previous subsections, the decaying cold dark matter
itself is simulated using particles. To keep things general, we allow for
having both stable and unstable cold dark matter within the same simulation,
and so we think of dcdm as an entirely new species, separate from the usual,
stable cdm.

If you're interested in the details of the physics of decaying cold dark
matter as well as its implementation in CO\ *N*\ CEPT, we refer to the paper
on
":doc:`Fully relativistic treatment of decaying cold dark matter in 𝘕-body simulations </publications>`".



.. raw:: html

   <h3>Particle-only simulations</h3>

We denote the rate of decay for the new decaying cold dark matter species by
:math:`\Gamma_{\text{dcdm}}`. We further wish to specify the amount of dcdm
today, :math:`\Omega_{\text{dcdm}}`, but as this depends highly on the decay
rate :math:`\Gamma_{\text{dcdm}}`, this is not a good parameter. Instead we
make use of :math:`\widetilde{\Omega}_{\text{dcdm}}`, which we define to be
the energy density parameter that dcdm *would* have had, had
:math:`\Gamma_{\text{dcdm}} = 0` (in which case dcdm and cdm would be
indistinguishable). Finally, rather than specifying
:math:`\Omega_{\text{cdm}}` and :math:`\widetilde{\Omega}_{\text{dcdm}}`
separately, we reparameterise these as the *total* (stable and decaying) cold
dark matter energy density
:math:`(\Omega_{\text{cdm}} + \widetilde{\Omega}_{\text{dcdm}})`, as well as
the fraction of this which is of the decaying kind;

.. math::

   f_{\text{dcdm}} \equiv
         \frac{\widetilde{\Omega}_{\text{dcdm}}}{\Omega_{\text{cdm}}
       + \widetilde{\Omega}_{\text{dcdm}}}\, .

Below you'll find a parameter file set up to run simulations with dcdm, which
you should save as e.g. ``params/tutorial``:

.. code-block:: python3
   :caption: params/tutorial :math:`\,` (decaying cold dark matter)
   :name: params-decaying-cold-dark-matter
   :emphasize-lines: 4-6, 13-17, 26-27, 30-35, 60, 76, 78-83, 90-94, 102-103

   # Non-parameter helper variable used to control the size of the simulation
   _size = 96

   # Non-parameter variables used to control the dcdm cosmology
   _Ω_cdm_plus_dcdm = 0.27  # Total amount of stable and decaying cold dark matter
   _Γ = 80*km/(s*Mpc)       # Decay rate

   # Input/output
   if _combine:
       initial_conditions = [
           # Non-linear (total) matter particles
           {
               'name'   : 'total matter',
               'species': (
                   'baryons + cold dark matter'
                   + (' + decaying cold dark matter' if _frac else '')
               ),
               'N'      : _size**3,
           }
       ]
   else:
       # Assume 0 < _frac < 1
       initial_conditions = [
           # Non-linear baryons and (stable) cold dark matter particles
           {
               'name'   : 'stable matter',
               'species': 'baryons + cold dark matter',
               'N'      : _size**3,
           },
           # Non-linear decaying cold dark matter particles
           {
               'name'   : 'decaying matter',
               'species': 'decaying cold dark matter',
               'N'      : _size**3,
           },
       ]
   for _species in _lin.split(','):
       if not _species:
           continue
       initial_conditions.append(
           # Linear fluid component(s)
           {
               'species'        : _species,
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': paths['output_dir'] + '/' + basename(paths['params']),
   }
   output_bases = {
       'powerspec': f'powerspec_{boxsize=}{_frac=}{_lin=}{_combine=}'
           .replace(' ', '').replace('"', '').replace("'", ''),
   }
   output_times = {
       'powerspec': 1,
   }
   powerspec_select = {
       'total matter': {'data': True, 'linear': True, 'plot': False},
       ('stable matter', 'decaying matter'): ...,
   }

   # Numerical parameters
   boxsize = 1*Gpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'p3m': 2*_size,
           },
       },
   }

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = (1 - _frac)*_Ω_cdm_plus_dcdm
   a_begin = 0.02
   if _frac:
       class_params = {
           # Decaying cold dark matter parameters
           'Omega_ini_dcdm': _frac*_Ω_cdm_plus_dcdm,
           'Gamma_dcdm'    : _Γ/(km/(s*Mpc)),
       }

   # Physics
   select_forces = {
       'particles': {'gravity': 'p3m'},
       'fluid'    : {'gravity': 'pm'},
   }
   if 'lapse' in _lin:
       select_forces.update({
           'decaying matter': 'lapse',
           'total matter'   : 'lapse',
       })

   # Simulation options
   primordial_amplitude_fixed = True

   # Non-parameter helper variables which should
   # be supplied as command-line parameters.
   _lin     = ''    # Linear species to include
   _frac    = 0     # Fraction of total cold dark matter which is decaying
   _combine = True  # Combine decaying and stable matter into a single component?

Begin by running this without any additional command-line parameters;

.. code-block:: bash

   ./concept -p params/tutorial

which performs a standard simulation with just stable matter (baryons and cold
dark matter).

In the parameter file, the dcdm parameters :math:`\Gamma_{\text{dcdm}}`,
:math:`(\Omega_{\text{cdm}} + \widetilde{\Omega}_{\text{dcdm}})` and
:math:`f_{\text{dcdm}}` are called ``_Γ``, ``_Ω_cdm_plus_dcdm`` and
``_frac``, respectively. A rather extreme value of
:math:`\Gamma_{\text{dcdm}} = 80\, \text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1}`
is used, corresponding to a dcdm mean particle lifetime
:math:`1/\Gamma_{\text{dcdm}}` comparable to the age of the universe, meaning
that the majority of the primordial dcdm population has decayed away at
:math:`a = 1`.

To run a simulation where some of the cold dark matter is decaying, say 70%,
specify ``_frac``:

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_frac = 0.7"

This new simulation still consists of just a single particle component, now
with a species of
``'baryons + cold dark matter + decaying cold dark matter'``. The decay is
taken into effect by continuously reducing the mass of each *N*-body particle
according to the decay rate, without changing the particle velocity. As the
component now represents three fundamental species, the effective "*N*-body
decay rate" used is

.. math::

   \Gamma_{\text{b}+\text{cdm}+\text{dcdm}}(a) =
       \frac{\bar{\rho}_{\text{dcdm}}(a)\Gamma_{\text{dcdm}}}{
             \bar{\rho}_{\text{b}}(a)
           + \bar{\rho}_{\text{cdm}}(a)
           + \bar{\rho}_{\text{dcdm}}(a)
       }\, .

To plot the usual relative power spectrum, this time between a cosmology with
and without decaying cold dark matter, make use of the below script:

.. code-block:: python3
   :caption: output/tutorial/plot.py :math:`\,` (decaying cold dark matter)
   :name: plot-decaying-cold-dark-matter

   import glob, os, re
   import numpy as np
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   ks, P_sims, P_lins = {}, {}, {}
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*'), key=os.path.getmtime):
       matches = re.findall(r'(?=_(.*?)=(.*?)_)', filename)
       if not matches or filename.endswith('.png'):
           continue
       for var, val in matches:
           try:
               exec(f'{var} = {val}')
           except:
               exec(f'{var} = "{val}"')
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       ks[boxsize] = k[mask]
       P_sims[boxsize, frac, lin, combine] = P_sim[mask]
       P_lins[boxsize, frac              ] = P_lin[mask]

   # Plot
   fig, ax = plt.subplots()
   linestyles = ['-', '--', ':', '-.']
   colors = {}
   for (boxsize, frac, lin, combine), P_sim in P_sims.items():
       if frac == 0:
           continue
       frac_nonzero = frac
       lin_ref = lin
       for lin_ignore in (*['decayradiation', 'darkradiation', 'dr'], 'lapse'):
           lin_ref = (lin_ref.replace(lin_ignore, '')
               .replace('++', '+').replace(',,', ',')
               .replace('+,', ',').replace(',+', ',')
               .strip('+,')
           )
       P_sim_ref = P_sims.get((boxsize, 0, lin_ref, True))
       if P_sim_ref is None:
           continue
       k = ks[boxsize]
       color, label = colors.get((lin, combine)), None
       if color is None:
           color = colors[lin, combine] = f'C{len(colors)%10}'
           label = f'simulation: {lin = }, {combine = }'
       P_rel = (P_sim/P_sim_ref - 1)*100
       linestyle = linestyles[
           sum(np.allclose(line.get_ydata(), P_rel, 1e-2) for line in ax.lines)
           %len(linestyles)
       ]
       if not combine:
           ylim = ax.get_ylim()
       ax.semilogx(k, P_rel, f'{color}{linestyle}', label=label)
       if not combine:
           ax.set_ylim(ylim)
   label = 'linear'
   for (boxsize, frac), P_lin in P_lins.items():
       if frac == 0:
           continue
       P_lin_ref = P_lins.get((boxsize, 0))
       if P_lin_ref is None:
           continue
       k = ks[boxsize]
       ax.semilogx(k, (P_lin/P_lin_ref - 1)*100, 'k--',
           label=label, linewidth=1,
       )
       label = None
   xdata = np.concatenate([line.get_xdata() for line in ax.lines])
   ax.set_xlim(min(xdata), max(xdata))
   ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   ax.set_ylabel(
       rf'$P_{{f_{{\mathrm{{dcdm}}}} = {frac_nonzero}}}'
       rf'/P_{{f_{{\mathrm{{dcdm}}}} = 0}} - 1\, [\%]$'
   )
   ax.legend(fontsize=8)
   fig.tight_layout()
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Save this script as e.g. ``output/tutorial/plot.py`` and run it using

.. code-block:: bash

   ./concept -m output/tutorial/plot.py

The resulting ``plot.png`` should show prominently the familiar non-linear
suppression dip on top of an already substantial drop in power due to the
decayed matter.



.. raw:: html

   <h3>Decay radiation</h3>

The plot resulting from the first two simulations show a familiar discrepancy
between the linear and non-linear result at low :math:`k`. As usual, we may
try to fix this by including the missing species as linear components during
the simulation:

.. code-block:: bash

   for frac in 0 0.7; do
       ./concept \
           -p params/tutorial \
           -c "_lin = 'photons + neutrinos + metric'" \
           -c "_frac = $frac"
   done

Once the above two simulations are complete, redo the plot. Adding the
photons, neutrinos and metric perturbations supplied about half of the missing
large-scale power needed to reach agreement with the linear prediction.

The remaining missing power should be supplied by further including the decay
radiation, of course only applicable for the dcdm simulation:

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "_lin = 'photons + neutrinos + decay radiation + metric'" \
       -c "_frac = 0.7"

Re-plotting after running the above, you should now see excellent agreement
with the linear result at large scales.

Studying :ref:`the parameter file <params-decaying-cold-dark-matter>`, we see
that the ``'species'`` of the ``'total matter'`` component gets set to
``'baryons + cold dark matter'`` when ``_frac`` equals ``0`` (corresponding to
unset) and ``'baryons + cold dark matter + decaying cold dark matter'``
otherwise. (Do not worry about the case of the variable ``_combine`` being
falsy. We shall make use of this special flag later.) We are used to
``'matter'`` being an alias for ``'baryons + cold dark matter'``, but really
it functions as a stand-in for *all* matter within the given cosmology,
including decaying cold dark matter. Go ahead and replace this needlessly
complicated expression for the ``'species'`` of ``'total matter'`` in
:ref:`the parameter file <params-decaying-cold-dark-matter>` with just
``'matter'``. Likewise, ``'radiation'`` includes not just ``'photons'`` and
(massless) ``'neutrinos'``, but also ``'decay radiation'``, when present. With
the aforementioned change to
:ref:`the parameter file <params-decaying-cold-dark-matter>` in place, try
rerunning both the dcdm and the reference simulation using simply

.. code-block:: bash

   for frac in 0 0.7; do
       ./concept \
           -p params/tutorial \
           -c "_lin = 'radiation + metric'" \
           -c "_frac = $frac"
   done

Updating the plot, we see that the new simulation results are indeed identical
to the previous ones.



.. raw:: html

   <h3>Additional general relativistic effects and multiple particle components</h3>

For most work involving dcdm simulations, the story ends here. For simulations
using very large boxes, or possibly extreme values of :math:`f_{\text{dcdm}}`
and :math:`\Gamma_{\text{dcdm}}`, additional effects from general relativity
show up, which one might wish to include in the simulations.

To begin the exploration of this extreme regime, run the same dcdm and
reference simulation as before, but in a much larger box:

.. code-block:: bash

   for frac in 0 0.7; do
       ./concept \
           -p params/tutorial \
           -c "boxsize = 30*Gpc" \
           -c "_lin = 'radiation + metric'" \
           -c "_frac = $frac"
   done

Rerunning ``plot.py``, we see that having the cold dark matter decay actually
leads to an *in*\ crease in power at very large scales. This increase in power
is replicated by the new dcdm simulation, though not quite enough to match the
linear result.

The continuous decay of all *N*-body particles happen in unison, according to
the set decay rate and the flow of cosmic time. Really though, each particle
should decay away in accordance with their own individually experienced flow
of proper time, which is affected by the local gravitational field. At linear
order, this general relativistic effect may be implemented as a correction
force applied to all decaying particles, with a strength proportional to their
decay rate. This force can be described as arising from a potential, which in
CO\ *N*\ CEPT is implemented as an energy density field from a fictitious
species --- much like the metric species --- called the *lapse* species. For
details on the physics of this lapse potential, see the paper on
":doc:`Fully relativistic treatment of decaying cold dark matter in 𝘕-body simulations </publications>`".

Just like the metric species needs to be assigned to a linear fluid component
in order to exist during the simulation, so does the lapse species. Simply
appending ``'+ lapse'`` to our ``_lin`` string of linear species is no good
though, as this would include the lapse potential as part of gravity,
affecting *all* non-linear components, decaying or not (and with a force not
satisfying the required proportionality of the decay rate). Instead, what we
need is to let lapse be its own separate linear fluid component. As we've seen
before, :ref:`the parameter file <params-decaying-cold-dark-matter>` has been
set up to allow separating linear components using '``,``', i.e. to properly
include the lapse component we should
use ``_lin = 'radiation + metric, lapse'``.

We further need to assign the new lapse force to the decaying matter
component. Studying the specification of ``select_forces`` in
:ref:`the parameter file <params-decaying-cold-dark-matter>`, we see that the
lapse force is already being assigned whenever ``_lin`` contains the substring
``'lapse'``. To run the large-box dcdm simulation with the lapse force
included then, simply do

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "boxsize = 30*Gpc" \
       -c "_lin = 'radiation + metric, lapse'" \
       -c "_frac = 0.7"

Re-plotting after completion of the above run, we see that the lapse force
indeed managed to supply the necessary power boost, and only at very large
scales, as required.

.. note::
   In the region of :math:`k` where the relative power spectra from the
   simulations in the small and large box meet, we would of course like them
   to agree, whereas in fact the large-box simulations are slightly off. This
   is simply a lack of resolution and can be corrected by running larger
   simulations (i.e. increasing ``_size``).

Being perhaps overly critical, we may conclude that the lapse force in fact
overdid its job, with the spectrum from the dcdm simulation now having
slightly too much power at very large scales. This small error arises from our
choice of combining the dcdm species together with the stable matter species
into a single particle component. Doing so in fact introduces new general
relativistic correction terms into the equations of motion for the particles,
which are not incorporated into CO\ *N*\ CEPT. For the physics of these
additional correction terms, we once again refer to the paper on
":doc:`Fully relativistic treatment of decaying cold dark matter in 𝘕-body simulations </publications>`".

To tackle this problem --- or at least confirm that it is indeed caused by
combining decaying and stable matter --- we may run a simulation which makes
use of two separate particle components; one for stable matter
(``'baryons + cold dark matter'``) and one for decaying matter
(``'decaying cold dark matter'``). This is done simply by listing each
particle component separately in the ``initial_conditions`` parameter in
:ref:`the parameter file <params-decaying-cold-dark-matter>`. Specifying
``_combine = False``, we see that
:ref:`our parameter file <params-decaying-cold-dark-matter>` file does exactly
this. We further want the produced power spectrum data file to contain the
combined power of the two particle components, rather than simply listing the
power spectra of each component separately. Looking at the specification of
``powerspec_select`` in
:ref:`the parameter file <params-decaying-cold-dark-matter>`, we see that
power spectra are to be produced of ``'total matter'`` and
``('stable matter', 'decaying matter')``. We are used to having these
specifications refer to our non-linear component through its species (usually
``'matter'``), but here we've chosen to refer by *name*, where each
(arbitrary) name is set as part of the component specification within
``initial_conditions``. The tuple syntax
``(<component 0>, <component 1>, ...)`` used within ``powerspec_select``
specifies the combined, total power spectrum of the listed components.

.. note::
   The same use of name referencing is also used when assigning the lapse
   force within ``select_forces``, as here we do not wish to also assign this
   force to ``'stable matter'``. It was this use of name referencing which
   earlier enabled us to reformulate the ``'species'`` of the total matter
   component, without this having any effect on the component selections
   within :ref:`the parameter file <params-decaying-cold-dark-matter>`.

As everything is already handled within
:ref:`the parameter file <params-decaying-cold-dark-matter>`, running the
two-particle-component simulation is then as simple as

.. code-block:: bash

   ./concept \
       -p params/tutorial \
       -c "boxsize = 30*Gpc" \
       -c "_lin = 'radiation + metric, lapse'" \
       -c "_frac = 0.7" \
       -c "_combine = False"

This of course increases the computation time drastically, as we now have
twice the number of particles and several times the number of force
evaluations. Once completed, update the plot one last time. You should now see
better agreement with linear theory on very large scales, but at the cost of
noise at "small" (relative to the enormous box) scales.

.. note::
   The noise is believed to stem from the use of grid (pre-)initial
   conditions, i.e. the initialization of the particle positions at Cartesian
   grid points (followed by a displacement according to the Zel'dovich
   approximation, using the full general relativistic transfer functions).
   When having multiple particle components, sticking to bare grid
   (pre-)initial conditions is far from optimal. In fact, CO\ *N*\ CEPT
   currently makes use of *interleaved* (pre-)initial condition grids
   (relatively shifted by half a grid cell) when running with two particle
   components, without which the results would be a lot worse.

   As a future enhancement of CO\ *N*\ CEPT, we hope to build in the option of
   using so-called *glass* (pre-)initial conditions, which should allow for
   seamless mixing of any number of particle components, even with an
   individual (and not even necessarily cubic) number of particles :math:`N`
   for each component.



.. _nonlinear_massive_neutrinos:

Non-linear massive neutrinos
............................
*Under construction!*

