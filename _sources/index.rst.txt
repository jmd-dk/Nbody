.. raw:: html

   <style type="text/css">
     span.bolditalic {
       font-weight: bold;
       font-style: italic;
     }
   </style>

.. role:: bolditalic
   :class: bolditalic



CONCEPT documentation
=====================

.. raw:: html

   <table class="docutils field-list" frame="void" rules="none">
   <col class="field-name" />
   <col class="field-body" />
   <tbody valign="top">
   <tr class="field-odd field"><th class="field-name">Author:</th><td class="field-body">Jeppe Mosgaard Dakin</td>
   </tr>
   <tr class="field-even field"><th class="field-name">Contact:</th><td class="field-body"><a class="reference external" href="mailto:dakin&#37;&#52;&#48;phys&#46;au&#46;dk">dakin<span>&#64;</span>phys<span>&#46;</span>au<span>&#46;</span>dk</a></td>
   </tr>
   <tr class="field-odd field"><th class="field-name">Web Site:</th><td class="field-body"><a class="reference external" href="https://github.com/jmd-dk/concept">https://github.com/jmd-dk/concept</a></td>
   </tr>
   <tr class="field-even field"><th class="field-name">Date:</th><td class="field-body">|today|</td>
   </tr>
   </tbody>
   </table>


This is the documentation for CO\ *N*\ CEPT, the **CO**\ smological
:bolditalic:`N`\ -body **C**\ od\ **E** in **P**\ y\ **T**\ hon.


.. topic:: Abstract

   CO\ *N*\ CEPT is a free and open-source simulation code for cosmological
   structure formation. The code should run on any Linux system, from
   massively parallel computer clusters to laptops. The code is
   written almost exclusively in Python, but achieves C-like performance
   thanks to Cython.

   CO\ *N*\ CEPT is capable of simulating matter particles evolving under
   self-gravity in an expanding background. It has multiple gravitational
   solvers to choose from, and has adaptive time integration built in.
   In addition to particles, fluids such as neutrinos are also implemented,
   and can be simulated at various levels of non-linearity. Completely linear
   perturbations in all species and the metric itself are fully supported,
   allowing for simulations consistent with general relativity.

   CO\ *N*\ CEPT comes with a sophisticated initial condition generator
   built in, and can output snapshots, power spectra and several kinds
   of renders.

   The `CLASS <http://class-code.net/>`_ code is fully
   integrated into CO\ *N*\ CEPT, supplying the needed information for e.g.
   initial condition generation and general relativistic corrections.



Content
-------
Below you will find the complete contents list.
If you're new to CO\ *N*\ CEPT, start with the
:doc:`tutorial <tutorial/tutorial>`, which briefly covers installation before
taking you on a guided tour through the CO\ *N*\ CEPT universe.


.. toctree::
   :maxdepth: 3

   tutorial/tutorial
   installation
   parameters
   utilities/utilities
   troubleshooting
   publications



