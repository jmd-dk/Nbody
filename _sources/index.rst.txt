.. role:: raw-html(raw)
   :format: html

.. role:: bolditalic
   :class: bolditalic



CONCEPT documentation
=====================

.. raw:: html

   <table class="docutils field-list" style="font-size:120%">
   <col class="field-name" />
   <col class="field-body" />
   <tbody valign="top">
   <tr><td class="field-name"><b>Release</b></td><td class="field-body">1.0.0 <i>(still under development)</i></td></tr>
   <tr><td class="field-name"><b>GitHub</b></td><td class="field-body"><a class="reference external" href="https://github.com/jmd-dk/concept/">https://github.com/jmd-dk/concept/</a>&emsp;<img src="https://github.com/jmd-dk/concept/workflows/build/badge.svg"/></td></tr>
   <tr><td class="field-name"><b>Author</b></td><td class="field-body">Jeppe Dakin, Aarhus University&emsp;<a class="reference external" href="mailto:dakin&#37;&#52;&#48;phys&#46;au&#46;dk">dakin<span>&#64;</span>phys<span>&#46;</span>au<span>&#46;</span>dk</a></td></tr>
   <tr class="field-even field"><td class="field-name"><b>Date</b></td><td class="field-body">|today|</td></tr>
   </tbody>
   </table>

:raw-html:`<font size="+1">This is the documentation for CO<i>N</i>CEPT, the <b>CO</b>smological <b><i>N</b></i>-body <b>C</b>od<b>E</b> in <b>P</b>y<b>T</b>hon</font>`



Abstract
--------
CO\ *N*\ CEPT is a free and open-source simulation code for cosmological
structure formation. The code should run on any Linux system, from massively
parallel computer clusters to laptops. The code is written almost exclusively
in Python, but achieves C-like performance through code transformation using a
custom transpiler/optimizer and `Cython <https://cython.org/>`_.

CO\ *N*\ CEPT is capable of simulating matter particles evolving under
self-gravity in an expanding background. It has multiple gravitational solvers
to choose from, and has adaptive time integration built in. In addition to
particles, the code is further able to evolve fluids at various levels of
non-linearity, providing the means for the inclusion of more exotic species
such as massive neutrinos. Completely linear perturbations in density fields
and the metric itself allows for simulations consistent with general
relativistic perturbation theory. Various non-standard
species --- such as decaying cold dark matter --- are fully supported.

CO\ *N*\ CEPT comes with a sophisticated initial condition generator built in,
and can output snapshots, power spectra and several kinds of renders.

The `CLASS <http://class-code.net/>`_ code is fully integrated into
CO\ *N*\ CEPT, supplying the needed information for e.g. initial condition
generation and general relativistic corrections.



Contents
--------
Below you will find the global table of contents for the documentation. If
you're new to CO\ *N*\ CEPT, start with the
:doc:`tutorial <tutorial/tutorial>`, which briefly covers installation before
taking you on a guided tour through the CO\ *N*\ CEPT universe.

.. toctree::
   :maxdepth: 2

   tutorial/tutorial
   installation
   command_line_options
   parameters/parameters
   utilities/utilities
   under_the_hood
   troubleshooting
   publications

