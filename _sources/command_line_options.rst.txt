Command-line options
====================
This page elaborates on the many options to the ``concept`` script, a
short description of which gets printed by invoking

.. code-block:: bash

   ./concept -h

If you have yet to play around with the the ``concept`` script (and thus
CO\ *N*\ CEPT), you are advised to take the
:doc:`tutorial </tutorial/first_simulations>`.

The command-line options are grouped into the following categories:

.. contents:: :local:



.. _basics:

Basics
------
The following basic command-line options are all you need to know in order for
running CO\ *N*\ CEPT simulations locally.

.. raw:: html

   <h4>
     Help:
     <code class="docutils literal notranslate"><span class="pre">
       -h
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --help
     </span></code>
   </h4>

Displays a short description of each command-line option and exits. This is
helpful if you forget the exact syntax. For learning about the usage of a
given option, this page is much preferable.


.. raw:: html

   <h4>
     Parameter file:
     <code class="docutils literal notranslate"><span class="pre">
       -p
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --params
     </span></code>
   </h4>

The ``-p`` or ``--param`` option is used to specify the parameter file to use.
Typically, parameter files are kept in the ``params`` directory. With the
parameter file ``my_params`` located in this directory, specifying this
parameter file would look like

.. code-block:: bash

   ./concept -p params/my_params

The many possible parameters to put inside parameter files are listed
:doc:`here</parameters/parameters>`. Parameters absent from the supplied
parameter file will take on default values. Leaving out the ``-p`` parameter
file specification when invoking ``concept``, *all* parameters take on their
default values, which does not result in actual simulation as no output is
then specified.



.. raw:: html

   <h4>
     Command-line parameters:
     <code class="docutils literal notranslate"><span class="pre">
       -c
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --command-line-param
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Number of processes:
     <code class="docutils literal notranslate"><span class="pre">
       -n
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --nprocs
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Utility:
     <code class="docutils literal notranslate"><span class="pre">
       -u
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --utility
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Version:
     <code class="docutils literal notranslate"><span class="pre">
       -v
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --version
     </span></code>
   </h4>

*Under construction!*



.. _remote_job_submission:

Remote job submission
---------------------
In addition to the :ref:`basic<basics>` options, the options below are used
for additional resource specification when submitting remote jobs. Note that
for remote jobs, additional possibilities for the ``-n`` option arise, as
documented above.

.. raw:: html

   <h4>
     Queue:
     <code class="docutils literal notranslate"><span class="pre">
       -q
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --queue
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Wall time:
     <code class="docutils literal notranslate"><span class="pre">
       -w
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --wall-time
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Memory:
     <code class="docutils literal notranslate"><span class="pre">
       --memory
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Job directive:
     <code class="docutils literal notranslate"><span class="pre">
       -j
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --job-directive
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     No watching:
     <code class="docutils literal notranslate"><span class="pre">
       --no-watching
     </span></code>
   </h4>

*Under construction!*



.. _other_modes_of_building_running:

Other modes of building/running
-------------------------------
The following options change the mode in which CO\ *N*\ CEPT is built and run.
When invoking the ``concept`` script, the default behavior is to check for
changes in the source code since the last build, in which case the code is
recompiled using the ``Makefile``. With the compiled code ready, the requested
CO\ *N*\ CEPT run is performed. In addition, when working on a remote server
(through ssh), rather than starting the run directly, it is submitted as a
remote job (which fails if you have not specified the ``-q``
:ref:`option<remote_job_submission>`).


.. raw:: html

   <h4>
     Local:
     <code class="docutils literal notranslate"><span class="pre">
       --local
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Pure Python:
     <code class="docutils literal notranslate"><span class="pre">
       --pure-python
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     No recompilation:
     <code class="docutils literal notranslate"><span class="pre">
       --no-recompilation
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     No optimization:
     <code class="docutils literal notranslate"><span class="pre">
       --no-optimization
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Unsafe building:
     <code class="docutils literal notranslate"><span class="pre">
       --unsafe-build
     </span></code>
   </h4>

*Under construction!*



.. _specials:

Specials
--------
The following special options are rarely used outside of development, though
knowledge about them may come in handy as they are very powerful.

.. raw:: html

   <h4>
     Test:
     <code class="docutils literal notranslate"><span class="pre">
       -t
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --test
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Main entry point:
     <code class="docutils literal notranslate"><span class="pre">
       -m
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --main
     </span></code>
   </h4>

*Under construction!*


.. raw:: html

   <h4>
     Interactive:
     <code class="docutils literal notranslate"><span class="pre">
       -i
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --interactive
     </span></code>
   </h4>

*Under construction!*

