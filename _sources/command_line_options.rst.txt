Command-line options
====================
This page elaborates on the many options to the ``concept`` script, a
short description of which gets printed by invoking

.. code-block:: bash

   ./concept -h

If you have yet to play around with the the ``concept`` script (and thus
CO\ *N*\ CEPT), you are advised to take the
:doc:`tutorial </tutorial/first_simulations>`.

Most options has a short and a long format --- e.g. ``-h`` and ``--help`` ---
which always does the same thing. Some however only has the long format. When
using the long format, you do not have to spell out the entire option, as long
as the first part uniquely identifies the full option. For example, ``--he``
may be used in place of ``--help``.

The command-line options are grouped into four categories as listed below:

.. contents::
   :local:
   :depth: 2



.. _basics:

Basics
------
The following basic command-line options are all you need to know in order for
running CO\ *N*\ CEPT simulations locally.



Help: ``-h``, ``--help``
........................
Displays a short description of each command-line option and exits. This is
helpful if you forget the exact syntax. For learning about the usage of a
given option, this page is much preferable.



Parameter file: ``-p``, ``--params``
....................................
Specifies the parameter file to use. Typically, parameter files are kept in
the ``params`` directory. With the parameter file ``my_params`` located in
this directory, specifying this parameter file would look like

.. code-block:: bash

   ./concept -p params/my_params

The many possible parameters to put inside parameter files are each described
:doc:`here</parameters/parameters>`. Parameters absent from the supplied
parameter file will take on default values. Leaving out the ``-p`` parameter
file specification when invoking ``concept``, *all* parameters take on their
default values, which does not result in an actual simulation as no output is
specified by default.



Command-line parameters: ``-c``, ``--command-line-param``
.........................................................
Directly specifies parameters to use, without referring to a parameter file.
E.g.

.. code-block:: bash

   ./concept -c "boxsize = 512*Mpc"

Often this is used in combination with a parameter file, e.g. if a suite of
similar simulations is to be run where only a few parameter values change
between the simulations. E.g.

.. code-block:: bash

   ./concept -p params/my_params -c "Ωb = 0.049" -c "Ωcdm = 0.27"

Note that the ``-c`` option may be specified multiple times.

Specifying command-line parameters while also making use of a parameter file
is equivalent to having the command-line parameters defined *at the bottom* of
the parameter file.

Be careful about trying to overwrite parameter values defined in a parameter
file using ``-c``: The parameter/variable will take on the value as defined
on the command-line from the top of the parameter file and down until the
parameter/variable is (re)assigned. During the actual simulation, the value
given by ``-c`` will be used. For a completely independent parameter there is
then no danger in overwriting its value using ``-c``, but if other parameters
are defined in terms of the first parameter, it is cleaner to just not have it
be defined both in the parameter file and on the command-line.



.. _number_of_processes:

Number of processes: ``-n``, ``--nprocs``
.........................................
Specifies the number of MPI processes to use, with each MPI process being
mapped to its own CPU core. To e.g. run with 8 cores:

.. code-block:: bash

   ./concept -n 8

If not specified, this defaults to 1.

You may write the number of processes as an expression. All of these specifies
that 8 processes should be used:

.. code-block:: bash

   ./concept -n 2*4
   ./concept -n 2**3
   ./concept -n "3 + 5"



.. _specifying_multiple_nodes:

Specifying multiple nodes
~~~~~~~~~~~~~~~~~~~~~~~~~
When running on a cluster with multiple compute nodes, you may also specify
the number of nodes to be used. The following examples all specify 8 MPI
processes distributed over 2 nodes each with 4 CPU cores:

.. code-block:: bash

   ./concept -n 2:4
   ./concept -n 2:2*2
   ./concept -n "2 2*2"
   ./concept -n "1 + 1 : 2**2"

Note that imhomogeneous layouts are not describable. If you leave out the node
specification (i.e. only supply a single number to ``-n``) and the cluster is
running Slurm, the specified total number of CPU cores may be distributed in
any which way between the available nodes. If the cluster is running
TORQUE/PBS, you must always explicitly specify the number of nodes as well as
the number of CPU cores/node.



Utility: ``-u``, ``--utility``
..............................
Signals that one of the CO\ *N*\ CEPT utilities is to be run instead of a
simulation. To run e.g. the ``powerspec`` utility, do

.. code-block:: bash

   ./concept -u powerspec </path/to/snapshot>

which will then produce a power spectrum of the snapshot file located at
``</path/to/snapshot>``.

Each utility comes with its own command-line options (for the ``powerspec``
utility, a required path to a snapshot), which you should specify together
with the normal ``concept`` command-line options. In the case of the
``powerspec`` utility, this could look like one of

.. code-block:: bash

   ./concept -n 4 -u powerspec </path/to/snapshot>
   ./concept -u powerspec </path/to/snapshot> -n 4

both of which will produce a power spectrum of the snapshot using 4 CPU cores.
Though not encouraged, you may even detach the utility options
(``</path/to/snapshot>``) from the utility specification (``-u powerspec``),
as in

.. code-block:: bash

   ./concept -n 4 </path/to/snapshot> -u powerspec
   ./concept </path/to/snapshot> -n 4 -u powerspec
   ./concept </path/to/snapshot> -u powerspec -n 4
   ./concept -u powerspec -n 4 </path/to/snapshot>

In short, do not worry about the order of ``concept`` and utility specific
command-line options.

You can read about the different utilities and their command-line interfaces
in the :doc:`Utilities</utilities/utilities>` chapter.



Version: ``-v``, ``--version``
..............................
Prints out the version of CO\ *N*\ CEPT that is installed.



.. _remote_job_submission:

Remote job submission
---------------------
On top of the :ref:`basic<basics>` options, the options below are used for
additional resource specification when submitting remote jobs. Note that
additional possibilities arise for the ``-n`` option when running on a cluster
with multple compute nodes, as documented
:ref:`above<specifying_multiple_nodes>`.



.. _queue:

Queue: ``-q``, ``--queue``
..........................
*Under construction!*



Wall time: ``-w``, ``--wall-time``
..................................
*Under construction!*



Memory: ``--memory``
....................
*Under construction!*



Job directive: ``-j``, ``--job-directive``
..........................................
*Under construction!*



No watching: ``--no-watching``
..............................
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
:ref:`option<queue>`).



Local: ``--local``
...................
*Under construction!*



.. _pure_python:

Pure Python: ``--pure-python``
..............................
*Under construction!*



No recompilation: ``--no-recompilation``
........................................
*Under construction!*



No optimization: ``--no-optimization``
......................................
*Under construction!*



Unsafe building: ``--unsafe-building``
......................................
*Under construction!*



.. _specials:

Specials
--------
The following special options are mostly useful for development. Even so,
knowledge about them may come in handy outside of development, as they are
very powerful.



.. _test:

Test: ``-t``, ``--test``
........................
*Under construction!*



Main entry point: ``-m``, ``--main``
....................................
*Under construction!*



Interactive: ``-i``, ``--interactive``
......................................
*Under construction!*

