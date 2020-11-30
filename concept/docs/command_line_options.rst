Command-line options
====================
This page elaborates on the many options to the ``concept`` script, used to
launch CO\ *N*\ CEPT.

If you have yet to play around with the ``concept`` script, you are advised to
take the :doc:`tutorial </tutorial/first_simulations>`.

Most options have a short and a long format --- e.g. ``-h`` and ``--help`` ---
which always does the same thing. Some however only have the long format. When
using the long format, you do not have to spell out the entire option, as long
as the first part uniquely identifies the full option. For example, ``--he``
may be used in place of ``--help``.

Many options require a value, e.g. ``-n 2`` or equivalently
``--nprocs 2``. Instead of a space, the option name and value may be separated
by an equal sign; ``-n=2`` or ``--nprocs=2``.

The command-line options are grouped into four categories as listed below.
Underneath each category heading, a new facet of the ``concept`` script is
described, which might be of interest even if you do not want to study each
option in detail.

.. contents::
   :local:
   :depth: 2



.. _basics:

Basics
------
The following basic command-line options are all you need to know in order to
run CO\ *N*\ CEPT simulations locally.



Help: ``-h``, ``--help``
........................
Displays a short description of each command-line option and exits:

.. code-block::

   ./concept -h

This is helpful if you have forgotten the name of some particular option. For
learning about the usage of a given option, this page is much preferable.



Parameter file: ``-p``, ``--params``
....................................
Specifies the parameter file to use:

.. code-block:: bash

   ./concept -p </path/to/parameter-file>

The path may be specified relative to the ``concept`` directory. Typically,
parameter files are kept in the ``concept/params`` directory. With the
parameter file ``my_params`` located in this directory, specifying this
parameter file then looks like

.. code-block:: bash

   ./concept -p params/my_params

The many possible parameters to put inside parameter files are each described
in the sections under :doc:`Parameters </parameters/parameters>`. Parameters
absent from the supplied parameter file will take on default values. Leaving
out the ``-p`` parameter file specification when invoking ``concept``, *all*
parameters take on their default values, which does not result in an actual
simulation as no output is specified by default.



Command-line parameters: ``-c``, ``--command-line-params``
..........................................................
Directly specifies parameters to use, without referring to a parameter file.
E.g.

.. code-block:: bash

   ./concept -c "boxsize = 512*Mpc"

Often this is used in combination with a parameter file, e.g. if a suite of
similar simulations is to be run where only a few parameter values change
between the simulations. E.g.

.. code-block:: bash

   ./concept -p params/my_params -c "Ωb = 0.049" -c "Ωcdm = 0.270"
   ./concept -p params/my_params -c "Ωb = 0.048" -c "Ωcdm = 0.271"

Note that the ``-c`` option may be specified multiple times.

Specifying command-line parameters while also making use of a parameter file
is equivalent to having the command-line parameters defined *at the bottom* of
the parameter file. Despite of this, the command-line parameters will become
available both during the simulation *and* to the other parameters in the
parameter file during read-in.

.. caution::
   Be careful about trying to overwrite parameter values defined in a
   parameter file using ``-c``. As stated above, specifying a command-line
   parameter with ``-c`` is equivalent to defining it at the bottom of the
   parameter file, which makes it take on the command-line value at the top of
   the parameter file and down until the parameter is (re)assigned (see
   :doc:`non-linear parsing of parameter file content </parameters/parameters>`
   for how this works). During the actual simulation, the value given by
   ``-c`` will be used.

   For a completely stand-alone parameter with no other parameters depending
   on its value, there is then no danger in overwriting its value using
   ``-c``. It might not obvious whether a given parameter is stand-alone in or
   not, and so it is generally cleaner to just not have any parameters be
   defined both in the parameter file and on the command-line.



.. _number_of_processes:

Number of processes: ``-n``, ``--nprocs``
.........................................
Specifies the number of MPI processes to use, with each MPI process being
mapped to its own CPU core (assuming enough of these are available). To run
using e.g. 8 processes:

.. code-block:: bash

   ./concept -n 8

If not specified, this defaults to 1.

You may write the number of processes as a (Python 3) expression. All of these
specifies that 8 processes should be used:

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

both of which will produce a power spectrum of the snapshot using 4 processes.
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
under :doc:`Utilities </utilities/utilities>`.

Note that while some utilies are always run locally, the ones that potentially
involves large computational resources are subject to the same
:ref:`remote submission behavior <remote_job_submission>` as standard
simulations.



Version: ``-v``, ``--version``
..............................
Prints out the version of CO\ *N*\ CEPT that is installed:

.. code-block:: bash

   ./concept -v



.. _remote_job_submission:

Remote job submission
---------------------
When running CO\ *N*\ CEPT on a cluster with a job scheduler (Slurm and
TORQUE/PBS supported), each invocation of ``concept`` submits the work to be
done as a remote job (unless this behavior is :ref:`overruled <local>`). If
running locally, none of these options have any effect.

On top of the :ref:`basic <basics>` options, the options below are used for
additional resource specification when submitting remote jobs. Note that
additional possibilities arise for the ``-n`` option when running on a cluster
with multple compute nodes, as documented
:ref:`above <specifying_multiple_nodes>`.



.. _queue:

Queue: ``-q``, ``--queue``
..........................
Specifies the name of the queue (called "partition" in Slurm) to which the
remote job is to be submitted:

.. code-block:: bash

   ./concept -q <queue>

If using Slurm, you can specify multiple queues:

.. code-block:: bash

   ./concept -q <queue1>,<queue2>,<queue3>

No remote job will be submitted if you do not supply this option. A job script
will however be produced, which you may edit and submit manually.



.. _walltime:

Wall time: ``-w``, ``--wall-time``
..................................
Specifies the maximum wall time (total computation time) within which the
remote job has to have completed. If the job runs for the entire specified
maximum wall time, it will be killed.

You have a lot of freedom in how you want to express this time:

.. code-block:: bash

   ./concept -w 2:30:00       #  2 hours and 30 minutes
   ./concept -w 45min         #              45 minutes
   ./concept -w 100m          #  1 hour  and 40 minutes
   ./concept -w "3 days"      # 72 hours
   ./concept -w 3:12:00:00    # 84 hours
   ./concept -w "2d + 7.5hr"  # 55 hours and 30 minutes

If not set, the system/queue usually have some default wall time limit set.



.. _memory:

Memory: ``--memory``
....................
Specified the amount of memory allocated for the remote job. If you assign
insufficient memory to a job, it will be killed (usually with a somewhat
cryptic error message) once its memory need exceeds the specified amount.

Examples of memory specifications:

.. code-block:: bash

   ./concept --memory 4GB     #   4 gigabytes
   ./concept --memory 2048MB  #   2 gigabytes
   ./concept --memory 8*2GB   #  16 gigabytes
   ./concept --memory 0.5*TB  # 512 gigabytes
   ./concept --mem 8G         #   8 gigabytes

Note that the specified memory is the total memory available to the job, to be
shared amongst all MPI processes / CPU cores, even when running on multiple
compute nodes.

If not set, the system usually have some default amount of memory to allocate
to each job.



Job directive: ``-j``, ``--job-directive``
..........................................
The specifications of system resources --- the designated
:ref:`queue(s) <queue>`, the
:ref:`number of nodes and processes <number_of_processes>`, the allowed
:ref:`wall time <walltime>` and the allocated :ref:`memory <memory>` --- gets
saved as "job directives" within a job script, which is then submitted to the
job scheduler. If you desire further fine tuning of system resources, you may
supply further such directives using this command-line option.

What job directives are available depend on the job scheduler, as well as the
local configuration of the cluster and queue. As an example, consider

.. code-block:: bash

   ./concept -j="--exclusive"

which tells Slurm to give the job exclusive access to the allocated nodes, so
that the nodes will not be shared with other running jobs which could otherwise
make use of still available resources.

The effect of this is simply to place the line

   #SBATCH \-\-exclusive

in the header of the job script, assuming Slurm, or

   #PBS \-\-exclusive

in the case of TORQUE/PBS (the specific example of ``--exclusive``
does not mean anything to TORQUE/PBS, though).

.. caution::
   Since the value ``--exclusive`` starts with '``-``', using
   ``-j --exclusive`` or ``-j "--exclusive"`` is not legal as the
   parser registres ``--exclusive`` as a whole new (and non-existent)
   option.

The ``-j`` option may be specified multiple times.



No watching: ``--no-watching``
..............................
After submitting a remote job, rather than put you back at the system prompt,
CO\ *N*\ CEPT will run the :doc:`watch utility </utilities/watch>` in order
for you to follow the progression of the job. This have no effect on the job
itself, and you may stop watching its printout using ``Ctrl``\ +\ ``C``.

If you have no desire to watch the job progression, you may specify this
option:

.. code-block:: bash

   ./concept --no-watching

in which case the watch utility will not be run at all.



.. _other_modes_of_building_running:

Other modes of building/running
-------------------------------
The following options change the mode in which CO\ *N*\ CEPT is built or run.
With the exception of the :ref:`\\\\-\\\\-local <local>` option, these are
mostly useful for development.

When invoking the ``concept`` script, the default behavior is to check for
changes in the source code since the last build, in which case the code is
recompiled using the ``Makefile``. With the compiled code ready, the requested
CO\ *N*\ CEPT run is performed. In addition, when working on a remote
server/cluster (through SSH), rather than starting the run directly, it is
submitted as a remote job.



.. _local:

Local: ``--local``
...................
Supply this option to disregard the fact that you are running on a remote
server/cluster. That is, do not submit the CO\ *N*\ CEPT run as a remote job
using the job scheduler, but run it directly as if you were running locally.

.. code-block:: bash

   ./concept --local



.. _pure_python:

Pure Python: ``--pure-python``
..............................
When this option is supplied, CO\ *N*\ CEPT is run directly from the Python
source files, disregarding the presence of any compiled modules:

.. code-block:: bash

   ./concept --pure-python

While handy during development, running actual simulations in pure Python mode
is impractical due to an enormous performance hit.

You can freely switch between running in compiled (the default) and pure
Python mode, without needing to recompile in-between.



No optimizations: ``--no-optimizations``
........................................
During compilation of CO\ *N*\ CEPT, a lot of optimizations are performed.
These include source code transformations performed by ``pyxpp.py`` as well
as standard C compiler optimizations, applied in the ``Makefile``. Though
these optimizations should not be disabled under normal circumstances, you may
do so by supplying this option:

.. code-block:: bash

   ./concept --no-optimizations

.. note::
   If CO\ *N*\ CEPT is already in a compiled state (built with or without
   optimizations), it will not be recompiled. To recompile without
   optimizations, you first have to remove the compiled files:

   .. code-block:: bash

      (source concept && make clean)



.. _native_optimizations:

Native optimizations: ``--native-optimizations``
................................................
The default optimizations performed during compilation are all portable, so
that the compiled code may be run on different (though not *too* different)
hardware. This is particularly useful on a cluster of multiple nodes with
different hardware, as CO\ *N*\ CEPT does not have to be recompiled when
switching nodes.

Supply this option if you are willing to use non-portable optimizations native
to your particular system, by which it is possible to squeeze a further few
percent performance increase out of the compilation:

.. code-block:: bash

   ./concept --native-optimizations

Specifically, this adds the ``-march=native`` compiler optimization.

.. note::
   If CO\ *N*\ CEPT is already in a compiled state (built with or without
   (native) optimizations), it will not be recompiled. To recompile with
   native optimizations, you first have to remove the compiled files:

   .. code-block:: bash

      (source concept && make clean)



No recompilation: ``--no-recompilation``
........................................
Any change to one of the source files will trigger automatic recompilation
upon the next invocation of ``concept`` (if not running in
:ref:`pure Python mode <pure_python>`). You may specify this option if you
want to run using a current, out-of-date compiled state of the code:

.. code-block:: bash

   ./concept --no-recompilation



Unsafe building: ``--unsafe-building``
......................................
By default the compilation process is carried out in a safe manner, meaning
that changes within a file triggers recompilation of all other files which
rely on the specific file in question. If you know that the change is entirely
internal to the given file, you may save yourself some compilation time by
supplying this option, in which case all interdependencies between the files
are ignored during compilation:

.. code-block:: bash

   ./concept --unsafe-building

.. caution::
   This option really *is* unsafe and may very well lead to a buggy build. To
   clean up after an unsuccessful build, do

   .. code-block:: bash

      (source concept && make clean)



.. _specials:

Specials
--------
The following options are mostly useful for development. As demonstrated by
the below examples though, "development" does not have to be restricted to the
CO\ *N*\ CEPT code itself. That is, these special options come in handy should
you wish to hook into the code, or simply check that the code is in a working
state.



.. _test:

Test: ``-t``, ``--test``
........................
CO\ *N*\ CEPT comes with an integration test suite, located in the
``concept/tests`` directory. Each subdirectory within this directory
implements a given test, the name of which equals the directory name. You may
use this option to run these tests, checking that the code works correctly.

To run e.g. the ``concept_vs_gadget_p3m`` test --- which runs a small
CO\ *N*\ CEPT P³M simulation and a (supposedly) equivalent GADGET TreePM
simulation, after which it compares the results --- do one of

.. code-block:: bash

   ./concept -t concept_vs_gadget_p3m
   ./concept -t tests/concept_vs_gadget_p3m

where the first form reference the test by name while the second reference the
test by its (relative) path. Note that tests are always performed locally,
i.e. not submitted as remote jobs even when working on a remote
cluster/server.

Once a test is complete, it will report either success or failure. Most
simulations also produce some artifacts within their directory, most notably
plots. You can clean up these artifacts by running the ``clean`` script within
the corresponding test subdirectory, e.g.

.. code-block:: bash

   tests/concept_vs_gadget_p3m/clean

The entire test suite may be run sing

.. code-block:: bash

   ./concept -t all

which runs each test sequentially. If one of the tests fails, the process
terminates immediately. To clean up after all tests, i.e. run the ``clean``
script within each subdirectory of the ``tests`` directory, do

.. code-block:: bash

   (source concept && make clean_tests)



.. _main_entry_point:

Main entry point: ``-m``, ``--main``
....................................
The job of the ``concept`` script is to set up the environment, build the
actual CO\ *N*\ CEPT code (if not running in
:ref:`pure Python mode <pure_python>`) and then launch it (or submit this
launch as a remote job). Skipping over many details, this final launch step
looks something like

.. code-block:: bash

   python -m main.so  # compiled mode
   python main.py     # pure Python mode

which fires up the main module, which in turn initiates the simulation. We can
use this option to switch out the default ``main`` module for some other main
entry point.

As an example, consider the ``hubble.py`` script below:

.. code-block:: python
   :caption: hubble.py
   :name: hubble

   from commons import *  # import everyting from the CO𝘕CEPT commons module

   h = H0/(100*units.km/(units.s*units.Mpc))
   print(f'{h = :g}')

We can run this script using

.. code-block:: bash

   ./concept -m hubble.py --pure-python

where :ref:`pure Python mode <pure_python>` is needed as the script itself is
not compiled. To see that this really does hook into the CO\ *N*\ CEPT
machinery:

.. code-block:: bash

   ./concept -m hubble.py -c "H0 = 72*km/(s*Mpc)" --pure-python

Using ``-m`` with a script as above is convenient for e.g. making plotting
scripts which need access to some CO\ *N*\ CEPT functionality. For small
printouts, we can specify the Python code directly as the value to ``-m``. To
e.g. get the value of the gravitational constant implemented in CO\ *N*\ CEPT,
we can do

.. code-block:: bash

   ./concept \
       -m "from commons import *; \
           unit = units.m**3/(units.kg*units.s**2); \
           print(G_Newton/unit); \
       " \
       --pure-python \
   | tail -n 1



Interactive: ``-i``, ``--interactive``
......................................
Normally when running ``concept``, you are send back to the system prompt once
the run has completed. You can use this option to instead end up in a live
Python prompt, with the :ref:`main <main_entry_point>` scope available for
exploration and further execution.

You could run a simulation interactively if you wanted to:

.. code-block:: bash

   ./concept \
       -c "a_begin = 0.1" \
       -c "initial_conditions = {'species': 'matter', 'N': 16**3}" \
       -i \
       --pure-python

.. code-block:: python

   >>> output_times['a']['render2D'] = render2D_times['a'] = (1, )
   >>> timeloop()

where ``>>>`` indicates input which should be typed at the interactive Python
prompt. We use :ref:`pure Python mode <pure_python>` as our interactive
commands are interpreted directly by Python.

While the above example requires knowledge of the internal code and serves no
real use outside of development, a perhaps more useful usage of interactive
mode is to combine it with :ref:`\\\\-\\\\-main <main_entry_point>` when
writing auxiliary scripts, or to use it purely exploratory. Say we did not
know the variable name and value of the gravitational constant implemented in
CO\ *N*\ CEPT, an we wanted to find out. We might go exploring, doing
something like

.. code-block:: bash

   ./concept -i --pure-python

.. code-block:: python

   >>> # What variables are available?
   >>> dir()
   >>> # I see one called 'G_Newton'
   >>> G_Newton
   >>> # Its value looks unrecognizable! It must not be given in SI units.
   >>> # I see something called 'units'
   >>> units
   >>> units.m
   >>> units.Mpc
   >>> # What if I try ...
   >>> G_Newton/(units.m**3/(units.kg*units.s**2))
   >>> # Success!
