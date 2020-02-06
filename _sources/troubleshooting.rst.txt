Troubleshooting
===============
This page contains solutions to and helpful information about possible issues
encountered when using CO\ *N*\ CEPT. If help from this page is insufficient
to solve your problem, do not hesitate to
`open an issue <https://github.com/jmd-dk/concept/issues>`_ or contact the
author at
:raw-html:`<td class="field-body"><a class="reference external" href="mailto:dakin&#37;&#52;&#48;phys&#46;au&#46;dk">dakin<span>&#64;</span>phys<span>&#46;</span>au<span>&#46;</span>dk</a></td>`.

Entries on this page:

.. contents::
   :local:
   :depth: 1



Installation failed
-------------------
We strive for a trivial installation process on as many Linux systems as
possible. If the
:ref:`standard installation process<standard_installation>`
(with every dependency allowed to be installed from scratch) keeps failing
for some inexplicable reason, you may try looking for a clue in the logged
installation output (of which there are a lot), in the ``install_log`` and
``install_log_err`` files.

It may happen that some dependency program fails to install due to some other
dependency not working correctly. You may try adding the ``--test`` option
when invoking the ``installer``, which test most of the dependency programs
after/during their individual installation. Carefully looking through the
installation log files for failed tests may then reveal something.



Terminal color output looks weird
---------------------------------
CO\ *N*\ CEPT includes a lot of color and other formatting in its terminal
output. While all modern terminal emulators on Linux fully support this, the
story is different on other platforms. If you run CO\ *N*\ CEPT from macOS
(via ssh into a cluster) and the colors appear all wrong, try installing
the superior `iTerm2 <https://www.iterm2.com/>`_ terminal emulator. If you
are running CO\ *N*\ CEPT from Windows (through e.g. putty), no proper
solution is known.

If you want to disable color and other formatted output alltogether, set

.. code-block:: python3

   enable_terminal_formatting = False

in your parameter files. Note that though this eliminates most formatting,
a few elements are still formatted.



The simulation hangs when calling CLASS
---------------------------------------
If the simulation hangs right at the beginning of the simulation, at the

   Calling CLASS in order to set the cosmic clock ...

step, it is probably because you have specified a cosmology that CLASS cannot
handle. When running CO\ *N*\ CEPT in compiled mode, CLASS hangs rather than
exiting with an error message. To see the CLASS error message, run
CO\ *N*\ CEPT in pure Python mode using the ``--pure-python``
:ref:`command-line option<other_modes_of_building_running>`.



Crashes or other bad behavior
-----------------------------
This entry is concerned with problems encountered when using CO\ *N*\ CEPT
*locally*. If your problem occurs only for remote jobs, please see the
'`Problems when running remotely`_' entry instead.

If you are unable to run or even compile CO\ *N*\ CEPT, try running in pure
Python mode by supplying the ``--pure-python`` flag to the ``concept`` script.
If this works, you are probably dealing with an erroneous compilation. Next,
remove the ``--pure-python`` flag and instead use ``--no-optimization``. This
runs CO\ *N*\ CEPT in compiled mode, but without any optimizations.

.. tip::
   To force a recompilation, you must first clean the code directory. You can
   do this by

   .. code-block:: bash

       (source concept && make clean)

If CO\ *N*\ CEPT is able to run without optimizations, it might be worth to
manually edit the ``Makefile`` and remove/alter the exact optimization(s) that
leads to trouble, allowing for successful runs without necessarily disabling
*all* optimizations with the ``--no-optimization`` option.

If you are able to start CO\ *N*\ CEPT runs, but they crash, hang, yield
obviously wrong results, or exhibit other bad behavior, it may be due to
improper installation or a code bug. To inspect the extent of the erroneous
behavior, try running the full CO\ *N*\ CEPT test suite via

.. code-block:: bash

   ./concept -t all

If any test is unsuccessful and you are running a stable version of
CO\ *N*\ CEPT (i.e. any release version, not 'master'), there is most probably
a problem with your installation. You can try reinstalling CO\ *N*\ CEPT along
with all of its dependencies, perhaps using compilers different from the ones
used the first time around.

If all tests passes despite the observed (and reproducible!) bad behavior,
you may have found a bug. Please report this.



Bad performance when using multiple processes/nodes
---------------------------------------------------
If you are running CO\ *N*\ CEPT on a cluster and experiences a significant
drop in performance as you increase the number of processes from e.g. 1 to 2
or 2 to 4, or when using 2 nodes instead of 1 with the same total number of
processes, the problem is likely that the MPI library used is not configured
to handle the network optimally.

Be sure to install CO\ *N*\ CEPT with
:ref:`optimal network performance on clusters<optimal_network_performance_on_clusters>`.
If you are observing bad network behavior even so, you should try changing the
MPI executor, as described in :ref:`this<problems_when_running_remotely>`
entry.


.. _problems_when_running_remotely:

Problems when running remotely
------------------------------
This entry is concerned with problems encountered specifically with remote
CO\ *N*\ CEPT jobs. If you have not tried out CO\ *N*\ CEPT locally, please do
this first. If you encounter problems here as well, please see the
'`Crashes or other bad behavior`_' entry.

Even if CO\ *N*\ CEPT runs fine on the front-end of a cluster (i.e. when
supplying the ``--local`` option to the ``concept`` script), you may
experience weird behavior or crashes when running remote jobs. Typically this
is either due to the remote nodes having different hardware architecture from
the front-end, or an improper choice of the MPI executor. Possible solutions to
both of these problems are provided below.

If your remote jobs run seemingly successfully for the first many time steps
but eventually crashes at some nondeterministic step, you should definately
try changing the MPI executor.



Different hardware architecture on front-end and remote node
............................................................
If CO\ *N*\ CEPT and its dependencies have been installed from the front-end,
these have been tailored to the architecture of the front-end. If the remote
node to which you are submitting the CO\ *N*\ CEPT job has a different
architecture, things may go wrong. The easy solution is then of course to
switch to using a different remote queue/partition with nodes that have
similar architecture to the one on the front-end.

One might think that disabling architecture-specific compiler optimizations in
the CO\ *N*\ CEPT ``Makefile`` (like ``-march=native``) would help, but since
many of the dependency programs are compiled with architecture-specific flags
as well, this rarely helps. That said, it is worth submitting a remote
CO\ *N*\ CEPT job without *any* optimizations via the ``--no-optimization``
option to the ``concept`` script, just to see what happens. Remember to clean
the code directory before running with ``--no-optimization``, to force
recompilation.

To really ensure compatibility with the architecture of a given node,
reinstall CO\ *N*\ CEPT --- including all of its dependencies --- from that
node. You may either do this by ssh'ing into the node and run the installation
manually, or you may submit the installation as a remote job. Below you will
find examples of Slurm and TORQUE/PBS job scripts for installing CO\ *N*\ CEPT.
In both cases, you may wish to add loading of modules or other environment
changes, and/or make use of a preinstalled MPI library as described under
:doc:`installation`. Once a CO\ *N*\ CEPT installation job has begun, you
can follow the installation process by executing

.. tabs::

   .. group-tab:: Slurm

      To submit a remote Slurm job for installing CO\ *N*\ CEPT, save the code
      below to e.g. ``jobscript`` (replacing ``<queue>`` with the partition in
      question) and execute ``sbatch jobscript``.

      .. code-block:: bash

         #!/usr/bin/env bash
         #SBATCH --job-name=install_concept
         #SBATCH --partition=<queue>
         #SBATCH --nodes=1
         #SBATCH --tasks-per-node=8
         #SBATCH --mem-per-cpu=2000M
         #SBATCH --time=12:00:00
         #SBATCH --output=/dev/null
         #SBATCH --error=/dev/null

         concept_version=master
         install_path="${HOME}/concept"

         installer="https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer"
         make_jobs="-j" bash <(wget -O- "${installer}") "${install_path}"

   .. group-tab:: TORQUE/PBS

      To submit a remote TORQUE/PBS job for installing CO\ *N*\ CEPT, save the
      code below to e.g. ``jobscript`` (replacing ``<queue>`` with the queue
      in question) and execute ``qsub jobscript``.

      .. code-block:: bash

         #!/usr/bin/env bash
         #PBS -N install_concept
         #PBS -q <queue>
         #PBS -l nodes=1:ppn=8
         #PBS -l walltime=12:00:00
         #PBS -o /dev/null
         #PBS -e /dev/null

         concept_version=master
         install_path="${HOME}/concept"

         installer="https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer"
         make_jobs="-j" bash <(wget -O- "${installer}") "${install_path}"

Once a CO\ *N*\ CEPT installation job has begun, you can follow the
installation process by executing

.. code-block:: bash

   tail -f <install_path>/install_log



.. _chosing_an_mpi_executor:

Choosing an MPI executor
........................
It may help to manually choose a different remote *MPI executor*. This is the
term used for e.g. ``mpiexec``/``mpirun`` in CO\ *N*\ CEPT, i.e. the
executable used to launch MPI programs.

To see which MPI executor is used when running remotely, check out the
``mpi_executor`` variable in the produced ``jobscript`` file. To manually set
the MPI executor, overwrite the dedicated ``mpi_executor`` varaible in the
``.env`` file (located one directory level above the ``concept`` directory,
i.e. at ``/path/to/concept_installation/.env``). Helpful suggestions for the
choice of MPI executor depends on the job schedular in use (Slurm or
TORQUE/PBS).

.. tabs::

   .. group-tab:: Slurm

      .. note::

         Even if you are using Slurm, it may be that your MPI library is not
         configured appropriately for ``srun`` to be able to correctly launch
         MPI jobs. This can happen e.g. if you are using an MPI library that
         was installed by the CO\ *N*\ CEPT ``installer`` script, as opposed
         to an MPI library configured and installed by a system administrator
         of the cluster. If the below does not work, try setting the MPI
         executor as though you were using TORQUE/PBS.

      If Slurm is used as the job schedular and the MPI library used was not
      installed by the ``installer`` script as part of the CO\ *N*\ CEPT
      installation, the MPI executor will be set to ``srun --cpu_bind=none``
      in jobscripts by default (or possibly
      ``srun --cpu_bind=none --mpi=openmpi`` if OpenMPI is used). The first
      thing to try is to leave out ``--cpu_bind=none``, i.e. setting

      .. code-block:: bash

         mpi_executor="srun"

      in the ``.env`` file. Submit a new job, and you should see the manually
      chosen MPI executor being respected by the ``jobscript``.

      If that did not fix the issue, try specyfing the MPI implementation in
      use, using the ``--mpi`` option to ``srun``. E.g. for OpenMPI, set

      .. code-block:: bash

         mpi_executor="srun --mpi=openmpi"

      in the ``.env`` file. To see which MPI implementations ``srun``
      supports, run

      .. code-block:: bash

         srun --mpi=list

      directly on the front-end. You may wish to try your luck on all
      supported MPI implementations. If you find one that works, do remember
      to test if it also works with the added ``--cpu_bind=none`` option, as
      this is preferred.

   .. group-tab:: TORQUE/PBS

      When TORQUE or PBS is used as the job schedular, the MPI executor will be
      set to one of ``mpiexec`` or ``mpirun`` by default, possibly with
      additional options. The first thing to try is to leave out these options,
      i.e. setting

      .. code-block:: bash

         mpi_executor="mpiexec"  # or "mpirun"

      in the ``.env`` file. Note that CO\ *N*\ CEPT sets the ``PATH`` so that
      ``mpiexec``/``mpirun`` are guaranteed to be those belonging to the
      correct MPI implementation (that specified in the ``.paths`` file). You
      are however allowed to specify absolute paths as well.

      Options to try out with ``mpiexec``/``mpirun`` include

      .. code-block:: bash

         mpi_executor="mpiexec --bind-to none"  # or "mpirun --bind-to none"

      and

      .. code-block:: bash

         mpi_executor="mpiexec -bind-to none"  # or "mpirun -bind-to none"

      (the difference being one or two dashes before ``bind``).

      If remote jobs still fail, you may look for other possible MPI executors,
      e.g. by running

      .. code-block:: bash

         (source concept && ls "${mpi_bindir}")

      (other possible MPI executors include ``mpiexec.hydra`` and ``orterun``).



It *still* does not work!
.........................
If you are still struggling, in particular if CO\ *N*\ CEPT does launch but
the MPI process binding/affinity is wrong, try removing some of the added
environment variables that gets set in the ``jobscript`` (under the
'Environment variables' heading). After altering the jobscript, submit it
manually using ``sbatch jobscript`` (Slurm) or ``qsub jobscript``
(TORQUE/PBS).

It is also possible that the cluster configuration just do not play nicely
with the current MPI implementation in use. If you installed CO\ *N*\ CEPT
using one of the MPI implementations present on the cluster, try again, using
another preinstalled MPI library. If you let CO\ *N*\ CEPT install its own
MPI, try switching from MPICH to OpenMPI or
vice versa (i.e. set ``mpi=openmpi`` or ``mpi=mpich`` when installing
CO\ *N*\ CEPT, as described :ref:`here<influential_environment_variables>`).

When installing CO\ *N*\ CEPT, try having as few modules loaded as possible,
in order to minimize the possibility of wrong MPI identification and linking.



.. _problems_when_using_multiple_nodes:

Problems when using multiple nodes
----------------------------------
If you observe a wrong process binding (i.e. it appears as though several
copies of CO\ *N*\ CEPT are running on top of each other, rather than all of
the MPI processes working together as a collective) when running CO\ *N*\ CEPT
across multiple nodes, you should try changing the MPI executor. See "choosing
an MPI executor" under :ref:`this<problems_when_running_remotely>` entry.

If you are able to run single-node CO\ *N*\ CEPT jobs remotely, but encounter
problems as soon as you request multiple nodes, it may be a permission
problem. For example, OpenMPI uses ssh to establish the connection between the
nodes, and so your local ``~/.ssh`` directory need to be configured properly.
Note that when using an MPI implementation preinstalled on the cluster, such
additional configuration from the user ought not to be necessary.

CO\ *N*\ CEPT comes with the ability to set up the ``~/.ssh`` as needed for
multi-node communication. Currently this feature resides as part of the
``installer`` script. To apply it, from the ``concept`` directory, execute

.. code:: bash

   ../installer --fix-ssh

Note that this will move all existing content of ``~/.ssh`` to
``~/.ssh_backup``. Also, any configuration you might have done will not be
reflected in the new content of ``~/.ssh``. If this indeed fixes the
multi-node problem and you want to preserve your original ssh configuration,
you must properly merge the original content of ``~/.ssh_backup`` back in with
the new content of ``~/.ssh``.

