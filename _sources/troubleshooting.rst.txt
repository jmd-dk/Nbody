Troubleshooting
===============
This page contains solutions to and helpful information about possible issues
encountered when running CO\ *N*\ CEPT. If help from this page is insufficient
to solve your problem, do not hesitate to
`open an issue <https://github.com/jmd-dk/concept/issues>`_ or contact the
author at dakin@phys.au.dk.

Entries on this page:

.. contents:: :local:



Installation failed
-------------------
We strive for a trivial installation process on as many Linux systems as
possible. If the :doc:`simple installation process</tutorial/installation>`
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

.. note::
   To force a recompilation, you must first clean the code directory. You can
   do this by

   .. code-block:: bash

       (source concept && make clean)

If CO\ *N*\ CEPT is able to run without optimizations, it might be worth to
manually edit the ``Makefile`` and remove/alter the exact optimization(s) that
leads to trouble, allowing for successful runs without necessarily disabling
*all* optimizations with the ``--no-optimization`` option.

If you are able to start CO\ *N*\ CEPT runs, but they crash, hang, yields
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



.. _problems_when_running_remotely:

Problems when running remotely
------------------------------
This entry is concerned with problems encountered specifically with remote
CO\ *N*\ CEPT jobs. If you have not tried out CO\ *N*\ CEPT locally, please do
this first. If you encounter problems here as well, please see the
'`Crashes or other bad behavior`_' entry.

Even if CO\ *N*\ CEPT runs fine on the frontend of a cluster (i.e. when
supplying the ``--local`` option to the ``concept`` script), you may
experience weird behavior or crashes when submitting remote jobs. Typically,
this is either due to the remote nodes having different hardware architecture
from the frontend, or an improper choice of the MPI executor. Possible
solutions to both of these problems are provided below.


.. raw:: html

   <h4>Different hardware architecture on frontend and remote node</h4>

If CO\ *N*\ CEPT and its dependencies have been installed from the frontend,
these have been tailored to the architecture of the frontend. If the remote
node to which you are submitting the CO\ *N*\ CEPT job has a different
architecture, things may go wrong. The easy solution is then of course to
switch to using a different remote queue/partition with nodes that have
similar architecture to the one on the frontend.

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
changes.

.. topic:: Slurm

   To submit a remote Slurm job for installing CO\ *N*\ CEPT, save the code
   below to e.g. ``jobscript`` (replacing ``<queue>`` with the partition in
   question) and execute ``sbatch jobscript``.

   .. code-block:: bash

      #!/usr/bin/env bash
      #SBATCH --job-name=install_concept
      #SBATCH --partition=<queue>
      #SBATCH --nodes=1
      #SBATCH --tasks-per-node=8
      #SBATCH --mem-per-cpu=5000M
      #SBATCH --time=24:00:00
      #SBATCH --output=/dev/null
      #SBATCH --error=/dev/null

      concept_version=master
      install_path="${HOME}/concept"

      make_jobs="-j" bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer) "${install_path}"

.. topic:: TORQUE/PBS

   To submit a remote TORQUE/PBS job for installing CO\ *N*\ CEPT, save the
   code below to e.g. ``jobscript`` (replacing ``<queue>`` with the queue in
   question) and execute ``qsub jobscript``.

   .. code-block:: bash

      #!/usr/bin/env bash
      #PBS -N install_concept
      #PBS -q <queue>
      #PBS -l nodes=1:ppn=8
      #PBS -l walltime=24:00:00
      #PBS -o /dev/null
      #PBS -e /dev/null

      concept_version=master
      install_path="${HOME}/concept"

      make_jobs="-j" bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer) "${install_path}"



.. raw:: html

   <h4>Choosing an MPI executor</h4>

It may help to manually choose a different remote *MPI executor*. This is the
term used for e.g. ``mpiexec``/``mpirun`` in CO\ *N*\ CEPT, i.e. the
executable used to launch MPI programs.

To see which MPI executor is used when running remotely, check out the
``mpi_executor`` variable in the produced ``jobscript`` file. To manually set
the MPI executor, overwrite the dedicated ``mpi_executor`` varaible in the
``.env`` file (located one directory level above the ``concept`` directory,
i.e. at ``/path/to/concept_installation/.env``). Helpful suggestions for the
choice of MPI executor depends on the job schedular in use.

.. topic:: Using Slurm

   If Slurm is used as the job schedular, the MPI executor will be set to
   ``srun --cpu_bind=none`` by default. The first thing to try is to leave
   out the ``--cpu_bind=none``, i.e. setting

   .. code-block:: bash

      mpi_executor="srun"

   in the ``.env`` file. Submit a new job, and you should see the manually chosen MPI
   executor being respected by the ``jobscript``.

   If that did not fix the issue, try specyfing the MPI implementation in use,
   using the ``--mpi`` option to ``srun``. E.g. for OpenMPI, set

   .. code-block:: bash

      mpi_executor="srun --mpi=openmpi"

   in the ``.env`` file. To see which MPI implementations ``srun`` supports,
   run

   .. code-block:: bash

      srun --mpi=list

   You may wish to try your luck on all supported MPI implementations. If you
   find one that works, do remember to test if it also works with the added
   ``--cpu_bind=none`` option, as this is preferred.

   If you are still unable to run remotely, you can try using ``mpiexec`` or
   ``mpirun`` for the MPI executor, as one would do when using TORQUE/PBS.


.. topic:: Using TORQUE/PBS

   When TORQUE or PBS is used as the job schedular, the MPI executor will be
   set to one of ``mpiexec`` or ``mpirun`` by default, possibly with
   additional options. The first thing to try is to leave out these options,
   i.e. setting

   .. code-block:: bash

      mpi_executor="mpiexec"  # or "mpirun"

   in the ``.env`` file. Note that CO\ *N*\ CEPT sets the ``PATH`` so that
   ``mpiexec``/``mpirun`` are guaranteed to be those belonging to the correct
   MPI implementation (that specified in the ``.paths`` file). You are however
   welcome to specify absolute paths as well.

   If remote jobs still fail, you may look for other possible MPI executors,
   by running

   .. code-block:: bash

      (source concept && ls "${mpi_bindir}")

   (other possible MPI executors include ``mpiexec.hydra`` and ``orterun``).


.. raw:: html

   <h4>It <em>still</em> does not work!</h4>

If you are still struggling, it is possible that the cluster configuration
just do not play nicely with the current MPI implementation in use. If you
installed CO\ *N*\ CEPT using one of the MPI implementations present on the
cluter, try again, using another preinstalled MPI library. If you let
CO\ *N*\ CEPT install its own MPI, try switching from MPICH to OpenMPI or
vice versa (i.e. set ``mpi=openmpi`` or ``mpi=mpich`` when installing
CO\ *N*\ CEPT, as described under :doc:`installation`).

When installing CO\ *N*\ CEPT, try having as few modules loaded as possible,
to minimize the possibilities of wrong MPI identification and linking.



Problems when using multiple nodes
----------------------------------
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










