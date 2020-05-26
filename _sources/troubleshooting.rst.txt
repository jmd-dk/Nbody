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
:ref:`standard installation process <standard_installation>`
(with every dependency allowed to be installed from scratch) keeps failing
for some inexplicable reason, you may try looking for a clue in the logged
installation output (of which there are a lot), in the ``install_log`` and
``install_log_err`` files.

One possible source of trouble is corrupted downloads. The ``installer``
script downloads the source code of every primary dependency into a directory
named ``tmp``. If you suspect a corrupt download, you can try deleting this
entire directory, which will trigger re-downloads.

It may happen that some dependency program fails to install due to some other
dependency not working correctly. You may try adding the ``--test`` option
when invoking the ``installer``, which test most of the dependency programs
after/during their individual installation. Carefully looking through the
installation log files for failed tests may then reveal something.



Compilation failed
------------------
Compilation of the CO\ *N*\ CPET code takes place as the last essential step
during :ref:`standard installation <standard_installation>`, as well as when
invoking ``concept`` after changes have been made to the source files. This
process may fail for several reasons, solutions to some are described in this
entry.

To check whether the problem is confined to the compilation process, run the
code in pure Python mode. From within the ``concept`` directory, do

.. code-block:: bash

   ./concept --pure-python --local

If this simple invocation of ``concept`` fails, the problem is not with the
compilation process itself, in which case this troubleshooting entry cannot
help you.



.. _insufficient_memory:

Insufficient memory
...................
The minimum memory needed in order for compilation of the code to succeed is
about 5 GB, though the exact number depends on the system. If you suspect the
cause of compilation errors might be insufficient memory, try out the below
steps from within the ``concept`` directory. Before attempting each step, run

.. code-block:: bash

   (source concept && make clean)

in order to clean up from the last, failed compilation attempt.

* Standard (parallel) compilation:

  .. code-block:: bash

     ./concept --local

* Compile each module in serial:

  .. code-block:: bash

     make_jobs="-j 1" ./concept --local

* Add extra swap memory: If you have root privileges on the system, you can
  temporarily increase the available memory by adding a swap file:

  .. code-block:: bash

     n=8
     sudo dd if=/dev/zero of=swapfile bs=1024 count=$((n*2**20))
     sudo chmod 600 swapfile
     sudo mkswap swapfile
     sudo swapon swapfile

  This will add an additional 8 GB of swap memory (taken from available disk
  space), which is plenty. If you do not have that much free disk space, you
  may try with a lower value of ``n``. With this increased amount of memory,
  try compiling the code again. If it still fails even when compiling
  serially, memory is not the problem. To clean up the swap file, do

  .. code-block:: bash

     sudo swapoff swapfile
     sudo rm -f swapfile

After successful compilation, CO\ *N*\ CEPT will run just as performant as had
the compilation taken place without trouble.



.. _dangerous_optimizations:

Dangerous optimizations
.......................
If the compilation errors were not due to
:ref:`insufficient memory <insufficient_memory>`, it may be that one or more
of the applied optimization flags causes trouble. Try compiling without any
optimizations:

.. code-block:: bash

   (source concept && make clean)  # cleanup
   ./concept --no-optimizations --local

If this works, the problem is definately with some of the optimization flags.
You should however not run CO\ *N*\ CEPT simulations with the compiled code in
a completely unoptimized state, as this reduces performance drastically.
Instead, experiment with removing individual optimization flags added to the
``optimizations`` variable within the ``Makefile``. Substitute the ``-O3``
flag with ``-O2``, then ``-O1``, then ``-O0``, before removing it completely.
For each attempt, recompile CO\ *N*\ CEPT with\ *out* ``--no-optimizations``.
Also, remember to clean up the old compilation before attempting anew.




Terminal color output looks weird
---------------------------------
CO\ *N*\ CEPT includes a lot of color and other formatting in its terminal
output. While most modern terminal emulators on Linux (GNOME Terminal,
Terminator, xterm, etc.) fully support this, the story is different on other
platforms.

If you are running CO\ *N*\ CEPT by ssh'ing into another machine from macOS
and the colors and/or text display badly, try installing the superior
`iTerm2 <https://www.iterm2.com/>`_ terminal emulator.

If you are running CO\ *N*\ CEPT by ssh'ing into another machine from Windows
(through e.g. PuTTY) and the colors and/or text display badly, no proper
solution is known.

If you are running CO\ *N*\ CEPT through the Windows subsystem for Linux and
the colors and/or text display badly, you can install a modern Linux terminal
within the Linux subsystem. Note that this requires a running X server on the
*Windows* side.

If you want to disable color and other formatted output alltogether, set

.. code-block:: python3

   enable_terminal_formatting = False

in your parameter files. Note that though this eliminates most formatting,
a few elements are still formatted.



Problems related to Python libraries and packages
-------------------------------------------------
If Python itself fails to start or raises ``ImportError`` when loading certain
packages, it may be due to improper configuration of the Python environment.
In particular the ``PYTHONPATH``, ``PYTHONHOME`` and ``PYTHONNOUSERSITE``
environment variables should be set (or unset) appropriately. If CO\ *N*\ CEPT
has its own dedicated Python installation, one safe choice is to follow
:ref:`this <eliminating_interference_from_foreign_Python_installations>`,
which isolate the dedicated Python installation so that other Python
installations on the system has no chance of interfering.

Note that if you have installed CO\ *N*\ CEPT using the ``installer`` and not
explicitly made use of a pre-existing Python installation, a dedicated Python
has been installation and the Python environment has already been set up
appropriately (see the ``.env`` file).



Error messages containing 'Read -1'
-----------------------------------
If you see error messages of the format

   Read -1, expected <int>, errno = <int>

whenever you run CO\ *N*\ CEPT using more than a single process, it is likely
a problem with OpenMPI, more specifically vader/CMA. If CO\ *N*\ CEPT
otherwise produces correct results, you can silence these messages by placing

.. code-block:: bash

   export OMPI_MCA_btl_vader_single_copy_mechanism=none

in the ``.env`` file (located one directory level above the ``concept``
directory, i.e. at ``/path/to/concept_installation/.env``) of your
CO\ *N*\ CEPT installation.



The simulation hangs when calling CLASS
---------------------------------------
If the simulation hangs right at the beginning of the simulation, at the

   Calling CLASS in order to set the cosmic clock ...

step, it is probably because you have specified a cosmology that CLASS cannot
handle. When running CO\ *N*\ CEPT in compiled mode, CLASS may hang rather
than exiting with an error message. To see the CLASS error message, run
CO\ *N*\ CEPT in pure Python mode using the ``--pure-python``
:ref:`command-line option <pure_python>`.



Crashes or other bad behavior
-----------------------------
This entry is concerned with problems encountered when using CO\ *N*\ CEPT
*locally*. If your problem occurs only for remote jobs, please see the
'`Problems when running remotely`_' entry instead.

If you are unable to even compile CO\ *N*\ CEPT, see the
'`Compilation failed`_' entry.

If you are able to start CO\ *N*\ CEPT runs, but they crash, hang, yield
obviously wrong results, or exhibit other bad behavior, it may be due to
improper installation or a code bug. To inspect the extent of the erroneous
behavior, try running the full CO\ *N*\ CEPT test suite via

.. code-block:: bash

   ./concept -t all

If any tests are unsuccessful and you are running a stable version of
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

Even if CO\ *N*\ CEPT runs fine on the front-end of a cluster (i.e. when
supplying the ``--local`` option to the ``concept`` script), you may
experience weird behavior or crashes when running remote jobs. Typically this
is either due to an improper choice of the MPI executor, or the remote nodes
having different hardware architecture from the front-end. Possible solutions
to both of these problems are provided below.



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
choice of MPI executor depends on the job scheduler in use (Slurm or
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

      If Slurm is used as the job scheduler and the MPI library used was not
      installed by the ``installer`` script as part of the CO\ *N*\ CEPT
      installation, the MPI executor will be set to ``srun --cpu_bind=none``
      in job scripts by default (or possibly
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

      When TORQUE or PBS is used as the job scheduler, the MPI executor will be
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



Different hardware architecture on front-end and remote node
............................................................
If CO\ *N*\ CEPT and its dependencies have been installed from the front-end,
these have been somewhat tailored to the architecture of the front-end. If the
remote node to which you are submitting the CO\ *N*\ CEPT job has a different
architecture, things may go wrong. The easy solution is then of course to
switch to using a different remote queue/partition with nodes that have
similar architecture to that of the front-end.

If you have installed CO\ *N*\ CEPT using the
:ref:`standard installation process <standard_installation>`, CO\ *N*\ CEPT
itself and all of its dependencies have been built in a somewhat portable
manner, meaning that CO\ *N*\ CEPT should run fine on architectures different
from that on the front-end, as long as its not *too* different.

The above portability is broken if you have built CO\ *N*\ CEPT using the
``--native-optimizations`` option. To rebuild the code without additional
non-portable optimizations (default build), do (from the ``concept``
directory)

.. code-block:: bash

   (source concept && make clean)  # cleanup
   ./concept --local               # rebuild

If rebuilding the code with only portable optimizations did not fix the
problem, it is worth submitting a remote CO\ *N*\ CEPT job without *any*
optimizations via the ``--no-optimizations`` option to the ``concept`` script,
just to see what happens. Remember to clean the code directory before running
with ``--no-optimizations``, to force recompilation. If this works, you should
experiment with the ``Makefile`` as described
:ref:`here <dangerous_optimizations>`, as running in a completely unoptimized
state is far from ideal.

To really ensure compatibility with the architecture of a given node,
reinstall CO\ *N*\ CEPT --- including all of its dependencies --- from that
node. You may either do this by ssh'ing into the node and run the installation
manually, or you may submit the installation as a remote job. Below you will
find examples of Slurm and TORQUE/PBS job scripts for installing CO\ *N*\ CEPT.
In both cases you may wish to change ``concept_version`` and ``install_path``,
load modules or perform other environment changes, and/or make use of a
pre-installed MPI library as described
:ref:`here <optimal_network_performance_on_clusters>`.

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



It *still* does not work!
.........................
If you are still struggling, in particular if CO\ *N*\ CEPT does launch but
the MPI process binding/affinity is wrong, try removing some of the added
environment variables that gets set in the ``jobscript`` (under the
'Environment variables' heading). After altering the job script, submit it
manually using ``sbatch jobscript`` (Slurm) or ``qsub jobscript``
(TORQUE/PBS).

It is also possible that the cluster configuration just do not play nicely
with the current MPI implementation in use. If you installed CO\ *N*\ CEPT
using one of the MPI implementations present on the cluster, try again, using
another pre-installed MPI library. If you let CO\ *N*\ CEPT install its own
MPI, try switching from MPICH to OpenMPI or vice versa, as described
:ref:`here <installing_mpich_or_openmpi>`.

When installing CO\ *N*\ CEPT, try having as few modules loaded as possible,
in order to minimize the possibility of wrong MPI identification and linking.



Bad performance when using multiple processes/nodes
---------------------------------------------------
If you are running CO\ *N*\ CEPT on a cluster and experiences a significant
drop in performance as you increase the number of processes from e.g. 1 to 2
or 2 to 4, or when using 2 nodes instead of 1 with the same total number of
processes, the problem is likely that the MPI library used is not configured
to handle the network optimally.

Be sure to install CO\ *N*\ CEPT with
:ref:`optimal network performance on clusters <optimal_network_performance_on_clusters>`.
If you are observing bad network behavior even so, you should try changing the
MPI executor, as described :ref:`here <chosing_an_mpi_executor>`.



.. _problems_when_using_multiple_nodes:

Problems when using multiple nodes
----------------------------------
If you observe a wrong process binding (i.e. it appears as though several
copies of CO\ *N*\ CEPT are running on top of each other, rather than all of
the MPI processes working together as a collective) when running CO\ *N*\ CEPT
across multiple nodes, you should try
:ref:`changing the MPI executor <chosing_an_mpi_executor>`.

If you are able to run single-node CO\ *N*\ CEPT jobs remotely, but encounter
problems as soon as you request multiple nodes, it may be a permission
problem. For example, OpenMPI uses SSH to establish the connection between the
nodes, and so your local ``~/.ssh`` directory need to be configured properly.
Note that when using an MPI implementation pre-installed on the cluster, such
additional configuration from the user ought not to be necessary.

CO\ *N*\ CEPT comes with the ability to set up the ``~/.ssh`` as needed for
multi-node communication. Currently this feature resides as part of the
``installer`` script. To apply it, from the ``concept`` directory, execute

.. code:: bash

   ../installer --fix-ssh

Note that this will move all existing content of ``~/.ssh`` to
``~/.ssh_backup``. Also, any configuration you might have done will not be
reflected in the new content of ``~/.ssh``. If this indeed fixes the
multi-node problem and you want to preserve your original SSH configuration,
you must properly merge the original content of ``~/.ssh_backup`` back in with
the new content of ``~/.ssh``.

