Troubleshooting
===============
This page contains solutions to and helpful information about possible issues
encountered when running CO\ *N*\ CEPT. Should you encounter a problem not
listed, do not hesitate to contact the author at dakin@phys.au.dk.



Installation failed
...................
We strive for a trivial installation process on as many Linux systems as
possible. If the :doc:`simple installation process</tutorial/installation>`
(with every dependency allowed to be installed from scratch) keeps failing,
please send an e-mail to dakin@phys.au.dk stating the details of your system,
including compiler brand + version. Do not forget to attach the
``install_log`` and ``install_log_err`` files produced during the attempted
installation.



The terminal color output looks weird
.....................................
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



.. _mpi-executor:

Problems when running remotely
..............................
If your remote jobs behave weirdly or doesn't run at all, while local jobs run
as expected, it may help to manually choose a different remote *MPI executor*.
This is the term used for e.g. ``mpiexec``/``mpirun`` in CO\ *N*\ CEPT, i.e.
the executable used to launch MPI programs.

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

   in the ``.env``file. Note that CO\ *N*\ CEPT sets the ``PATH`` so that
   ``mpiexec``/``mpirun`` are guaranteed to be those belonging to the MPI
   correct MPI implementation (that specified in the ``.paths`` file). You
   are however welcome to specify absolute paths as well.

   If remote jobs still fail, you may look for other possible MPI executors,
   by running

   .. code-block:: bash

      (source concept && ls "${mpi_bindir}")

   (other possible MPI executors include ``mpiexec.hydra`` and ``orterun``).

If you are still struggling, it is possible that the cluster configuration
just do not play nicely with the current MPI implementation in use. If you
installed CO\ *N*\ CEPT using one of the MPI implementations present on the
cluter, try again, using another preinstalled MPI library. If you let
CO\ *N*\ CEPT install its own MPI, try switching from MPICH to OpenMPI or
vice versa (i.e. set ``mpi=openmpi`` or ``mpi=mpich`` when installing
CO\ *N*\ CEPT, as described under :doc:`installation`).

When installing CO\ *N*\ CEPT, try having as few modules loaded as possible,
to minimize the possibilities of wrong MPI identification and linking.



Problems when running on multiple nodes
.......................................
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










