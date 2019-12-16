Working on remote clusters
--------------------------
If you are running CO\ *N*\ CEPT on your local machine, you should skip this
section.

If you are running CO\ *N*\ CEPT on a remote machine, i.e. logged in to a
server/cluster via ``ssh``, you've so far had to supply the additional
``--local`` option to the ``concept`` script. This is because CO\ *N*\ CEPT
has built-in support for submission of jobs to a job schedular / queueing system
/ resource manager (specifically Slurm, TORQUE and PBS), and this is the
default behavior when working remotely. If you are working remotely but do not
intend to use a job shedular, keep using ``--local`` and skip the rest of this
section.



.. topic:: Submitting jobs

   If you try to run a simulation *without* the ``--local`` option while
   logged into a remote server, CO\ *N*\ CEPT will error out immediately,
   letting you know that it has created an almost complete *job script*,
   simply called ``jobscript`` and placed in the ``concept`` directory. This
   job script is a great starting point if you want to control the job
   submission yourself. All that needs to be changed/added are the directives
   at the top of the jobscript.

   To automatically submit a complete job script, you need to specify the
   *queue* (called *partition* in Slurm) in which to submit the job using the
   ``-q`` option to ``concept``. Submitting a simulation using e.g. a
   parameter file named ``params/tutorial`` using 8 cores then looks like

   .. code-block:: bash

      ./concept -p params/tutorial -n 8 -q <queue>  # Replace <queue> with queue name

   The 8 cores may be distributed over several *nodes* of the cluster. If you
   wish to control the number of nodes and number of cores per node, use e.g.
   ``-n 1:8`` to request 1 node with 8 cores, or ``-n 2:4`` to request 2 nodes
   each with 4 cores.

   .. note::

      If remote CO\ *N*\ CEPT jobs mysteriously fail, check out the
      ':ref:`problems_when_running_remotely`' troubleshooting entry.

   To specify a memory requirement, further supply ``--memory <memory>``,
   where ``<memory>`` is the *total* memory required collectively by all cores
   on all nodes. Examples of legal memory specifications include
   ``--memory 8192MB``, ``--memory 8192M``, ``--memory 8G``,
   ``--memory 2*4G``, which all specify 8 gigabytes, i.e. 1 gigabyte per core
   for ``-n 8`` or ``-n 2:4``.

   To specify a walltime limit, i.e. a maximum time within which the
   simulation is expected to be completed, further supply the
   ``-w <walltime>`` option. Examples of legal walltime specifications include
   ``-w 60min``, ``-w 60m``, ``-w 1hr``, ``-w 1h``, which all request one hour
   of walltime.

   A complete CO\ *N*\ CEPT job submission could then look like

   .. code-block:: bash

      ./concept -p params/tutorial -n 8 -q somequeue --mem 8G -w 1h


   .. tip::

      Note that in the above, ``--memory`` is shortened to ``--mem``.
      Generally, as long as no conflict occurs with other options, you may
      shorten any option to ``concept`` in this manner. Also, the order in
      which the options are supplied does not matter.



.. topic:: The watch utility

   Once a job is submitted, CO\ *N*\ CEPT will notify you that you may now kill
   (``Ctrl`` + ``C``) the process. If you don't, the submitted job is
   continually monitored, and its output will be printed to the screen once it
   starts running, as if you were running the simulation locally. This is
   handled by the *watch utility*, which is automatically called after job
   submission. It works by continually printing out updates to the log file in
   close to real time.

   If you don't want to watch the job after submission, you may supply the
   ``--no-watching`` option to ``concept`` instead of having to kill the
   process after submission.

   You may manually run the watch utility at any later time, like so:

   .. code-block:: bash

      ./concept -u watch <ID>  # Replace <ID> with job ID of remote running job

   The job ID --- and hence log filename --- of submitted jobs is determined by
   the job schedular, and is printed as soon as the job is submitted. You may
   also leave out any job ID when running the ``watch`` utility, in which case
   the latest submitted, running job will be watched. Again, to exit, simply
   press ``Ctrl`` + ``C``.



.. topic:: Using a pre-installed MPI library

   The :doc:`installation procedure</tutorial/installation>` described in this
   tutorial installed CO\ *N*\ CEPT along with every dependency, with no
   regard for possibly preinstalled libraries. Though generally recommended,
   for running serious, multi-node simulations one should make use of an MPI
   library native to the cluster, in order to ensure
   :ref:`optimal network performance<optimal_network_performance_on_clusters>`.






















