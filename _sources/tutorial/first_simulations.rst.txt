Your first simulations
----------------------

The ``concept`` script
......................

In a terminal, navigate to the ``concept`` directory within your chosen
installation directory. Unless otherwise specified, all code examples to come
is meant to be executed from within this directory. Here you'll find the
various ``*.py`` files making up the code, as well as a few other files, the
most important of which is a script similarly named ``concept``,
used to launch the code.

Try invoking the script using

.. code-block:: bash

   ./concept --pure-python --local

This will fire up CO\ *N*\ CEPT, which should display a few lines of colorful
text before shutting back down.

.. note::
   If the color output looks hideous, chances are that you're running
   CO\ *N*\ CEPT through ssh on macOS or putty on Windows. In the case of
   macOS, switching to a terminal emulator with better color support
   --- such as `iTerm2 <https://www.iterm2.com/>`_ --- fixes the issue.

The ``--local`` option specifies that we wish to run CO\ *N*\ CEPT on the local
machine, as opposed to submitting the simulation as a job on a remote cluster.
If you *are* working locally, you may omit the ``--local`` option, just as this
tutorial will do from now on. If you are working remotely, please add the
``--local`` option yourself to all future invocations of the ``concept``
script. Later this tutorial covers how to submit CO\ *N*\ CEPT jobs when
:doc:`working on a cluster <working_remotely>`.

As we did not provide any parameters to CO\ *N*\ CEPT, it shut down emmideately.
We can add parameters using the ``-c`` option, like so:

.. code-block:: bash

   ./concept \
       -c 'initial_conditions = {"name": "matter", "species": "matter particles", "N": 32**3}' \
       -c 'a_begin = 0.02' \
       -c 'output_times = {"powerspec": 1}' \
       --pure-python

A simulation with :math:`N = 32^3` particles will now be run, with initial
conditions generated at a scale factor of :math:`a = 0.02` and a power
spectrum produced at :math:`a = 1`. Once completed, check the ``output``
directory, where you'll find the power spectrum data and plot.

The above simulation should take a couple of minutes. Now rerun the simulation,
this time without the ``--pure-python`` option. If the CO\ *N*\CEPT code has
yet to be compiled, this will now be done. This process takes a few minutes.
Once done, the simulation will be run in *compiled* mode, as opposed to *pure
Python* mode. You should see a dramatic performance enhancement, in particular
for the actual time stepping. For each time step, the computation time is
written in parentheses to the right. Also, the last printed line of a
simulation contains the total execution time.


.. topic:: Log files

   The printed output of all CO\ *N*\ CEPT runs gets logged in the ``logs``
   directory. Each run (or *job*) gets a unique integer ID, which is also used
   as the filename for the logged output. The log filename of any CO\ *N*\ CEPT
   run is stated when the program starts, and the job ID is written again at
   the end. Also, the job ID is included in the header of the power spectrum
   data files.

   Using the logs, try comparing the time step and total timings for the pure
   Python and compiled run you just performed of the same simulation.

   .. tip::
      To view e.g. the logged output of CO\ *N*\ CEPT run 1 with proper
      coloring, use ``less -r logs/1``. Arrow keys to navigate, ``q`` to quit.

For running actual simulations, compiled mode is always preferable to pure
Python mode.



Don't forget about gravity
..........................
The reason for the rapid time stepping (and to the keen eye, a low power
spectrum) is because CO\ *N*\ CEPT does not assign forces implicitly.
We must explicitly state that the ``"matter"`` component is under the
influence of gravity, where ``"matter"`` is the arbitrary name we have given
to our matter particles:

.. code-block:: bash

   ./concept \
       -c 'initial_conditions = {"name": "matter", "species": "matter particles", "N": 32**3}' \
       -c 'a_begin = 0.02' \
       -c 'output_times = {"powerspec": 1}' \
       -c 'select_forces = {"matter": {"gravity": "p3m"}}'

Finally, *this* runs a full simulation. Also, the simulation time is once again
a few minutes. We can bring it down by replacing ``"p3m"`` with ``"pm"``.
which switches out the P³M method for the much faster but also (at small scales)
less accurate PM method.

Sticking to the PM method, we may want to increase the simuation size,
say to :math:`N = 64^3` particles. At the same time, let's make sure that the
mesh for the gravitational potential :math:`\varphi` has at least the same
number of cells:

.. code-block:: bash

   ./concept \
       -c 'initial_conditions = {"name": "matter", "species": "matter particles", "N": 64**3}' \
       -c 'a_begin = 0.02' \
       -c 'output_times = {"powerspec": 1}' \
       -c 'select_forces = {"matter": {"gravity": "pm"}}' \
       -c 'φ_gridsize = 64'

(yes, really copy the greek letter directly into your termianl. It'll be fine.)

.. note::
   The potential grid :math:`\varphi` is reused for power spectrum computations,
   and so ``φ_gridsize`` also determines the smallest scale at which power is
   measured.

The above simulation runs rather quickly. Were we to use ``"p3m"`` in place of
``"pm"``, the simulation time would would rise dramatically. Perhaps
surprisingly, when using the P³M method we may gain quite a bit of performance
by *increasing* the resolution of the potential grid,
using e.g. ``φ_gridsize = 128``. See for yourself!

Finally, we should of course make use of multiple CPU cores to further cut
down on the computation time. To run with e.g. 4 cores, execute

.. code-block:: bash

   ./concept \
       -c 'initial_conditions = {"name": "matter", "species": "matter particles", "N": 64**3}' \
       -c 'a_begin = 0.02' \
       -c 'output_times = {"powerspec": 1}' \
       -c 'select_forces = {"matter": {"gravity": "p3m"}}' \
       -c 'φ_gridsize = 128' \
       -n 4

To see a complete list of possible options to the ``concept`` script, invoke

.. code-block:: bash

   ./concept -h






