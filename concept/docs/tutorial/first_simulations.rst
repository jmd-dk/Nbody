Your first simulations
----------------------
In a terminal, navigate to the ``concept`` directory within your chosen
installation directory. Unless otherwise specified, all code examples to come
is meant to be executed from within this directory. Here you'll find the
various ``*.py`` files making up the code, as well as a few other files, the
most important of which is a script similarly named ``concept``,
used to launch the code.

Try invoking the script using

.. code-block:: bash

   ./concept --local

This will fire up CO\ *N*\ CEPT, which should display a few lines of colourful
text before shutting back down. The ``--local`` option specifies that we wish
to run CO\ *N*\ CEPT on the local machine, as opposed to submitting the
simulation as a job on a remote cluster. If you *are* working locally, you may
omit the ``--local`` option, just as this tutorial will do from now on. If you
are working remotely, please add the ``--local`` option yourself to all future
invocations of the ``concept`` script. Later this tutorial covers how to submit
CO\ *N*\ CEPT jobs to the scheduler when
:doc:`working on a cluster <working_remotely>`.

As we did not provide any parameters to CO\ *N*\ CEPT, it shut down
immediately. We can add parameters using the ``-c`` option, like so:

.. code-block:: bash

   ./concept \
       -c "initial_conditions = {'species': 'matter', 'N': 64**3}" \
       -c "a_begin = 0.02" \
       -c "output_times = {'powerspec': 1}"

A simulation with :math:`N = 64^3` matter particles will now be run, with
initial conditions generated at a scale factor of :math:`a = 0.02` and a power
spectrum produced at :math:`a = 1`. Once completed, check the ``output``
directory, where you'll find the power spectrum data (a plain text file)
and plot.

.. tip::
   Once a simulation is complete, its total computation time is shown in the
   last line of output



.. raw:: html

   <h2>Don't forget about gravity</h2>

The above simulation should take about a minute to complete, with almost none
of the time spent on the actual simulation (time stepping), but rather on the
generation of initial conditions. The reason for this rapid time stepping is
that CO\ *N*\ CEPT does not assign interactions implicitly, meaning that no
gravitational forces were applied during the simulation. This also explains
the low simulation power spectrum compared to the linear power spectrum (which
do include the effects of gravity), both of which are present in
``output/powerspec_a=1.00`` and ``output/powerspec_a=1.00.png``.

Let's try again, this time specifying a size for the gravitational potential
grid:

.. code-block:: bash

   ./concept \
       -c "initial_conditions = {'species': 'matter', 'N': 64**3}" \
       -c "a_begin = 0.02" \
       -c "output_times = {'powerspec': 1}" \
       -c "potential_options = 128"

The simulation time will increase noticeably. Once done, take a look at the
output ``output/powerspec_a=1.00.png`` again, and you should see that the
simulated power spectrum has caught up to the linear one.

The role of the potential grid will be discussed in detail
:doc:`in a little while <gravity>`.



.. raw:: html

   <h2>Going multi-core</h2>

To cut down on the computation time we may make use of multiple CPU cores. To
run with e.g. 4 CPU cores, execute

.. code-block:: bash

   ./concept \
       -c "initial_conditions = {'species': 'matter', 'N': 64**3}" \
       -c "a_begin = 0.02" \
       -c "output_times = {'powerspec': 1}" \
       -c "potential_options = 128" \
       -n 4

.. note::
   A choice of e.g. ``-n 3`` is illegal as :math:`N = 64^3` is not divisible
   by :math:`3`. Further such restrictions exist. If an illegal number of
   processes is chosen, CO\ *N*\ CEPT exits with a helpful error message.

To see a complete list of possible options to the ``concept`` script, invoke

.. code-block:: bash

   ./concept -h

For full documentation on each option, consult
:doc:`Command-line options </command_line_options>`.

