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

   ./concept --pure-python --local

This will fire up CO\ *N*\ CEPT, which should display a few lines of colorful
text before shutting back down. The ``--local`` option specifies that we wish
to run CO\ *N*\ CEPT on the local machine, as opposed to submitting the
simulation as a job on a remote cluster. If you *are* working locally, you may
omit the ``--local`` option, just as this tutorial will do from now on. If you
are working remotely, please add the ``--local`` option yourself to all future
invocations of the ``concept`` script. Later this tutorial covers how to submit
CO\ *N*\ CEPT jobs when :doc:`working on a cluster <working_remotely>`.

As we did not provide any parameters to CO\ *N*\ CEPT, it shut down
immediately. We can add parameters using the ``-c`` option, like so:

.. code-block:: bash

   ./concept \
       -c 'initial_conditions = {"name": "matter", "species": "matter particles", "N": 64**3}' \
       -c 'a_begin = 0.02' \
       -c 'output_times = {"powerspec": 1}'

A simulation with :math:`N = 64^3` matter particles will now be run, with
initial conditions generated at a scale factor of :math:`a = 0.02` and a power
spectrum produced at :math:`a = 1`. Once completed, check the ``output``
directory, where you'll find the power spectrum data and plot.



Don't forget about gravity
..........................
The above simulation should take about a minute or two to complete, with
almost all of the time spent setting up the initial conditions, rather than on
the time stepping. The reason for this rapid time stepping is that
CO\ *N*\ CEPT does not assign forces implicitly, meaning that no gravitational
forces were applied during the simulation. This also explains the low power of
the simulation power spetrum compared to the linear power spectrum (which do
include the effects of gravity), both of which are shown in
``output/powerspec_a=1.00.png``.

Let's try again, this time explicitly stating that the ``"matter"`` component
is under the influence of gravity, where ``"matter"`` is the arbitrary name
we give to our matter particles:

.. code-block:: bash

   ./concept \
       -c 'initial_conditions = {"name": "matter", "species": "matter particles", "N": 64**3}' \
       -c 'a_begin = 0.02' \
       -c 'output_times = {"powerspec": 1}' \
       -c 'select_forces = {"matter": "gravity"}'

The simulation time will increase somewhat. Once done, take a look at the
output ``output/powerspec_a=1.00.png`` again, and you should see that the
simulated power spectrum has catched up to the linear one.

To cut down the simulation time we may make use of multiple CPU cores. To run
with e.g. 4 cores, execute

.. code-block:: bash

   ./concept \
       -c 'initial_conditions = {"name": "matter", "species": "matter particles", "N": 64**3}' \
       -c 'a_begin = 0.02' \
       -c 'output_times = {"powerspec": 1}' \
       -c 'select_forces = {"matter": "gravity"}' \
       -n 4

To see a complete list of possible options to the ``concept`` script, invoke

.. code-block:: bash

   ./concept -h

