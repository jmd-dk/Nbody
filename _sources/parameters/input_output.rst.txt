Input/output
------------

.. raw:: html

   <h3>
     <code class="docutils literal notranslate"><span class="pre">
       initial_conditions
     </span></code>
   </h3>

=== =============== =
\ \ **Type**        ``list`` of ``dict``\ 's :math:`\ |\ ` ``str``
--- --------------- -
\ \ **Default**     ``''``
--- --------------- -
\ \ **Description** Specifies the initial conditions of the simulation, either
                    as a ``list`` of ``dict``\ 's --- each specifying a
                    component to generate initial conditions for --- or
                    alternatively as a ``str`` specifying the path to an
                    existing snapshot.
--- --------------- -
\ \ **Examples**    Generate initial conditions consisting of a single
                    component comprised of :math:`128^3` matter particles. The
                    particle positions and momenta will be set according to the
                    combined baryonic and cold dark matter transfer functions:

                    .. code-block:: python3

                       initial_conditions = [{
                          'name'   : 'matter component',
                          'species': 'matter',
                          'N'      : 128**3,
                       }]

                    Generate initial conditions where baryonic and cold dark
                    matter particles are realized separately as individual
                    components, each comprised of :math:`64^3` particles:

                    .. code-block:: python3

                       initial_conditions = [
                           {
                               'name'   : 'dark matter component',
                               'species': 'cold dark matter',
                               'N'      : 64**3,
                           },
                           {
                               'name'   : 'baryonic component',
                               'species': 'baryons',
                               'N'      : 64**3,
                           },
                       ]

                    Generate initial conditions consisting of a combined
                    matter component with :math:`64^3` particles, as well as a
                    fluid component containing the combined linear energy
                    density perturbations of the photons, the neutrinos and
                    the metric, having a linear grid size of :math:`64` (and
                    thus :math:`64^3` fluid elements):

                    .. code-block:: python3

                       initial_conditions = [
                           {
                               'name'   : 'matter component',
                               'species': 'matter',
                               'N'      : 64**3,
                           },
                           {
                               'name'           : 'linear component',
                               'species'        : 'photons + neutrinos + metric',
                               'gridsize'       : 64,
                               'boltzmann order': -1,  # Completely linear
                           },
                       ]

                    In all of the above, the ``'name'`` given to each component
                    is used only for referencing by other parameters, and may
                    be omitted. For further examples on the
                    possible component specifications, check out the
                    ':doc:`beyond matter-only </tutorial/beyond_matter_only>`'
                    part of the tutorial.

                    Load initial conditions from an existng snapshot:

                    .. code-block:: python3

                       initial_conditions = '/path/to/snapshot'

=== =============== =



.. raw:: html

   <h3>
     <code class="docutils literal notranslate"><span class="pre">
       snapshot_type
     </span></code>
   </h3>

=== =============== =
\ \ **Type**        ``str``
--- --------------- -
\ \ **Default**     ``'standard'``
--- --------------- -
\ \ **Description** Specifies the snapshot format to use when dumping snapshots.
                    CO\ *N*\ CEPT currently understands two snapshot formats;
                    ``'standard'``, which is its own, well-structured
                    `HDF5 <https://www.hdfgroup.org/>`_ format, and
                    ``'gadget2'``, which is the binary Fortran format of
                    `GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/>`_,
                    specifically the *second* type (``SnapFormat = 2`` in
                    GADGET-2). Note that the value of ``snapshot_type`` does
                    not affect which snapshots may be *read*, e.g. used as
                    ``initial_conditions``.
--- --------------- -
\ \ **Example**     Dump output snapshots in GADGET-2 format:

                    .. code-block:: python3

                       snapshot_type = 'gadget2'

=== =============== =



*Under construction!*

