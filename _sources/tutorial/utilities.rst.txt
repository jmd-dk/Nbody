Using utilities
---------------
If you have followed this tutorial so far, you've encounted the *play*, the
*info* and the *watch* utility; side programs baked into CO\ *N*\ CEPT. This
section further demonstrates the use of the *powerspec*, *render2D*,
*render3D*, *CLASS* and *update* utility.

All utilities are invoked via

.. code-block:: bash

   ./concept -u <utility> <utility-options>

possibly in combination with standard ``concept`` options (e.g. ``-p``). The
utilities are really executable scripts found in the ``utilities`` directory,
and so alternatively you may invoke them directly:

.. code-block:: bash

   utilities/<utility> <utility-options>

The former calling convention is generally preferred, but in one instance you
*have* to call the utilities directly, namely when you want to get help on the
possible options specific to a given utility:

.. code-block:: bash

   utilities/<utility> -h

Note that any utility that (potentially) performs heavy computations are
treated in the same manner as a regular simulation when it comes to the
local/remote behaviour. Thus, if you are working on a remote server and you
want the utility to run directly as opposed to be submitted as a remote job,
remember to supply ``--local``. The utilities that always runs locally are
the play and update utility. The info utility is special in this regard, in
that it runs locally except when the ``--stat`` option is given (see
``utilities/info -h``).





The powerspec and render utilities
..................................
*Under construction!*



The CLASS utility
.................
*Under construction!*



The update utility
..................
*Under construction!*




