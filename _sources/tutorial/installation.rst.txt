Effortless installation
-----------------------
.. tip::
   If you just want to quickly try out CO\ *N*\ CEPT and are familiar with
   Docker, you may choose to run the code off of a provided Docker image,
   bypassing the below installation. See the Docker tab
   :ref:`here <supported_platforms>`.

If this is your first time trying out CO\ *N*\ CEPT, it is advised to do the
installation on your local machine, not on a remote cluster.

.. note::
   Though CO\ *N*\ CEPT may be installed on Windows through the
   `Windows Subsytem for Linux <https://docs.microsoft.com/en-us/windows/wsl/about>`_,
   installing on a native Linux system is recommended. Don't bother attempting the
   installation on macOS.

To install the newest version of CO\ *N*\ CEPT along with all of its
dependencies, open a terminal and invoke

.. code-block:: bash

   bash <(wget -O- --no-check-certificate https://raw.githubusercontent.com/jmd-dk/concept/master/installer)

You will be prompted for an installation directory.

.. note::
   The initial ``bash`` is required regardless of your shell

The installation will take an hour or two. You can kill the installation at
any time --- it will pick up from where it left off if you re-enter the
installation command.

For full documentation on the installation process, consult
:doc:`Installation </installation>`.

