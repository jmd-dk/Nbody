Installation
============
This page contains detailed information about how to install CO\ *N*\ CEPT
along with its many dependencies. Unless you have some special need, it is
recommended to just follow the simple,
:ref:`standard installation<standard_installation>` instructions.

The :ref:`standard<standard_installation>` way of installing CO\ *N*\ CEPT
utilizes the ``installer`` script, which installs CO\ *N*\ CEPT and all its
dependencies. Many more details about how to use this script for more
customized installation is available under the
':ref:`installer script in-depth<the_installer_script_in_depth>`' entry.

Though not recommended, you may choose to not make use of the ``installer``
script, in which case you need to install all of the
:ref:`dependencies<dependencies>` yourself. After
`downloading <https://github.com/jmd-dk/concept>`_ the CO\ *N*\ CEPT source
code, you then need to
:ref:`specify environment information<the_paths_and_env_files>` by editing
the ``.paths`` and ``.env`` files.


Entries on this page:

.. contents:: :local:



.. _standard_installation:

Standard installation
---------------------
The easiest way to install CO\ *N*\ CEPT along with all of its dependencies is
to use the CO\ *N*\ CEPT ``installer`` script. This script is part of the
CO\ *N*\ CEPT source code, and so if you have CO\ *N*\ CEPT already downloaded,
you may find and run this script. However, the simplest way to run the script
is to just execute

.. code-block:: bash

   concept_version=master
   bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer)

in a terminal, which will fetch and run the ``installer`` script directly from
the CO\ *N*\ CEPT GitHub repository, specifically the master version of the
script, which will install the corresponding master version of CO\ *N*\ CEPT.
Check out the CO\ *N*\ CEPT
`releases <https://github.com/jmd-dk/concept/releases>`_ for other available
release versions, or use ``concept_version=master`` for the absolute newest
(and unstable!) version.

.. note::
   The initial ``bash`` in the above command is required regardless of your
   shell

The ``installer`` will prompt you for an installation directory, as well as
for permission to install each :ref:`system dependency<system_dependencies>`,
should any be missing. At the very beginning of the installation, a table of
software to be installed will be shown.

CO\ *N*\ CEPT along with all of the libraries it needs will be installed into
the chosen installation directory. To completely remove the CO\ *N*\ CEPT
installation, simply remove this directory.

The installation will take an hour or two on modern hardware. Should the
installation process end prematurely (e.g. if you kill it yourself), simply
rerun the installation command and it will pick up from where it was.




.. _the_installer_script_in_depth:

The ``installer`` script in-depth
---------------------------------
The ``installer`` script may be run with additional options or influential
environment variables, which can be used to e.g. request for a specific
version of a dependency to be installed, or to skip the installation of a
particular dependency and instead make use of a preinstalled version of the
same library. Before discussing such features, it is good to know exactly
what dependencies are installed.


Programs installed
..................
The ``installer`` partitions all dependencies into *library* and *system*
dependencies. Unless explicitly specified, the ``installer`` installs all
library dependencies regardless of whether these already exist on the system.
The system dependencies consist of standard tools commonly installed
system-wide. If the ``installer`` detects a missing system dependency, it will
prompt for system-wide (root) installation through the package manager on the
system.

The ``installer`` script is able to bootstrap itself up from just bash, GNU
core utilities and a package manager (or just bash and GNU core utilities if
all system dependencies are already present). In addition, it builds all
library dependencies from source, increasing the total number of dependencies
(as many dependencies have other dependencies, and so on). For the absolute
minimum dependency requirements needed to *run* CO\ *N*\ CEPT, see the
:ref:`dependencies<dependencies>` entry.

The complete set of system dependencies needed/installed by the ``installer``
is:

* **GNU tools**: awk, grep, sed, gzip, tar, wget.
* **Build tools**: gcc, g++, gfortran, glibc, GNU make, as, ld, Linux headers.

The complete list of all library dependencies (and their dependency relations)
installed by the ``installer`` is given below:

* **zlib**
* **libpng**
* **FreeType**
* **BLAS** and **LAPACK**
* **MPI**
* **FFTW 3**
* **FFTW 2**
* **ncurses**
* **libffi**
* **Perl**
* **OpenSSL** (depends on Perl)
* **GSL** (depends on BLAS)
* **HDF5** (depends on zlib)
* **GADGET** (depends on MPI, GSL, FFTW 2)
* **Python 3** (depends on zlib)

  - **pip** (depends on OpenSSL, libffi), **setuptools** and **wheel**;
    needed to install the packages below.

    - **Blessings** (depends on ncurses)
    - **Cython**
    - **CythonGSL** (depends on Cython, GSL)
    - **pytest**
    - **NumPy** (depends on BLAS, LAPACK, pytest)
    - **SciPy** (depends on BLAS, LAPACK, pytest)
    - **matplotlib** (depends on libpng, FreeType)
    - **MPI4Py** (depends on MPI, Cython)
    - **H5Py** (depends on HDF5 and MPI)
    - **Sphinx**
    - **sphinx_rtd_theme**

* **CLASS** + **classy** (depends on Cython, NumPy)

Finally, CO\ *N*\ CEPT itself depends on MPI, FFTW, GADGET, Python, Blessings, Cython, CythonGSL, NumPy, SciPy, matplotlib, MPI4Py, H5Py, classy, Sphinx, sphinx_rtd_theme.

The ``installer`` installs the `OpenBLAS <https://github.com/xianyi/OpenBLAS>`_
library in order to provide both BLAS and LAPACK. For MPI, MPICH (default) or
OpenMPI is installed.


Command-line options
....................
When invoking the ``installer`` --- whether a local copy or directly off of
GitHub --- you may supply optional command-line arguments, the most useful of
which is probably the installation path. That is,

.. code-block:: bash

   ./installer /path/to/concept

(when running a local copy of the ``installer``) or something like

.. code-block:: bash

   concept_version=master
   bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer) /path/to/concept

(when running directly off of GitHub) will not prompt you for an installation
directory but instead use the supplied ``/path/to/concept``.


Other command-line options to ``installer`` are listed below.


.. raw:: html

   <h6>
     Help:
     <code class="docutils literal notranslate"><span class="pre">
       -h
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --help
     </span></code>
   </h6>

This prints out a short description of how to use the ``installer`` script
and then exits. Generally though, this documentation page is much preferable.


.. raw:: html

   <h6>
     Test:
     <code class="docutils literal notranslate"><span class="pre">
       -t
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --test
     </span></code>
   </h6>

With this option, close to all dependency programs will be tested after their
individual installations. CO\ *N*\ CEPT itself will similarly be tested. On
failure (fatal or non-fatal) of any test, a log file of the test output will
be placed in the installation subdirectory of the given program. Any test
failures will be reported at the end of the entire installation process.

This option is helpful for debugging if it is known that the installation
results in a non-functioning CO\ *N*\ CEPT, but it is unknown which of the
many dependencies does not function correctly. Generally though, this option
is not recommended as it increases the installation time by a couple of hours.

The tests performed on the CO\ *N*\ CEPT code itself are those of

.. code:: python3

   ./concept -t all

and so may also be run at any time after the installation, whether or not the
``-t`` option was used for th installation. See the ``concept``
:ref:`test<specials>` option for further details.


.. raw:: html

   <h6>
     Yes:
     <code class="docutils literal notranslate"><span class="pre">
       -y
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --yes
     </span></code>
   </h6>

Assume "yes" as answer to all prompts and run non-interactively. Currently
these include only the system-wide installations of system dependencies,
should any be missing. Note that this require root access.

.. raw:: html

   <h6>
     Fix ssh:
     <code class="docutils literal notranslate"><span class="pre">
       --fix-ssh
     </span></code>
   </h6>

.. warning::
   Do *not* use this option if you seek to install CO\ *N*\ CEPT

Invoking the installer with this option will not install CO\ *N*\ CEPT, but
instead attempt to configure the users ``~/.ssh`` directory for use with
remote jobs running on multiple nodes, as described
:ref:`here<problems_when_using_multiple_nodes>`.


Influential environment variables
.................................
*Under construction!*




.. _dependencies:

Dependencies
------------
This entry lists the dependency stack of CO\ *N*\ CEPT. Knowledge about this
stack is not needed if using the installation script (*highly* recommended!),
but it is important if for some reason you want to build (parts of) this stack
yourself.

Typically the exact version of any given dependency is not crucial. An effort
is made to ensure that CO\ *N*\ CEPT functions with the newest stable versions
of each dependency. As many of the CO\ *N*\ CEPT dependencies also depend on
each other, finding a working set of dependency versions may be non-trivial.
You may draw inspiration from the current or older versions of the
`installation script <https://github.com/jmd-dk/concept/blob/master/installer>`_
(look for 'Specification of software versions').


.. topic:: Python dependencies

   The CO\ *N*\ CEPT source depends explicitly on
   `Python <https://www.python.org/>`_ version 3.6 or newer, together with the
   following Python packages (many of which have heavy dependencies of their
   own):

   * `Cython <https://cython.org/>`_: Needed for transpilation
     (*cythonization*) of the pure Python source code of CO\ *N*\ CEPT into
     equivalent C code.

   * `NumPy <https://www.numpy.org/>`_: Provides the basic array types used
     for representing all primary data, together with various array/math
     functions.

   * `SciPy <https://www.scipy.org/>`_: Provides various numerical methods and
     additional math functions.

   * `Matplotlib <https://matplotlib.org/>`_: Provides plotting functionality
     for 2D and 3D figures, as well as color transformations used for colored
     terminal output.

   * `MPI4Py <https://mpi4py.readthedocs.io/>`_: Provides Python bindings for
     MPI, used for all inter-process communication.

   * `H5Py <https://www.h5py.org/>`_: Provides Python bindings for HDF5, used
     for various binary input/output.

   * `CythonGSL <https://github.com/twiecki/CythonGSL>`_: Provides Cython
     bindings for GSL, used for more performant replacements of some
     NumPy/SciPy functionalities when running CO\ *N*\ CEPT in compiled mode.

   * `Blessings <https://github.com/erikrose/blessings>`_: Provides terminal
     formatting.

   In addition, the `Sphinx <http://www.sphinx-doc.org/>`_ and
   `sphinx_rtd_theme <https://sphinx-rtd-theme.readthedocs.io/>`_ Python
   packages are needed to build the documentation, but may otherwise be left
   out.



.. topic:: Other primary dependencies

   In addition to Python, the Python packages listed above and their
   respective dependencies, CO\ *N*\ CEPT further depends explicitly on
   `FFTW <http://www.fftw.org/>`_ 3 for its distributed FFT capabilities.

   .. note::
      CO\ *N*\ CEPT does not make use of the Python bindings
      `pyFFTW <https://github.com/pyFFTW/pyFFTW>`_ for FFTW, as these do not
      include the distributed (MPI) FFT's needed. Instead, CO\ *N*\ CEPT
      provides its own minimal wrapper, ``fft.c``. This is the only C file in
      the primary CO\ *N*\ CEPT source code.

   If building FFTW yourself, remember to link against an MPI library.
   The same goes for building HDF5 and installing MPI4Py and H5Py. Also, the
   MPI library has to conform to the MPI-3 standard.

   For testing, CO\ *N*\ CEPT compares itself against
   `GADGET <https://wwwmpa.mpa-garching.mpg.de/gadget/>`_, specifically
   version 2.0.7. If you do not care about running the CO\ *N*\ CEPT test
   suite, you do not have to install GADGET.

   The last non-standard depenency of CO\ *N*\ CEPT is the
   `CLASS <http://class-code.net/>`_ code, along with its Python wrapper
   'classy'. When installing using the installation script, CLASS and classy
   are patched in order to enable larger and new kinds of output, fix bugs and
   improve interoperability with CO\ *N*\ CEPT. If installing without the use
   of the installation script, you will have to obtain the patched
   CLASS + classy by some other means. A good solution is to install
   everything else first, including CO\ *N*\ CEPT itself, and then make use of
   the CO\ *N*\ CEPT ``update`` utility to install and patch CLASS + classy:

   .. code-block:: bash

      ./concept -u update --class <version>

   Here, ``<version>`` should be replaced with the CO\ *N*\ CEPT version whose
   ``installer`` script should be used to install and patch CLASS + classy.


.. _system_dependencies:

.. topic:: System dependencies

   In addition to the many dependencies above, CO\ *N*\ CEPT further uses
   a lot of *system dependencies*, by which is meant programs that usually is
   installed system-wide. These include
   `bash <https://www.gnu.org/software/bash/>`_ 3.0 or newer and the
   `GNU core utilities <https://www.gnu.org/software/coreutils/>`_, which are
   the only two dependencies not installed by the ``installer`` script.

   Other system dependencies needed for the core CO\ *N*\ CEPT functionality
   are awk, grep and sed. Also, the ``installer`` script and ``update``
   utility further makes use of gzip, tar and wget. That is, you may run
   simulations without these last three components installed. If running the
   ``installer`` script or ``update`` utility without these, you will be
   prompted for system-wide (root) installation.

   .. note::

      Several implementations exist for the above system dependencies.
      CO\ *N*\ CEPT specifically needs the GNU implementations, i.e. what
      is commonly found on Linux systems. The fact that macOS uses the BSD
      implementations of these programs is the primary reason for
      CO\ *N*\ CEPT not yet being ported to this platform.

   Lastly, CO\ *N*\ CEPT needs standard tools for compiling and linking C
   (C99) code. An ``mpicc`` C compiler/linker should be bundled with the MPI
   library used. The GNU make utility is also needed.



.. _the_paths_and_env_files:


The ``.paths`` and ``.env`` files
---------------------------------
*Under construction!*


