# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2020 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from mesh import interpolate_in_vectorgrid, tabulate_vectorgrid')



# Cython function for computing Ewald correction
@cython.header(
    # Argument
    x='double',
    y='double',
    z='double',
    # Locals
    dist='double',
    dist_x='double',
    dist_y='double',
    dist_z='double',
    dist2='double',
    force_x='double',
    force_y='double',
    force_z='double',
    h2='int',
    kx='double',
    ky='double',
    kz='double',
    r3='double',
    scalarpart='double',
    sumindex_x='int',
    sumindex_y='int',
    sumindex_z='int',
    returns='double*',
)
def summation(x, y, z):
    """This function performs the Ewald summation given the distance
    x, y, z between two particles, normalized so that
    0 <= |x|, |y|, |z| < 1 (corresponding to boxsize = 1). The equation
    being solved corresponds to (8) in Ralf Klessen's 'GRAPESPH with
    Fully Periodic Boundary Conditions: Fragmentation of Molecular
    Clouds', though it is written without the normalization. The actual
    Ewald force is then given by force/boxsize**2. What is returned is
    the Ewald correction, corresponding to (9) in the mentioned paper.
    That is, the return value is not the total force, but the force from
    all particle images except the nearest one. Note that this nearest
    image need not be the actual particle.
    """
    # The Ewald force vector components
    force_x = force_y = force_z = 0
    # The image is on top of the particle: No force
    if x == 0 and y == 0 and z == 0:
        ewald_force[0] = 0
        ewald_force[1] = 0
        ewald_force[2] = 0
        return ewald_force
    # Remove the direct force, as we
    # are interested in the correction only
    r3 = x**2 + y**2 + z**2
    r3 *= sqrt(r3)
    force_x += x*ℝ[1/r3]
    force_y += y*ℝ[1/r3]
    force_z += z*ℝ[1/r3]
    # The short range (real space) sum
    for sumindex_x in range(n_lower, n_upper):
        dist_x = x - sumindex_x
        for sumindex_y in range(n_lower, n_upper):
            dist_y = y - sumindex_y
            for sumindex_z in range(n_lower, n_upper):
                dist_z = z - sumindex_z
                dist2 = ℝ[ℝ[dist_x**2] + dist_y**2] + dist_z**2
                if dist2 > ℝ[maxdist**2]:
                    continue
                dist = sqrt(dist2)
                scalarpart = -dist**(-3)*(erfc(dist*ℝ[1/(2*rs)])
                    + dist*ℝ[1/(sqrt(π)*rs)]*exp(dist2*ℝ[-1/(4*rs**2)]))
                force_x += dist_x*scalarpart
                force_y += dist_y*scalarpart
                force_z += dist_z*scalarpart
    # The long range (Fourier space) sum
    for sumindex_x in range(h_lower, h_upper):
        kx = ℝ[2*π]*sumindex_x
        for sumindex_y in range(h_lower, h_upper):
            ky = ℝ[2*π]*sumindex_y
            for sumindex_z in range(h_lower, h_upper):
                h2 = ℤ[ℤ[sumindex_x**2] + sumindex_y**2] + sumindex_z**2
                if h2 > maxh2 or h2 == 0:
                    continue
                kz = ℝ[2*π]*sumindex_z
                k2 = ℝ[ℝ[kx**2] + ky**2] + kz**2
                scalarpart = ℝ[-4*π]/k2*exp(-k2*ℝ[rs**2])*sin(kx*x + ky*y + kz*z)
                force_x += kx*scalarpart
                force_y += ky*scalarpart
                force_z += kz*scalarpart
    # Pack and return Ewald force
    ewald_force[0] = force_x
    ewald_force[1] = force_y
    ewald_force[2] = force_z
    return ewald_force
# Vector used as the return value
# of the summation function.
cython.declare(ewald_force='double*')
ewald_force = malloc(3*sizeof('double'))

# Master function of this module. Returns the Ewald force correction.
@cython.header(
    # Arguments
    x='double',
    y='double',
    z='double',
    # Locals
    dim='int',
    force='double*',
    isnegative_x='bint',
    isnegative_y='bint',
    isnegative_z='bint',
    returns='double*',
)
def ewald(x, y, z):
    """This function performs a look up of the Ewald correction to the
    fully periodic gravitational force (corresponding to 1/r²) on a
    particle due to some other particle at a position (x, y, z) relative
    to the first particle. It is important that the passed coordinates
    are of the nearest periodic image of the other particle, and not
    necessarily of the particle itself. This means that
    0 <= |x|, |y|, |z| < boxsize/2. The returned value is thus the force
    arising on the first particle due to all periodic images of the
    second particle, except for the nearest one.
    """
    # Only the positive octant of the box is tabulated. Flip the sign of
    # the coordinates so that they reside inside this octant.
    if x > 0:
        isnegative_x = False
    else:
        x *= -1
        isnegative_x = True
    if y > 0:
        isnegative_y = False
    else:
        y *= -1
        isnegative_y = True
    if z > 0:
        isnegative_z = False
    else:
        z *= -1
        isnegative_z = True
    # Look up Ewald force and do a CIC interpolation. Since the
    # coordinates are to the nearest image, they must be scaled by
    # 2/boxsize to reside in the range 0 <= x, y, z < 1. Furthermore, a
    # scaling of (ewald_gridsize - 1) is applied to achieve
    # 0 <= x, y, z < grid.shape - 1, as required by
    # interpolate_in_vectorgrid().
    force = interpolate_in_vectorgrid(
        grid,
        x*ℝ[2/boxsize*(ewald_gridsize - 1)*(1 - machine_ϵ)],
        y*ℝ[2/boxsize*(ewald_gridsize - 1)*(1 - machine_ϵ)],
        z*ℝ[2/boxsize*(ewald_gridsize - 1)*(1 - machine_ϵ)],
        2,  # CIC interpolation
    )
    # Put the sign back in for negative input
    if isnegative_x:
        force[0] *= -1
    if isnegative_y:
        force[1] *= -1
    if isnegative_z:
        force[2] *= -1
    # The tabulated force is for a unit box. Do rescaling.
    for dim in range(3):
        force[dim] *= ℝ[1/boxsize**2]
    return force

# Function for loading the Ewald grid from disk.
# The result is stored as the global variable 'grid',
# which will be fetched when called repeatedly.
@cython.header(
    found_on_disk='bint',
    shape=tuple,
    returns='double[:, :, :, ::1]',
)
def get_ewald_grid():
    global grid
    # If the Ewald grid already exist in memory, return it
    if grid is not None:
        return grid
    # Let the master process read in the Ewald grid from disk,
    # if it exists.
    shape = (ewald_gridsize, )*3 + (3, )
    found_on_disk = False
    if master:
        if os.path.isfile(filename):
            # Ewald grid already tabulated. Load it from disk.
            found_on_disk = True
            with open_hdf5(filename, mode='r') as hdf5_file:
                grid = hdf5_file['data'][...].reshape(shape)
    found_on_disk = bcast(found_on_disk)
    if found_on_disk:
        # Ewald grid loaded by the master process.
        # Broadcast it to all slave processes.
        if not master:
            grid = empty(shape, dtype=C2np['double'])
        Bcast(grid)
    else:
        # No tabulated Ewald grid found. Compute it. The factor 0.5
        # ensures that only the first octant of the box is tabulated.
        grid = tabulate()
    return grid

# Function for tabulation of the Ewald grid
@cython.pheader(grid='double[:, :, :, ::1]', returns='double[:, :, :, ::1]')
def tabulate():
    masterprint(f'Tabulating Ewald grid of size {ewald_gridsize} ...')
    grid = tabulate_vectorgrid(ewald_gridsize, summation, 0.5/(ewald_gridsize - 1), filename)
    masterprint('done')
    return grid



# The global Ewald grid and its path on disk
cython.declare(grid='double[:, :, :, ::1]', filename=str)
grid = None
filename = f'{paths["reusables_dir"]}/ewald/{ewald_gridsize}.hdf5'

# Set parameters for the Ewald summation at import time
cython.declare(
    rs='double',
    maxdist='double',
    maxh2='int',
    h_lower='int',
    h_upper='int',
    n_lower='int',
    n_upper='int',
)
# The values chosen match those listed in the article mentioned in the
# docstring of the summation function. These are also those used
# (in effect) in GADGET2.
rs = 0.25  # Corresponds to alpha = 2
maxdist = 3.6
maxh2 = 10
# Derived constants
h_lower = int(-sqrt(maxh2))      # GADGET: -4 (same here for maxh2 = 10)
h_upper = int(+sqrt(maxh2)) + 1  # GADGET:  5 (same here for maxh2 = 10)
n_lower = int(-(maxdist + 1))    # GADGET: -4 (same here for maxdist = 3.6)
n_upper = int(maxdist + 1) + 1   # GADGET:  5 (same here for maxdist = 3.6)
