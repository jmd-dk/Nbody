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
cimport('from ewald import ewald')
cimport('from interactions import particle_particle')

# Function implementing pairwise gravity (full/periodic)
@cython.header(
    # Arguments
    interaction_name=str,
    receiver='Component',
    supplier='Component',
    ᔑdt_rungs=dict,
    rank_supplier='int',
    only_supply='bint',
    pairing_level=str,
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier_paired='Py_ssize_t**',
    tile_indices_supplier_paired_N='Py_ssize_t*',
    extra_args=dict,
    # Locals
    apply_to_i='bint',
    apply_to_j='bint',
    force_ij='double*',
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    gravity_factor='double',
    gravity_factors='const double[::1]',
    gravity_factors_ptr='const double*',
    i='Py_ssize_t',
    j='Py_ssize_t',
    particle_particle_t_begin='double',
    particle_particle_t_final='double',
    r3='double',
    rung_index_i='signed char',
    rung_index_j='signed char',
    softening2='double',
    subtiling_r='Tiling',
    x_ji='double',
    y_ji='double',
    z_ji='double',
    Δmomx_r='double*',
    Δmomx_s='double*',
    Δmomy_r='double*',
    Δmomy_s='double*',
    Δmomz_r='double*',
    Δmomz_s='double*',
    returns='void',
)
def gravity_pairwise(
    interaction_name, receiver, supplier, ᔑdt_rungs, rank_supplier, only_supply, pairing_level,
    tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
    extra_args,
):
    # Extract momentum update buffers
    Δmomx_r = receiver.Δmomx
    Δmomy_r = receiver.Δmomy
    Δmomz_r = receiver.Δmomz
    Δmomx_s = supplier.Δmomx
    Δmomy_s = supplier.Δmomy
    Δmomz_s = supplier.Δmomz
    # The combined, squared softening length
    softening2 = (0.5*(receiver.softening_length + supplier.softening_length))**2
    # Construct array of factors used for momentum updates:
    #   Δmom = -r⃗/r³*G*mass_r*mass_s*Δt/a.
    # In the general case of decaying particles,
    # the mass of each particle is
    #   mass(a) = component.mass*a**(-3*component.w_eff(a=a)).
    # Below we integrate over the time dependence.
    # The array should be indexed with the rung_index
    # of the receiver/supplier particle.
    gravity_factors = G_Newton*receiver.mass*supplier.mass*ᔑdt_rungs[
        'a**(-3*w_eff₀-3*w_eff₁-1)', receiver.name, supplier.name]
    gravity_factors_ptr = cython.address(gravity_factors[:])
    # Loop over all (receiver, supplier) particle pairs (i, j)
    j = -1
    for i, j, rung_index_i, rung_index_j, x_ji, y_ji, z_ji, apply_to_i, apply_to_j, particle_particle_t_begin, subtiling_r in particle_particle(
        receiver, supplier, pairing_level,
        tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
        rank_supplier, interaction_name, only_supply,
    ):
        # Translate coordinates so they correspond to the nearest image
        if x_ji > ℝ[0.5*boxsize]:
            x_ji -= boxsize
        elif x_ji < ℝ[-0.5*boxsize]:
            x_ji += boxsize
        if y_ji > ℝ[0.5*boxsize]:
            y_ji -= boxsize
        elif y_ji < ℝ[-0.5*boxsize]:
            y_ji += boxsize
        if z_ji > ℝ[0.5*boxsize]:
            z_ji -= boxsize
        elif z_ji < ℝ[-0.5*boxsize]:
            z_ji += boxsize
        # The Ewald correction force for all images except the
        # nearest one, which might not be the actual particle.
        force_ij = ewald(x_ji, y_ji, z_ji)
        # Add in the force from the particle's nearest image
        r3 = x_ji**2 + y_ji**2 + z_ji**2 + softening2
        r3 *= sqrt(r3)
        forcex_ij = force_ij[0] - x_ji*ℝ[1/r3]
        forcey_ij = force_ij[1] - y_ji*ℝ[1/r3]
        forcez_ij = force_ij[2] - z_ji*ℝ[1/r3]
        # Momentum change of particle i due to particle j
        with unswitch(3):
            if apply_to_i:
                gravity_factor = gravity_factors_ptr[rung_index_i]
                Δmomx_r[i] += forcex_ij*gravity_factor
                Δmomy_r[i] += forcey_ij*gravity_factor
                Δmomz_r[i] += forcez_ij*gravity_factor
        # Momentum change of particle j due to particle i
        with unswitch:
            if 𝔹[not only_supply]:
                with unswitch(2):
                    if apply_to_j:
                        gravity_factor = gravity_factors_ptr[rung_index_j]
                        Δmomx_s[j] -= forcex_ij*gravity_factor
                        Δmomy_s[j] -= forcey_ij*gravity_factor
                        Δmomz_s[j] -= forcez_ij*gravity_factor
    # Add computation time to the running total,
    # for use with automatic subtiling refinement.
    if j != -1:
        particle_particle_t_final = time()
        subtiling_r.computation_time += particle_particle_t_final - particle_particle_t_begin

# Function implementing pairwise gravity (short-range only)
@cython.header(
    # Arguments
    interaction_name=str,
    receiver='Component',
    supplier='Component',
    ᔑdt_rungs=dict,
    rank_supplier='int',
    only_supply='bint',
    pairing_level=str,
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier_paired='Py_ssize_t**',
    tile_indices_supplier_paired_N='Py_ssize_t*',
    extra_args=dict,
    # Locals
    apply_to_i='bint',
    apply_to_j='bint',
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    gravity_factors='const double[::1]',
    gravity_factors_ptr='const double*',
    i='Py_ssize_t',
    j='Py_ssize_t',
    particle_particle_t_begin='double',
    particle_particle_t_final='double',
    r2='double',
    rung_index_i='signed char',
    rung_index_j='signed char',
    shortrange_factor='double',
    shortrange_index='Py_ssize_t',
    softening2='double',
    subtiling_r='Tiling',
    total_factor='double',
    x_ji='double',
    y_ji='double',
    z_ji='double',
    Δmomx_r='double*',
    Δmomx_s='double*',
    Δmomy_r='double*',
    Δmomy_s='double*',
    Δmomz_r='double*',
    Δmomz_s='double*',
    returns='void',
)
def gravity_pairwise_shortrange(
    interaction_name, receiver, supplier, ᔑdt_rungs, rank_supplier, only_supply, pairing_level,
    tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
    extra_args,
):
    # Extract momentum update buffers
    Δmomx_r = receiver.Δmomx
    Δmomy_r = receiver.Δmomy
    Δmomz_r = receiver.Δmomz
    Δmomx_s = supplier.Δmomx
    Δmomy_s = supplier.Δmomy
    Δmomz_s = supplier.Δmomz
    # The combined, squared softening length
    softening2 = (0.5*(receiver.softening_length + supplier.softening_length))**2
    # Construct array of factors used for momentum updates:
    #   Δmom = -r⃗/r³*G*mass_r*mass_s*Δt/a.
    # In the general case of decaying particles,
    # the mass of each particle is
    #   mass(a) = component.mass*a**(-3*component.w_eff(a=a)).
    # Below we integrate over the time dependence.
    # The array should be indexed with the rung_index
    # of the receiver/supplier particle.
    gravity_factors = G_Newton*receiver.mass*supplier.mass*ᔑdt_rungs[
        'a**(-3*w_eff₀-3*w_eff₁-1)', receiver.name, supplier.name]
    gravity_factors_ptr = cython.address(gravity_factors[:])
    # Loop over all (receiver, supplier) particle pairs (i, j)
    j = -1
    for i, j, rung_index_i, rung_index_j, x_ji, y_ji, z_ji, apply_to_i, apply_to_j, particle_particle_t_begin, subtiling_r in particle_particle(
        receiver, supplier, pairing_level,
        tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
        rank_supplier, interaction_name, only_supply,
    ):
        # Translate coordinates so they correspond to the nearest image
        if x_ji > ℝ[0.5*boxsize]:
            x_ji -= boxsize
        elif x_ji < ℝ[-0.5*boxsize]:
            x_ji += boxsize
        if y_ji > ℝ[0.5*boxsize]:
            y_ji -= boxsize
        elif y_ji < ℝ[-0.5*boxsize]:
            y_ji += boxsize
        if z_ji > ℝ[0.5*boxsize]:
            z_ji -= boxsize
        elif z_ji < ℝ[-0.5*boxsize]:
            z_ji += boxsize
        r2 = x_ji**2 + y_ji**2 + z_ji**2 + softening2
        # If the particle pair is separated by a distance larger
        # than the range of the short-range force,
        # ignore this interaction completely.
        if r2 > ℝ[shortrange_params['gravity']['range']**2]:
            continue
        # Compute the short-range force. Here the "force" is in units
        # of inverse length squared, given by
        # force = -r⃗/r³ (x/sqrt(π) exp(-x²/4) + erfc(x/2)),
        # where x = r/scale with scale the long/short-range
        # force split scale.
        # We have this whole expression except for r⃗ already tabulated.
        shortrange_index = int(r2*ℝ[(shortrange_table_size - 1)/shortrange_table_maxr2])
        shortrange_factor = shortrange_table_ptr[shortrange_index]
        # Momentum change of particle i due to particle j
        with unswitch(3):
            if apply_to_i:
                total_factor = shortrange_factor*gravity_factors_ptr[rung_index_i]
                Δmomx_r[i] += x_ji*total_factor
                Δmomy_r[i] += y_ji*total_factor
                Δmomz_r[i] += z_ji*total_factor
        # Momentum change of particle j due to particle i
        with unswitch:
            if 𝔹[not only_supply]:
                with unswitch(2):
                    if apply_to_j:
                        total_factor = shortrange_factor*gravity_factors_ptr[rung_index_j]
                        Δmomx_s[j] -= x_ji*total_factor
                        Δmomy_s[j] -= y_ji*total_factor
                        Δmomz_s[j] -= z_ji*total_factor
    # Add computation time to the running total,
    # for use with automatic subtiling refinement.
    if j != -1:
        particle_particle_t_final = time()
        subtiling_r.computation_time += particle_particle_t_final - particle_particle_t_begin

# Helper function for the gravity_pairwise_shortrange function,
# which initializes the global shortrange table.
@cython.header(
    # Locals
    component='Component',
    i='Py_ssize_t',
    r='double[::1]',
    r_center='double',
    shortrange_table_nonconst='double[::1]',
    x_center='double',
    returns='void',
)
def tabulate_shortrange_gravity():
    """This function tabulates the short-range factor
    y = -r⁻³(x/sqrt(π)exp(-x²/4) + erfc(x/2))
    in the global shortrange_table array. The tabulation is quadratic
    in r, which is the distance between two particles, while x = r/scale
    with scale the long/short-range force split scale.
    We only need the tabulation for 0 <= r <= range, where range
    is the maximum reach of the short-range force.
    """
    global shortrange_table, shortrange_table_ptr
    if shortrange_table is not None:
        return
    # The distances at which the tabulation will be carried out,
    # quadratically spaced.
    r = np.sqrt(linspace(0, shortrange_table_maxr2, shortrange_table_size))
    # Create the table. The i'th element of shortrange_table really
    # corresponds to the value at r[i+½]. Nearest grid point lookups
    # can then be performed by cheap floor (int casting) indexing.
    shortrange_table_nonconst = empty(shortrange_table_size, dtype=C2np['double'])
    for i in range(shortrange_table_size):
        if i == ℤ[shortrange_table_size - 1]:
            r_center = r[i]
        else:
            r_center = 0.5*(r[i] + r[i+1])
        x_center = r_center*ℝ[1/shortrange_params['gravity']['scale']]
        shortrange_table_nonconst[i] = -r_center**(-3)*(
            ℝ[1/sqrt(π)]*x_center*exp(-ℝ[0.5*x_center]**2) + erfc(ℝ[0.5*x_center])
        )
    # Assign global variables
    shortrange_table = shortrange_table_nonconst
    shortrange_table_ptr = cython.address(shortrange_table[:])
# Global variables used by the tabulate_shortrange_gravity()
# and gravity_pairwise_shortrange() functions.
cython.declare(
    shortrange_table='const double[::1]',
    shortrange_table_ptr='const double*',
    shortrange_table_size='Py_ssize_t',
    shortrange_table_maxr2='double',
)
shortrange_table = None
shortrange_table_ptr = NULL
shortrange_table_size = 2**14  # Lower value improves caching, but leads to inaccurate lookups
shortrange_table_maxr2 = (1 + 1e-3)*shortrange_params['gravity']['range']**2

# Function implementing pairwise gravity (non-periodic)
@cython.header(
    # Arguments
    interaction_name=str,
    receiver='Component',
    supplier='Component',
    ᔑdt_rungs=dict,
    rank_supplier='int',
    only_supply='bint',
    pairing_level=str,
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier_paired='Py_ssize_t**',
    tile_indices_supplier_paired_N='Py_ssize_t*',
    extra_args=dict,
    # Locals
    apply_to_i='bint',
    apply_to_j='bint',
    forcex_ij='double',
    forcey_ij='double',
    forcez_ij='double',
    gravity_factor='double',
    gravity_factors='const double[::1]',
    gravity_factors_ptr='const double*',
    i='Py_ssize_t',
    j='Py_ssize_t',
    particle_particle_t_begin='double',
    particle_particle_t_final='double',
    r3='double',
    rung_index_i='signed char',
    rung_index_j='signed char',
    subtiling_r='Tiling',
    x_ji='double',
    y_ji='double',
    z_ji='double',
    Δmomx_r='double*',
    Δmomx_s='double*',
    Δmomy_r='double*',
    Δmomy_s='double*',
    Δmomz_r='double*',
    Δmomz_s='double*',
    returns='void',
)
def gravity_pairwise_nonperiodic(
    interaction_name, receiver, supplier, ᔑdt_rungs, rank_supplier, only_supply, pairing_level,
    tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
    extra_args,
):
    # Extract momentum update buffers
    Δmomx_r = receiver.Δmomx
    Δmomy_r = receiver.Δmomy
    Δmomz_r = receiver.Δmomz
    Δmomx_s = supplier.Δmomx
    Δmomy_s = supplier.Δmomy
    Δmomz_s = supplier.Δmomz
    # Construct array of factors used for momentum updates:
    #   Δmom = -r⃗/r³*G*mass_r*mass_s*Δt/a.
    # In the general case of decaying particles,
    # the mass of each particle is
    #   mass(a) = component.mass*a**(-3*component.w_eff(a=a)).
    # Below we integrate over the time dependence.
    # The array should be indexed with the rung_index
    # of the receiver/supplier particle.
    gravity_factors = G_Newton*receiver.mass*supplier.mass*ᔑdt_rungs[
        'a**(-3*w_eff₀-3*w_eff₁-1)', receiver.name, supplier.name]
    gravity_factors_ptr = cython.address(gravity_factors[:])
    # Loop over all (receiver, supplier) particle pairs (i, j)
    j = -1
    for i, j, rung_index_i, rung_index_j, x_ji, y_ji, z_ji, apply_to_i, apply_to_j, particle_particle_t_begin, subtiling_r in particle_particle(
        receiver, supplier, pairing_level,
        tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
        rank_supplier, interaction_name, only_supply,
    ):
        # The direct force on particle i from particle j
        r3 = (x_ji**2 + y_ji**2 + z_ji**2
            + ℝ[(0.5*(receiver.softening_length + supplier.softening_length))**2]
        )
        r3 *= sqrt(r3)
        forcex_ij = x_ji*ℝ[-1/r3]
        forcey_ij = y_ji*ℝ[-1/r3]
        forcez_ij = z_ji*ℝ[-1/r3]
        # Momentum change to particle i due to particle j
        with unswitch(3):
            if apply_to_i:
                gravity_factor = gravity_factors_ptr[rung_index_i]
                Δmomx_r[i] += forcex_ij*gravity_factor
                Δmomy_r[i] += forcey_ij*gravity_factor
                Δmomz_r[i] += forcez_ij*gravity_factor
        # Momentum change of particle j due to particle i
        with unswitch:
            if 𝔹[not only_supply]:
                with unswitch(2):
                    if apply_to_j:
                        gravity_factor = gravity_factors_ptr[rung_index_j]
                        Δmomx_s[j] -= forcex_ij*gravity_factor
                        Δmomy_s[j] -= forcey_ij*gravity_factor
                        Δmomz_s[j] -= forcez_ij*gravity_factor
    # Add computation time to the running total,
    # for use with automatic subtiling refinement.
    if j != -1:
        particle_particle_t_final = time()
        subtiling_r.computation_time += particle_particle_t_final - particle_particle_t_begin

# Function implementing the gravitational potential (in Fouier space).
# Here k2 = k² is the squared magnitude of the wave vector,
# in physical units.
@cython.header(
    k2='double',
    returns='double',
)
def gravity_potential(k2):
    return ℝ[-4*π*G_Newton]/k2

# Function implementing just the long-range part
# of the gravitational potential (in Fouier space).
# Here k2 = k² is the squared magnitude of the wave vector,
# in physical units.
@cython.header(
    k2='double',
    returns='double',
)
def gravity_longrange_potential(k2):
    return exp(k2*ℝ[-shortrange_params['gravity']['scale']**2])*gravity_potential(k2)

