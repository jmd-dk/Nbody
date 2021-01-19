/*
This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
Copyright © 2015–2021 Jeppe Mosgaard Dakin.

CO𝘕CEPT is free software: You can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CO𝘕CEPT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/

The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
The latest version of CO𝘕CEPT is available at
https://github.com/jmd-dk/concept/
*/



#include <fftw3-mpi.h>
#include <string.h>

/* This file defines the functions fftw_setup and fftw_clean, which
 * together with fftw_execute (included in fftw3-mpi.h) constitutes the
 * necessary functions for using FFTW to do parallel, real, 3D in-place
 * transforms through Cython.
 */

/* Note on indexing
 *
 * In real space, before any transformations,
 * provided that we execute this on the correct process, where
 *   gridstart_local_i <= i && i < gridstart_local_i + gridsize_local_i,
 * we have
 *   grid[i, j, k] --> grid[
 *       ((i - gridstart_local_i)*gridsize_j + j)*gridsize_padding + k
 *   ]
 * That is, each process can acces every element in its slab by
 *   for (i = 0; i < gridsize_local_i; i++) {
 *       for (j = 0; j < gridsize_j; j++) {
 *           for (k = 0; k < gridsize_k; k++) {
 *               // This is the [i + gridstart_local_i, j, k]'th element
 *               double element = grid[
 *                   (i*gridsize_j + j)*gridsize_padding + k
 *               ];
 *           }
 *       }
 *   }
 *
 * After a forward, in-place r2c transformation,
 * provided that we execute this on the correct process, where
 *   gridstart_local_j <= j && j < gridstart_local_j + gridsize_local_j,
 * we have
 *   grid[i, j, k] --> grid[
 *       ((j - gridstart_local_j)*gridsize_i + i)*gridsize_padding + k
 *   ]
 * That is, each process can acces every element in its slab by
 *   for (i = 0; i < gridsize_i; i++) {
 *       for (j = 0; j < gridsize_local_j; j++) {
 *           for (k = 0; k < gridsize_k; k++) {
 *               // This is the [i, j + gridstart_local_j, k]'th element
 *               double element = grid[
 *                   (j*gridsize_i + i)*gridsize_padding + k
 *               ];
 *           }
 *       }
 *   }
 */

struct fftw_return_struct {
    ptrdiff_t gridsize_local_i;
    ptrdiff_t gridsize_local_j;
    ptrdiff_t gridstart_local_i;
    ptrdiff_t gridstart_local_j;
    double* grid;
    fftw_plan plan_forward;
    fftw_plan plan_backward;
};

struct plans_struct {
    fftw_plan forward;
    fftw_plan backward;
    int reused;
};

struct plans_struct make_plans(
    ptrdiff_t gridsize_i,
    ptrdiff_t gridsize_j,
    ptrdiff_t gridsize_k,
    double* grid,
    unsigned rigor_flag,
    int fftw_wisdom_reuse,
    char* wisdom_filename
);

/* This function initializes fftw_mpi, allocates a grid,
 * decides the local lengths and starting indices and
 * creates forwards and backwards plans.
 */
struct fftw_return_struct fftw_setup(
    ptrdiff_t gridsize_i,
    ptrdiff_t gridsize_j,
    ptrdiff_t gridsize_k,
    char* fftw_wisdom_rigor,
    int fftw_wisdom_reuse,
    char* wisdom_filename
) {
    /* Arguments to this function:
     * - Linear gridsize of dimension 1.
     * - Linear gridsize of dimension 2.
     * - Linear gridsize of dimension 3.
     * - FFTW planning-rigour, determining the optimization level of
     *   of the wisdom. In order of patience:
     *     "estimate", "measure", "patient", "exhaustive".
     * - Flag specifying whether or not to use pre-existing FFTW wisdom.
     */

    /* Size of last dimension with padding */
    ptrdiff_t gridsize_padding = 2*(gridsize_k/2 + 1);

    /* Convert fftw_wisdom_rigor to unsigned integer flag */
    unsigned rigor_flag = FFTW_ESTIMATE;
    if (strcmp(fftw_wisdom_rigor, "estimate") == 0)
        rigor_flag = FFTW_ESTIMATE;
    else if (strcmp(fftw_wisdom_rigor, "measure") == 0)
        rigor_flag = FFTW_MEASURE;
    else if (strcmp(fftw_wisdom_rigor, "patient") == 0)
        rigor_flag = FFTW_PATIENT;
    else if (strcmp(fftw_wisdom_rigor, "exhaustive") == 0)
        rigor_flag = FFTW_EXHAUSTIVE;

    /* Initialize parallel FFTW (note that MPI_Init should not be
     * called, as MPI is already running via MPI4Py).
     * This function may be called multiple times in one MPI session
     * without errors; only the first call will have any effect.
     */
    fftw_mpi_init();

    /* Declaration and allocation of the (local part of the) grid. This
     * also initializes gridsize_local_(i/j) and gridstart_local_(i/j).
     */
    ptrdiff_t gridsize_local_i, gridstart_local_i;
    ptrdiff_t gridsize_local_j, gridstart_local_j;
    double* grid = fftw_alloc_real(fftw_mpi_local_size_3d_transposed(
        gridsize_i,
        gridsize_j,
        gridsize_padding,
        MPI_COMM_WORLD,
        &gridsize_local_i,
        &gridstart_local_i,
        &gridsize_local_j,
        &gridstart_local_j
    ));

    /* Create the two plans, gathering wisdom in the process.
     * Because the generated wisdom is individual to each process,
     * the created plans may not be constructed from identical wisdom
     * on all processes, even though we gather and broadcast the wisdom
     * after plan creation. This is OK if the wisdom is not be reused.
     * If wisdom should be reused and already exists on the disk,
     * the wisdom is read and used directly to create the plans.
     * If wisdom should be reused and it is not present on disk,
     * we carry out the plan creation process twice, with a wisdom dump
     * to the disk in between. This way, the final plans all end up the
     * same as if the wisdom already existed on disk prior to calling
     * this function.
    */
    struct plans_struct plans;
    fftw_plan plan_forward;
    fftw_plan plan_backward;
    int wisdom_iteration;
    for (wisdom_iteration = 0; wisdom_iteration < 1 + fftw_wisdom_reuse; wisdom_iteration++) {
        /* Destroy plans and forget wisdom from first iteration */
        if (wisdom_iteration > 0) {
            fftw_destroy_plan(plan_forward);
            fftw_destroy_plan(plan_backward);
            fftw_forget_wisdom();
        }
        /* Create the two plans */
        plans = make_plans(
            gridsize_i,
            gridsize_j,
            gridsize_k,
            grid,
            rigor_flag,
            fftw_wisdom_reuse,
            wisdom_filename
        );
        plan_forward  = plans.forward;
        plan_backward = plans.backward;
        /* No further iteration needed if wisdom was read from disk */
        if (plans.reused)
            break;
    }

    /* Pack and return grid and plans */
    struct fftw_return_struct fftw_struct = {
        gridsize_local_i,
        gridsize_local_j,
        gridstart_local_i,
        gridstart_local_j,
        grid,
        plan_forward,
        plan_backward
    };
    return fftw_struct;
}

/* Helper function for creating the plans */
struct plans_struct make_plans(
    ptrdiff_t gridsize_i,
    ptrdiff_t gridsize_j,
    ptrdiff_t gridsize_k,
    double* grid,
    unsigned rigor_flag,
    int fftw_wisdom_reuse,
    char* wisdom_filename
) {
    /* Process identification */
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int master_rank = 0;
    int master = (rank == master_rank);

    /* Read in previous wisdom and broadcast it */
    int reused = 0;
    if (fftw_wisdom_reuse) {
        if (master)
            reused = fftw_import_wisdom_from_filename(wisdom_filename);
        fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);
    }
    MPI_Bcast(&reused, 1, MPI_INT, master_rank, MPI_COMM_WORLD);

    /* Create the two plans */
    fftw_plan plan_forward = fftw_mpi_plan_dft_r2c_3d(
        gridsize_i,
        gridsize_j,
        gridsize_k,
        grid,
        (fftw_complex*) grid,
        MPI_COMM_WORLD,
        rigor_flag | FFTW_MPI_TRANSPOSED_OUT
    );
    fftw_plan plan_backward = fftw_mpi_plan_dft_c2r_3d(
        gridsize_i,
        gridsize_j,
        gridsize_k,
        (fftw_complex*) grid,
        grid,
        MPI_COMM_WORLD,
        rigor_flag | FFTW_MPI_TRANSPOSED_IN
    );
    /* The wisdom generated above (if not reusing pre-existing) is
     * individual to each process. Collect all wisdom into the master
     * process, which then selects one of them. Then broadcast the
     * selected one back out again.
     */
    fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
    fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);

    /* Save newly acquired wisdom to disk, if it is to be reused */
    if (master && fftw_wisdom_reuse && ! reused) {
        fftw_export_wisdom_to_filename(wisdom_filename);
    }

    /* Return the two plans */
    struct plans_struct plans = {
        plan_forward,
        plan_backward,
        reused
    };
    return plans;
}

/* Call this function when all FFTW work is done */
void fftw_clean(
    double* grid,
    fftw_plan plan_forward,
    fftw_plan plan_backward
) {
    fftw_free(grid);
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_mpi_cleanup();
}
