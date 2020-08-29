/*
This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
Copyright © 2015–2020 Jeppe Mosgaard Dakin.

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
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

/* This file defines the functions fftw_setup and fftw_clean, which
together with fftw_execute (included in fftw3-mpi.h) constitutes the
necessary functions for using FFTW to do parallel, real, 3D in-place
transforms through Cython.
*/

/* Note on indexing.
- In real space, before any transformations,
  provided that we execute this on the correct process, where
  (gridstart_local_i <= i && i < gridstart_local_i + gridsize_local_i):
  grid[i, j, k] --> grid[((i - gridstart_local_i)*gridsize_j + j)
                         *gridsize_padding + k]
  That is, each process can acces every element in its slab by:
  for (i = 0; i < gridsize_local_i; ++i){
      for (j = 0; j < gridsize_j; ++j){
          for (k = 0; k < gridsize_k; ++k){
              // This is the [i + gridstart_local_i, j, k]'th element
              double element = grid[(i*gridsize_j + j)
                                    *gridsize_padding + k];
          }
      }
  }
- After a forward, in-place r2c transformation,
  provided that we execute this on the correct process, where
  (gridstart_local_j <= j && j < gridstart_local_j + gridsize_local_j):
  grid[i, j, k] --> grid[((j - gridstart_local_j)*gridsize_i + i)
                         *gridsize_padding + k]
  That is, each process can acces every element in its slab by:
  for (i = 0; i < gridsize_i; ++i){
      for (j = 0; j < gridsize_local_j; ++j){
          for (k = 0; k < gridsize_k; ++k){
              // This is the [i, j + gridstart_local_j, k]'th element
              double element = grid[(j*gridsize_i + i)
                                    *gridsize_padding + k];
          }
      }
  }
*/

// Struct for return type of the fftw_setup function
struct fftw_return_struct{
    ptrdiff_t gridsize_local_i;
    ptrdiff_t gridsize_local_j;
    ptrdiff_t gridstart_local_i;
    ptrdiff_t gridstart_local_j;
    double* grid;
    fftw_plan plan_forward;
    fftw_plan plan_backward;
};
// This function initializes fftw_mpi, allocates a grid,
// desides the local lengths and starting indices and
// creates forwards and backwards plans.
struct fftw_return_struct fftw_setup(ptrdiff_t gridsize_i,
                                     ptrdiff_t gridsize_j,
                                     ptrdiff_t gridsize_k,
                                     char* fftw_wisdom_rigor,
                                     bool fftw_wisdom_reuse,
                                     char* wisdom_filename){
    // Arguments to this function:
    // - Linear gridsize of dimension 1.
    // - Linear gridsize of dimension 2.
    // - Linear gridsize of dimension 3.
    // - FFTW planning-rigor, determining the optimization level of
    //   of the wisdom. In order of patience:
    //   "estimate", "measure", "patient", "exhaustive".
    // - Flag specifying whether or not to use pre-existing FFTW wisdom.

    // Size of last dimension with padding
    ptrdiff_t gridsize_padding = 2*(gridsize_k/2 + 1);

    // Initialize parallel fftw (note that MPI_Init should not be
    // called, as MPI is already running via MPI4Py).
    // Note that this function may be called multiple times in one MPI
    // session without errors; only the first call will have any effect.
    fftw_mpi_init();

    // Process identification
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int root = 0;
    bool master = (rank == root);

    // Declaration and allocation of the (local part of the) grid. This
    // also initializes gridsize_local_(i/j) and gridstart_local_(i/j).
    ptrdiff_t gridsize_local_i, gridstart_local_i,
              gridsize_local_j, gridstart_local_j;
    double* grid = fftw_alloc_real(
                       fftw_mpi_local_size_3d_transposed(gridsize_i,
                                                         gridsize_j,
                                                         gridsize_padding,
                                                         MPI_COMM_WORLD,
                                                         &gridsize_local_i,
                                                         &gridstart_local_i,
                                                         &gridsize_local_j,
                                                         &gridstart_local_j)
                                   );

    // The master process reads in previous wisdom and broadcasts it
    int previous_wisdom = 0;
    if (fftw_wisdom_reuse){
        if (master){
            previous_wisdom = fftw_import_wisdom_from_filename(wisdom_filename);
        }
        fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);
    }

    // Convert fftw_wisdom_rigor to integer flag
    int rigor_flag = FFTW_ESTIMATE;
    if (strcmp(fftw_wisdom_rigor, "estimate") == 0){
        rigor_flag = FFTW_ESTIMATE;
    }
    else if (strcmp(fftw_wisdom_rigor, "measure") == 0){
        rigor_flag = FFTW_MEASURE;
    }
    else if (strcmp(fftw_wisdom_rigor, "patient") == 0){
        rigor_flag = FFTW_PATIENT;
    }
    else if (strcmp(fftw_wisdom_rigor, "exhaustive") == 0){
        rigor_flag = FFTW_EXHAUSTIVE;
    }

    // Create the two plans
    fftw_plan plan_forward  = fftw_mpi_plan_dft_r2c_3d(gridsize_i,
                                                       gridsize_j,
                                                       gridsize_k,
                                                       grid,
                                                       (fftw_complex*) grid,
                                                       MPI_COMM_WORLD,
                                                       rigor_flag | FFTW_MPI_TRANSPOSED_OUT);
    fftw_plan plan_backward = fftw_mpi_plan_dft_c2r_3d(gridsize_i,
                                                       gridsize_j,
                                                       gridsize_k,
                                                       (fftw_complex*) grid,
                                                       grid,
                                                       MPI_COMM_WORLD,
                                                       rigor_flag | FFTW_MPI_TRANSPOSED_IN);

    // If new wisdom is acquired, the master process saves it to disk
    fftw_mpi_gather_wisdom(MPI_COMM_WORLD);
    if (master && ! previous_wisdom){
        fftw_export_wisdom_to_filename(wisdom_filename);
    }

    // Return a struct with variables
    struct fftw_return_struct fftw_struct = {gridsize_local_i,
                                             gridsize_local_j,
                                             gridstart_local_i,
                                             gridstart_local_j,
                                             grid,
                                             plan_forward,
                                             plan_backward};
    return fftw_struct;
}

// Call this function when all FFT work is done
void fftw_clean(double* grid, fftw_plan plan_forward, fftw_plan plan_backward){
    fftw_free(grid);
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_mpi_cleanup();
}
