// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// intel compiler warns "cast discards qualifiers from target type"
// disabling this warning here
#pragma warning(push)
#pragma warning(disable : 2203)
#include <mpi.h>
#pragma warning(pop)

#include "../tests/common.hpp"

/*
  Demonstrates some basics regarding tensor trains
  Here we assume you have been introduced to the basics of tensor trains
*/

constexpr boba::execution_space space = boba::default_execution_space;

int main(int argc, char* argv[])
{

  bool check = 1;

  MPI_Init(nullptr, nullptr);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0)
  {
    boba::splash();

    boba_print(world_size);
    boba_print(world_rank);
  }
  boba::init();

  // Get the tensor
  constexpr int dimension = boba::is_ci_mode() ? 4 : 7;
  auto sizes = ::boba::filled_array<dimension>(boba::is_ci_mode() ? 3_z : 5_z);

  boba::Tensor<dimension, space, double> tensor_A(sizes);

  //
  // Generate tensor on rank 0
  //
  if (world_rank == 0)
  {
    tensor_A.fill_with(1.0);
  }

  if (world_rank == 0)
  {
    boba_print("MPI_Bcast initiate");
  }

  MPI_Bcast(tensor_A.data(), tensor_A.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank == 0)
  {
    boba_print("MPI_Bcast complete");
  }

  double expected_local_tensor_norm;

  if (world_rank == 0)
  {
    expected_local_tensor_norm = ::boba::norm_frobenius(tensor_A);
  }

  auto local_tensor_norm = ::boba::norm_frobenius(tensor_A);

  MPI_Bcast(&expected_local_tensor_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (not(::boba::is_tiny(expected_local_tensor_norm - local_tensor_norm)))
  {
    boba_print(expected_local_tensor_norm);
    boba_print(local_tensor_norm);
    boba_error("MPI_Bcast failed!");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0)
  {
    boba_print("MPI_Bcast success!");
  }

  double svd_tolerance = 1.0e-10;

  // Organize total work to do

  ::boba::Multiindexer<dimension> all_permutations(::boba::filled_array<dimension>(static_cast<::boba::index_t>(dimension)));
  size_t global_work_length = all_permutations.size();
  size_t work_length_per_worker = ::boba::ceil(double(global_work_length) / double(world_size));

  if (world_rank == 0)
  {
    boba_print(global_work_length);
    boba_print(work_length_per_worker);
  }

  size_t local_work_start = world_rank * work_length_per_worker;
  size_t local_work_end = ::boba::min(global_work_length, local_work_start + work_length_per_worker);

  //
  // Given this ranks' work, find the optimal permutation
  //
  double local_optimal_tt_cr = 0.0;
  size_t local_optimal_permutation_id = local_work_start;

  checkpoint();
  for (size_t I = local_work_start; I < local_work_end; I++)
  {
    auto permutation_mid = all_permutations.multiindex(I);
    bool is_valid = is_valid_permutation(permutation_mid);

    if (is_valid)
    {
      ::boba::Tensor<dimension, ::boba::host_space, double> tensor_copy = tensor_A;
      permute(tensor_copy, permutation_mid);

      auto tt = compress_to_TensorTrain(tensor_copy, svd_tolerance);
      auto cr = tt.compression_rate();
      if (cr > local_optimal_tt_cr)
      {
        local_optimal_tt_cr = cr;
        local_optimal_permutation_id = I;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  //
  // Now that each process has done its local work, by finding the optimal CR within its range,
  // we must gather the results to rank 0 and do a final comparison to find the optimal cr
  //

  ::boba::Vector<::boba::host_space, double> gathered_cr_values({static_cast<::boba::index_t>(world_size)});
  ::boba::Vector<::boba::host_space, ::boba::index_t> gathered_id_values({static_cast<::boba::index_t>(world_size)});

  MPI_Gather(&local_optimal_tt_cr, 1, MPI_DOUBLE, gathered_cr_values.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_optimal_permutation_id, 1, MPI_UNSIGNED_LONG, gathered_id_values.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  auto [global_optimal_tt_cr, winning_rank] = gathered_cr_values.max_loc_reduce();

  auto gathered_id_values_view = gathered_id_values.view();
  size_t global_optimal_permutation_id = static_cast<size_t>(gathered_id_values_view(winning_rank));

  auto global_optimal_permutation = all_permutations.multiindex(global_optimal_permutation_id);

  if (world_rank == 0)
  {
    boba_print(global_optimal_permutation_id);
    boba_print(global_optimal_tt_cr);
    boba_print(global_optimal_permutation);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Done!
  MPI_Finalize();
  boba::finalize();
  return final_check(check);
}
