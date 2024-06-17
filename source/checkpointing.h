//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "multicomponent_vector.h"
#include "offline_data.h"

#include <deal.II/base/utilities.h>
#include <deal.II/distributed/solution_transfer.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/core/demangle.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace ryujin
{
  namespace Checkpointing
  {
    template <int dim>
    constexpr bool have_distributed_triangulation =
        std::is_same<typename Discretization<dim>::Triangulation,
                     dealii::parallel::distributed::Triangulation<dim>>::value;

    /**
     * Performs a resume operation. Given a @p base_name the function tries
     * to locate correponding checkpoint files and will read in the saved
     * mesh and reinitializes the Discretization object.
     *
     * @ingroup Miscellaneous
     */
    template <int dim>
    void load_mesh(Discretization<dim> &discretization,
                   const std::string &base_name)
    {
      if constexpr (have_distributed_triangulation<dim>) {
        discretization.refinement() = 0; /* do not refine */
        discretization.prepare(base_name);
        discretization.triangulation().load(base_name + "-checkpoint.mesh");
      } else {
        AssertThrow(false, dealii::ExcNotImplemented());
        __builtin_trap();
      }
    }


    /**
     * Performs a resume operation. Given a @p base_name the function tries
     * to locate correponding checkpoint files and will read in the saved
     * state @p U at saved time @p t with saved output cycle @p output_cycle.
     *
     * @ingroup Miscellaneous
     */
    template <int dim, typename Number, int n_comp, int simd_length>
    void load_state_vector(
        const OfflineData<dim, Number> &offline_data,
        const std::string &base_name,
        Vectors::MultiComponentVector<Number, n_comp, simd_length> &U,
        Number &t,
        unsigned int &output_cycle,
        const MPI_Comm &mpi_communicator)
    {
      if constexpr (have_distributed_triangulation<dim>) {
        const auto &dof_handler = offline_data.dof_handler();

        /* Create temporary scalar component vectors: */

        const auto &scalar_partitioner = offline_data.scalar_partitioner();

        using ScalarVector = typename Vectors::ScalarVector<Number>;
        std::array<ScalarVector, n_comp> state_vector;
        for (auto &it : state_vector) {
          it.reinit(scalar_partitioner);
        }

        /* Create SolutionTransfer object, attach state vector and deserialize:
         */

        dealii::parallel::distributed::SolutionTransfer<dim, ScalarVector>
            solution_transfer(dof_handler);

        std::vector<ScalarVector *> ptr_state;
        std::transform(state_vector.begin(),
                       state_vector.end(),
                       std::back_inserter(ptr_state),
                       [](auto &it) { return &it; });

        solution_transfer.deserialize(ptr_state);

        unsigned int d = 0;
        for (auto &it : state_vector) {
          U.insert_component(it, d++);
        }
        U.update_ghost_values();

        /* Read in and broadcast metadata: */

        std::string name = base_name + "-checkpoint";

        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
          std::string meta = name + ".metadata";

          std::ifstream file(meta, std::ios::binary);
          boost::archive::binary_iarchive ia(file);
          ia >> t >> output_cycle;
        }

        int ierr;
        if constexpr (std::is_same_v<Number, double>)
          ierr = MPI_Bcast(&t, 1, MPI_DOUBLE, 0, mpi_communicator);
        else
          ierr = MPI_Bcast(&t, 1, MPI_FLOAT, 0, mpi_communicator);
        AssertThrowMPI(ierr);

        ierr = MPI_Bcast(&output_cycle, 1, MPI_UNSIGNED, 0, mpi_communicator);
        AssertThrowMPI(ierr);

        ierr = MPI_Barrier(mpi_communicator);
        AssertThrowMPI(ierr);

      } else {
        AssertThrow(false, dealii::ExcNotImplemented());
        __builtin_trap();
      }
    }


    /**
     * Writes out a checkpoint to disk. Given a @p base_name and a current
     * state @p U at time @p t and output cycle @p output_cycle the function
     * writes out the state to disk using boost::archive for serialization.
     *
     * @todo Some day, we should refactor this into a class and do something
     * smarter...
     *
     * @ingroup Miscellaneous
     */
    template <int dim, typename Number, int n_comp, int simd_length>
    void write_checkpoint(
        const OfflineData<dim, Number> &offline_data,
        const std::string &base_name,
        const Vectors::MultiComponentVector<Number, n_comp, simd_length> &U,
        const Number t,
        const unsigned int output_cycle,
        const MPI_Comm &mpi_communicator)
    {
      if constexpr (have_distributed_triangulation<dim>) {
        const auto &triangulation =
            offline_data.discretization().triangulation();
        const auto &dof_handler = offline_data.dof_handler();

        /* Copy state into scalar component vectors: */

        const auto &scalar_partitioner = offline_data.scalar_partitioner();

        using ScalarVector = typename Vectors::ScalarVector<Number>;
        std::array<ScalarVector, n_comp> state_vector;
        unsigned int d = 0;
        for (auto &it : state_vector) {
          it.reinit(scalar_partitioner);
          U.extract_component(it, d++);
        }

        /* Create SolutionTransfer object, attach state vector and write out: */

        dealii::parallel::distributed::SolutionTransfer<dim, ScalarVector>
            solution_transfer(dof_handler);

        std::vector<const ScalarVector *> ptr_state;
        std::transform(state_vector.begin(),
                       state_vector.end(),
                       std::back_inserter(ptr_state),
                       [](auto &it) { return &it; });
        solution_transfer.prepare_for_serialization(ptr_state);

        std::string name = base_name + "-checkpoint";

        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
          for (const std::string suffix :
               {".mesh", ".mesh_fixed.data", ".mesh.info", ".metadata"})
            if (std::filesystem::exists(name + suffix))
              std::filesystem::rename(name + suffix, name + suffix + "~");
        }

        triangulation.save(name + ".mesh");

        /* Metadata: */

        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
          std::string meta = name + ".metadata";
          std::ofstream file(meta, std::ios::binary | std::ios::trunc);
          boost::archive::binary_oarchive oa(file);
          oa << t << output_cycle;
        }

        const int ierr = MPI_Barrier(mpi_communicator);
        AssertThrowMPI(ierr);

      } else {
        AssertThrow(false, dealii::ExcNotImplemented());
        __builtin_trap();
      }
    }
  } // namespace Checkpointing
} // namespace ryujin
