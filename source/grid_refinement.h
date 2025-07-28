//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <algorithm>
#include <compile_time_options.h>

#include "convenience_macros.h"
#include "initial_values.h"
#include "mpi_ensemble.h"
#include "offline_data.h"
#include "sparse_matrix_simd.h"
#include "state_vector.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

#include <deal.II/distributed/grid_refinement.h>
#include <functional>

namespace ryujin
{
  namespace GridMarking
  {
    /**
     * @p refine_threshold The normalized threshold for which all cells with
     * indicators above this value are marked for refinement
     * @p coarsen_threshold The normalized threshold for which all cells with
     * indicators below this value are marked for coarsening
     */
    template <int dim, int spacedim, class CriteriaT, class Real>
    void refine_and_coarsen_fixed_tolerance(
        dealii::Triangulation<dim, spacedim> &tria,
        const CriteriaT &criteria,
        const Real &refine_threshold,
        const Real &coarsen_threshold)
    {
      Assert(criteria.size() == tria.n_active_cells(),
             dealii::ExcInternalError());


      // Get the Max, Min over all ranks of the criteria
      int n_locally_owned_active_cells = 0;
      if (const auto parallel_tria = dynamic_cast<
              const dealii::parallel::TriangulationBase<dim, spacedim> *>(
              &tria))
        n_locally_owned_active_cells =
            parallel_tria->n_locally_owned_active_cells();
      else
        n_locally_owned_active_cells = tria.n_active_cells();

      dealii::Vector<Real> locally_owned_indicators(
          n_locally_owned_active_cells);

      unsigned int owned_index = 0;
      for (const auto &cell : tria.active_cell_iterators() |
                                  dealii::IteratorFilters::LocallyOwnedCell()) {
        locally_owned_indicators(owned_index) =
            criteria(cell->active_cell_index());
        ++owned_index;
      }

      MPI_Comm mpi_communicator = tria.get_communicator();
      const auto local_min = *std::min_element(locally_owned_indicators.begin(),
                                               locally_owned_indicators.end());
      const auto local_max = *std::max_element(locally_owned_indicators.begin(),
                                               locally_owned_indicators.end());

      Real input[2] = {local_min, -local_max};
      Real output[2] = {0, 0};


      MPI_Allreduce(input, output, 2, MPI_DOUBLE, MPI_MIN, mpi_communicator);

      const auto normalization_factor = std::abs(output[0] + output[1]);

      if (normalization_factor < 1e-14)
        return;

      const auto new_refine_threshold = refine_threshold * normalization_factor;
      const auto new_coarsen_threshold =
          coarsen_threshold * normalization_factor;

      for (const auto &cell : tria.active_cell_iterators())
        if ((dynamic_cast<
                 dealii::parallel::DistributedTriangulationBase<dim, spacedim>
                     *>(&tria) == nullptr ||
             cell->is_locally_owned())) {
          // refinement
          if (std::fabs(criteria(cell->active_cell_index())) >=
              new_refine_threshold) {
            if (cell->coarsen_flag_set())
              cell->clear_coarsen_flag();
            cell->set_refine_flag();
          } else if (std::fabs(criteria(cell->active_cell_index())) <=
                         new_coarsen_threshold &&
                     !cell->refine_flag_set())
            cell->set_coarsen_flag();
        }
    }


    template <int dim, int spacedim, class CriteriaT, class Real>
    void refine_and_coarsen_fixed_tolerance_by_consensus(
        dealii::Triangulation<dim, spacedim> &tria,
        const std::vector<CriteriaT> &criteria_list,
        const Real &refine_threshold,
        const Real &coarsen_threshold)
    {
      Real new_coarsen_threshold = coarsen_threshold;
      for (const auto &criteria : criteria_list) {
        refine_and_coarsen_fixed_tolerance(
            tria, criteria, refine_threshold, new_coarsen_threshold);
        new_coarsen_threshold = 0;
      }
    }
  } // namespace GridMarking

} /* namespace ryujin */
