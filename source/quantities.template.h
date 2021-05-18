//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "openmp.h"
#include "quantities.h"
#include "scope.h"
#include "scratch_data.h"
#include "simd.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>

DEAL_II_NAMESPACE_OPEN
template <int rank, int dim, typename Number>
bool operator<(const Tensor<rank, dim, Number> &left,
               const Tensor<rank, dim, Number> &right)
{
  return std::lexicographical_compare(
      left.begin_raw(), left.end_raw(), right.begin_raw(), right.end_raw());
}
DEAL_II_NAMESPACE_CLOSE

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  Quantities<dim, Number>::Quantities(
      const MPI_Comm &mpi_communicator,
      const ryujin::ProblemDescription &problem_description,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const std::string &subsection /*= "Quantities"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , problem_description_(&problem_description)
      , offline_data_(&offline_data)
  {
    add_parameter("interior manifolds",
                  interior_manifolds_,
                  "List of level set functions describing interior manifolds. "
                  "The description is used to only output point values for "
                  "vertices belonging to a certain level set.");

    boundary_manifolds_.push_back({"upper_boundary", "y - 1.0"});
    boundary_manifolds_.push_back({"lower_boundary", "y"});
    add_parameter("boundary manifolds",
                  boundary_manifolds_,
                  "List of level set functions describing boundary. The "
                  "description is used to only output point values for "
                  "boundary vertices belonging to a certain level set.");
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::prepare()" << std::endl;
#endif

    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(offline_data_->dof_handler(),
                                            relevant_dofs);
    const auto scalar_partitioner =
        std::make_shared<Utilities::MPI::Partitioner>(
            offline_data_->dof_handler().locally_owned_dofs(),
            relevant_dofs,
            offline_data_->dof_handler().get_communicator());

    /*
     * Create interior maps:
     */

    // FIXME
    AssertThrow(interior_manifolds_.empty(),
                dealii::ExcMessage("Not implemented. Output for interior "
                                   "manifolds has not been written yet."));

    const unsigned int n_owned = offline_data_->n_locally_owned();

    interior_maps_.clear();
    std::transform(
        interior_manifolds_.begin(),
        interior_manifolds_.end(),
        std::back_inserter(interior_maps_),
        [this, n_owned](auto it) {
          const auto &[name, expression] = it;
          FunctionParser<dim> level_set_function(expression);

          std::map<types::global_dof_index, Point<dim>> map;

          const auto &dof_handler = offline_data_->dof_handler();
          const auto &scalar_partitioner = offline_data_->scalar_partitioner();

          for (auto &cell : dof_handler.active_cell_iterators()) {

            /* skip non-local cells */
            if (!cell->is_locally_owned())
              continue;

            for (auto v : GeometryInfo<dim>::vertex_indices()) {

              const auto position = cell->vertex(v);

              /* only record points sufficiently close to the level set */
              if (std::abs(level_set_function.value(position)) > 1.e-12)
                continue;

              const auto global_index = cell->vertex_dof_index(v, 0);
              const auto index =
                  scalar_partitioner->global_to_local(global_index);

              /* skip nonlocal */
              if (index >= n_owned)
                continue;

              /* skip constrained */
              if (offline_data_->affine_constraints().is_constrained(
                      offline_data_->scalar_partitioner()->local_to_global(
                          index)))
                continue;

              map.insert({index, position});
            }
          }
          return std::make_tuple(name, map);
        });

    /*
     * Create boundary maps:
     */

    boundary_maps_.clear();
    std::transform(
        boundary_manifolds_.begin(),
        boundary_manifolds_.end(),
        std::back_inserter(boundary_maps_),
        [this, n_owned](auto it) {
          const auto &[name, expression] = it;
          FunctionParser<dim> level_set_function(expression);

          std::multimap<types::global_dof_index, boundary_description> map;

          for (const auto &entry : offline_data_->boundary_map()) {
            /* skip nonlocal */
            if (entry.first >= n_owned)
              continue;

            /* skip constrained */
            if (offline_data_->affine_constraints().is_constrained(
                    offline_data_->scalar_partitioner()->local_to_global(
                        entry.first)))
              continue;

            const auto position = std::get<4>(entry.second);
            if (std::abs(level_set_function.value(position)) < 1.e-12)
              map.insert(entry);
          }
          return std::make_tuple(name, map);
        });

    /*
     *
     */
  }


  template <int dim, typename Number>
  void Quantities<dim, Number>::compute(const vector_type &U,
                                        const Number t,
                                        std::string name,
                                        unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::compute()" << std::endl;
#endif
  }

} /* namespace ryujin */
