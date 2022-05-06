//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "simd.h"
#include "vtu_output.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <atomic>
#include <chrono>
#include <fstream>

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  VTUOutput<dim, Number>::VTUOutput(
      const MPI_Comm &mpi_communicator,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const ryujin::Postprocessor<dim, Number> &postprocessor,
      const std::string &subsection /*= "VTUOutput"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , offline_data_(&offline_data)
      , postprocessor_(&postprocessor)
  {
    use_mpi_io_ = false;
    add_parameter("use mpi io",
                  use_mpi_io_,
                  "If enabled write out one vtu file via MPI IO using "
                  "write_vtu_in_parallel() instead of independent output files "
                  "via write_vtu_with_pvtu_record()");

    add_parameter("manifolds",
                  manifolds_,
                  "List of level set functions. The description is used to "
                  "only output cells that intersect the given level set.");
  }


  template <int dim, typename Number>
  void VTUOutput<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "VTUOutput<dim, Number>::prepare()" << std::endl;
#endif

    // nothing at the moment
  }


  template <int dim, typename Number>
  void VTUOutput<dim, Number>::schedule_output(const vector_type &U,
                                               std::string name,
                                               Number t,
                                               unsigned int cycle,
                                               bool output_full,
                                               bool output_levelsets)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "VTUOutput<dim, Number>::schedule_output()" << std::endl;
#endif

    const auto &affine_constraints = offline_data_->affine_constraints();

    /* Copy state vector: */

    unsigned int d = 0;
    for (auto &it : state_vector_) {
      it.reinit(offline_data_->scalar_partitioner());
      U.extract_component(it, d++);
      affine_constraints.distribute(it);
      it.update_ghost_values();
    }

    /* prepare DataOut: */

    auto data_out = std::make_unique<dealii::DataOut<dim>>();
    auto data_out_levelsets = std::make_unique<dealii::DataOut<dim>>();

    const auto &discretization = offline_data_->discretization();
    const auto &mapping = discretization.mapping();
    const auto patch_order = discretization.finite_element().degree - 1;

    if (output_full) {
      data_out->attach_dof_handler(offline_data_->dof_handler());

      for (unsigned int i = 0; i < problem_dimension; ++i)
        data_out->add_data_vector(state_vector_[i],
                                  HyperbolicSystem::component_names<dim>[i]);

      constexpr auto n_quantities = Postprocessor<dim, Number>::n_quantities;
      for (unsigned int i = 0; i < n_quantities; ++i)
        data_out->add_data_vector(postprocessor_->quantities()[i],
                                  postprocessor_->component_names[i]);

      data_out->build_patches(mapping, patch_order);

      DataOutBase::VtkFlags flags(
          t, cycle, true, DataOutBase::VtkFlags::best_speed);
      data_out->set_flags(flags);
    }

    if (output_levelsets && manifolds_.size() != 0) {
      data_out_levelsets->attach_dof_handler(offline_data_->dof_handler());

      for (unsigned int i = 0; i < problem_dimension; ++i)
        data_out_levelsets->add_data_vector(
            state_vector_[i], HyperbolicSystem::component_names<dim>[i]);

      constexpr auto n_quantities = Postprocessor<dim, Number>::n_quantities;
      for (unsigned int i = 0; i < n_quantities; ++i)
        data_out_levelsets->add_data_vector(postprocessor_->quantities()[i],
                                            postprocessor_->component_names[i]);

      /*
       * Specify an output filter that selects only cells for output that are
       * in the viscinity of a specified set of output planes:
       */

      std::vector<std::shared_ptr<FunctionParser<dim>>> level_set_functions;
      for (const auto &expression : manifolds_)
        level_set_functions.emplace_back(
            std::make_shared<FunctionParser<dim>>(expression));

      data_out_levelsets->set_cell_selection(
          [level_set_functions](const auto &cell) {
            if (!cell->is_active() || cell->is_artificial())
              return false;

            for (const auto &function : level_set_functions) {

              unsigned int above = 0;
              unsigned int below = 0;

              for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                   ++v) {
                const auto vertex = cell->vertex(v);
                constexpr auto eps = std::numeric_limits<Number>::epsilon();
                if (function->value(vertex) >= 0. - 100. * eps)
                  above++;
                if (function->value(vertex) <= 0. + 100. * eps)
                  below++;
                if (above > 0 && below > 0)
                  return true;
              }
            }
            return false;
          });

      data_out_levelsets->build_patches(mapping, patch_order);

      DataOutBase::VtkFlags flags(
          t, cycle, true, DataOutBase::VtkFlags::best_speed);
      data_out_levelsets->set_flags(flags);
    }

    if (use_mpi_io_) {
      /* MPI-based synchronous IO */
      if (output_full) {
        data_out->write_vtu_in_parallel(
            name + "_" + Utilities::to_string(cycle, 6) + ".vtu",
            mpi_communicator_);
      }
      if (output_levelsets && manifolds_.size() != 0) {
        data_out_levelsets->write_vtu_in_parallel(
            name + "-levelsets_" + Utilities::to_string(cycle, 6) + ".vtu",
            mpi_communicator_);
      }

    } else {

      /* Write out individual files per rank */

      if (output_full) {
        data_out->write_vtu_with_pvtu_record(
            "", name, cycle, mpi_communicator_, 6);
      }
      if (output_levelsets && manifolds_.size() != 0) {
        data_out_levelsets->write_vtu_with_pvtu_record(
            "", name + "-levelsets", cycle, mpi_communicator_, 6);
      }
    }

    /* Explicitly delete pointer to free up memory early: */
    data_out.reset();
    data_out_levelsets.reset();
  }

} /* namespace ryujin */
