//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "selected_components_extractor.h"
#include "vtu_output.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


namespace ryujin
{
  using namespace dealii;


  template <typename Description, int dim, typename Number>
  VTUOutput<Description, dim, Number>::VTUOutput(
      const MPI_Comm &mpi_communicator,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const Postprocessor<Description, dim, Number> &postprocessor,
      const InitialPrecomputedVector &initial_precomputed,
      const ScalarVector &alpha,
      const std::string &subsection /*= "VTUOutput"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , postprocessor_(&postprocessor)
      , initial_precomputed_(initial_precomputed)
      , alpha_(alpha)
  {
    use_mpi_io_ = true;
    add_parameter("use mpi io",
                  use_mpi_io_,
                  "If enabled write out one vtu file via MPI IO using "
                  "write_vtu_in_parallel() instead of independent output files "
                  "via write_vtu_with_pvtu_record()");

    add_parameter("manifolds",
                  manifolds_,
                  "List of level set functions. The description is used to "
                  "only output cells that intersect the given level set.");

    std::copy(std::begin(View::component_names),
              std::end(View::component_names),
              std::back_inserter(vtu_output_quantities_));

    std::copy(std::begin(View::initial_precomputed_names),
              std::end(View::initial_precomputed_names),
              std::back_inserter(vtu_output_quantities_));

    add_parameter("vtu output quantities",
                  vtu_output_quantities_,
                  "List of conserved, primitive, precomputed, or postprocessed "
                  "quantities that will be written to the vtu files.");
  }


  template <typename Description, int dim, typename Number>
  void VTUOutput<Description, dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "VTUOutput<dim, Number>::prepare()" << std::endl;
#endif

    SelectedComponentsExtractor<Description, dim, Number>::check(
        vtu_output_quantities_);
  }


  template <typename Description, int dim, typename Number>
  void VTUOutput<Description, dim, Number>::schedule_output(
      const StateVector &state_vector,
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

    /*
     * Extract quantities and store in ScalarVectors so that we can call
     * DataOut::add_data_vector()
     */

    auto selected_components =
        SelectedComponentsExtractor<Description, dim, Number>::extract(
            *hyperbolic_system_,
            state_vector,
            initial_precomputed_,
            alpha_,
            vtu_output_quantities_);

    /* prepare DataOut: */

    auto data_out = std::make_unique<dealii::DataOut<dim>>();
    data_out->attach_dof_handler(offline_data_->dof_handler());

    for (unsigned int d = 0; d < selected_components.size(); ++d) {
      affine_constraints.distribute(selected_components[d]);
      selected_components[d].update_ghost_values();
      data_out->add_data_vector(selected_components[d],
                                vtu_output_quantities_[d],
                                DataOut<dim>::type_dof_data);
    }

    const auto n_quantities = postprocessor_->n_quantities();
    for (unsigned int i = 0; i < n_quantities; ++i)
      data_out->add_data_vector(postprocessor_->quantities()[i],
                                postprocessor_->component_names()[i],
                                DataOut<dim>::type_dof_data);

    DataOutBase::VtkFlags flags(t,
                                cycle,
                                true,
#if DEAL_II_VERSION_GTE(9, 5, 0)
                                DataOutBase::CompressionLevel::best_speed);
#else
                                DataOutBase::VtkFlags::best_speed);
#endif
    data_out->set_flags(flags);

    const auto &discretization = offline_data_->discretization();
    const auto &mapping = discretization.mapping();
    const auto patch_order =
        std::max(1u, discretization.finite_element().degree) - 1u;

    /* Perform output: */

    if (output_full) {
      data_out->build_patches(mapping, patch_order);

      if (use_mpi_io_) {
        /* MPI-based synchronous IO */
        data_out->write_vtu_in_parallel(
            name + "_" + Utilities::to_string(cycle, 6) + ".vtu",
            mpi_communicator_);
      } else {
        data_out->write_vtu_with_pvtu_record(
            "", name, cycle, mpi_communicator_, 6);
      }
    }

    if (output_levelsets && manifolds_.size() != 0) {
      /*
       * Specify an output filter that selects only cells for output that are
       * in the viscinity of a specified set of output planes:
       */

      std::vector<std::shared_ptr<FunctionParser<dim>>> level_set_functions;
      for (const auto &expression : manifolds_)
        level_set_functions.emplace_back(
            std::make_shared<FunctionParser<dim>>(expression));

      data_out->set_cell_selection([level_set_functions](const auto &cell) {
        if (!cell->is_active() || cell->is_artificial())
          return false;

        for (const auto &function : level_set_functions) {

          unsigned int above = 0;
          unsigned int below = 0;

          for (unsigned int v : cell->vertex_indices()) {
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

      data_out->build_patches(mapping, patch_order);

      if (use_mpi_io_) {
        /* MPI-based synchronous IO */
        data_out->write_vtu_in_parallel(
            name + "-levelsets_" + Utilities::to_string(cycle, 6) + ".vtu",
            mpi_communicator_);
      } else {
        data_out->write_vtu_with_pvtu_record(
            "", name + "-levelsets", cycle, mpi_communicator_, 6);
      }
    }

    /* Explicitly delete pointer to free up memory early: */
    data_out.reset();
  }

} /* namespace ryujin */
