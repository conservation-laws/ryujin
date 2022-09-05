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
      const ryujin::HyperbolicModule<dim, Number> &hyperbolic_module,
      const ryujin::Postprocessor<dim, Number> &postprocessor,
      const std::string &subsection /*= "VTUOutput"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , offline_data_(&offline_data)
      , hyperbolic_module_(&hyperbolic_module)
      , postprocessor_(&postprocessor)
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

    std::copy(std::begin(HyperbolicSystem::component_names<dim>),
              std::end(HyperbolicSystem::component_names<dim>),
              std::back_inserter(vtu_output_quantities_));

    // FIXME
//     std::copy(std::begin(HyperbolicSystem::precomputed_names<dim>),
//               std::end(HyperbolicSystem::precomputed_names<dim>),
//               std::back_inserter(vtu_output_quantities_));

    add_parameter("vtu output quantities",
                  vtu_output_quantities_,
                  "List of conserved, primitive, or precomputed quantities "
                  "that will be written to the vtu files.");
  }


  template <int dim, typename Number>
  void VTUOutput<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "VTUOutput<dim, Number>::prepare()" << std::endl;
#endif

    /* Populate quantities mapping: */

    quantities_mapping_.clear();

    for (const auto &entry : vtu_output_quantities_) {
      {
        /* Conserved quantities: */

        constexpr auto &names = HyperbolicSystem::component_names<dim>;
        const auto pos = std::find(std::begin(names), std::end(names), entry);
        if (pos != std::end(names)) {
          const auto index = std::distance(std::begin(names), pos);
          quantities_mapping_.push_back(std::make_tuple(
              entry, [index](scalar_type &result, const vector_type &U) {
                U.extract_component(result, index);
              }));
          continue;
        }
      }

      {
        /* Primitive quantities: */

        constexpr auto &names =
            HyperbolicSystem::primitive_component_names<dim>;
        const auto pos = std::find(std::begin(names), std::end(names), entry);
        if (pos != std::end(names)) {
          AssertThrow(false, dealii::ExcNotImplemented());
          continue;
        }
      }

      // FIXME
#if 0
      {
        /* Precomputed quantities: */

        constexpr auto &names = HyperbolicSystem::precomputed_names<dim>;
        const auto pos = std::find(std::begin(names), std::end(names), entry);
        if (pos != std::end(names)) {
          const auto index = std::distance(std::begin(names), pos);
          quantities_mapping_.push_back(std::make_tuple(
              entry, [this, index](scalar_type &result, const vector_type &) {
                const auto &precomputed =
                    hyperbolic_module_->precomputed_values();
                precomputed.extract_component(result, index);
              }));
          continue;
        }
      }
#endif

      {
        /* Special indicator value: */

        if (entry == "alpha") {
          quantities_mapping_.push_back(std::make_tuple(
              entry, [this](scalar_type &result, const vector_type &) {
                const auto &alpha = hyperbolic_module_->alpha();
                result = alpha;
              }));
          continue;
        }
      }

      AssertThrow(false, ExcMessage("Invalid component name »" + entry + "«"));
    }

    quantities_.resize(quantities_mapping_.size());
    for (auto &it : quantities_)
      it.reinit(offline_data_->scalar_partitioner());
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

    /* Copy quantities: */

    Assert(quantities_.size() == quantities_mapping_.size(),
           ExcInternalError());
    for (unsigned int d = 0; d < quantities_.size(); ++d) {
      const auto &lambda = std::get<1>(quantities_mapping_[d]);
      lambda(quantities_[d], U);
      affine_constraints.distribute(quantities_[d]);
      quantities_[d].update_ghost_values();
    }

    /* prepare DataOut: */

    auto data_out = std::make_unique<dealii::DataOut<dim>>();
    data_out->attach_dof_handler(offline_data_->dof_handler());

    for (unsigned int d = 0; d < quantities_.size(); ++d) {
      const auto &entry = std::get<0>(quantities_mapping_[d]);
      data_out->add_data_vector(quantities_[d], entry);
    }

    const auto n_quantities = postprocessor_->n_quantities();
    for (unsigned int i = 0; i < n_quantities; ++i)
      data_out->add_data_vector(postprocessor_->quantities()[i],
                                postprocessor_->component_names()[i]);

    DataOutBase::VtkFlags flags(
        t, cycle, true, DataOutBase::VtkFlags::best_speed);
    data_out->set_flags(flags);

    const auto &discretization = offline_data_->discretization();
    const auto &mapping = discretization.mapping();
    const auto patch_order = discretization.finite_element().degree - 1;

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
