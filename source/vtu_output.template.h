//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include "computing_timer.h"
#include "vtu_output.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  VTUOutput<dim, Number>::VTUOutput(
      const MPIEnsemble &mpi_ensemble,
      const Interface::OfflineData<dim, Number> &offline_data,
      const Interface::HyperbolicSystem &hyperbolic_system,
      const Interface::Postprocessor<dim, Number> &postprocessor,
      const std::string &subsection /*= "VTUOutput"*/)
      : ParameterAcceptor(subsection)
      , mpi_ensemble_(mpi_ensemble)
      , offline_data_(&offline_data)
      , postprocessor_(postprocessor)
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

    const auto &state_names = hyperbolic_system.state_names(dim);
    vtu_output_quantities_ = state_names;

    const auto &initial_names =
        hyperbolic_system.initial_precomputed_names(dim);
    vtu_output_quantities_.insert(vtu_output_quantities_.end(),
                                  initial_names.begin(),
                                  initial_names.end());

    Assert(state_names.size() > 0,
           dealii::ExcMessage("Need at least one scalar quantitity"));
    vtu_output_quantities_.push_back("schlieren_" + state_names[0]);

    add_parameter(
        "vtu output quantities",
        vtu_output_quantities_,
        "List of conserved, primitive, precomputed, or postprocessed "
        "(schlieren_<name>, vorticity_<name>) quantities that will be "
        "written to the vtu files.");

    beta_ = 10.;
    add_parameter("schlieren beta",
                  beta_,
                  "Beta factor used in the exponential scale for the schlieren "
                  "and vorticity plots");

    recompute_bounds_ = true;
    add_parameter(
        "schlieren recompute bounds",
        recompute_bounds_,
        "Recompute bounds for every output cycle. If set to false, bounds once "
        "at the beginning and reused thereafter.");
  }


  template <int dim, typename Number>
  void VTUOutput<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "VTUOutput<dim, Number>::prepare()" << std::endl;
#endif

    selection_ = postprocessor_.prepare(vtu_output_quantities_);
    bounds_.clear();
  }


  template <int dim, typename Number>
  void VTUOutput<dim, Number>::schedule_output(
      const StateVector<Number> &state_vector,
      std::string name,
      Number t [[maybe_unused]],
      unsigned int cycle,
      bool output_full,
      bool output_levelsets)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "VTUOutput<dim, Number>::schedule_output()" << std::endl;
#endif

    /*
     * Extract quantities and store in ScalarHostVectors so that we can
     * call DataOut::add_data_vector()
     */

    auto selected_components = postprocessor_.extract(selection_, state_vector);

    /*
     * Normalize postprocessed quantities on an exponential scale:
     */

    const auto n_selected =
        static_cast<unsigned int>(selected_components.size());

    const auto is_postprocessed = [&](const unsigned int k) {
      const auto &name = vtu_output_quantities_[k];
      return name.rfind("schlieren_", 0) == 0 ||
             name.rfind("vorticity_", 0) == 0;
    };

    if (recompute_bounds_ || bounds_.size() != n_selected) {
      bounds_.assign(
          n_selected,
          std::make_pair(Number(0.), std::numeric_limits<Number>::max()));

      const unsigned int n_owned = offline_data_->n_locally_owned();

      for (unsigned int k = 0; k < n_selected; ++k) {
        if (!is_postprocessed(k))
          continue;

        auto &[q_max, q_min] = bounds_[k];
        for (unsigned int i = 0; i < n_owned; ++i) {
          const auto q = selected_components[k].local_element(i);
          q_max = std::max(q_max, std::abs(q));
          q_min = std::min(q_min, std::abs(q));
        }
        q_max = dealii::Utilities::MPI::max(
            q_max, mpi_ensemble_.ensemble_communicator());
        q_min = dealii::Utilities::MPI::min(
            q_min, mpi_ensemble_.ensemble_communicator());
        Assert(q_max >= q_min, dealii::ExcInternalError());
      }
    }

    {
      constexpr Number eps = std::numeric_limits<Number>::epsilon();
      constexpr Number floor = std::max(Number(1.0e-10), eps);

      const unsigned int n_owned = offline_data_->n_locally_owned();

      for (unsigned int k = 0; k < n_selected; ++k) {
        if (!is_postprocessed(k))
          continue;

        const auto &[q_max, q_min] = bounds_[k];
        for (unsigned int i = 0; i < n_owned; ++i) {
          auto &q = selected_components[k].local_element(i);
          /* clip off everything that is below the noise "floor": */
          const auto ratio = std::max(Number(0.), std::abs(q) - q_min - floor) /
                             std::max(q_max - q_min, eps);

          const auto magnitude = Number(1.) - std::exp(-beta_ * ratio);
          q = std::copysign(magnitude, q);
        }
      }
    }

    /*
     * Attach data vectors to DataOut object:
     */

    auto data_out = std::make_unique<dealii::DataOut<dim>>();

    const auto attach_data_vector = [&](auto &data, const auto &name) {
      const auto &dof_handler_cg = offline_data_->dof_handler_cg();
      const auto &dof_handler_dg = offline_data_->dof_handler_dg();

      if (data.size() == dof_handler_cg.n_dofs()) {
        offline_data_->affine_constraints_cg().distribute(data);
        data.update_ghost_values();
        data_out->add_data_vector(dof_handler_cg, data, name);

      } else if (data.size() == dof_handler_dg.n_dofs()) {
        offline_data_->affine_constraints_dg().distribute(data);
        data.update_ghost_values();
        data_out->add_data_vector(dof_handler_dg, data, name);

      } else {
        Assert(
            false,
            dealii::ExcMessage("The selected solution component »" + name +
                               "« is associated with an unknown dof handler"));
      }
    };

    for (unsigned int d = 0; d < selected_components.size(); ++d) {
      attach_data_vector(selected_components[d], vtu_output_quantities_[d]);
    }

    DataOutBase::VtkFlags flags(
        t, cycle, true, DataOutBase::CompressionLevel::best_speed);
    data_out->set_flags(flags);

    const auto &discretization = offline_data_->discretization();
    const auto &mapping = discretization.mapping();
    const auto patch_order =
        std::max(1u, discretization.polynomial_degree()) - 1u;

    /* Perform output: */

    if (output_full) {
      data_out->build_patches(
          mapping, patch_order, DataOut<dim>::curved_inner_cells);

      if (use_mpi_io_) {
        /* MPI-based synchronous IO */
        data_out->write_vtu_in_parallel(
            name + "_" + Utilities::to_string(cycle, 6) + ".vtu",
            mpi_ensemble_.ensemble_communicator());
      } else {
        data_out->write_vtu_with_pvtu_record(
            "", name, cycle, mpi_ensemble_.ensemble_communicator(), 6);
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

      data_out->build_patches(
          mapping, patch_order, DataOut<dim>::curved_inner_cells);

      if (use_mpi_io_) {
        /* MPI-based synchronous IO */
        data_out->write_vtu_in_parallel(
            name + "-levelsets_" + Utilities::to_string(cycle, 6) + ".vtu",
            mpi_ensemble_.ensemble_communicator());
      } else {
        data_out->write_vtu_with_pvtu_record(
            "",
            name + "-levelsets",
            cycle,
            mpi_ensemble_.ensemble_communicator(),
            6);
      }
    }

    /* Explicitly delete pointer to free up memory early: */
    data_out.reset();
  }

} /* namespace ryujin */
