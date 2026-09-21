//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include "computing_timer.h"
#include "error_evaluation.h"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <fstream>
#include <iomanip>

namespace ryujin
{
  using namespace dealii;

  template <typename Description, int dim, typename Number>
  ErrorEvaluation<Description, dim, Number>::ErrorEvaluation(
      const MPIEnsemble &mpi_ensemble,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const InitialPrecomputedVector &initial_precomputed,
      const std::string &subsection /*= "ErrorEvaluation"*/)
      : ParameterAcceptor(subsection)
      , mpi_ensemble_(mpi_ensemble)
      , offline_data_(&offline_data)
      , selected_components_extractor_(offline_data,
                                       hyperbolic_system,
                                       parabolic_system,
                                       initial_precomputed)
      , base_name_("")
  {
    std::copy(std::begin(View::component_names),
              std::end(View::component_names),
              std::back_inserter(error_quantities_));

    add_parameter("error quantities",
                  error_quantities_,
                  "List of conserved, primitive, precomputed, or parabolic "
                  "quantities used in the computation of the error norms.");

    error_normalize_ = true;
    add_parameter("error normalize",
                  error_normalize_,
                  "Flag to control whether the error should be normalized by "
                  "the corresponding norm of the analytic solution.");

    error_norms_ = {"Linf", "L1", "L2"};
    add_parameter("error norms",
                  error_norms_,
                  "List of norms that are computed and reported (in the given "
                  "order). Valid choices are Linf, L1, and L2.");
  }


  template <typename Description, int dim, typename Number>
  void
  ErrorEvaluation<Description, dim, Number>::prepare(const std::string &name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "ErrorEvaluation<dim, Number>::prepare()" << std::endl;
#endif

    base_name_ = name;

    selected_components_extractor_.prepare(error_quantities_);

    AssertThrow(!error_norms_.empty(),
                dealii::ExcMessage("No error norms selected."));
    for (const auto &norm : error_norms_) {
      AssertThrow(norm == "Linf" || norm == "L1" || norm == "L2",
                  dealii::ExcMessage("Invalid norm: \"" + norm +
                                     "\" is not one of Linf, L1, or L2."));
    }

    /* Reset the log file and write out a header: */

    if (mpi_ensemble_.world_rank() != 0)
      return;

    std::ofstream output(base_name_ + "-error.log",
                         std::ofstream::out | std::ofstream::trunc);

    output << "# " << description() << " summed over quantities: ";
    for (std::size_t k = 0; k < error_quantities_.size(); ++k)
      output << (k == 0 ? "" : ", ") << error_quantities_[k];
    output << "\n";

    if (error_normalize_)
      output << "# (each error is normalized by the corresponding norm of the "
                "analytic solution)\n";

    output << "# time t";
    for (const auto &norm : error_norms_)
      output << "\t" << norm;
    output << "\n" << std::flush;
  }


  template <typename Description, int dim, typename Number>
  std::vector<Number> ErrorEvaluation<Description, dim, Number>::compute(
      const StateVector &state_vector, const StateVector &analytic) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "ErrorEvaluation<dim, Number>::compute()" << std::endl;
#endif

    /* Ensure that the state vectors are resident on the host memory space. */
    if constexpr (have_separate_memory_spaces) {
      ComputingTimer::Scope scope("time step [X] _ - memory space transfers");
      for (const auto *vector : {&state_vector, &analytic}) {
        const auto &[U, precomputed, parabolic] = *vector;
        U.template copy_to_memory_space<dealii::MemorySpace::Host>();
        precomputed.template copy_to_memory_space<dealii::MemorySpace::Host>();
      }
    }

    const auto &discretization = offline_data_->discretization();
    const auto &dof_handler = offline_data_->dof_handler();

    Vector<Number> difference_per_cell(
        discretization.triangulation().n_active_cells());

    /* Compute the selected norm of a scalar vector: */
    const auto compute_norm = [&](const ScalarHostVector &vector,
                                  const std::string &norm) -> Number {
      if (norm == "Linf")
        return vector.linfty_norm();

      VectorTools::integrate_difference(discretization.mapping(),
                                        dof_handler,
                                        vector,
                                        Functions::ZeroFunction<dim, Number>(),
                                        difference_per_cell,
                                        discretization.quadrature_high_order(),
                                        norm == "L1" ? VectorTools::L1_norm
                                                     : VectorTools::L2_norm);

      if (norm == "L1")
        return Utilities::MPI::sum(difference_per_cell.l1_norm(),
                                   mpi_ensemble_.ensemble_communicator());

      return Number(std::sqrt(
          Utilities::MPI::sum(std::pow(difference_per_cell.l2_norm(), 2),
                              mpi_ensemble_.ensemble_communicator())));
    };

    selected_components_extractor_.prepare_extraction(analytic);
    auto analytic_components = selected_components_extractor_.view().extract();

    selected_components_extractor_.prepare_extraction(state_vector);
    auto error_components = selected_components_extractor_.view().extract();

    std::vector<Number> norms(error_norms_.size(), Number(0.));

    /* Loop over all selected components: */
    for (std::size_t k = 0; k < error_quantities_.size(); ++k) {
      auto &analytic_component = analytic_components[k];
      auto &error_component = error_components[k];

      analytic_component.update_ghost_values();

      /* Populate constrained dofs due to periodicity: */
      offline_data_->affine_constraints().distribute(error_component);
      error_component.update_ghost_values();
      error_component -= analytic_component;

      for (std::size_t n = 0; n < error_norms_.size(); ++n) {
        const auto error = compute_norm(error_component, error_norms_[n]);
        norms[n] += error_normalize_ ? error / compute_norm(analytic_component,
                                                            error_norms_[n])
                                     : error;
      }
    }

    /*
     * Sum up over all participating MPI ranks. Note: we only perform this
     * operation on "peer" ranks zero:
     */

    if (mpi_ensemble_.ensemble_rank() == 0 && mpi_ensemble_.n_ensembles() > 1)
      for (auto &norm : norms)
        norm = Utilities::MPI::sum(
            norm, mpi_ensemble_.ensemble_leader_communicator());

    return norms;
  }


  template <typename Description, int dim, typename Number>
  void ErrorEvaluation<Description, dim, Number>::write_out(
      const StateVector &state_vector,
      const StateVector &analytic,
      const Number t) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "ErrorEvaluation<dim, Number>::write_out()" << std::endl;
#endif

    const auto norms = compute(state_vector, analytic);

    if (mpi_ensemble_.world_rank() != 0)
      return;

    std::ofstream output(base_name_ + "-error.log",
                         std::ofstream::out | std::ofstream::app);
    output << std::scientific << std::setprecision(14);

    output << t;
    for (const auto &norm : norms)
      output << "\t" << norm;
    output << "\n" << std::flush;
  }


  template <typename Description, int dim, typename Number>
  void ErrorEvaluation<Description, dim, Number>::print_summary(
      std::ostream &stream,
      const Number t,
      const dealii::types::global_dof_index n_global_dofs,
      const std::vector<Number> &norms) const
  {
    if (mpi_ensemble_.world_rank() != 0)
      return;

    stream << description() << " at final time \n";
    stream << std::setprecision(16);
    stream << "#dofs = " << n_global_dofs << std::endl;
    stream << "t     = " << t << std::endl;

    for (std::size_t n = 0; n < error_norms_.size(); ++n) {
      const auto &name = error_norms_[n];
      stream << name << std::string(6 - name.size(), ' ') << "= " << norms[n]
             << std::endl;
    }
  }


  template <typename Description, int dim, typename Number>
  std::string ErrorEvaluation<Description, dim, Number>::description() const
  {
    std::string result =
        error_normalize_ ? "Normalized consolidated " : "Consolidated ";

    /* Join the norms in natural language: "Linf, L1, and L2" */
    const auto n = error_norms_.size();
    for (std::size_t i = 0; i < n; ++i) {
      if (i > 0)
        result += (n == 2) ? " and " : (i + 1 == n ? ", and " : ", ");
      result += error_norms_[i];
    }

    return result + " errors";
  }

} /* namespace ryujin */
