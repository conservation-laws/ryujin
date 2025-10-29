//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include "electrostatic_configuration_library.h"
#include "parabolic_module.h"

#include <convenience_macros.h>
#include <instrumentation.h>
#include <openmp.h>
#include <scope.h>
#include <simd.h>

#include <deal.II/numerics/vector_tools_interpolate.h>


namespace ryujin
{
  namespace EulerPoisson
  {
    using namespace dealii;

    template <int dim, typename Number>
    ParabolicModule<dim, Number>::ParabolicModule(
        const MPIEnsemble &mpi_ensemble,
        std::map<std::string, dealii::Timer> &computing_timer,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const ParabolicSystem &parabolic_system,
        const InitialValues<Description, dim, Number> &initial_values,
        const std::string &subsection)
        : ParameterAcceptor(subsection)
        , mpi_ensemble_(mpi_ensemble)
        , computing_timer_(computing_timer)
        , hyperbolic_system_(&hyperbolic_system)
        , parabolic_system_(&parabolic_system)
        , offline_data_(&offline_data)
        , initial_values_(&initial_values)
        , id_violation_strategy_(IDViolationStrategy::warn)
        , cycle_(0)
        , n_iterations_(0)
        , n_restarts_(0)
        , n_corrections_(0)
        , n_warnings_(0)
        , potential_initialized_(false)
    {
      gauss_law_restart_strategy_ = GaussLawRestartStrategy::no_restart;
      add_parameter("gauss law restart strategy",
                    gauss_law_restart_strategy_,
                    "Strategy used when restarting the gauss law. Options are "
                    "\'no restart\', \'full restart\', \'correction\', "
                    "\'static no restart\', and \'static full restart\'.");

      use_gmg_ = false;
      add_parameter("multigrid", use_gmg_, "Use geometric multigrid");

      gmg_max_iter_ = 15;
      add_parameter("multigrid - max iter",
                    gmg_max_iter_,
                    "Maximal number of CG iterations with GMG smoother");

      gmg_smoother_range_ = 8.;
      add_parameter("multigrid - chebyshev range",
                    gmg_smoother_range_,
                    "Chebyshev smoother: eigenvalue range parameter");

      gmg_smoother_max_eig_ = 2.0;
      add_parameter("multigrid - chebyshev max eig",
                    gmg_smoother_max_eig_,
                    "Chebyshev smoother: maximal eigenvalue");

      gmg_smoother_degree_ = 3;
      add_parameter("multigrid - chebyshev degree",
                    gmg_smoother_degree_,
                    "Chebyshev smoother: degree");

      gmg_smoother_n_cg_iter_ = 10;
      add_parameter(
          "multigrid - chebyshev cg iter",
          gmg_smoother_n_cg_iter_,
          "Chebyshev smoother: number of CG iterations to approximate "
          "eigenvalue");

      gmg_min_level_ = 0;
      add_parameter(
          "multigrid - min level",
          gmg_min_level_,
          "Minimal mesh level to be visited in the geometric multigrid "
          "cycle where the coarse grid solver (Chebyshev) is called");

      tolerance_ = Number(1.0e-12);
      add_parameter("tolerance", tolerance_, "Tolerance for linear solvers");

      tolerance_linfty_norm_ = false;
      add_parameter("tolerance linfty norm",
                    tolerance_linfty_norm_,
                    "Use the l_infty norm instead of the l_2 norm for the "
                    "stopping criterion");

      ElectrostaticConfigurationLibrary::
          populate_electrostatic_configuration_list<dim, Number>(
              electrostatic_configuration_list_,
              parabolic_system_->subsection());

      const auto populate = [this]() {
        bool initialized = false;
        for (auto &it : electrostatic_configuration_list_)

          if (it->name() == parabolic_system_->electrostatic_configuration()) {
            selected_electrostatic_configuration_ = it;
            initialized = true;
            break;
          }

        AssertThrow(initialized,
                    dealii::ExcMessage(
                        "Could not find an electrostatic configuration "
                        "description with name \"" +
                        parabolic_system_->electrostatic_configuration() +
                        "\""));
      };

      ParameterAcceptor::parse_parameters_call_back.connect(populate);
      populate();
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::prepare()
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::prepare()" << std::endl;
#endif

      const auto &discretization = offline_data_->discretization();
      AssertThrow(discretization.ansatz() == Ansatz::dg_q1 ||
                      discretization.ansatz() == Ansatz::cg_q1,
                  dealii::ExcMessage("The Euler-Poisson module currently only "
                                     "supports cG/dg Q1 finite elements."));

      AssertThrow(!offline_data_->dof_handler().has_hp_capabilities(),
                  dealii::ExcMessage(
                      "The Euler-Poisson module currently does not support "
                      "DoFHandlers set up with hp capabilities."));

      potential_initialized_ = false;

      /*
       * (Re)initialize matrix free object:
       */

      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
          MatrixFree<dim, Number>::AdditionalData::none;

      // First index CG, second index hyperbolic ansatz
      std::vector<const dealii::DoFHandler<dim> *> dof_handlers = {
          &offline_data_->dof_handler_cg(), &offline_data_->dof_handler()};
      std::vector<const dealii::AffineConstraints<Number> *>
          affine_constraints = {&offline_data_->affine_constraints_cg(),
                                &offline_data_->affine_constraints()};

      // First index full quadrature, second index lumped quadrature
      std::vector<dealii::Quadrature<1>> quadratures = {
          discretization.quadrature_1d()[0],
          discretization.nodal_quadrature_1d()[0]};

      matrix_free_.reinit(discretization.mapping(),
                          dof_handlers,
                          affine_constraints,
                          quadratures,
                          additional_data);

      /*
       * (Re)initialize auxiliary vectors:
       */

      const auto &potential_partitioner =
          matrix_free_.get_dof_info(0).vector_partitioner;
      potential_rhs_.reinit(potential_partitioner);

      const auto &scalar_partitioner =
          matrix_free_.get_dof_info(1).vector_partitioner;
      density_.reinit(scalar_partitioner);
      background_density_.reinit(scalar_partitioner);

      magnetic_field_.reinit(dim == 2 ? 1 : dim);
      for (unsigned int i = 0; i < magnetic_field_.n_blocks(); ++i)
        magnetic_field_.block(i).reinit(scalar_partitioner);

      if (!selected_electrostatic_configuration_->is_time_dependent()) {
        /*
         * Interpolate auxiliary vectors for background fields:
         */

        // FIXME: maybe use a matrix-free loop

        dealii::VectorTools::interpolate(
            discretization.mapping(),
            offline_data_->dof_handler(),
            dealii::ScalarFunctionFromFunctionObject<dim, Number>(
                [&](const dealii::Point<dim> &p) {
                  return selected_electrostatic_configuration_
                      ->background_density(p, 0);
                }),
            background_density_);

        for (unsigned int k = 0; k < (dim == 2 ? 1 : dim); ++k) {
          dealii::VectorTools::interpolate(
              discretization.mapping(),
              offline_data_->dof_handler(),
              to_function<dim, Number>(
                  [&](const dealii::Point<dim> &p) {
                    return selected_electrostatic_configuration_
                        ->magnetic_field(p, 0);
                  },
                  k),
              magnetic_field_.block(k));
        }
      }
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::reinit_state_vector(
        StateVector &state_vector) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::reinit_state_vector()"
                << std::endl;
#endif

      auto &[U, precomputed, V] = state_vector;
      V.reinit(1);

      auto &potential = V.block(0);
      const auto &partitioner = matrix_free_.get_dof_info(0).vector_partitioner;
      potential.reinit(partitioner);

#ifdef DEBUG
      /* Poison potential: */
      constexpr auto nan = std::numeric_limits<Number>::signaling_NaN();
      for (unsigned int i = 0; i < partitioner->locally_owned_size(); ++i)
        potential.local_element(i) = nan;
#endif
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::prepare_state_vector(
        StateVector &state_vector, Number t) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::prepare_state_vector()"
                << std::endl;
#endif

      /*
       * We (re)compute the potential on the first step and if the restart
       * strategy is set to full_restart or static_full_restart.
       */

      if (potential_initialized_ ||
          (gauss_law_restart_strategy_ ==
           GaussLawRestartStrategy::full_restart) ||
          (gauss_law_restart_strategy_ ==
           GaussLawRestartStrategy::static_full_restart)) {
        compute_potential(t, state_vector);
        potential_initialized_ = true;
      }
    }


    template <int dim, typename Number>
    template <int stages>
    void ParabolicModule<dim, Number>::backward_euler_step(
        const StateVector &old_state_vector,
        const Number old_t,
        std::array<std::reference_wrapper<const StateVector>,
                   stages> /*stage_state_vectors*/,
        const std::array<Number, stages> /*stage_weights*/,
        StateVector &new_state_vector,
        Number tau) const
    {
      step(old_state_vector,
           old_t,
           new_state_vector,
           tau,
           /*crank_nicolson_extrapolation = */ false);
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::crank_nicolson_step(
        const StateVector &old_state_vector,
        const Number old_t,
        StateVector &new_state_vector,
        Number tau) const
    {
      try {
        /* Backward Euler step to half time step, and extrapolate: */

        step(old_state_vector,
             old_t,
             new_state_vector,
             tau / Number(2.),
             /*crank_nicolson_extrapolation = */ true);

      } catch (Correction) {

        /*
         * Under very rare circumstances we might fail to perform a Crank
         * Nicolson step because the extrapolation step produced
         * inadmissible states. We could correct the update now by
         * performing a limiting step (either convex limiting, or flux
         * corrected transport)... but *meh*, just perform a backward Euler
         * step:
         */
        step(old_state_vector,
             old_t,
             new_state_vector,
             tau,
             /*crank_nicolson_extrapolation = */ false);
      }
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::compute_potential(
        const Number /*t*/, StateVector & /*state_vector*/) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::compute_potential()"
                << std::endl;
#endif
      const auto &discretization = offline_data_->discretization();

      if (selected_electrostatic_configuration_->is_time_dependent()) {
        dealii::VectorTools::interpolate(
            discretization.mapping(),
            offline_data_->dof_handler(),
            dealii::ScalarFunctionFromFunctionObject<dim, Number>(
                [&](const dealii::Point<dim> &p) {
                  return selected_electrostatic_configuration_
                      ->background_density(p, 0);
                }),
            background_density_);
      }

      return;
    }


    template <int dim, typename Number>
    void
    ParabolicModule<dim, Number>::step(const StateVector &old_state_vector,
                                       const Number /*t*/,
                                       StateVector &new_state_vector,
                                       Number tau [[maybe_unused]],
                                       const bool crank_nicolson_extrapolation
                                       [[maybe_unused]]) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::step()" << std::endl;
#endif
      const auto &discretization = offline_data_->discretization();

      if (selected_electrostatic_configuration_->is_time_dependent()) {
        for (unsigned int k = 0; k < (dim == 2 ? 1 : dim); ++k) {
          dealii::VectorTools::interpolate(
              discretization.mapping(),
              offline_data_->dof_handler(),
              to_function<dim, Number>(
                  [&](const dealii::Point<dim> &p) {
                    return selected_electrostatic_configuration_
                        ->magnetic_field(p, 0);
                  },
                  k),
              magnetic_field_.block(k));
        }
      }

#ifdef DEBUG_OUTPUT
      std::cout << "        perform time-step with tau = " << tau << std::endl;
      if (crank_nicolson_extrapolation)
        std::cout << "        and extrapolate to t + 2 * tau" << std::endl;
#endif

      new_state_vector = old_state_vector;
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::print_solver_statistics(
        std::ostream &output) const
    {
      output << "        [ " << std::setprecision(2) << std::fixed
             << n_iterations_ << (use_gmg_ ? " GMG phi ]" : " CG phi ]")
             << std::endl;
    }

  } // namespace EulerPoisson
} /* namespace ryujin */
