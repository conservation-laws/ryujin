//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include "parabolic_module.h"

#include <instrumentation.h>
#include <openmp.h>
#include <scope.h>
#include <simd.h>

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
        const std::string &subsection /*= "ParabolicModule"*/)
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
    {
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
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::prepare()
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::prepare()" << std::endl;
#endif
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::reinit_state_vector(
        StateVector & /*state_vector*/) const
    {
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::prepare_state_vector(
        StateVector & /*state_vector*/, Number /*t*/) const
    {
      /*
       * There is no parabolic part of the state vector for Navier-Stokes,
       * so we do nothing.
       */
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
      /* Backward Euler step to half time step, and extrapolate: */

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
    void ParabolicModule<dim, Number>::step(
        const StateVector &old_state_vector,
        const Number /*t*/,
        StateVector &new_state_vector,
        Number tau [[maybe_unused]],
        const bool crank_nicolson_extrapolation [[maybe_unused]]) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::step()" << std::endl;
#endif

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
