//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <convenience_macros.h>
#include <discretization.h>
#include <patterns_conversion.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <functional>

namespace ryujin
{
  /**
   * Description of a @p dim dimensional parabolic operator for the
   * compressible Navier-Stokes equations.
   *
   * We have a (1 + dim) dimensional state space \f$[\textbf v, e]\f$,
   * where \f$\textbf v\f$ is the velocity, and
   * \f$e\f$ is the internal energy.
   *
   * @ingroup DissipationModule
   */
  class ParabolicSystem final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    // clang-format off
    template<int dim>
    static constexpr unsigned int problem_dimension = 2 + dim;
    // clang-format on

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    template <int dim, typename Number>
    using hyperbolic_state_type =
        dealii::Tensor<1, problem_dimension<dim>, Number>;

    /**
     * The dimension of the parabolic state space.
     */
    template <int dim>
    static constexpr unsigned int parabolic_problem_dimension = 1 + dim;

    /**
     * The storage type used for a parabolic state vector \f$\boldsymbol U\f$.
     */
    template <int dim, typename Number>
    using parabolic_state_type =
        dealii::Tensor<1, parabolic_problem_dimension<dim>, Number>;

    /**
     * The number of decoupled linear systems to solve
     */
    static constexpr unsigned int n_implicit_systems = 2;

    /**
     * Constructor
     */
    ParabolicSystem(const std::string &subsection = "ParabolicSystem");

    void parse_parameters_callback();

    /**
     * @name Run time options
     */
    //@{

    /**
     * Given a conserved state vector @p hyperbolic_state and a (lumped)
     * mass @p m, return the corresponding parabolic state vector
     * \f$[v, e]\f$ and right hand side \f$[m \rho v, m \rho e]\f$.
     */
    template <int problem_dim, typename Number>
    std::array<dealii::Tensor<1, problem_dim - 1, Number>, 2>
    compute_parabolic_state_and_rhs(
        const dealii::Tensor<1, problem_dim, Number> &hyperbolic_state,
        const Number &m) const;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    double mu_;
    ACCESSOR_READ_ONLY(mu)

    double lambda_;
    ACCESSOR_READ_ONLY(lambda)

    double cv_inverse_kappa_;
    ACCESSOR_READ_ONLY(cv_inverse_kappa)

    //@}
  };

  /* Inline definitions */

  template <int pd, typename Number>
  DEAL_II_ALWAYS_INLINE inline std::array<dealii::Tensor<1, pd - 1, Number>, 2>
  ParabolicSystem::compute_parabolic_state_and_rhs(
      const dealii::Tensor<1, pd, Number> &hyperbolic_state,
      const Number &m) const
  {
    constexpr int dim = pd - 2;

    const auto rho = HyperbolicSystem::density(hyperbolic_state);
    const auto M = HyperbolicSystem::momentum(hyperbolic_state);
    const auto rho_e = HyperbolicSystem::internal_energy(hyperbolic_state);

    dealii::Tensor<1, pd - 1, Number> state, rhs;
    for (unsigned int d = 0; d < dim; ++d) {
      state[d] = M[d] / rho;
      rhs[d] = m * M[d];
    }
    state[dim] = rho_e / rho;
    rhs[dim] = m * rho_e;

    return {state, rhs};
  }

} /* namespace ryujin */
