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
    using state_type = dealii::Tensor<1, problem_dimension<dim>, Number>;

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
     * An array holding all component names of the conserved state as a string.
     */
    template <int dim>
    static const std::array<std::string, parabolic_problem_dimension<dim>>
        parabolic_component_names;

    /**
     * The number of decoupled linear systems to solve
     */
    static constexpr unsigned int n_implicit_systems = 2;

    /**
     * The block size of the decoupled systems.
     *
     * First we have a velocity block of size dim followed by a scalar
     * internal energy block (size 1).
     */
    template <int dim>
    static constexpr std::array<unsigned int, n_implicit_systems> //
        block_sizes = {dim, 1};

    /**
     * An array holding a string naming all linear subsystems
     */
    static const std::array<std::string, n_implicit_systems>
        implicit_system_names;

    /**
     * Constructor
     */
    ParabolicSystem(const std::string &subsection = "ParabolicSystem");

    void parse_parameters_callback();

    /**
     * @name Routines used transfering states, and assembling the diagonal
     * part of the linear system.
     */
    //@{

    /**
     * Given a conserved state vector @p hyperbolic_state return the
     * corresponding parabolic state vector \f$[v, e]\f$.
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim - 1, Number> to_parabolic_state(
        const dealii::Tensor<1, problem_dim, Number> &hyperbolic_state) const;

    /**
     * Given a conserved state vector @p hyperbolic_state and a parabolic
     * state @p parabolic_state this function returns a hyperbolic_state in
     * conserved quantities composed of velocity and internal energy from
     * the parabolic state and density from the hyperbolic state.
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim + 1, Number> from_parabolic_state(
        dealii::Tensor<1, problem_dim + 1, Number> U,
        const dealii::Tensor<1, problem_dim, Number> &parabolic_state) const;

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

    /**
     * Given a conserved state vector @p hyperbolic_state and a (lumped)
     * mass @p m, return an appropriate diagonal scaling for the linear
     * system, in our case \f$(\rho m_i)^{-1}\f$.
     */
    template <int problem_dim, typename Number>
    Number compute_diagonal_scaling(
        const dealii::Tensor<1, problem_dim, Number> &hyperbolic_state,
        const Number &m) const;

    /**
     * Given a conserved state vector @p hyperbolic_state and a (lumped)
     * mass @p m, return the "diagonal action" of the linear subsystem. In
     * our case this is a multiplication by \f$(m_i \rho)\f$.
     */
    template <int problem_dim, typename Number>
    Number compute_diagonal_action(
        const dealii::Tensor<1, problem_dim, Number> &hyperbolic_state,
        const Number &m) const;

    //@}
    /**
     * @name Routines used for enforcing and applying boundary conditions
     */
    //@{

    /**
     * Apply boundary conditions.
     *
     * For the parabolic diffusion problem we have:
     *
     *  - Dirichlet boundary conditions by prescribing the velocity field
     *    and internal energy described by the conserved state
     *    get_dirichlet_data().
     *
     *  - Slip boundary conditions where we remove the normal component of
     *    the velocity.
     *
     *  - No slip boundary conditions where we set the velocity to 0.
     */
    template <int problem_dim, typename Number, typename Lambda>
    dealii::Tensor<1, problem_dim, Number> apply_boundary_conditions(
        dealii::types::boundary_id id,
        dealii::Tensor<1, problem_dim, Number> U,
        const dealii::Tensor<1, problem_dim - 1, Number> &normal,
        Lambda get_dirichlet_data) const;

    /**
     * "Propagate the action" of boundary conditions in the linear
     * operator.
     *
     * For the parabolic diffusion problem we have:
     *
     *  - Dirichlet boundary conditions: Copy the src state to the
     *    destination state.
     *
     *  - No slip boundary conditions: Copy the velocity component of the
     *    src state to the destination state.
     *    the velocity.
     *
     *  - Slip boundary conditions: Copy the velocity normal component of
     *    the source state over to the dst state.
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> apply_boundary_action(
        dealii::types::boundary_id id,
        dealii::Tensor<1, problem_dim, Number> dst,
        const dealii::Tensor<1, problem_dim - 1, Number> &normal,
        const dealii::Tensor<1, problem_dim, Number> &src) const;

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
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, pd - 1, Number>
  ParabolicSystem::to_parabolic_state(
      const dealii::Tensor<1, pd, Number> &hyperbolic_state) const
  {
    constexpr int dim = pd - 2;

    const auto rho = HyperbolicSystem::density(hyperbolic_state);
    const auto M = HyperbolicSystem::momentum(hyperbolic_state);
    const auto rho_e = HyperbolicSystem::internal_energy(hyperbolic_state);

    dealii::Tensor<1, pd - 1, Number> state;
    for (unsigned int d = 0; d < dim; ++d) {
      state[d] = M[d] / rho;
    }
    state[dim] = rho_e / rho;

    return state;
  }


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


  template <int pd, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ParabolicSystem::compute_diagonal_scaling(
      const dealii::Tensor<1, pd, Number> &hyperbolic_state,
      const Number &m) const
  {
    const auto rho = HyperbolicSystem::density(hyperbolic_state);
    return Number(1.) / (rho * m);
  }


  template <int pd, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ParabolicSystem::compute_diagonal_action(
      const dealii::Tensor<1, pd, Number> &hyperbolic_state,
      const Number &m) const
  {
    const auto rho = HyperbolicSystem::density(hyperbolic_state);
    return (rho * m);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim + 1, Number>
  ParabolicSystem::from_parabolic_state(
      dealii::Tensor<1, problem_dim + 1, Number> U,
      const dealii::Tensor<1, problem_dim, Number> &parabolic_state) const
  {
    constexpr int dim = problem_dim - 1;

    const auto rho = HyperbolicSystem::density(U);
    auto E = rho * parabolic_state[dim];
    for (unsigned int d = 0; d < dim; ++d) {
      const auto v = parabolic_state[d];
      E += Number(0.5) * rho * v * v;
      U[1 + d] = rho * v;
    }
    U[1 + dim] = E;

    return U;
  }


  template <int problem_dim, typename Number, typename Lambda>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ParabolicSystem::apply_boundary_conditions(
      dealii::types::boundary_id id,
      dealii::Tensor<1, problem_dim, Number> V,
      const dealii::Tensor<1, problem_dim - 1, Number> &normal,
      Lambda get_dirichlet_data) const
  {
    constexpr auto dim = problem_dim - 1;

    if (id == Boundary::slip) {
      /* Remove normal component of velocity: */
      dealii::Tensor<1, dim> velocity;
      for (unsigned int d = 0; d < dim; ++d)
        velocity[d] = V[d];
      velocity -= (velocity * normal) * normal;
      for (unsigned int d = 0; d < dim; ++d)
        V[d] = velocity[d];

    } else if (id == Boundary::no_slip) {

      /* Set velocity to zero: */
      for (unsigned int d = 0; d < dim; ++d) {
        V[d] = Number(0.);
      }

    } else if (id == Boundary::dirichlet) {

      /* Prescribe velocity and internal energy: */
      const auto U = get_dirichlet_data();
      V = to_parabolic_state(U);
    }

    return V;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ParabolicSystem::apply_boundary_action(
      dealii::types::boundary_id id,
      dealii::Tensor<1, problem_dim, Number> dst,
      const dealii::Tensor<1, problem_dim - 1, Number> &normal,
      const dealii::Tensor<1, problem_dim, Number> &src) const
  {
    constexpr auto dim = problem_dim - 1;

    if (id == Boundary::slip) {
      /* Replace normal component of velocity: */
      dealii::Tensor<1, dim> velocity_dst, velocity_src;
      for (unsigned int d = 0; d < dim; ++d) {
        velocity_dst[d] = dst[d];
        velocity_src[d] = src[d];
      }
      velocity_dst += ((velocity_src - velocity_dst) * normal) * normal;
      for (unsigned int d = 0; d < dim; ++d)
        dst[d] = velocity_dst[d];

    } else if (id == Boundary::no_slip) {
      /* Replace velocity: */
      for (unsigned int d = 0; d < dim; ++d) {
        dst[d] = src[d];
      }

    } else if (id == Boundary::dirichlet) {
      /* Replace velocity and internal energy: */
      dst = src;
    }

    return dst;
  }

} /* namespace ryujin */
