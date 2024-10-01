//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    template <typename ScalarNumber = double>
    class RiemannSolverParameters : public dealii::ParameterAcceptor
    {
    public:
      RiemannSolverParameters(const std::string &subsection = "/RiemannSolver")
          : ParameterAcceptor(subsection)
      {
      }
    };


    /**
     * A fast approximative solver for the 1D Riemann problem. The solver
     * ensures that the estimate \f$\lambda_{\text{max}}\f$ that is returned
     * for the maximal wavespeed is a strict upper bound.
     *
     * The solver is based on @cite ClaytonGuermondPopov-2022.
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number = double>
    class RiemannSolver
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number>;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      /**
       * Number of components in a primitive state, we store \f$[\rho, v,
       * p, a, gamma]\f$, thus, 5.
       */
      static constexpr unsigned int riemann_data_size = 5;

      /**
       * The array type to store the expanded primitive state for the
       * Riemann solver \f$[\rho, v, p, a]\f$
       */
      using primitive_type = typename std::array<Number, riemann_data_size>;

      using precomputed_type = typename View::precomputed_type;

      using PrecomputedVector = typename View::PrecomputedVector;

      using Parameters = RiemannSolverParameters<ScalarNumber>;

      //@}
      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      RiemannSolver(const HyperbolicSystem &hyperbolic_system,
                    const Parameters &parameters,
                    const PrecomputedVector &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * For two given 1D primitive states riemann_data_i and
       * riemann_data_j, compute an estimate for an upper bound of the
       * maximum wavespeed lambda.
       */
      Number compute(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimate for an upper bound of the maximum wavespeed
       * lambda.
       */
      Number compute(const state_type &U_i,
                     const state_type &U_j,
                     const unsigned int i,
                     const unsigned int *js,
                     const dealii::Tensor<1, dim, Number> &n_ij) const;

      //@}

    protected:
      /** @name Internal functions used in the Riemann solver */
      //@{

      /**
       * FIXME
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      Number c(const Number gamma_Z) const;

      /**
       * FIXME
       *
       * Cost: 0x pow, 1x division, 0x sqrt
       */
      Number
      alpha(const Number &rho, const Number &gamma, const Number &a) const;

      /**
       * Compute the best available, but expensive, upper bound on the
       * expansion-shock case as described in §5.4, Eqn. (5.7) and (5.8) in
       * @cite ClaytonGuermondPopov-2022
       *
       * Cost: 5x pow, 11x division, 1x sqrt
       */
      Number p_star_RS_full(const primitive_type &riemann_data_i,
                            const primitive_type &riemann_data_j) const;

      /**
       * Compute the best available, but expensive, upper bound on the
       * shock-shock case as described in §5.5, Eqn. (5.10) and (5.12) in
       * @cite ClaytonGuermondPopov-2022
       *
       * Cost: 2x pow, 9x division, 3x sqrt
       */
      Number p_star_SS_full(const primitive_type &riemann_data_i,
                            const primitive_type &riemann_data_j) const;

      /*
       * Compute only the failsafe the failsafe bound for \f$\tilde
       * p_2^\ast\f$ (5.11) in @cite ClaytonGuermondPopov-2022
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */
      Number p_star_failsafe(const primitive_type &riemann_data_i,
                             const primitive_type &riemann_data_j) const;

      /*
       * Compute a simultaneous upper bound on (5.7) second formula for
       * \tilde p_2^\ast (5.8) first formula for \tilde p_1^\ast (5.11)
       * formula for \tilde p_2^\ast in @cite ClaytonGuermondPopov-2022
       *
       * Cost: 3x pow, 9x division, 2x sqrt
       *
       * @todo improve documentation
       */
      Number p_star_interpolated(const primitive_type &riemann_data_i,
                                 const primitive_type &riemann_data_j) const;


#ifndef DOXYGEN
      /*
       * FIXME
       */
      Number f(const primitive_type &riemann_data, const Number p_star) const;


      /*
       * FIXME
       */
      Number phi(const primitive_type &riemann_data_i,
                 const primitive_type &riemann_data_j,
                 const Number p_in) const;
#endif


      /**
       * See @cite ClaytonGuermondPopov-2022
       *
       * The approximate Riemann solver is based on a function phi(p) that is
       * montone increasing in p, concave down and whose (weak) third
       * derivative is non-negative and locally bounded. Because we
       * actually do not perform any iteration for computing our wavespeed
       * estimate we can get away by only implementing a specialized
       * variant of the phi function that computes phi(p_max). It inlines
       * the implementation of the "f" function and eliminates all
       * unnecessary branches in "f".
       *
       * Cost: 0x pow, 2x division, 2x sqrt
       */
      Number phi_of_p_max(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j) const;


      /**
       * See @cite GuermondPopov2016 page 912, (3.7)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      Number lambda1_minus(const primitive_type &riemann_data,
                           const Number p_star) const;


      /**
       * See @cite GuermondPopov2016 page 912, (3.8)
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      Number lambda3_plus(const primitive_type &primitive_state,
                          const Number p_star) const;


      /**
       * See @cite GuermondPopov2016 page 912, (3.9)
       *
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and a guess p_2, compute an upper bound
       * for lambda.
       *
       * Cost: 0x pow, 2x division, 2x sqrt (inclusive)
       */
      Number compute_lambda(const primitive_type &riemann_data_i,
                            const primitive_type &riemann_data_j,
                            const Number p_star) const;


      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, and a
       * (normalized) "direction" n_ij, first compute the corresponding
       * projected state in the corresponding 1D Riemann problem, and then
       * compute and return the Riemann data [rho, u, p, a] (used in the
       * approximative Riemann solver).
       */
      primitive_type
      riemann_data_from_state(const state_type &U,
                              const Number &p,
                              const Number &gamma,
                              const dealii::Tensor<1, dim, Number> &n_ij) const;

    private:
      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;
      const PrecomputedVector &precomputed_values;
      //@}
    };
  } // namespace EulerAEOS
} /* namespace ryujin */
