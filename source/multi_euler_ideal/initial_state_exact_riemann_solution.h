//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include "hyperbolic_system.h"

#include <cmath>
#include <deal.II/base/tensor.h>

#include <initial_state_library.h>

// #define DEBUG_SOLUTION

namespace ryujin
{
  namespace MultiSpeciesEulerInitialStates
  {
    using namespace MultiSpeciesEuler;

    /**
     * The exact Riemann solution.
     *
     * This initial class computes the analytic solution for the
     * compressible multi-species Euler equations with an ideal gas equation of
     * state for each species.
     *
     * @note This class is currently only implemented for exactly 2 species.
     * A compile-time assertion will fail if n_species != 2.
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @ingroup MultiSpeciesEulerEquations
     */

    template <typename Description, int dim, typename Number>
    class ExactRiemannSolution : public InitialState<Description, dim, Number>
    {
    public:
      //@{

      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      using ScalarNumber = typename View::ScalarNumber;

      /* Primitive state for 2 species: (Y_0, rho, u, p) */
      static constexpr unsigned int primitive_dim = 4;
      using primitive_state_type = dealii::Tensor<1, primitive_dim, Number>;

      ExactRiemannSolution(const HyperbolicSystem &hyperbolic_system,
                           const std::string subsection)
          : InitialState<Description, dim, Number>("exact riemann solution",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        Assert(n_species == 2,
               dealii::ExcMessage(
                   "ExactRiemannSolution is only implemented for 2 species"));

        primitive_left_[0] = 0.5;
        primitive_left_[1] = 1.4;
        primitive_left_[2] = 0.0;
        primitive_left_[3] = 1.0;
        this->add_parameter(
            "primitive state left",
            primitive_left_,
            "Initial 1d primitive state (Y_0, rho, u, p) on the left");

        primitive_right_[0] = 0.5;
        primitive_right_[1] = 1.4;
        primitive_right_[2] = 0.0;
        primitive_right_[3] = 1.0;
        this->add_parameter(
            "primitive state right",
            primitive_right_,
            "Initial 1d primitive state (Y_0, rho, u, p) on the right");

        // Prepare the Riemann data
        const auto prepare_riemann_data = [&]() {
          // Convert to alpha_0 rho_0, alpha_1 rho_1, u, p
          primitive_left_ = transform(primitive_left_);
          primitive_right_ = transform(primitive_right_);

          const Number p_L = primitive_left_[3];
          const Number p_R = primitive_right_[3];

          p_star_ = compute_pstar(p_L, p_R, primitive_left_, primitive_right_);

          const Number u_L = primitive_left_[2];
          u_star_ = u_L - fZofP(p_star_, primitive_left_);

#ifdef DEBUG_SOLUTION
          const Number u_R = primitive_right_[2];
          std::cout << "left data          = " << primitive_left_
                    << "\nright data       = " << primitive_right_
                    << "\np_star           = " << p_star_
                    << "\nu_star           = " << u_star_
                    << "\nVerifying u_star = "
                    << u_R + fZofP(p_star_, primitive_right_) << std::endl;
#endif

          lambda_left_minus_ = lambda(p_star_, primitive_left_, -1.);
          lambda_left_plus_ =
              lambda_intermediate(p_star_, primitive_left_, -1.);
          lambda_right_minus_ =
              lambda_intermediate(p_star_, primitive_right_, 1.);
          lambda_right_plus_ = lambda(p_star_, primitive_right_, 1.);


#ifdef DEBUG_SOLUTION
          std::cout << "lambda_left_minus    =  " << lambda_left_minus_
                    << "\nlambda_left_plus   =  " << lambda_left_plus_
                    << "\nlambda_right_minus = " << lambda_right_minus_
                    << "\nlambda_right_plus  =  " << lambda_right_plus_
                    << std::endl;
#endif
        };

        this->parse_parameters_call_back.connect(prepare_riemann_data);
        prepare_riemann_data();
      }


      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        const double &x = point[0];

        const Number xi = x / t;

        primitive_state_type primitive_state;

        if (t < 1.e-14 && x < 0.) {
          primitive_state = primitive_left_;
#ifdef DEBUG_SOLUTION
          std::cout << "Left primitive state: " << primitive_state << std::endl;
#endif

        } else if (t < 1.e-14 && x > 0.) {
          primitive_state = primitive_right_;
#ifdef DEBUG_SOLUTION
          std::cout << "Right primitive state: " << primitive_state
                    << std::endl;
#endif

        } else if (xi < lambda_left_minus_) {
          /* Left state: */
          primitive_state = primitive_left_;
#ifdef DEBUG_SOLUTION
          std::cout << "Left primitive state: " << primitive_state << std::endl;
#endif

        } else if (xi < lambda_left_plus_) {
          const auto c_LL =
              expansion_solution(p_star_, xi, primitive_left_, -1.);
          primitive_state = c_LL;
#ifdef DEBUG_SOLUTION
          std::cout << "Left expansion state: " << primitive_state << std::endl;
#endif

        } else if (xi < u_star_) {
          primitive_state = cstar_solution(p_star_, u_star_, primitive_left_);

          const Number p_L = primitive_left_[3];
          if (p_star_ < p_L)
            primitive_state = expansion_solution(
                p_star_, lambda_left_plus_, primitive_left_, -1.);
#ifdef DEBUG_SOLUTION
          std::cout << "Left cstar state: " << primitive_state << std::endl;
#endif

        } else if (xi < lambda_right_minus_) {
          primitive_state = cstar_solution(p_star_, u_star_, primitive_right_);

          const Number p_R = primitive_right_[3];
          if (p_star_ < p_R)
            primitive_state = expansion_solution(
                p_star_, lambda_right_minus_, primitive_right_, 1.);
#ifdef DEBUG_SOLUTION
          std::cout << "Right cstar state: " << primitive_state << std::endl;
#endif

        } else if (xi < lambda_right_plus_) {
          primitive_state =
              expansion_solution(p_star_, xi, primitive_right_, 1.);
#ifdef DEBUG_SOLUTION
          std::cout << "Right expansion state: " << primitive_state
                    << std::endl;
#endif

        } else {
          /* Right state: */
          primitive_state = primitive_right_;
#ifdef DEBUG_SOLUTION
          std::cout << "Right primitive state: " << primitive_state
                    << std::endl;
#endif
        }

        return view.from_initial_state(primitive_state);
      }

    private:
      //@}
      /**
       * Runtime parameters
       */
      //@{

      primitive_state_type primitive_left_;
      primitive_state_type primitive_right_;

      //@}
      /**
       * Private fields
       */
      //@{

      const HyperbolicSystem &hyperbolic_system_;

      Number p_star_;
      Number u_star_;
      Number lambda_left_minus_;
      Number lambda_left_plus_;
      Number lambda_right_minus_;
      Number lambda_right_plus_;

      //@}
      /**
       * Internal helper functions for solving the exact Riemann problem
       */
      //@{

      // Used to transform data to (alpha_rho_0, alpha_rho_1, u, p)
      primitive_state_type transform(primitive_state_type &temp_in) const
      {
        primitive_state_type result;
        result[0] = temp_in[0] * temp_in[1];        // = alpha_0 rho_0;
        result[1] = (1. - temp_in[0]) * temp_in[1]; // = alpha_1 rho_1;

        for (unsigned int i = 2; i < 4; ++i)
          result[i] = temp_in[i];

        return result;
      }

      Number fZofP(const Number &p_in,
                   const primitive_state_type &data_in) const
      {
        // Need hyperbolic system for particular quantities
        const auto view = hyperbolic_system_.template view<dim, Number>();
        const auto conserved_in = view.from_initial_state(data_in);

        // Get left/right data
        const Number rho_Z = view.density(conserved_in);
        const Number p_Z = data_in[3];
        const Number gamma_ = view.gamma_mixture(conserved_in);

        const Number c_Z = std::sqrt(gamma_ * p_Z / rho_Z);

        const Number A_Z = 2. / (gamma_ + 1.) / rho_Z;
        const Number B_Z = (gamma_ - 1.) / (gamma_ + 1.) * p_Z;

        const Number exp = 0.5 * (gamma_ - 1.) / gamma_;
        Number left_brach = 2. * c_Z / (gamma_ - 1.);
        left_brach *= (std::pow(p_in / p_Z, exp) - 1.);

        Number f_of_p = (p_in - p_Z) * std::sqrt(A_Z / (p_in + B_Z));

        if (p_in <= p_Z)
          f_of_p = left_brach;

        return f_of_p;
      }


      Number dfZofP(const Number &p_in,
                    const primitive_state_type &data_in) const
      {
        // Need hyperbolic system for particular quantities
        const auto view = hyperbolic_system_.template view<dim, Number>();
        const auto conserved_in = view.from_initial_state(data_in);

        // Get left/right data
        const Number rho_Z = view.density(conserved_in);
        const Number p_Z = data_in[3];
        const Number gamma_Z = view.gamma_mixture(conserved_in);

        const Number c_Z = std::sqrt(gamma_Z * p_Z / rho_Z);

        const Number A_Z = 2. / (gamma_Z + 1.) / rho_Z;
        const Number B_Z = (gamma_Z - 1.) / (gamma_Z + 1.) * p_Z;

        Number exp = 0.5 * (gamma_Z - 1.) / gamma_Z;
        Number left_brach = 2. * c_Z / (gamma_Z - 1.) * exp;
        exp -= 1.;

        left_brach *= std::pow(p_in / p_Z, exp - 1.) / p_Z;

        Number right_branch = std::pow(A_Z / (p_in + B_Z), 1.5);
        right_branch *= (2. * B_Z + p_in + p_Z) / (2. * A_Z);

        Number df_of_p = right_branch;

        if (p_in <= p_Z)
          df_of_p = left_brach;

        return df_of_p;
      }


      Number dphi(const Number &p_in,
                  const primitive_state_type &data_left,
                  const primitive_state_type &data_right) const
      {
        return dfZofP(p_in, data_left) + dfZofP(p_in, data_right);
      }


      Number phi(const Number &p_in,
                 const primitive_state_type &data_left,
                 const primitive_state_type &data_right) const
      {
        const Number u_L = data_left[2];
        const Number u_R = data_right[2];

        return fZofP(p_in, data_right) + fZofP(p_in, data_left) + u_R - u_L;
      }


      Number lambda(const Number &p_in,
                    const primitive_state_type &data_in,
                    const Number &sign) const
      {
        // Need hyperbolic system for particular quantities
        const auto view = hyperbolic_system_.template view<dim, Number>();
        const auto conserved_in = view.from_initial_state(data_in);

        // Get left/right data
        const Number rho_Z = view.density(conserved_in);
        const Number u_Z = data_in[2];
        const Number p_Z = data_in[3];
        const Number gamma_Z = view.gamma_mixture(conserved_in);

        const Number c_Z = std::sqrt(gamma_Z * p_Z / rho_Z);

        const Number radicand =
            1. + 0.5 * (gamma_Z + 1.) / gamma_Z * std::max(p_in / p_Z - 1., 0.);

        return u_Z + sign * c_Z * std::sqrt(radicand);
      }


      Number lambda_intermediate(const Number &p_in,
                                 const primitive_state_type &data_in,
                                 const Number &sign) const
      {
        // Need hyperbolic system for particular quantities
        const auto view = hyperbolic_system_.template view<dim, Number>();
        const auto conserved_in = view.from_initial_state(data_in);

        // Get left/right data
        const Number rho_Z = view.density(conserved_in);
        const Number u_Z = data_in[2];
        const Number p_Z = data_in[3];
        const Number gamma_Z = view.gamma_mixture(conserved_in);

        const Number c_Z = std::sqrt(gamma_Z * p_Z / rho_Z);

        const auto lambda_value = lambda(p_in, data_in, sign);

        const Number f_of_p = fZofP(p_in, data_in);

        const Number exp = 0.5 * (gamma_Z - 1.) / gamma_Z;
        const Number expansion_speed =
            u_Z + sign * (f_of_p + c_Z * std::pow(p_in / p_Z, exp));

        Number result = lambda_value;
        if (p_in < p_Z)
          result = expansion_speed;

        return result;
      }


      primitive_state_type
      cstar_solution(const Number &p_star,
                     const Number &u_star,
                     const primitive_state_type &data_in) const
      {
        // Need hyperbolic system for particular quantities
        const auto view = hyperbolic_system_.template view<dim, Number>();
        const auto conserved_in = view.from_initial_state(data_in);

        // Get left/right data
        const Number rho_Z = view.density(conserved_in);
        const Number p_Z = data_in[3];
        const Number gamma_Z = view.gamma_mixture(conserved_in);

        // Define rho_star
        const Number p_ratio = p_star / p_Z;
        const Number gamma_ratio = (gamma_Z - 1.) / (gamma_Z + 1.);

        const Number numerator = rho_Z * (p_ratio + gamma_ratio);
        const Number denominator = gamma_ratio * p_ratio + 1.;

        Number rho_star = numerator / denominator;

        auto result = data_in;
        result[0] = data_in[0] / rho_Z;
        result[1] = rho_star;
        result[2] = u_star;
        result[3] = p_star;

        return transform(result);
      }


      primitive_state_type
      expansion_solution(const Number & /*p_star*/,
                         const Number &xi,
                         const primitive_state_type &data_in,
                         const Number &sign) const
      {
        // Need hyperbolic system for particular quantities
        const auto view = hyperbolic_system_.template view<dim, Number>();
        const auto conserved_in = view.from_initial_state(data_in);

        // Get left/right data
        const Number rho_Z = view.density(conserved_in);
        const Number u_Z = data_in[2];
        const Number p_Z = data_in[3];
        const Number gamma_Z = view.gamma_mixture(conserved_in);

        const Number c_Z = std::sqrt(gamma_Z * p_Z / rho_Z);

        // Define rho_expansion
        const Number gamma_ratio = (gamma_Z - 1.) / (gamma_Z + 1.);

        const Number first = 2. / (gamma_Z + 1.);
        const Number second = gamma_ratio / c_Z * (u_Z - xi);
        const Number exp = 2. / (gamma_Z - 1.);

        Number rho_expansion = rho_Z * std::pow(first - sign * second, exp);

        // Define p_expansion
        const Number factor = p_Z / std::pow(rho_Z, gamma_Z);
        const Number p_expansion = factor * std::pow(rho_expansion, gamma_Z);

        // Define u_expansion
        const Number u_expansion = u_Z + sign * fZofP(p_expansion, data_in);

        auto result = data_in;
        result[0] = data_in[0] / rho_Z;
        result[1] = rho_expansion;
        result[2] = u_expansion;
        result[3] = p_expansion;

        return transform(result);
      }


      /**
       * Compute pstar using the quadratic_newton_step()
       */
      double compute_pstar(double p_1,
                           double p_2,
                           primitive_state_type data_1,
                           primitive_state_type data_2)
      {
        constexpr Number eps = std::numeric_limits<Number>::epsilon();

        // Ensure that p_1 <= p_2

        if (p_1 > p_2) {
          std::swap(p_1, p_2);
          std::swap(data_1, data_2);
        }

#ifdef DEBUG
        const double phi_1 = phi(p_1, data_1, data_2);
        const double phi_2 = phi(p_2, data_1, data_2);
        Assert(phi_1 * phi_2 <= 0.,
               dealii::ExcMessage(
                   "Euler::ExactRiemannSolver: failed to compute p_star."));
#endif

        //
        // We simply compute the root of phi with a bisection method down
        // to machine precision. This is not terribly efficient but luckily
        // happens only once during initialization.
        //

#ifdef DEBUG_SOLUTION
        std::cout << "Computing p_star with a bisection method." << std::endl;
#endif

        unsigned int iter = 0;
        for (; iter < 200; ++iter) {

          // Check for convergence:
          if (std::abs(p_2 - p_1) < 10. * eps * std::max(p_1, p_2)) {
            break;
          }

          const double phi_2 = phi(p_2, data_1, data_2);

#ifdef DEBUG_SOLUTION
          const double phi_1 = phi(p_1, data_1, data_2);

          std::cout << "\niter: " << iter << "\n";
          std::cout << "p_1: " << p_1 << "\n";
          std::cout << "p_2: " << p_2 << "\n";
          std::cout << "phi_1: " << phi_1 << "\n";
          std::cout << "phi_2: " << phi_2 << "\n";
#endif

          const auto p_m = 0.5 * (p_2 + p_1);
          const double phi_m = phi(p_m, data_1, data_2);

          if (phi_m * phi_2 >= 0.) {
            p_2 = p_m;
          } else {
            p_1 = p_m;
          }
        }

#ifdef DEBUG_SOLUTION
        const double phi_2 = phi(p_2, data_1, data_2);
        std::cout << "After " << iter << " iterations:"
                  << "\np_star =      " << p_2 << "\nphi(p_star) = " << phi_2
                  << "\n|p_2 - p_1| = " << std::abs(p_2 - p_1) << std::endl;
#endif

        return p_2;
      }

      //@}
    };
  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
