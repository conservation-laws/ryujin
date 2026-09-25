//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include <compile_time_options.h>

#include <initial_state_library.h>

#include <deal.II/base/tensor.h>

#include <cmath>

// #define DEBUG_SOLUTION

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * The exact Riemann solution.
     *
     * This initial class computes the analytic solution for the
     * compressible Euler equations with ideal gas equation of state.
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @ingroup EulerEquations
     */

    template <typename Description, int dim, typename Number>
    class ExactRiemannSolution : public InitialState<Description, dim, Number>
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;

      using ScalarNumber = typename View::ScalarNumber;


      ExactRiemannSolution(const HyperbolicSystem &hyperbolic_system,
                           const std::string subsection)
          : InitialState<Description, dim, Number>("exact riemann solution",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        gamma_ = 1.4;
        if constexpr (!View::have_gamma) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        primitive_left_[0] = 1.4;
        primitive_left_[1] = 0.0;
        primitive_left_[2] = 1.0;
        this->add_parameter("primitive state left",
                            primitive_left_,
                            "1d primitive state [rho, u, p] (for the "
                            "polytropic gas EOS) on the left");

        primitive_right_[0] = 1.4;
        primitive_right_[1] = 0.0;
        primitive_right_[2] = 1.0;
        this->add_parameter("primitive state right",
                            primitive_right_,
                            "1d primitive state [rho, u, p] (for the "
                            "polytropic gas EOS) on the right");

        // Convert the primitive states to conserved states
        const auto prepare_riemann_data = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();
          if constexpr (View::have_gamma) {
            gamma_ = view.gamma();
          }

          p_star_ = compute_pstar(primitive_left_, primitive_right_);

          const Number u_L = primitive_left_[1];
          u_star_ = u_L - fZofP(p_star_, primitive_left_);

#ifdef DEBUG_SOLUTION
          const Number u_R = primitive_right_[1];
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
          std::cout << "lambda_left_minus  =  " << lambda_left_minus_
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

        dealii::Tensor<1, 3, Number> primitive_state;

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

          const Number p_L = primitive_left_[2];
          if (p_star_ < p_L)
            primitive_state = expansion_solution(
                p_star_, lambda_left_plus_, primitive_left_, -1.);
#ifdef DEBUG_SOLUTION
          std::cout << "Left cstar state: " << primitive_state << std::endl;
#endif

        } else if (xi < lambda_right_minus_) {
          primitive_state = cstar_solution(p_star_, u_star_, primitive_right_);

          const Number p_R = primitive_right_[2];
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

        using state_type_1d =
            typename HyperbolicSystem::template View<1, Number>::state_type;
        static_assert(state_type_1d::dimension <=
                      dealii::Tensor<1, 3, Number>::dimension);

        state_type_1d result;
        for (unsigned int i = 0; i < state_type_1d::dimension; ++i)
          result[i] = primitive_state[i];
        return view.from_initial_state(result);
      }

    private:
      //@}
      /**
       * @name Run time options
       */
      //@{

      Number gamma_;

      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;

      //@}
      /**
       * @name Internal data
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
       * @name Internal methods
       */
      //@{

      Number fZofP(const Number &p_in,
                   const dealii::Tensor<1, 3, Number> &data_in) const
      {
        // Get left/right data
        const Number rho_Z = data_in[0];
        const Number p_Z = data_in[2];

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
                    const dealii::Tensor<1, 3, Number> &data_in) const
      {
        // Get left/right data
        const Number rho_Z = data_in[0];
        const Number p_Z = data_in[2];

        const Number c_Z = std::sqrt(gamma_ * p_Z / rho_Z);

        const Number A_Z = 2. / (gamma_ + 1.) / rho_Z;
        const Number B_Z = (gamma_ - 1.) / (gamma_ + 1.) * p_Z;

        Number exp = 0.5 * (gamma_ - 1.) / gamma_;
        Number left_brach = 2. * c_Z / (gamma_ - 1.) * exp;
        exp -= 1.;

        left_brach *= std::pow(p_in / p_Z, exp) / p_Z;

        Number right_branch = std::pow(A_Z / (p_in + B_Z), 1.5);
        right_branch *= (2. * B_Z + p_in + p_Z) / (2. * A_Z);

        Number df_of_p = right_branch;

        if (p_in <= p_Z)
          df_of_p = left_brach;

        return df_of_p;
      }


      Number dphi(const Number &p_in,
                  const dealii::Tensor<1, 3, Number> &data_left,
                  const dealii::Tensor<1, 3, Number> &data_right) const
      {
        return dfZofP(p_in, data_left) + dfZofP(p_in, data_right);
      }


      Number phi(const Number &p_in,
                 const dealii::Tensor<1, 3, Number> &data_left,
                 const dealii::Tensor<1, 3, Number> &data_right) const
      {
        const Number u_L = data_left[1];
        const Number u_R = data_right[1];

        return fZofP(p_in, data_right) + fZofP(p_in, data_left) + u_R - u_L;
      }


      Number lambda(const Number &p_in,
                    const dealii::Tensor<1, 3, Number> &data_in,
                    const Number &sign) const
      {
        // Get left/right data
        const Number rho_Z = data_in[0];
        const Number u_Z = data_in[1];
        const Number p_Z = data_in[2];

        const Number c_Z = std::sqrt(gamma_ * p_Z / rho_Z);

        const Number radicand =
            1. + 0.5 * (gamma_ + 1.) / gamma_ * std::max(p_in / p_Z - 1., 0.);

        return u_Z + sign * c_Z * std::sqrt(radicand);
      }


      Number lambda_intermediate(const Number &p_in,
                                 const dealii::Tensor<1, 3, Number> &data_in,
                                 const Number &sign) const
      {
        const Number rho_Z = data_in[0];
        const Number u_Z = data_in[1];
        const Number p_Z = data_in[2];

        const Number c_Z = std::sqrt(gamma_ * p_Z / rho_Z);

        const auto lambda_value = lambda(p_in, data_in, sign);

        const Number f_of_p = fZofP(p_in, data_in);

        const Number exp = 0.5 * (gamma_ - 1.) / gamma_;
        const Number expansion_speed =
            u_Z + sign * (f_of_p + c_Z * std::pow(p_in / p_Z, exp));

        Number result = lambda_value;
        if (p_in < p_Z)
          result = expansion_speed;

        return result;
      }


      dealii::Tensor<1, 3, Number>
      cstar_solution(const Number &p_star,
                     const Number &u_star,
                     const dealii::Tensor<1, 3, Number> &data_in) const
      {
        const Number rho_Z = data_in[0];
        const Number p_Z = data_in[2];

        // Define rho_star
        const Number p_ratio = p_star / p_Z;
        const Number gamma_ratio = (gamma_ - 1.) / (gamma_ + 1.);

        const Number numerator = rho_Z * (p_ratio + gamma_ratio);
        const Number denominator = gamma_ratio * p_ratio + 1.;

        Number rho_star = numerator / denominator;

        auto result = data_in;
        result[0] = rho_star;
        result[1] = u_star;
        result[2] = p_star;

        return result;
      }


      dealii::Tensor<1, 3, Number>
      expansion_solution(const Number & /*p_star*/,
                         const Number &xi,
                         const dealii::Tensor<1, 3, Number> &data_in,
                         const Number &sign) const
      {
        const Number rho_Z = data_in[0];
        const Number u_Z = data_in[1];
        const Number p_Z = data_in[2];

        const Number c_Z = std::sqrt(gamma_ * p_Z / rho_Z);

        // Define rho_expansion
        const Number gamma_ratio = (gamma_ - 1.) / (gamma_ + 1.);

        const Number first = 2. / (gamma_ + 1.);
        const Number second = gamma_ratio / c_Z * (u_Z - xi);
        const Number exp = 2. / (gamma_ - 1.);

        Number rho_expansion = rho_Z * std::pow(first - sign * second, exp);

        // Define p_expansion
        const Number factor = p_Z / std::pow(rho_Z, gamma_);
        const Number p_expansion = factor * std::pow(rho_expansion, gamma_);

        // Define u_expansion
        const Number u_expansion = u_Z + sign * fZofP(p_expansion, data_in);

        auto result = data_in;
        result[0] = rho_expansion;
        result[1] = u_expansion;
        result[2] = p_expansion;

        return result;
      }


      /**
       * The two-rarefaction approximation \f$\tilde p^\ast\f$ to
       * \f$p^\ast\f$, i.e., the (closed-form) root of the two-rarefaction
       * branch \f$\phi_R\f$ of \f$\phi\f$.
       *
       * See @cite GuermondPopov2016b, page 914, (4.3) (with covolume
       * \f$b=0\f$); this is also equation (4.103) in Toro, Chapter 4.7.2.
       *
       * By @cite GuermondPopov2016b, Lemma 4.3 we have \f$p^\ast <
       * \tilde p^\ast\f$ for the physical range \f$1 < \gamma \le 5/3\f$,
       * which is what makes \f$\tilde p^\ast\f$ usable as an upper bound
       * for the bracketing interval in Algorithm 1.
       */
      double p_tilde_star(const dealii::Tensor<1, 3, Number> &data_left,
                          const dealii::Tensor<1, 3, Number> &data_right) const
      {
        const Number rho_L = data_left[0];
        const Number u_L = data_left[1];
        const Number p_L = data_left[2];

        const Number rho_R = data_right[0];
        const Number u_R = data_right[1];
        const Number p_R = data_right[2];

        const Number c_L = std::sqrt(gamma_ * p_L / rho_L);
        const Number c_R = std::sqrt(gamma_ * p_R / rho_R);

        const Number exp = 0.5 * (gamma_ - 1.) / gamma_;

        const Number numerator = c_L + c_R - 0.5 * (gamma_ - 1.) * (u_R - u_L);
        const Number denominator =
            c_L * std::pow(p_L, -exp) + c_R * std::pow(p_R, -exp);

        return std::pow(numerator / denominator, 1. / exp);
      }


      /**
       * Compute the intermediate ("star") pressure \f$p^\ast\f$, i.e., the
       * unique root of \f$\phi\f$, see @cite GuermondPopov2016b, page 912,
       * (3.3).
       *
       * The bracketing interval \f$[p_1, p_2]\f$ with \f$p_1 \le p^\ast \le
       * p_2\f$ is initialized following @cite GuermondPopov2016b,
       * Algorithm 1 ("Initialization"); the root is then computed by
       * bisection down to machine precision. (The paper continues with the
       * quadratic Newton iteration of its Algorithm 2, which converges much
       * faster. We do not need the speed here: this happens exactly once
       * during initialization, and unlike RiemannSolver we want the root
       * itself and not merely a bound on \f$\lambda_{\max}\f$.)
       *
       * @note The two states must be passed in left/right order and must
       * never be transposed. The concavity of \f$\phi\f$, which is what
       * justifies the bracketing and the Newton step of Algorithm 1 (see
       * @cite GuermondPopov2016b, Theorem 4.1), does require the two
       * pressures to be ordered, but ordering them as *scalars* (line 1 of
       * Algorithm 1, \f$p_{\min}\f$ and \f$p_{\max}\f$ below) is enough:
       * the sum \f$f(p,L) + f(p,R)\f$ is symmetric, so a left/right
       * transposition of the two states leaves \f$\phi\f$ concave but
       * shifts it by the constant \f$-2(u_R - u_L)\f$, which moves the root
       * whenever \f$u_L \neq u_R\f$.
       */
      double compute_pstar(const dealii::Tensor<1, 3, Number> &data_left,
                           const dealii::Tensor<1, 3, Number> &data_right)
      {
        constexpr Number eps = std::numeric_limits<Number>::epsilon();

        /* Algorithm 1, line 1: */

        const double p_min = std::min(data_left[2], data_right[2]);
        const double p_max = std::max(data_left[2], data_right[2]);

        /*
         * The non-vacuum condition, see @cite GuermondPopov2016b, page 912,
         * (3.6): phi(0) < 0 is equivalent to
         *
         *   u_R - u_L < 2 c_L / (gamma - 1) + 2 c_R / (gamma - 1).
         *
         * If it is violated the Riemann solution contains a vacuum region
         * and there is no star state to compute.
         */

        AssertThrow(
            phi(Number(0.), data_left, data_right) < 0.,
            dealii::ExcMessage(
                "Euler::ExactRiemannSolution: the prescribed left and right "
                "states violate the non-vacuum condition; the exact Riemann "
                "solution contains a vacuum region and has no star state."));

        const double phi_p_min = phi(p_min, data_left, data_right);
        const double phi_p_max = phi(p_max, data_left, data_right);

        const double p_tilde = p_tilde_star(data_left, data_right);

        double p_1, p_2;

        if (phi_p_max < 0.) {

          /*
           * Algorithm 1, lines 8-9: two shocks, p_max < p^* < \tilde p^*.
           */

          p_1 = p_max;
          p_2 = p_tilde;

        } else if (phi_p_min <= 0.) {

          /*
           * Algorithm 1, lines 10-11: one shock and one rarefaction,
           * p_min <= p^* <= min(p_max, \tilde p^*). This branch also covers
           * the special case phi(p_max) == 0 of Algorithm 1, lines 5-6.
           */

          p_1 = p_min;
          p_2 = std::min(p_max, p_tilde);

        } else {

          /*
           * Two rarefactions, 0 < p^* < p_min.
           *
           * Algorithm 1, lines 2-3 short-circuit this case by setting
           * p^* = 0, which is legitimate when all one needs is the bound
           * (3.7)-(3.8) on lambda_max, but not here: we want the actual
           * solution. We therefore bracket with [0, p_min] instead, which
           * is a valid bracket because phi(0) < 0 (non-vacuum condition
           * above) and phi(p_min) > 0.
           */

          p_1 = 0.;
          p_2 = p_min;
        }

        /*
         * Safeguard: the bound p^* < \tilde p^* of Lemma 4.3 rests on
         * Theorem 4.1, which by @cite GuermondPopov2016b, Remark 4.2 is
         * false outside the physical range gamma \in (1, 5/3]. For
         * gamma > 5/3 the upper bracket may therefore fall short of p^*;
         * widen it until phi changes sign.
         */

        for (unsigned int i = 0; phi(p_2, data_left, data_right) < 0.; ++i) {
          p_2 *= 2.;
          AssertThrow(i < 200,
                      dealii::ExcMessage("Euler::ExactRiemannSolution: failed "
                                         "to bracket p_star."));
        }

        Assert(phi(p_1, data_left, data_right) <= 0. &&
                   phi(p_2, data_left, data_right) >= 0.,
               dealii::ExcMessage(
                   "Euler::ExactRiemannSolver: failed to compute p_star."));

        //
        // We simply compute the root of phi with a bisection method down
        // to machine precision. This is not terribly efficient but luckily
        // happens only once during initialization.
        //

#ifdef DEBUG_SOLUTION
        std::cout << "Computing p_star with a bisection method.\n"
                  << "p_tilde_star: " << p_tilde << "\n"
                  << "initial bracket: [" << p_1 << ", " << p_2 << "]"
                  << std::endl;
#endif

        unsigned int iter = 0;
        for (; iter < 200; ++iter) {

          // Check for convergence:
          if (std::abs(p_2 - p_1) < 10. * eps * std::max(p_1, p_2)) {
            break;
          }

#ifdef DEBUG_SOLUTION
          const double phi_1 = phi(p_1, data_left, data_right);
          const double phi_2 = phi(p_2, data_left, data_right);

          std::cout << "\niter: " << iter << "\n";
          std::cout << "p_1: " << p_1 << "\n";
          std::cout << "p_2: " << p_2 << "\n";
          std::cout << "phi_1: " << phi_1 << "\n";
          std::cout << "phi_2: " << phi_2 << "\n";
#endif

          /*
           * phi is strictly monotone increasing and we maintain the
           * invariant phi(p_1) <= 0 <= phi(p_2), so we can simply test the
           * sign of phi at the midpoint.
           */

          const auto p_m = 0.5 * (p_2 + p_1);

          if (phi(p_m, data_left, data_right) >= 0.) {
            p_2 = p_m;
          } else {
            p_1 = p_m;
          }
        }

        const double p_star = 0.5 * (p_1 + p_2);

#ifdef DEBUG_SOLUTION
        std::cout << "After " << iter << " iterations:"
                  << "\np_star =      " << p_star
                  << "\nphi(p_star) = " << phi(p_star, data_left, data_right)
                  << "\n|p_2 - p_1| = " << std::abs(p_2 - p_1) << std::endl;
#endif

        return p_star;
      }

      //@}
    };
  } // namespace EulerInitialStates
} // namespace ryujin
