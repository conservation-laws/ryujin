//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * An Analytic solution for the compressible Euler equations with
     * polytropic gas equation of state consisting of a smooth rarefaction
     * wave. This analytic solution is discussed in detail in Section 5.3
     * of @cite GuermondEtAl2018
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @ingroup EulerEquations
     */

    template <typename Description, int dim, typename Number>
    class Rarefaction : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      using state_type_1d = std::array<Number, 4>;

      Rarefaction(const HyperbolicSystem &hyperbolic_system,
                  const std::string subsection)
          : InitialState<Description, dim, Number>("rarefaction", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        gamma_ = 1.4;
        if constexpr (!View::have_gamma) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        /*
         * Compute the speed of sound:
         */
        const auto speed_of_sound = [&](const Number rho, const Number p) {
          return std::sqrt(gamma_ * p / rho);
        };

        /*
         * Compute the rarefaction right side:
         */
        const auto rarefaction_right_state = [this, speed_of_sound](
                                                 const auto primitive_left,
                                                 const Number rho_right) {
          const auto &[rho_left, u_left, p_left, c_left] = primitive_left;
          state_type_1d primitive_right{{rho_right, 0., 0.}};

          /* Isentropic condition: pR = (rhoR/rhoL)^{gamma} * pL */
          primitive_right[2] = std::pow(rho_right / rho_left, gamma_) * p_left;

          const auto c_right = speed_of_sound(rho_right, primitive_right[2]);
          primitive_right[3] = c_right;

          /* 1-Riemann invariant: uR + 2 cR/(gamma -1) = uL + 2 cL/(gamma -1) */
          primitive_right[1] =
              u_left + 2.0 * (c_left - c_right) / (gamma_ - 1.0);

          return primitive_right;
        };

        const auto compute_constants =
            [this, speed_of_sound, rarefaction_right_state]() {
              const auto view = hyperbolic_system_.template view<dim, Number>();
              if constexpr (View::have_gamma) {
                gamma_ = view.gamma();
              }

              /*
               * Initial left and right states (rho, u, p, c):
               */

              const Number rho_left = 3.0;
              const Number p_left = 1.0;
              const Number c_left = speed_of_sound(rho_left, p_left);
              const Number u_left = c_left; /* verify */
              const Number rho_right = 0.5;

              primitive_left_ = {rho_left, c_left, p_left, c_left};
              primitive_right_ =
                  rarefaction_right_state(primitive_left_, rho_right);

              /*
               * Populate constants:
               */

              k1 = 2.0 / (gamma_ + 1.0);
              k2 = ((gamma_ - 1.0) / ((gamma_ + 1.0) * c_left));
              density_exponent = 2.0 / (gamma_ - 1.0);
              k3 = c_left + ((gamma_ - 1.0) / 2.0) * u_left;
              pressure_exponent = 2.0 * gamma_ / (gamma_ - 1.0);
            };

        this->parse_parameters_call_back.connect(compute_constants);
        compute_constants();
      } /* Constructor */

      state_type compute(const dealii::Point<dim> &point, Number delta_t) final
      {
        /*
         * Compute rarefaction solution:
         */

        const auto &[rho_left, u_left, p_left, c_left] = primitive_left_;
        const auto &[rho_right, u_right, p_right, c_right] = primitive_right_;

        const double x = point[0];
        const auto t_0 = 0.2 / (u_right - u_left);
        const auto t = t_0 + delta_t;

        state_type_1d primitive;

        if (x <= t * (u_left - c_left)) {
          primitive = primitive_left_;

        } else if (x <= t * (u_right - c_right)) {

          /* Self-similar variable: */
          const double chi = x / t;

          primitive[0] =
              rho_left * std::pow(k1 + k2 * (u_left - chi), density_exponent);
          primitive[1] = k1 * (k3 + chi);
          primitive[2] =
              p_left * std::pow(k1 + k2 * (u_left - chi), pressure_exponent);

        } else {
          primitive = primitive_right_;
        }

        state_type conserved_state;
        {
          const auto &[rho, u, p, c] = primitive;
          conserved_state[0] = rho;
          conserved_state[1] = rho * u;
          conserved_state[dim + 1] =
              p / Number(gamma_ - 1.) + Number(0.5) * rho * u * u;
        }
        return conserved_state;
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;
      Number gamma_;

      state_type_1d primitive_left_;
      state_type_1d primitive_right_;
      Number k1;
      Number k2;
      Number density_exponent;
      Number k3;
      Number pressure_exponent;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
