//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace Euler
  {
    struct Description;
  }

  namespace EulerInitialStates
  {
    /**
     * The rarefaction
     * @todo Documentation
     *
     * @ingroup EulerEquations
     */

    template <typename Description, int dim, typename Number>
    class Rarefaction : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      Rarefaction(const HyperbolicSystem &hyperbolic_system,
                  const std::string subsection)
          : InitialState<Description, dim, Number>("rarefaction", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        gamma_ = 1.4;
        if constexpr (!std::is_same_v<Description, Euler::Description>) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        this->parse_parameters_call_back.connect([&]() {
          if constexpr (std::is_same_v<Description, Euler::Description>) {
            gamma_ = hyperbolic_system_.gamma();
          }
        });
      } /* Constructor */

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        using state_type_1d = std::array<Number, 3>;

        /*
         * Compute the speed of sound:
         */
        const auto speed_of_sound = [&](const Number rho, const Number p) {
          return std::sqrt(gamma_ * p / rho);
        };

        /*
         * Compute the rarefaction right side:
         */
        const auto rarefaction_right_state = [&](const auto primitive_left,
                                                 const Number rho_right) {
          const auto &[rho_left, u_left, p_left] = primitive_left;
          state_type_1d primitive_right{{rho_right, 0., 0.}};

          /* Isentropic condition: pR = (rhoR/rhoL)^{gamma} * pL */
          primitive_right[2] = std::pow(rho_right / rho_left, gamma_) * p_left;

          const auto c_left = speed_of_sound(rho_left, p_left);
          const auto c_right = speed_of_sound(rho_right, primitive_right[2]);

          /* 1-Riemann invariant: uR + 2 cR/(gamma -1) = uL + 2 cL/(gamma -1) */
          primitive_right[1] =
              u_left + 2.0 * (c_left - c_right) / (gamma_ - 1.0);

          return primitive_right;
        };

        /* Initial left and right states (rho, u, p): */
        const state_type_1d primitive_left{
            {3.0, speed_of_sound(3.0, 1.0), 1.0}};
        const state_type_1d primitive_right =
            rarefaction_right_state(primitive_left, /*rho_right*/ 0.5);

        /*
         * Compute rarefaction solution:
         */

        const auto &[rho_left, u_left, p_left] = primitive_left;
        const auto c_left = speed_of_sound(rho_left, p_left);
        const auto &[rho_right, u_right, p_right] = primitive_right;
        const auto c_right = speed_of_sound(rho_right, p_right);

        /* Constants: */
        const Number k1 = 2.0 / (gamma_ + 1.0);
        const Number k2 = ((gamma_ - 1.0) / ((gamma_ + 1.0) * c_left));
        const Number density_exponent = 2.0 / (gamma_ - 1.0);
        const Number k3 = c_left + ((gamma_ - 1.0) / 2.0) * u_left;
        const Number pressure_exponent = 2.0 * gamma_ / (gamma_ - 1.0);

        const double &x = point[0];

        state_type_1d primitive;

        if (x <= t * (u_left - c_left)) {
          primitive = primitive_left;

        } else if (x <= t * (u_right - c_right)) {

          /* Self-similar variable: */
          const double chi = x / t;

          primitive[0] =
              rho_left * std::pow(k1 + k2 * (u_left - chi), density_exponent);
          primitive[1] = k1 * (k3 + chi);
          primitive[2] =
              p_left * std::pow(k1 + k2 * (u_left - chi), pressure_exponent);

        } else {
          primitive = primitive_right;
        }

        const dealii::Tensor<1, 3, Number> result{
            {primitive[0], primitive[1], primitive[2]}};

        // FIXME: update primitive

        return hyperbolic_system_.from_primitive_state(
            hyperbolic_system_.expand_state(result));
      }

    private:
      const HyperbolicSystemView hyperbolic_system_;
      Number gamma_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
