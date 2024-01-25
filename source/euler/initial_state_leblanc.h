//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <cmath>
#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * The Le Blanc shocktube.
     *
     * An Analytic solution for the compressible Euler equations with
     * polytropic gas equation of state and \f$\gamma = 5./3\f$.
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @ingroup EulerEquations
     */

    template <typename Description, int dim, typename Number>
    class LeBlanc : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      using ScalarNumber = typename HyperbolicSystemView::ScalarNumber;

      LeBlanc(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("leblanc", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
      } /* Constructor */

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        /*
         * The Le Blanc shock tube:
         */

        /* Initial left and right states (rho, u, p): */
        using state_type_1d = std::array<Number, 3>;
        constexpr state_type_1d primitive_left{1., 0., Number(2. / 3. * 1.e-1)};
        constexpr state_type_1d primitive_right{
            1.e-3, 0., Number(2. / 3. * 1.e-10)};

        /* The intermediate wave-speeds appearing on the Riemann fan: */
        constexpr Number rarefaction_speed = 0.49578489518897934;
        constexpr Number contact_velocity = 0.62183867139173454;
        constexpr Number right_shock_speed = 0.82911836253346982;

        /*
         * Velocity and pressure are constant across the middle discontinuity,
         * only the density jumps: it's a contact wave!
         */
        constexpr Number pre_contact_density = 5.4079335349316249e-02;
        constexpr Number post_contact_density = 3.9999980604299963e-03;
        constexpr Number contact_pressure = 0.51557792765096996e-03;

        state_type_1d primitive_state;
        const double &x = point[0];

        if (x <= -1.0 / 3.0 * t) {
          /* Left state: */
          primitive_state = primitive_left;

        } else if (x < rarefaction_speed * t) {
          /* Expansion data (with self-similar variable chi): */
          const double chi = x / t;
          primitive_state[0] = std::pow(0.75 - 0.75 * chi, 3.0);
          primitive_state[1] = 0.75 * (1.0 / 3.0 + chi);
          primitive_state[2] = (1.0 / 15.0) * std::pow(0.75 - 0.75 * chi, 5.0);

        } else if (x < contact_velocity * t) {
          primitive_state[0] = pre_contact_density;
          primitive_state[1] = contact_velocity;
          primitive_state[2] = contact_pressure;

        } else if (x < right_shock_speed * t) {
          /* Contact-wave data (velocity and pressure are continuous): */
          primitive_state[0] = post_contact_density;
          primitive_state[1] = contact_velocity;
          primitive_state[2] = contact_pressure;

        } else {
          /* Right state: */
          primitive_state = primitive_right;
        }

        state_type conserved_state;
        {
          const auto &[rho, u, p] = primitive_state;
          conserved_state[0] = rho;
          conserved_state[1] = rho * u;
          conserved_state[dim + 1] =
              p / ScalarNumber(5. / 3. - 1.) + ScalarNumber(0.5) * rho * u * u;
        }

        return conserved_state;
      }

    private:
      const HyperbolicSystemView hyperbolic_system_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
