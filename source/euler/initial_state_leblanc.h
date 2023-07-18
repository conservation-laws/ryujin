//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"
#include <initial_state.h>

namespace ryujin
{
  namespace Euler
  {
    struct Description;

    /**
     * The rarefaction
     * @todo Documentation
     *
     * @ingroup EulerEquations
     */

    template <int dim, typename Number>
    class LeBlanc : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystemView = HyperbolicSystem::View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      LeBlanc(const HyperbolicSystemView &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("leblanc", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
      } /* Constructor */

      state_type compute(const dealii::Point<dim> &point, Number t) final;

    private:
      const HyperbolicSystemView hyperbolic_system;
    };


    template <int dim, typename Number>
    auto LeBlanc<dim, Number>::compute(const dealii::Point<dim> &point,
                                       Number t) -> state_type
    {
      /*
       * The LeBlanc shock tube:
       */

      /* Initial left and right states (rho, u, p): */
      using state_type_1d = dealii::Tensor<1, 3, Number>;
      const state_type_1d primitive_left{{1.0, 0.0, 1.0 / 15.0}};
      const state_type_1d primitive_right{
          {0.001, 0.0, 2.0 / 3.0 * std::pow(10.0, -10.0)}};

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

      state_type_1d primitive;
      const double &x = point[0];

      if (x <= -1.0 / 3.0 * t) {
        /* Left state: */
        primitive = primitive_left;

      } else if (x < rarefaction_speed * t) {
        /* Expansion data (with self-similar variable chi): */
        const double chi = x / t;
        primitive[0] = std::pow(0.75 - 0.75 * chi, 3.0);
        primitive[1] = 0.75 * (1.0 / 3.0 + chi);
        primitive[2] = (1.0 / 15.0) * std::pow(0.75 - 0.75 * chi, 5.0);

      } else if (x < contact_velocity * t) {
        primitive[0] = pre_contact_density;
        primitive[1] = contact_velocity;
        primitive[2] = contact_pressure;

      } else if (x < right_shock_speed * t) {
        /* Contact-wave data (velocity and pressure are continuous): */
        primitive[0] = post_contact_density;
        primitive[1] = contact_velocity;
        primitive[2] = contact_pressure;

      } else {
        /* Right state: */
        primitive = primitive_right;
      }

      return hyperbolic_system.from_primitive_state(
          hyperbolic_system.expand_state(primitive));
    }

  } // namespace Euler
} // namespace ryujin
