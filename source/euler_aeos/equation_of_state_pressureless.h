//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "equation_of_state.h"

namespace ryujin
{
  namespace EquationOfStateLibrary
  {
    /**
     * Trivial equation of state for pressureless Euler
     *
     * @ingroup EulerEquations
     */
    class Pressureless : public EquationOfState
    {
    public:
      using EquationOfState::pressure;
      using EquationOfState::specific_internal_energy;
      using EquationOfState::speed_of_sound;
      using EquationOfState::temperature;

      Pressureless(const std::string &subsection)
          : EquationOfState("pressureless", subsection)
      {
        gamma_ = 7. / 5.;
        this->add_parameter("gamma", gamma_, "The ratio of specific heats");
      }

      /**
       * The pressure is given by
       * \f{align}
       *   p = 0.0
       *  \f}
       */
      double pressure(double rho [[maybe_unused]],
                      double e [[maybe_unused]]) const final
      {
        return 0.0;
      }

      /**
       * The specific internal energy is given by
       * \f{align}
       *   e = 0
       * \f}
       */
      double specific_internal_energy(double rho [[maybe_unused]],
                                      double p [[maybe_unused]]) const final
      {
        return 0.0;
      }

      /**
       * The temperature is given by
       * \f{align}
       *   T = 0
       * \f}
       */
      double temperature(double /*rho*/, double e [[maybe_unused]]) const final
      {
        return 0.0;
      }

      /**
       * The cold curve bound is given by
       * \f{align}
       *   e_cold = 0
       * \f}
       */
      double cold_curve_bound(double /*rho*/) const final
      {
        return 0.0;
      }

      /**
       * The specific entropy is given by
       * \f{align}
       *   s = 0
       * \f}
       */
      double specific_entropy(double /*rho*/, double /*e*/) const final
      {
        return 0.0;
      }

      /**
       * The speed of sound is given by
       * \f{align}
       *   c^2 = 0
       * \f}
       */
      double speed_of_sound(double /*rho*/,
                            double e [[maybe_unused]]) const final
      {
        return 0.0;
      }

    private:
      double gamma_;
    };
  } // namespace EquationOfStateLibrary
} // namespace ryujin
