//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <deal.II/base/tensor.h>
#include <initial_state.h>

namespace ryujin
{
  namespace InitialStateLibrary
  {
    /**
     * Ritter's dam break solution. This is one-dimensional dam break without
     * friction. See: Sec.~7.3 in Guermond et al 2018.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename state_type>
    class RitterDamBreak : public InitialState<dim, Number, state_type, 1>
    {
    public:
      RitterDamBreak(const HyperbolicSystem &hyperbolic_system,
                     const std::string subsection)
          : InitialState<dim, Number, state_type, 1>("ritter_dam_break",
                                                     subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        t_initial_ = 0.;
        this->add_parameter("time initial",
                            t_initial_,
                            "Time at which initial state is prescribed");
        dam_location = 5.;
        this->add_parameter(
            "dam location",
            dam_location,
            "Location of pseudo-dam that separates two water depths");
        left_depth = 0.005;
        this->add_parameter("left water depth",
                            left_depth,
                            "Depth of water to the left of pseudo-dam");
        right_depth = 0.;
        this->add_parameter("right water depth",
                            right_depth,
                            "Depth of water to the right of pseudo-dam");
      }

      virtual state_type compute(const dealii::Point<dim> &point,
                                 Number t) final override
      {
        if constexpr (dim == 1) {
          AssertThrow(false, dealii::ExcNotImplemented());
          return state_type();
        }

        const auto g = hyperbolic_system.gravity();
        const Number x = point[0];

        // Explicitly define initial left state
        left_state[0] = left_depth;
        left_state[1] = Number(0.);
        left_state[2] = Number(0.);

        // Explicitly define initial right state
        right_state[0] = right_depth;
        right_state[1] = Number(0.);
        right_state[2] = Number(0.);

        // Return initial state if t_initial_ = 0
        if (t_initial_ <= 1.e-10) {
          final_state = (x <= dam_location ? left_state : right_state);
          return final_state;
        }

        AssertThrow(t + t_initial_ > 0.,
                    dealii::ExcMessage("Expansion must be computed at a time "
                                       "greater than 0."));

        // Else we compute the expansion profiles at t + t_initial
        const Number aL = std::sqrt(g * left_depth);
        const Number xA = dam_location - (t + t_initial_) * aL;
        const Number xB = dam_location + Number(2.) * (t + t_initial_) * aL;

        const Number tmp = aL - (x - dam_location) / (2. * (t + t_initial_));

        const Number h_expansion = 4. / (9. * g) * tmp * tmp;
        const Number v_expansion =
            2. / 3. * ((x - dam_location) / (t + t_initial_) + aL);

        if (x <= xA) {
          final_state = left_state;
        } else if (x <= xB) {
          final_state[0] = h_expansion;
          final_state[1] = h_expansion * v_expansion;
          final_state[2] = Number(0.);
        } else {
          final_state = right_state;
        }

        return final_state;
      }

      /* Default bathymetry of 0 */

    private:
      const HyperbolicSystem &hyperbolic_system;

      Number t_initial_;

      Number dam_location;
      Number left_depth;
      Number right_depth;

      state_type final_state;
      state_type left_state;
      state_type right_state;
    };

  } // namespace InitialStateLibrary
} // namespace ryujin
