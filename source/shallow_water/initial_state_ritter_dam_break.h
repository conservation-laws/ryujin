//
// SPDX-License-Identifier: MIT or BSD-3-Clause
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2023 by the ryujin authors
// Copyright (C) 2023 - 2023 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * Ritter's dam break solution. This is one-dimensional dam break without
     * friction. See Section 7.3 in @cite GuermondEtAl2018SW for details.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class RitterDamBreak : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename HyperbolicSystemView::state_type;

      RitterDamBreak(const HyperbolicSystem &hyperbolic_system,
                     const std::string subsection)
          : InitialState<Description, dim, Number>("ritter dam break",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        dealii::ParameterAcceptor::parse_parameters_call_back.connect(
            std::bind(&RitterDamBreak::parse_parameters_callback, this));

        t_initial_ = 0.1;
        this->add_parameter("time initial",
                            t_initial_,
                            "Time at which initial state is prescribed");

        left_depth = 0.005;
        this->add_parameter("left water depth",
                            left_depth,
                            "Depth of water to the left of pseudo-dam (x<0)");
      }

      void parse_parameters_callback()
      {
        AssertThrow(t_initial_ > 0.,
                    dealii::ExcMessage("Expansion must be computed at an "
                                       "initial time greater than 0."));
      }

      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto g = hyperbolic_system_.gravity();

        const auto x = point[0];

        const Number aL = std::sqrt(g * left_depth);
        const Number xA = -(t + t_initial_) * aL;
        const Number xB = Number(2.) * (t + t_initial_) * aL;

        const Number tmp = aL - x / (2. * (t + t_initial_));

        const Number h_expansion = 4. / (9. * g) * tmp * tmp;
        const Number v_expansion = 2. / 3. * (x / (t + t_initial_) + aL);

        if (x <= xA)
          return state_type{{left_depth, Number(0.)}};
        else if (x <= xB)
          return state_type{{h_expansion, h_expansion * v_expansion}};
        else
          return state_type{{Number(0.), Number(0.)}};
      }

      /* Default bathymetry of 0 */

    private:
      const HyperbolicSystemView hyperbolic_system_;

      Number t_initial_;
      Number left_depth;
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
