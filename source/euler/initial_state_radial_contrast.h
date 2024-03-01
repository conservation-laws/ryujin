//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * A modification of the "contrast" initial state. Now, we have an
     * initial state formed by a contrast of a given "left" and "right"
     * primitive state where the "left" state is "inside" the radius, R,
     * and the "right" state is outside.
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @todo Transform a nonzero initial velocity into a radial momentum
     * scaling with 1 / r^(dim -1).
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class RadialContrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      RadialContrast(const HyperbolicSystem &hyperbolic_system,
                     const std::string &subsection)
          : InitialState<Description, dim, Number>("radial contrast",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        primitive_inner_[0] = 1.4;
        primitive_inner_[1] = 0.0;
        primitive_inner_[2] = 1.;
        this->add_parameter(
            "primitive state inner",
            primitive_inner_,
            "Initial 1d primitive state (rho, u, p) on the inner disk");

        primitive_outer_[0] = 1.4;
        primitive_outer_[1] = 0.0;
        primitive_outer_[2] = 1.;
        this->add_parameter(
            "primitive state outer",
            primitive_outer_,
            "Initial 1d primitive state (rho, u, p) on the outer annulus");

        radius_ = 0.5;
        this->add_parameter("radius", radius_, "Radius of radial area");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();
          state_inner_ = view.from_initial_state(primitive_inner_);
          state_outer_ = view.from_initial_state(primitive_outer_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }

      auto compute(const dealii::Point<dim> &point, Number /*t*/)
          -> state_type final
      {
        return (point.norm() > radius_ ? state_outer_ : state_inner_);
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      dealii::Tensor<1, 3, Number> primitive_inner_;
      dealii::Tensor<1, 3, Number> primitive_outer_;
      double radius_;

      state_type state_inner_;
      state_type state_outer_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
