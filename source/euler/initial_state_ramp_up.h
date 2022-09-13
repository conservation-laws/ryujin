//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

namespace ryujin
{
  namespace InitialStateLibrary
  {
    /**
     * A time-dependent state given by an initial state @p primite_left_
     * valid for \f$ t \le t_{\text{left}} \f$ and a final state @p
     * primite_right_ valid for \f$ t \ge t_{\text{right}} \f$. In between,
     * a smooth interpolation is performed.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename state_type>
    class RampUp : public InitialState<dim, Number, state_type>
    {
    public:
      RampUp(const HyperbolicSystem &hyperbolic_system,
             const std::string subsection)
          : InitialState<dim, Number, state_type>("ramp up", subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        primitive_initial_[0] = hyperbolic_system.gamma();
        primitive_initial_[1] = 0.0;
        primitive_initial_[2] = 1.;
        this->add_parameter("primitive state initial",
                            primitive_initial_,
                            "Initial 1d primitive state (rho, u, p)");

        primitive_final_[0] = hyperbolic_system.gamma();
        primitive_final_[1] = 3.0;
        primitive_final_[2] = 1.;
        this->add_parameter("primitive state final",
                            primitive_final_,
                            "Final 1d primitive state (rho, u, p)");

        t_initial_ = 0.;
        this->add_parameter("time initial",
                            t_initial_,
                            "Time until which initial state is prescribed");

        t_final_ = 1.;
        this->add_parameter("time final",
                            t_final_,
                            "Time from which on the final state is attained)");
      }

      virtual state_type compute(const dealii::Point<dim> & /*point*/,
                                 Number t) final override
      {
        dealii::Tensor<1, 3, Number> primitive;

        if (t <= t_initial_) {
          primitive = primitive_initial_;
        } else if (t >= t_final_) {
          primitive = primitive_final_;
        } else {
          const Number factor =
              std::cos(0.5 * M_PI * (t - t_initial_) / (t_final_ - t_initial_));

          const Number alpha = factor * factor;
          const Number beta = Number(1.) - alpha;
          primitive = alpha * primitive_initial_ + beta * primitive_final_;
        }

        const auto temp = hyperbolic_system.from_primitive_state(primitive);
        return hyperbolic_system.template expand_state<dim>(temp);
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      dealii::Tensor<1, 3, Number> primitive_initial_;
      dealii::Tensor<1, 3, Number> primitive_final_;

      Number t_initial_;
      Number t_final_;
    };
  } // namespace InitialStateLibrary
} // namespace ryujin
