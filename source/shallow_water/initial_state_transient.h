//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2024 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace ShallowWaterInitialStates
  {
    /**
     * Various experiment configurations described in @cite Martinez2018.
     *
     * @ingroup ShallowWaterEquations
     */
    template <typename Description, int dim, typename Number>
    class TankExperiments : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      TankExperiments(const HyperbolicSystem &hyperbolic_system,
                      const std::string subsection)
          : InitialState<Description, dim, Number>("transient experiments",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        dealii::ParameterAcceptor::parse_parameters_call_back.connect(
            std::bind(&TankExperiments::parse_parameters_callback, this));

        state_left_[0] = 1.;
        state_left_[1] = 0.0;
        this->add_parameter("flow state left",
                            state_left_,
                            "Initial 1d flow state (h, q) on the left");

        state_right_[0] = 1.;
        state_right_[1] = 0.0;
        this->add_parameter("flow state right",
                            state_right_,
                            "Initial 1d flow state (h, q) on the right");

        which_case_ = "G1";
        this->add_parameter("experimental configuration",
                            which_case_,
                            "Either 'G1', 'G2', 'G3' or 'none' for bathymetry "
                            "configuration");
      }

      void parse_parameters_callback()
      {
        AssertThrow(
            which_case_ == "G1" || which_case_ == "G2" || which_case_ == "G3" ||
                which_case_ == "none",
            dealii::ExcMessage("Case must be 'G1', 'G2', 'G3' or 'none'"));
      }

      state_type compute(const dealii::Point<dim> &point, Number /* t */) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        if constexpr (dim == 1) {
          AssertThrow(false, dealii::ExcNotImplemented());
          __builtin_trap();
        }

        const auto temp = point[0] > 1.e-8 ? state_right_ : state_left_;
        return view.expand_state(temp);
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<Description, dim, Number>::
              precomputed_state_type final
      {
        /* Compute bathymetry: */
        return {compute_bathymetry(point)};
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      DEAL_II_ALWAYS_INLINE
      inline Number compute_bathymetry(const dealii::Point<dim> &point) const
      {
        const auto &x = point[0], &y = point[1];

        /* Bathymetry base is the same for all configurations */
        Number bath = 0.;
        if (x >= 0. && x <= 326. / 100.)
          bath = -0.00092 * x;
        else if (x > 326. / 100.)
          bath = -0.0404 * (x - 326. / 100.) - 0.00092 * 326. / 100.;

        if (which_case_ == "none")
          return bath;

        /* Initialize obstacle to 0 */
        Number obstacle = 0.;


        // G1 -- rectangular obstacle
        if (which_case_ == "G1") {

          Number obstacle_length = 16.3 / 100.;
          Number obstacle_width = 8. / 100.;

          Number xc = 205. / 100. + (16.3 / 2. / 100.); // obstacle center

          if (std::abs((x - xc) / obstacle_length + y / obstacle_width) +
                  std::abs((x - xc) / obstacle_length - y / obstacle_width) <=
              1.)
            obstacle = 7. / 100.;
        } else if (which_case_ == "G2") { // circular bump + rectangle

          // circular bump
          double xc = 184.5 / 100. + 31. / 2. / 100.;
          const double radicand =
              positive_part(1. - std::pow((x - xc) / (31. / 2. / 100.), 2));

          const double semi_circle = 7.3 / 100. * std::sqrt(radicand);

          obstacle = std::max(semi_circle, 0.);

          // rectangular obstacle
          double obstacle_length = 16.3 / 100.;
          double obstacle_width = 8. / 100.;

          xc = 235. / 100. + (16.3 / 2. / 100.); // obstacle center

          if (std::abs((x - xc) / obstacle_length + y / obstacle_width) +
                  std::abs((x - xc) / obstacle_length - y / obstacle_width) <=
              1.)
            obstacle = 7. / 100.;
        } else if (which_case_ == "G3") { // narrowing half-circle + rectangle

          // narrowing half-circles canal
          double xc = 194 / 100. + 31. / 2. / 100.;
          const double radicand =
              positive_part(1. - std::pow((x - xc) / (31. / 2. / 100.), 2));
          const double semi_circle = 7.3 / 100. * std::sqrt(radicand);

          if (y < semi_circle - 24. / 2. / 100. &&
              std::abs(x - xc) <= 31. / 2. / 100.)
            obstacle = 21. / 100.;

          if (y > -semi_circle + 24. / 2. / 100. &&
              std::abs(x - xc) <= 31. / 2. / 100.)
            obstacle = 21. / 100.;

          // rectangular obstacle
          double obstacle_length = 16.3 / 100.;
          double obstacle_width = 8. / 100.;

          xc = 235. / 100. + (16.3 / 2. / 100.); // obstacle center

          if (std::abs((x - xc) / obstacle_length + y / obstacle_width) +
                  std::abs((x - xc) / obstacle_length - y / obstacle_width) <=
              1.)
            obstacle = 7. / 100.;
        }

        return bath + obstacle;
      }

      dealii::Tensor<1, 2, Number> state_left_;
      dealii::Tensor<1, 2, Number> state_right_;

      std::string which_case_;
      std::string flow_type_;
    };

  } // namespace ShallowWaterInitialStates
} // namespace ryujin
