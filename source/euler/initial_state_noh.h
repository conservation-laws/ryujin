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
     * The Noh problem
     *
     * This initial state sets up the classical Noh problem introduced in
     * @cite Noh1987
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class Noh : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      Noh(const HyperbolicSystem &hyperbolic_system,
          const std::string &subsection)
          : InitialState<Description, dim, Number>("noh", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        gamma_ = 1.4;
        if constexpr (!View::have_gamma) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        rho0_ = 1.0;
        this->add_parameter(
            "reference density", rho0_, "The reference density");

        /*
         * Exact solution assumes this value is negative, but we are just
         * switching u0 to -u0 by hand in the formulas.
         */
        u0_ = 1.0;
        this->add_parameter("reference velocity magnitude",
                            u0_,
                            "The reference velocity magnitude");


        p0_ = 1.e-12;
        this->add_parameter(
            "reference pressure", p0_, "The reference pressure");

        this->parse_parameters_call_back.connect([&]() {
          if constexpr (View::have_gamma) {
            const auto view = hyperbolic_system_.template view<dim, Number>();
            gamma_ = view.gamma();
          }
        });
      }

      /* Compute solution */
      auto compute(const dealii::Point<dim> &point, Number t)
          -> state_type final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        const auto norm = point.norm();
        const auto min = 10. * std::numeric_limits<Number>::min();

        /* Initialize primitive variables */
        Number rho = rho0_;
        auto vel = -u0_ * point / (norm + min);
        Number p = p0_;

        /* Define exact solutions */
        const auto D = u0_ * (gamma_ - 1.) / 2.;
        const bool in_interior = t == Number(0.) ? false : norm / t < D;

        if (in_interior) {
          rho = rho0_ * std::pow((gamma_ + 1.) / (gamma_ - 1.), dim);
          vel = 0. * point;
          p = 0.5 * rho0_ * u0_ * u0_;
          p *= std::pow(gamma_ + 1., dim) / std::pow(gamma_ - 1., dim - 1);
        } else {
          rho = rho0_ * std::pow(1. + t / (norm + min), dim - 1);
        }

        /* Set final state */
        if constexpr (dim == 1)
          return view.from_initial_state(state_type{{rho, vel[0], p}});
        else if constexpr (dim == 2)
          return view.from_initial_state(state_type{{rho, vel[0], vel[1], p}});
        else
          return view.from_initial_state(
              state_type{{rho, vel[0], vel[1], vel[2], p}});
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;
      Number gamma_;
      Number rho0_;
      Number u0_;
      Number p0_;
    };
  } // namespace EulerInitialStates
} // namespace ryujin
