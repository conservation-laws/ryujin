//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

namespace ryujin
{
  namespace Euler
  {
    namespace InitialStateLibrary
    {
      /**
       * An analytic solution of the compressible Navier Stokes system
       * @todo Documentation
       *
       * @ingroup InitialValues
       */
      template <int dim, typename Number, typename state_type>
      class BeckerSolution : public InitialState<dim, Number, state_type>
      {
      public:
        BeckerSolution(const HyperbolicSystem &hyperbolic_system,
                       const std::string subsection)
            : InitialState<dim, Number, state_type>("becker solution",
                                                    subsection)
            , hyperbolic_system(hyperbolic_system)
        {
          dealii::ParameterAcceptor::parse_parameters_call_back.connect(
              std::bind(&BeckerSolution<dim, Number, state_type>::
                            parse_parameters_callback,
                        this));

          velocity_ = 0.2;
          this->add_parameter(
              "velocity galilean frame",
              velocity_,
              "Velocity used to apply a Galilean transformation "
              "to the otherwise stationary solution");

          velocity_left_ = 1.0;
          this->add_parameter(
              "velocity left", velocity_left_, "Left limit velocity");

          velocity_right_ = 7. / 27.;
          this->add_parameter(
              "velocity right", velocity_right_, "Right limit velocity");

          density_left_ = 1.0;
          this->add_parameter(
              "density left", density_left_, "Left limit density");

          mu_ = 0.01;
          this->add_parameter("mu", mu_, "Shear viscosity");
        }

        void parse_parameters_callback()
        {
          const double gamma = hyperbolic_system.gamma();

          AssertThrow(
              velocity_left_ > velocity_right_,
              dealii::ExcMessage("The left limiting velocity must be greater "
                                 "than the right limiting velocity"));
          AssertThrow(velocity_left_ > 0.,
                      dealii::ExcMessage(
                          "The left limiting velocity must be positive"));

          const double velocity_origin =
              std::sqrt(velocity_left_ * velocity_right_);

          /* Prefactor as given in: (7.1) */

          const double Pr = 0.75;
          const double factor = 2. * gamma / (gamma + 1.) //
                                * mu_ / (density_left_ * velocity_left_ * Pr);

          psi = [=](double x, double v) {
            const double c_l =
                velocity_left_ / (velocity_left_ - velocity_right_);
            const double c_r =
                velocity_right_ / (velocity_left_ - velocity_right_);
            const double log_l = std::log(velocity_left_ - v) -
                                 std::log(velocity_left_ - velocity_origin);
            const double log_r = std::log(v - velocity_right_) -
                                 std::log(velocity_origin - velocity_right_);

            const double value = factor * (c_l * log_l - c_r * log_r) - x;

            const double derivative = factor * (-c_l / (velocity_left_ - v) -
                                                c_r / (v - velocity_right_));

            return std::make_tuple(value, derivative);
          };

          /* Determine cut-off points: */

          constexpr double tol = 1.e-12;

          const double x_left = std::get<0>(
              psi(0., (1. - tol) * velocity_left_ + tol * velocity_right_));

          const double x_right = std::get<0>(
              psi(0., tol * velocity_left_ + (1. - tol) * velocity_right_));

          const double norm = (x_right - x_left) * tol;

          /* Root finding algorithm: */

          find_velocity = [=](double x) {
            /* Return extremal cases: */
            if (x <= x_left)
              return double(velocity_left_);
            if (x >= x_right)
              return double(velocity_right_);

            /* Interpolate initial guess: */
            const auto nu =
                0.5 * std::tanh(10. * (x - 0.5 * (x_right + x_left)) /
                                (x_right - x_left));
            double v =
                velocity_left_ * (0.5 - nu) + velocity_right_ * (nu + 0.5);

            auto [f, df] = psi(x, v);

            while (std::abs(f) > norm) {
              const double v_next = v - f / df;

              /* Also break if we made no progress: */
              if (std::abs(v_next - v) <
                  tol * 0.5 * (velocity_right_ + velocity_left_)) {
                v = v_next;
                break;
              }

              if (v_next < velocity_right_)
                v = 0.5 * (velocity_right_ + v);
              else if (v_next > velocity_left_)
                v = 0.5 * (velocity_left_ + v);
              else
                v = v_next;

              const auto [new_f, new_df] = psi(x, v);
              f = new_f;
              df = new_df;
            }

            return v;
          };
        }

        virtual state_type compute(const dealii::Point<dim> &point,
                                   Number t) final override
        {
          /* (7.2) */
          const double gamma = hyperbolic_system.gamma();
          const double R_infty = (gamma + 1) / (gamma - 1);

          /* (7.3) */
          const double x = point[0] - velocity_ * t;
          const double v = find_velocity(x);
          Assert(v >= velocity_right_, dealii::ExcInternalError());
          Assert(v <= velocity_left_, dealii::ExcInternalError());
          const double rho = density_left_ * velocity_left_ / v;
          Assert(rho > 0., dealii::ExcInternalError());
          const double e = 1. / (2. * gamma) *
                           (R_infty * velocity_left_ * velocity_right_ - v * v);
          Assert(e > 0., dealii::ExcInternalError());

          return hyperbolic_system.template expand_state<dim>(
              HyperbolicSystem::state_type<1, Number>{
                  {Number(rho),
                   Number(rho * (velocity_ + v)),
                   Number(rho *
                          (e + 0.5 * (velocity_ + v) * (velocity_ + v)))}});
        }

      private:
        const HyperbolicSystem &hyperbolic_system;

        Number velocity_;
        Number velocity_left_;
        Number velocity_right_;
        Number density_left_;
        Number mu_;
        std::function<std::tuple<double, double>(double, double)> psi;
        std::function<double(double)> find_velocity;
      };

    } // namespace InitialStateLibrary
  } // namespace Euler
} // namespace ryujin
