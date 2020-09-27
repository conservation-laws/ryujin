//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef INITIAL_STATE_TEMPLATE_H
#define INITIAL_STATE_TEMPLATE_H

#include "initial_state.h"

namespace ryujin
{
  namespace InitialStates
  {
    /**
     * Uniform initial state defined by a given primitive state.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number>
    class Uniform : public InitialState<dim, Number>
    {
    public:
      using typename InitialState<dim, Number>::rank1_type;

      Uniform(const ProblemDescription &problem_description,
              const std::string subsection)
          : InitialState<dim, Number>(
                problem_description, "uniform", subsection)
      {
        primitive_[0] = this->problem_description.gamma();
        primitive_[1] = 3.0;
        primitive_[2] = 1.;
        this->add_parameter("primitive state",
                            primitive_,
                            "Initial 1d primitive state (rho, u, p)");
      }

      virtual rank1_type compute(const dealii::Point<dim> & /*point*/,
                                 Number /*t*/) final override
      {
        return this->problem_description.template from_primitive_state<dim>(
            primitive_);
      }

    private:
      dealii::Tensor<1, 3, Number> primitive_;
    };


    /**
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive state.
     *
     * @note This class does not evolve a possible shock front in time. If
     * you need correct time-dependent Dirichlet data use @ref ShockFront
     * instead.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number>
    class Contrast : public InitialState<dim, Number>
    {
    public:
      using typename InitialState<dim, Number>::rank1_type;

      Contrast(const ProblemDescription &problem_description,
               const std::string subsection)
          : InitialState<dim, Number>(
                problem_description, "contrast", subsection)
      {
        primitive_left_[0] = this->problem_description.gamma();
        primitive_left_[1] = 0.0;
        primitive_left_[2] = 1.;
        this->add_parameter(
            "primitive state left",
            primitive_left_,
            "Initial 1d primitive state (rho, u, p) on the left");

        primitive_right_[0] = this->problem_description.gamma();
        primitive_right_[1] = 0.0;
        primitive_right_[2] = 1.;
        this->add_parameter(
            "primitive state right",
            primitive_right_,
            "Initial 1d primitive state (rho, u, p) on the right");
      }

      virtual rank1_type compute(const dealii::Point<dim> &point,
                                 Number /*t*/) final override
      {
        return this->problem_description.template from_primitive_state<dim>(
            point[0] > 0. ? primitive_right_ : primitive_left_);
      }

    private:
      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;
    };


    /**
     * An S1/S3 shock front
     *
     * @todo Documentation
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number>
    class ShockFront : public InitialState<dim, Number>
    {
    public:
      using typename InitialState<dim, Number>::rank1_type;

      ShockFront(const ProblemDescription &problem_description,
                 const std::string subsection)
          : InitialState<dim, Number>(
                problem_description, "shockfront", subsection)
      {
        dealii::ParameterAcceptor::parse_parameters_call_back.connect(std::bind(
            &ShockFront<dim, Number>::parse_parameters_callback, this));

        primitive_right_[0] = this->problem_description.gamma();
        primitive_right_[1] = 0.0;
        primitive_right_[2] = 1.;
        this->add_parameter("primitive state",
                            primitive_right_,
                            "Initial 1d primitive state (rho, u, p) before the "
                            "shock (to the right)");

        mach_number_ = 2.0;
        this->add_parameter(
            "mach number",
            mach_number_,
            "Mach number of shock front (S1, S3 = mach * a_L/R)");
      }

      void parse_parameters_callback()
      {
        /* Compute post-shock state and S3: */

        const auto gamma = this->problem_description.gamma();
        const auto b = this->problem_description.b();

        const auto &rho_R = primitive_right_[0];
        const auto &u_R = primitive_right_[1];
        const auto &p_R = primitive_right_[2];
        /* a_R^2 = gamma * p / rho / (1 - b * rho) */
        const Number a_R = std::sqrt(gamma * p_R / rho_R / (1 - b * rho_R));
        const Number mach_R = u_R / a_R;

        S3_ = mach_number_ * a_R;
        const Number delta_mach = mach_R - mach_number_;

        const Number rho_L =
            rho_R * (gamma + Number(1.)) * delta_mach * delta_mach /
            ((gamma - Number(1.)) * delta_mach * delta_mach + Number(2.));
        const Number u_L =
            (Number(1.) - rho_R / rho_L) * S3_ + rho_R / rho_L * u_R;
        const Number p_L = p_R *
                           (Number(2.) * gamma * delta_mach * delta_mach -
                            (gamma - Number(1.))) /
                           (gamma + Number(1.));

        primitive_left_[0] = rho_L;
        primitive_left_[1] = u_L;
        primitive_left_[2] = p_L;
      }

      virtual rank1_type compute(const dealii::Point<dim> &point,
                                 Number t) final override
      {
        const Number position_1d = Number(point[0] - S3_ * t);
        return this->problem_description.template from_primitive_state<dim>(
            position_1d > 0. ? primitive_right_ : primitive_left_);
      }

    private:
      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;
      Number mach_number_;
      Number S3_;
    };


    /**
     * The isentropic vortex
     * @todo Documentation
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number>
    class IsentropicVortex : public InitialState<dim, Number>
    {
    public:
      using typename InitialState<dim, Number>::rank1_type;

      IsentropicVortex(const ProblemDescription &problem_description,
                       const std::string subsection)
          : InitialState<dim, Number>(
                problem_description, "isentropic vortex", subsection)
      {
        mach_number_ = 2.0;
        this->add_parameter(
            "mach number", mach_number_, "Mach number of isentropic vortex");

        beta_ = 5.0;
        this->add_parameter("beta", beta_, "vortex strength beta");
      }

      virtual rank1_type compute(const dealii::Point<dim> &point,
                                 Number t) final override
      {
        const auto gamma = this->problem_description.gamma();

        if constexpr (dim == 2) {
          auto point_bar = point;
          point_bar[0] -= mach_number_ * t;

          const Number r_square = Number(point_bar.norm_square());

          const Number factor = beta_ / Number(2. * M_PI) *
                                exp(Number(0.5) - Number(0.5) * r_square);

          const Number T = Number(1.) - (gamma - Number(1.)) /
                                            (Number(2.) * gamma) * factor *
                                            factor;

          const Number u = mach_number_ - factor * Number(point_bar[1]);
          const Number v = factor * Number(point_bar[0]);

          const Number rho = ryujin::pow(T, Number(1.) / (gamma - Number(1.)));
          const Number p = ryujin::pow(rho, gamma);
          const Number E =
              p / (gamma - Number(1.)) + Number(0.5) * rho * (u * u + v * v);

          return rank1_type({rho, rho * u, rho * v, E});

        } else {
          AssertThrow(false, dealii::ExcNotImplemented());
          return rank1_type();
        }
      }

    private:
      Number mach_number_;
      Number beta_;
    };


    /**
     * An analytic solution of the compressible Navier Stokes system
     * @todo Documentation
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number>
    class BeckerSolution : public InitialState<dim, Number>
    {
    public:
      using typename InitialState<dim, Number>::rank1_type;

      BeckerSolution(const ProblemDescription &problem_description,
                     const std::string subsection)
          : InitialState<dim, Number>(
                problem_description, "becker solution", subsection)
      {
        dealii::ParameterAcceptor::parse_parameters_call_back.connect(std::bind(
            &BeckerSolution<dim, Number>::parse_parameters_callback, this));

        velocity_ = 0.2;
        this->add_parameter("velocity galilean frame",
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
        const double gamma = this->problem_description.gamma();

        AssertThrow(
            velocity_left_ > velocity_right_,
            dealii::ExcMessage("The left limiting velocity must be greater "
                               "than the right limiting velocity"));
        AssertThrow(
            velocity_left_ > 0.,
            dealii::ExcMessage("The left limiting velocity must be positive"));

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
          const auto nu = 0.5 * std::tanh(10. * (x - 0.5 * (x_right + x_left)) /
                                          (x_right - x_left));
          double v = velocity_left_ * (0.5 - nu) + velocity_right_ * (nu + 0.5);

          auto [f, df] = psi(x, v);
          unsigned int iter = 0;

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
            iter++;
          }

          return v;
        };
      }

      virtual rank1_type compute(const dealii::Point<dim> &point,
                                 Number t) final override
      {
        /* (7.2) */
        const double gamma = this->problem_description.gamma();
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

        return rank1_type(
            {Number(rho),
             Number(rho * (velocity_ + v)),
             Number(0.),
             Number(rho * (e + 0.5 * (velocity_ + v) * (velocity_ + v)))});
      }

    private:
      Number velocity_;
      Number velocity_left_;
      Number velocity_right_;
      Number density_left_;
      Number mu_;
      std::function<std::tuple<double, double>(double, double)> psi;
      std::function<double(double)> find_velocity;
    };

  } /* namespace IinitialStates */
} /* namespace ryujin */

#endif /* INITIAL_STATE_TEMPLATE_H */
