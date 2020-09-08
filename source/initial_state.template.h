//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef INITIAL_STATE_TEMPLATE_H
#define INITIAL_STATE_TEMPLATE_H

#include "initial_state.h"

namespace
{
  using namespace ryujin;

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline auto
  from_primitive_state(const dealii::Tensor<1, 3, Number> &state_1d)
  {
    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;
    constexpr auto gamma = ProblemDescription<1, Number>::gamma;

    const auto &rho = state_1d[0];
    const auto &u = state_1d[1];
    const auto &p = state_1d[2];

    rank1_type state;

    state[0] = rho;
    state[1] = rho * u;
    state[dim + 1] = p / (gamma - Number(1.)) + Number(0.5) * rho * u * u;

    return state;
  };
}

namespace ryujin
{
  namespace InitialStates
  {
    /**
     * Uniform initial state defined by a given primitive state.
     *
     * @relates InitialState
     */
    template <int dim, typename Number>
    class Uniform : public InitialState<dim, Number>
    {
    public:
      using typename InitialState<dim, Number>::rank1_type;

      Uniform(const std::string subsection)
          : InitialState<dim, Number>("uniform", subsection)
      {
        constexpr auto gamma = ProblemDescription<1, Number>::gamma;
        primitive_[0] = gamma;
        primitive_[1] = 3.0;
        primitive_[2] = 1.;
        this->add_parameter("primitive state",
                            primitive_,
                            "Initial 1d primitive state (rho, u, p)");
      }

      virtual rank1_type compute(const dealii::Point<dim> & /*point*/,
                                 Number /*t*/) final override
      {
        return from_primitive_state<dim>(primitive_);
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
     * @relates InitialState
     */
    template <int dim, typename Number>
    class Contrast : public InitialState<dim, Number>
    {
    public:
      using typename InitialState<dim, Number>::rank1_type;

      Contrast(const std::string subsection)
          : InitialState<dim, Number>("contrast", subsection)
      {
        constexpr auto gamma = ProblemDescription<1, Number>::gamma;
        primitive_left_[0] = gamma;
        primitive_left_[1] = 0.0;
        primitive_left_[2] = 1.;
        this->add_parameter(
            "primitive state left",
            primitive_left_,
            "Initial 1d primitive state (rho, u, p) on the left");

        primitive_right_[0] = gamma;
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
        if (point[0] > 0.)
          return from_primitive_state<dim>(primitive_right_);
        else
          return from_primitive_state<dim>(primitive_left_);
      }

    private:
      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;
    };


    /**
     * An S1/S3 shock front
     *
     * @fixme Documentation
     *
     * @relates InitialState
     */
    template <int dim, typename Number>
    class ShockFront : public InitialState<dim, Number>
    {
    public:
      using typename InitialState<dim, Number>::rank1_type;

      ShockFront(const std::string subsection)
          : InitialState<dim, Number>("shockfront", subsection)
      {
        dealii::ParameterAcceptor::parse_parameters_call_back.connect(std::bind(
            &ShockFront<dim, Number>::parse_parameters_callback, this));

        constexpr auto gamma = ProblemDescription<1, Number>::gamma;
        primitive_right_[0] = gamma;
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

        constexpr auto gamma = ProblemDescription<1, Number>::gamma;
        constexpr auto b = ProblemDescription<1, Number>::b;

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
        if (position_1d > 0.)
          return from_primitive_state<dim>(primitive_right_);
        else
          return from_primitive_state<dim>(primitive_left_);
      }

    private:
      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;
      Number mach_number_;
      Number S3_;
    };


    /**
     * The isentropic vortex
     * @fixme Documentation
     *
     * @relates InitialState
     */
    template <int dim, typename Number>
    class IsentropicVortex : public InitialState<dim, Number>
    {
    public:
      using typename InitialState<dim, Number>::rank1_type;

      IsentropicVortex(const std::string subsection)
          : InitialState<dim, Number>("isentropic vortex", subsection)
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
        constexpr auto gamma = ProblemDescription<1, Number>::gamma;

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

  } /* namespace IinitialStates */
} /* namespace ryujin */

#endif /* INITIAL_STATE_TEMPLATE_H */
