#ifndef PROBLEM_DESCRIPTION_TEMPLATE_H
#define PROBLEM_DESCRIPTION_TEMPLATE_H

#include "problem_description.h"
#include <iostream>

namespace grendel
{
  using namespace dealii;

  template <int dim>
  ProblemDescription<dim>::ProblemDescription(const std::string &subsection)
      : ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&ProblemDescription<dim>::parse_parameters_callback, this));

    gamma_ = 7. / 5.;
    add_parameter("gamma", gamma_, "Gamma");

    b_ = 0.0;
    add_parameter("b", b_, "b aka bcovol");

    cfl_update_ = 1.00;
    add_parameter("cfl update", cfl_update_, "CFL constant used for update");

    cfl_max_ = 1.00;
    add_parameter("cfl max", cfl_max_, "Maximal admissible CFL constant");


    initial_state_ = "shock front";
    add_parameter("initial state",
                  initial_state_,
                  "Initial state. Valid names are \"shock front\", "
                  "\"sod contrast\", \"uniform\", \"smooth solution\", or \"vortex\".");

    initial_direction_[0] = 1.;
    add_parameter("initial - direction",
                  initial_direction_,
                  "Initial direction of shock front, sod contrast, or vortex");

    initial_position_[0] = 1.;
    add_parameter("initial - position",
                  initial_position_,
                  "Initial position of shock front, sod contrast, or vortex");

    initial_mach_number_ = 2.0;
    add_parameter(
        "initial - mach number",
        initial_mach_number_,
        "Initial mach number of shock front, uniform state, or vortex");

    initial_vortex_beta_ = 5.0;
    add_parameter(
        "initial - vortex beta",
        initial_vortex_beta_,
        "Initial vortex strength beta");
  }


  template <int dim>
  void ProblemDescription<dim>::parse_parameters_callback()
  {
    /*
     * First, let's normalize the direction:
     */

    AssertThrow(
        initial_direction_.norm() != 0.,
        ExcMessage("Initial shock front direction is set to the zero vector."));
    initial_direction_ /= initial_direction_.norm();

    /*
     * Create a small lambda that translates a 1D state (rho, u, p) into an
     * nD state:
     */

    const auto from_1d_state =
        [=](const std::array<double, 3> &state_1d) -> rank1_type {
      const auto &[rho, u, p] = state_1d;

      rank1_type state;

      state[0] = rho;
      for (unsigned int i = 0; i < dim; ++i)
        state[1 + i] = rho * u * initial_direction_[i];
      state[dim + 1] = p / (gamma_ - 1.) + 0.5 * rho * u * u;

      return state;
    };


    /*
     * Now populate the initial_state_internal function object:
     */

    if (initial_state_ == "shock front") {

      /*
       * A mach shock front:
       *
       * FIXME: Add reference to literature
       */

      const double rho_R = gamma_;
      const double u_R = 0.;
      const double p_R = 1.;

      /*   c^2 = gamma * p / rho / (1 - b * rho) */
      const double a_R = std::sqrt(gamma_ * p_R / rho_R / (1. - b_ * rho_R));
      const double mach = initial_mach_number_;
      const double S3 = mach * a_R;

      const double rho_L = rho_R * (gamma_ + 1.) * mach * mach /
                           ((gamma_ - 1.) * mach * mach + 2.);
      double u_L = (1. - rho_R / rho_L) * S3 + rho_R / rho_L * u_R;
      double p_L =
          p_R * (2. * gamma_ * mach * mach - (gamma_ - 1.)) / (gamma_ + 1.);

      initial_state_internal = [=](const dealii::Point<dim> &point, double t) {
        AssertThrow(t == 0.,
                    ExcMessage("No analytic solution for t > 0. available"));

        const double position_1d =
            (point - initial_position_) * initial_direction_;

        if (position_1d > 0.) {
          return from_1d_state({rho_R, u_R, p_R});
        } else {
          return from_1d_state({rho_L, u_L, p_L});
        }
      };

    } else if (initial_state_ == "uniform") {

      /*
       * A uniform flow:
       */

      initial_state_internal = [=](const dealii::Point<dim> & /*point*/,
                                   double t) {
        AssertThrow(t == 0.,
                    ExcMessage("No analytic solution for t > 0. available"));

        return from_1d_state({gamma_, initial_mach_number_, 1.});
      };

    } else if (initial_state_ == "sod contrast") {

      /*
       * Contrast of the Sod shock tube:
       */

      initial_state_internal = [=](const dealii::Point<dim> &point, double t) {
        AssertThrow(t == 0.,
                    ExcMessage("No analytic solution for t > 0. available"));

        const double position_1d =
            (point - initial_position_) * initial_direction_;

        if (position_1d > 0.) {
          return from_1d_state({0.125, 0.0, 0.1});
        } else {
          return from_1d_state({1.0, 0.0, 1.0});
        }
      };

    } else if (initial_state_ == "smooth") {

      /*
       * Smooth 1D solution:
       */

      initial_state_internal = [=](const dealii::Point<dim> &point, double t) {
        const double x = point[0];
        const double x_0 = -0.1;
        const double x_1 = 0.1;

        double rho = 1.;
        if (x - t > x_0 && x - t < x_1)
          rho = 1. + 64. * std::pow(x_1 - x_0, -6.) * std::pow(x - t - x_0, 3) *
                         std::pow(x_1 - x + t, 3);

        return from_1d_state({rho, 1.0, 1.0});
      };

    } else if (initial_state_ == "vortex") {

      /*
       * 2D isentropic vortex problem. See section 5.6 of Euler-convex
       * limiting paper by Guermond et al.
       */

      if constexpr(dim == 2) {
        initial_state_internal = [=](const dealii::Point<dim> &point,
                                     double t) {
          const auto point_bar = point - initial_position_ -
                                 initial_direction_ * initial_mach_number_ * t;
          const double r_square = point_bar.norm_square();

          const double beta = initial_vortex_beta_;

          const double T =
              1. - (gamma_ - 1.) * beta * beta /
                       (8. * gamma_ * M_PI * M_PI * exp(1. - r_square));

          const double u =
              initial_direction_[0] * initial_mach_number_ -
              beta / (2. * M_PI) * exp((1. - r_square) / 2.) * point_bar[1];

          const double v =
              initial_direction_[1] * initial_mach_number_ +
              beta / (2. * M_PI) * exp((1. - r_square) / 2.) * point_bar[0];

          const double rho = std::pow(T, 1. / (gamma_ - 1.));
          const double p = std::pow(rho, gamma_);
          const double E = p / (gamma_ - 1.) + 0.5 * rho * u * u;

          return rank1_type({rho, u, v, E});
        };

      } else {

        AssertThrow(false, dealii::ExcNotImplemented());
      }

    } else {

      AssertThrow(false, dealii::ExcMessage("Unknown initial state."));
    }
  }

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_TEMPLATE_H */
