#ifndef PROBLEM_DESCRIPTION_TEMPLATE_H
#define PROBLEM_DESCRIPTION_TEMPLATE_H

#include "problem_description.h"

namespace grendel
{
  using namespace dealii;

  template <int dim>
  ProblemDescription<dim>::ProblemDescription(const std::string &subsection)
      : ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&ProblemDescription<dim>::parse_parameters_callback, this));

    gamma_ = 7./5.;
    add_parameter("gamma", gamma_, "Gamma");

    b_ = 0.0;
    add_parameter("b", b_, "b aka bcovol");

    cfl_constant_ = 1.00;
    add_parameter("cfl constant", cfl_constant_, "CFL constant C");


    initial_state_ = "shock front";
    add_parameter("initial state",
                  initial_state_,
                  "Initial state. Valid names are \"shock front\", "
                  "\"sod contrast\", or \"smooth solution\".");

    initial_direction_[0] = 1.;
    add_parameter("initial - direction",
                  initial_direction_,
                  "Initial direction of shock, or contrast front");

    initial_position_[0] = 1.;
    add_parameter("initial - position",
                  initial_position_,
                  "Initial position of shock, or contrast front");

    initial_shock_front_mach_number_ = 2.0;
    add_parameter("shock front - mach number",
                  initial_shock_front_mach_number_,
                  "Shock Front: Mach number");
  }


  template <int dim>
  void ProblemDescription<dim>::parse_parameters_callback()
  {
    /*
     * First, let's normalize the direction:
     */

    AssertThrow(initial_direction_.norm() != 0.,
                ExcMessage("no direction, initial shock front direction is set "
                           "to the zero vector."));
    initial_direction_ /= initial_direction_.norm();

    /*
     * Now populate the "left" and "right" state functions:
     */

    if (initial_state_ == "shock front") {

      // FIXME: Add reference to literature

      const double rho_R = gamma_;
      const double u_R = 0.;
      const double p_R = 1.;

      /*   c^2 = gamma * p / rho / (1 - b * rho) */
      const double a_R = std::sqrt(gamma_ * p_R / rho_R / (1. - b_ * rho_R));
      const double mach = initial_shock_front_mach_number_;
      const double S3 = mach * a_R;

      const double rho_L = rho_R * (gamma_ + 1.) * mach * mach /
                           ((gamma_ - 1.) * mach * mach + 2.);
      double u_L = (1. - rho_R / rho_L) * S3 + rho_R / rho_L * u_R;
      double p_L =
          p_R * (2. * gamma_ * mach * mach - (gamma_ - 1.)) / (gamma_ + 1.);

      state_1d_L_ = [=](const double, const double t) -> std::array<double, 3> {
        AssertThrow(t == 0.,
                    ExcMessage("No analytic solution for t > 0. available"));

        return {rho_L, u_L, p_L};
      };

      state_1d_R_ = [=](const double, const double t) -> std::array<double, 3> {
        AssertThrow(t == 0.,
                    ExcMessage("No analytic solution for t > 0. available"));

        return {rho_R, u_R, p_R};
      };

    } else if (initial_state_ == "contrast") {

      /* Contrast of the Sod shock tube: */

      state_1d_L_ = [](const double, const double t) -> std::array<double, 3> {
        AssertThrow(t == 0.,
                    ExcMessage("No analytic solution for t > 0. available"));

        // rho, u, p
        return {1.0, 0.0, 1.0};
      };

      state_1d_R_ = [](const double, const double t) -> std::array<double, 3> {
        AssertThrow(t == 0.,
                    ExcMessage("No analytic solution for t > 0. available"));

        // rho, u, p
        return {0.125, 0.0, 0.1};
      };

    } else if (initial_state_ == "smooth") {

      state_1d_L_ = [](const double x,
                       const double t) -> std::array<double, 3> {
        const double x_0 = -0.1; // 0.1;
        const double x_1 = 0.1;  // 0.3;

        double rho = 1.;
        if (x - t > x_0 && x - t < x_1)
          rho = 1. + 64. * std::pow(x_1 - x_0, -6.) * std::pow(x - t - x_0, 3) *
                         std::pow(x_1 - x + t, 3);

        // rho, u, p
        return {rho, 1.0, 1.0};
      };
      state_1d_R_ = state_1d_L_;

    } else {

      AssertThrow(false, dealii::ExcMessage("Unknown initial state."));
    }
  }


  template <int dim>
  typename ProblemDescription<dim>::rank1_type
  ProblemDescription<dim>::initial_state(const dealii::Point<dim> &point,
                                         double t) const
  {
    /*
     * Translate to conserved quantities and return the corresponding
     * state:
     */

    const double position_1d =
        (point - initial_position_) * initial_direction_;

    const auto [rho, u, p] = (position_1d > 0.) ? state_1d_R_(position_1d, t)
                                                : state_1d_L_(position_1d, t);

    rank1_type state;

    state[0] = rho;
    for (unsigned int i = 0; i < dim; ++i)
      state[1 + i] = rho * u * initial_direction_[i];
    state[dim + 1] = p / (gamma_ - 1.) + 0.5 * rho * u * u;

    return state;
  }

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_TEMPLATE_H */
