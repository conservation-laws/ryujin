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
    add_parameter(
        "initial state",
        initial_state_,
        "Initial state. Valid names are \"shock front\", or \"contrast\".");

    initial_direction_[0] = 1.;
    add_parameter("initial - direction",
                  initial_direction_,
                  "Initial direction of shock front, or  contrast");

    initial_position_[0] = 1.;
    add_parameter("initial - position",
                  initial_position_,
                  "Initial position of shock front, or  contrast");

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
     * Now compute the initial states:
     */

    double rho_R = 0., u_R = 0., p_R = 0., rho_L = 0., u_L = 0., p_L = 0.;

    if (initial_state_ == "shock front") {

      // FIXME: Add reference to literature

      rho_R = gamma_;
      u_R = 0.;
      p_R = 1.;

      /*   c^2 = gamma * p / rho / (1 - b * rho) */
      const double a_R = std::sqrt(gamma_ * p_R / rho_R / (1. - b_ * rho_R));
      const double mach = initial_shock_front_mach_number_;
      const double S3 = mach * a_R;

      rho_L = rho_R * (gamma_ + 1.) * mach * mach /
              ((gamma_ - 1.) * mach * mach + 2.);
      u_L = (1. - rho_R / rho_L) * S3 + rho_R / rho_L * u_R;
      p_L = p_R * (2. * gamma_ * mach * mach - (gamma_ - 1.)) / (gamma_ + 1.);

    } else if (initial_state_ == "contrast") {

      /* Contrast of the Sod shock tube: */
      rho_L = 1.0;
      u_L = 0.0;
      p_L = 1.0;
      rho_R = 0.125;
      u_R = 0.0;
      p_R = 0.1;

    } else {

      AssertThrow(false, dealii::ExcMessage("Unknown initial state."));
    }

    /*
     * And translate to nD states:
     */

    initial_state_L_[0] = rho_L;
    initial_state_R_[0] = rho_R;
    for (unsigned int i = 0; i < dim; ++i) {
      initial_state_L_[1 + i] = rho_L * initial_direction_[i] * u_L;
      initial_state_R_[1 + i] = rho_R * initial_direction_[i] * u_R;
    }
    initial_state_L_[dim + 1] = p_L / (gamma_ - 1.) + 0.5 * rho_L * u_L * u_L;
    initial_state_R_[dim + 1] = p_R / (gamma_ - 1.) + 0.5 * rho_R * u_R * u_R;
  }


  template <int dim>
  typename ProblemDescription<dim>::rank1_type
  ProblemDescription<dim>::initial_state(const dealii::Point<dim> &point) const
  {
    const double x = point[0];
    const double x_0 = 0.1;
    const double x_1 = 0.3;
    const double t = 0.;

    double rho = 1.;
    if(x - t > x_0  && x - t < x_1)
      rho = 1. + 64. * std::pow(x_1 - x_0, -6.) * std::pow(x - t - x_0, 3) *
                     std::pow(x_1 - x + t, 3);
    const double u = 1.;
    const double p = 1.;

    rank1_type state;

    state[0] = rho;
    state[1] = rho * u;
    state[dim + 1] = p / (gamma_ - 1.) + 0.5 * rho * u * u;

    return state;
  }

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_TEMPLATE_H */
