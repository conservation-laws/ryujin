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
                  "Initial state. Valid names are \"shock front\".");

    initial_shock_front_mach_number_ = 2.0;
    add_parameter("shock front - mach number",
                  initial_shock_front_mach_number_,
                  "Shock Front: Mach number");

    initial_shock_front_direction_[0] = 1.;
    add_parameter("shock front - direction",
                  initial_shock_front_direction_,
                  "Shock Front: direction");

    initial_shock_front_position_[0] = 1.;
    add_parameter("shock front - position",
                  initial_shock_front_position_,
                  "Shock Front: position");
  }


  template <int dim>
  void ProblemDescription<dim>::parse_parameters_callback()
  {
    /*
     * Let's compute the two initial states:
     */

    // FIXME: Validate
    // FIXME: Add reference to literature

    const double mach = initial_shock_front_mach_number_;

    const double rho_R = gamma_;
    const double u_R = 0.;
    const double p_R = 1.;

    /*   c^2 = gamma * p / rho / (1 - b * rho) */
    const double c_R = std::sqrt(gamma_ * p_R / rho_R / (1. - b_ * rho_R));
    const double S3 = mach * c_R;

    const double rho_L = rho_R * (gamma_ + 1.) * mach * mach /
                         ((gamma_ - 1.) * mach * mach + 2.);
    const double u_L =  (1. - rho_R / rho_L) * S3 + rho_R / rho_L * u_R;
    const double p_L =
        p_R * (2. * gamma_ * mach * mach - (gamma_ - 1.)) / (gamma_ + 1.);

    /*
     * And translate to 3D states:
     */

    initial_shock_front_state_L_[0] = rho_L;
    initial_shock_front_state_R_[0] = rho_R;
    for (unsigned int i = 0; i < dim; ++i) {
      initial_shock_front_state_L_[1 + i] =
          rho_L * initial_shock_front_direction_[i] * u_L;
      initial_shock_front_state_R_[1 + i] =
          rho_R * initial_shock_front_direction_[i] * u_R;
    }
    initial_shock_front_state_L_[dim + 1] =
        p_L / (gamma_ - 1.) + 0.5 * rho_L * u_L * u_L;
    initial_shock_front_state_R_[dim + 1] =
        p_R / (gamma_ - 1.) + 0.5 * rho_R * u_R * u_R;
  }

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_TEMPLATE_H */
