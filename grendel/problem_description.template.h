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
                  "Initial direction of shock, or contrast front");

    initial_position_[0] = 1.;
    add_parameter("initial - position",
                  initial_position_,
                  "Initial position of shock, or contrast front");

    initial_shock_front_mach_number_ = 2.0;
    add_parameter("shock front - mach number",
                  initial_shock_front_mach_number_,
                  "Shock Front: Mach number");

    initial_uniform_mach_number_ = 3.0;
    add_parameter("uniform - mach number",
                  initial_uniform_mach_number_,
                  "Uniform: Mach number");
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

    } else if (initial_state_ == "uniform") {

      state_1d_L_ = [=](const double, const double t) -> std::array<double, 3> {
        AssertThrow(t == 0.,
                    ExcMessage("No analytic solution for t > 0. available"));

        return {gamma_, initial_uniform_mach_number_, 1.};
      };

      state_1d_R_ = state_1d_L_;

    } else if (initial_state_ == "sod contrast") {

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

    } else if (initial_state_ == "vortex") {

      /*
        2D isentropic vortex problem. See section 5.6 of Euler-convex limiting
        paper by Guermond et al. Note that the paper might have typos. Also note
        this is not in the framework of left/right states.
      */
      initial_state_2D = [=](const double x,
                             const double y,
                             const double t) -> std::array<double, 4> {
        // define PI
        const double PI = std::atan(1.0) * 4.0;
        // set center point vortex center initialized at (5,5) and set
        // definition of r^2
        const double xBar = x - 5.0 - 2.0 * t;
        const double yBar = y - 5.0;
        const double rSquared = std::pow(xBar, 2) + std::pow(yBar, 2);
        // free stream values, Inf for infinity
        const double uInf = 2.0;
        const double vInf = 0.0;
        const double TInf = 1.0;
        // vortex strength
        const double beta = 5.0;
        // define flow perturbuations here
        double deltaU = -beta / (2.0 * PI) * exp((1.0 - rSquared) / 2.0) * yBar;
        double deltaV = beta / (2.0 * PI) * exp((1.0 - rSquared) / 2.0) * xBar;
        double deltaT = -(gamma_ - 1.0) * std::pow(beta, 2) /
                        (8.0 * gamma_ * std::pow(PI, 2)) * exp(1.0 - rSquared);
        // exact functions defined here
        double Temp = TInf + deltaT;
        double rho = std::pow(Temp, 1.0 / (gamma_ - 1.0));
        double u = uInf + deltaU;
        double v = vInf + deltaV;
        double p = std::pow(rho, gamma_);
        // rho, u, v, p
        return {rho, u, v, p};
      };
    } else {

      AssertThrow(false, dealii::ExcMessage("Unknown initial state."));
    }
  } // namespace grendel


  template <int dim>
  typename ProblemDescription<dim>::rank1_type
  ProblemDescription<dim>::initial_state(const dealii::Point<dim> &point,
                                         double t) const
  {
    /*
     * Translate to conserved quantities and return the corresponding
     * state:
     */

    rank1_type state;
    const double position_1d = (point - initial_position_) * initial_direction_;

    if (initial_state_ != "vortex") {
      const auto [rho, u, p] = (position_1d > 0.) ? state_1d_R_(position_1d, t)
                                                  : state_1d_L_(position_1d, t);
      state[0] = rho;
      for (unsigned int i = 0; i < dim; ++i) {
        state[1 + i] = rho * u * initial_direction_[i];
        state[dim + 1] = p / (gamma_ - 1.) + 0.5 * rho * u * u;
      }
    } else {
      const auto [rho2d, u2d, v2d, p2d] =
          initial_state_2D(point[0], point[1], t);
      state[0] = rho2d;
      state[1] = rho2d * u2d;
      state[2] = rho2d * v2d;
      state[3] = p2d / (gamma_ - 1.) + 0.5 * rho2d * (u2d * u2d + v2d * v2d);
    }
    return state;
  }

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_TEMPLATE_H */
