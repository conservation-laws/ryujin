#ifndef INITIAL_VALUES_TEMPLATE_H
#define INITIAL_VALUES_TEMPLATE_H

#include "initial_values.h"

#include <random>

namespace grendel
{
  using namespace dealii;

  template <int dim>
  InitialValues<dim>::InitialValues(const std::string &subsection)
      : ParameterAcceptor(subsection)
      , initial_state(initial_state_)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&InitialValues<dim>::parse_parameters_callback, this));

    configuration_ = "uniform";
    add_parameter(
        "configuration",
        configuration_,
        "Configuration. Valid options are \"uniform\", \"shock front\", "
        "\"sod contrast\", \"isentropic vortex\"");

    initial_direction_[0] = 1.;
    add_parameter("initial - direction",
                  initial_direction_,
                  "Initial direction of shock front, sod contrast, or vortex");

    initial_position_[0] = 1.;
    add_parameter("initial - position",
                  initial_position_,
                  "Initial position of shock front, sod contrast, or vortex");

    initial_1d_state_[0] = gamma;
    initial_1d_state_[1] = 0.;
    initial_1d_state_[2] = 1.;
    add_parameter("initial - 1d state",
                  initial_1d_state_,
                  "Initial 1d state (rho, u, p) of the uniform, shock front "
                  "configurations");

    initial_mach_number_ = 2.0;
    add_parameter("initial - mach number",
                  initial_mach_number_,
                  "Mach number of shock front (S1, S3 = mach * a_L/R), or "
                  "isentropic vortex");

    initial_vortex_beta_ = 5.0;
    add_parameter("vortex - beta",
                  initial_vortex_beta_,
                  "Isentropic vortex strength beta");

    perturbation_ = 0.;
    add_parameter("perturbation",
                  perturbation_,
                  "Add a random perturbation of the specified magnitude to the "
                  "initial state.");
  }


  template <int dim>
  void InitialValues<dim>::parse_parameters_callback()
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
        [=](const dealii::Tensor<1, 3, double> &state_1d) -> rank1_type {
      const auto &rho = state_1d[0];
      const auto &u = state_1d[1];
      const auto &p = state_1d[2];

      rank1_type state;

      state[0] = rho;
      for (unsigned int i = 0; i < dim; ++i)
        state[1 + i] = rho * u * initial_direction_[i];
      state[dim + 1] = p / (gamma - 1.) + 0.5 * rho * u * u;

      return state;
    };


    /*
     * Now populate the initial_state_ function object:
     */

    if (configuration_ == "uniform") {

      /*
       * A uniform flow:
       */

      initial_state_ = [=](const dealii::Point<dim> & /*point*/, double /*t*/) {
        return from_1d_state(initial_1d_state_);
      };

    } else if (configuration_ == "shock front") {

      /*
       * Mach shock front S1/S3:
       */

      const auto &rho_R = initial_1d_state_[0];
      const auto &u_R = initial_1d_state_[1];
      const auto &p_R = initial_1d_state_[2];
      const double mach_S = initial_mach_number_;

      /* a_R^2 = gamma * p / rho / (1 - b * rho) */
      const double a_R = std::sqrt(gamma * p_R / rho_R / (1 - b * rho_R));
      const double mach_R = u_R / a_R;

      const double S3 = mach_S * a_R;
      const double delta_mach = mach_R - mach_S;

      const double rho_L = rho_R * (gamma + 1.) * delta_mach * delta_mach /
                           ((gamma - 1.) * delta_mach * delta_mach + 2.);
      double u_L = (1. - rho_R / rho_L) * S3 + rho_R / rho_L * u_R;
      double p_L = p_R * (2. * gamma * delta_mach * delta_mach - (gamma - 1.)) /
                   (gamma + 1.);

      dealii::Tensor<1, 3, double> initial_1d_state_L{{rho_L, u_L, p_L}};

      initial_state_ = [=](const dealii::Point<dim> &point, double t) {
        const double position_1d =
            (point - initial_position_) * initial_direction_ - S3 * t;

        if (position_1d > 0.) {
          return from_1d_state(initial_1d_state_);
        } else {
          return from_1d_state(initial_1d_state_L);
        }
      };

    } else if (configuration_ == "sod contrast") {

      /*
       * Contrast of the Sod shock tube:
       */

      dealii::Tensor<1, 3, double> initial_1d_state_L{{0.125, 0.0, 0.1}};
      dealii::Tensor<1, 3, double> initial_1d_state_R{{1.0, 0.0, 1.0}};

      initial_state_ = [=](const dealii::Point<dim> &point, double /*t*/) {
        const double position_1d =
            (point - initial_position_) * initial_direction_;

        if (position_1d > 0.) {
          return from_1d_state(initial_1d_state_L);
        } else {
          return from_1d_state(initial_1d_state_R);
        }
      };

    } else if (configuration_ == "isentropic vortex") {

      /*
       * 2D isentropic vortex problem. See section 5.6 of Euler-convex
       * limiting paper by Guermond et al.
       */

      if constexpr (dim == 2) {
        initial_state_ = [=](const dealii::Point<dim> &point, double t) {
          const auto point_bar = point - initial_position_ -
                                 initial_direction_ * initial_mach_number_ * t;
          const double r_square = point_bar.norm_square();

          const double factor =
              initial_vortex_beta_ / (2. * M_PI) * exp(0.5 - 0.5 * r_square);

          const double T = 1. - (gamma - 1.) / (2. * gamma) * factor * factor;

          const double u = initial_direction_[0] * initial_mach_number_ -
                           factor * point_bar[1];

          const double v = initial_direction_[1] * initial_mach_number_ +
                           factor * point_bar[0];

          const double rho = std::pow(T, 1. / (gamma - 1.));
          const double p = std::pow(rho, gamma);
          const double E = p / (gamma - 1.) + 0.5 * rho * (u * u + v * v);

          return rank1_type({rho, rho * u, rho * v, E});
        };

      } else {

        AssertThrow(false, dealii::ExcNotImplemented());
      }

    } else {

      AssertThrow(false, dealii::ExcMessage("Unknown initial state."));
    }

    /*
     * Add a random perturbation to the original function object:
     */
    if (perturbation_ != 0.) {
      initial_state_ = [old_state = this->initial_state_,
                        perturbation = this->perturbation_](
                           const dealii::Point<dim> &point, double t) {
        static std::default_random_engine generator;
        static std::uniform_real_distribution<double> distribution(-1., 1.);
        auto draw = std::bind(distribution, generator);

        auto state = old_state(point, t);
        for (unsigned int i = 0; i < problem_dimension; ++i)
          state[i] *= (1. + perturbation * draw());

        return state;
      };
    }
  }

} /* namespace grendel */

#endif /* INITIAL_VALUES_TEMPLATE_H */
