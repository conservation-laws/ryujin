//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <initial_state_library.h>

namespace ryujin
{
  namespace MultiSpeciesEulerInitialStates
  {
    using namespace MultiSpeciesEuler;

    /**
     * A modification of the "contrast" initial state. Now, we have an
     * initial state formed by a contrast of a given "left" and "right"
     * primitive state where the "left" state is "inside" the radius, R,
     * and the "right" state is outside.
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @ingroup MultiSpeciesEulerEquations
     */
    template <typename Description, int dim, typename Number>
    class RadialContrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      /* 1D primitive state: (Y_0, ..., Y_{n-2}, rho, u, p) */
      static constexpr unsigned int primitive_dim = n_species + 2;
      using primitive_state_type = dealii::Tensor<1, primitive_dim, Number>;

      RadialContrast(const HyperbolicSystem &hyperbolic_system,
                     const std::string &subsection)
          : InitialState<Description, dim, Number>("radial contrast",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        use_radial_velocity_ = false;
        this->add_parameter(
            "use radial velocity",
            use_radial_velocity_,
            "If set to true, we transform a non-zero velocity into a radial "
            "velocity with scaling 1 / r^(dim - 1)");

        /* Default: equal mass fractions */
        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_left_[k] = Number(1.) / Number(n_species);
        temp_left_[n_species - 1] = 1.4;  /* rho */
        temp_left_[n_species] = 0.0;      /* u */
        temp_left_[n_species + 1] = 1.0;  /* p */
        this->add_parameter(
            "primitive state left",
            temp_left_,
            "Initial 1d primitive state (Y_0, ..., Y_{n-2}, rho, u, p) inside "
            "radial area");

        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_right_[k] = Number(1.) / Number(n_species);
        temp_right_[n_species - 1] = 1.4;  /* rho */
        temp_right_[n_species] = 0.0;      /* u */
        temp_right_[n_species + 1] = 1.0;  /* p */
        this->add_parameter(
            "primitive state right",
            temp_right_,
            "Initial 1d primitive state (Y_0, ..., Y_{n-2}, rho, u, p) outside "
            "radial area");

        radius_ = 1.0;
        this->add_parameter("radius", radius_, "Radius of radial area");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();

          const auto primitive_left_ = extend_primitive(temp_left_);
          const auto primitive_right_ = extend_primitive(temp_right_);

          state_left_ = view.from_initial_state(primitive_left_);
          state_right_ = view.from_initial_state(primitive_right_);
        };

        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      };

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        auto result =
            (point.norm() > radius_ ? state_right_ : state_left_);

        if (point.norm() > 0. && use_radial_velocity_) {
          const auto view = hyperbolic_system_.template view<dim, Number>();
          const auto &temp =
              (point.norm() > radius_ ? temp_right_ : temp_left_);
          const auto rho = view.density(result);
          const auto u = temp[n_species]; /* scalar velocity */
          for (unsigned int d = 0; d < dim; ++d)
            result[n_species + d] = rho * u * point[d] / point.norm();
        }

        return result;
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      primitive_state_type temp_left_;
      primitive_state_type temp_right_;

      state_type state_left_;
      state_type state_right_;

      double radius_;
      bool use_radial_velocity_;

    };


  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
