//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
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
     * An initial state formed by two contrasts of given "left", "middle"
     * and "right" primitive states. The user defines the lengths of the left
     * and middle regions. The rest of the domain is populated with the right
     * region. For single species, this initial state (default values) can be
     * used to replicate the classical Woodward-Colella colliding blast wave
     * problem described in @cite Woodward1984
     *
     * @note The @p t argument is ignored. This class always returns the
     * initial configuration.
     *
     * @ingroup MultiSpeciesEulerEquations
     */
    template <typename Description, int dim, typename Number>
    class ThreeStateContrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      /* 1D primitive state: (Y_0, ..., Y_{n-2}, rho, u, p) */
      static constexpr unsigned int primitive_dim = n_species + 2;
      using primitive_state_type = dealii::Tensor<1, primitive_dim, Number>;

      ThreeStateContrast(const HyperbolicSystem &hyperbolic_system,
                         const std::string &subsection)
          : InitialState<Description, dim, Number>("three state contrast",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        /* Default: equal mass fractions */
        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_left_[k] = Number(1.) / Number(n_species);
        temp_left_[n_species - 1] = 1.;   /* rho */
        temp_left_[n_species] = 0.;       /* u */
        temp_left_[n_species + 1] = 1.e3; /* p */
        this->add_parameter(
            "primitive state left",
            temp_left_,
            "Initial 1d primitive state (Y_0, ..., Y_{n-2}, rho, u, p) on the "
            "left");

        left_length_ = 0.1;
        this->add_parameter("left region length",
                            left_length_,
                            "The length of the left region");

        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_middle_[k] = Number(1.) / Number(n_species);
        temp_middle_[n_species - 1] = 1.;    /* rho */
        temp_middle_[n_species] = 0.;        /* u */
        temp_middle_[n_species + 1] = 1.e-2; /* p */
        this->add_parameter(
            "primitive state middle",
            temp_middle_,
            "Initial 1d primitive state (Y_0, ..., Y_{n-2}, rho, u, p) in the "
            "middle");

        middle_length_ = 0.8;
        this->add_parameter("middle region length",
                            middle_length_,
                            "The length of the middle region");

        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_right_[k] = Number(1.) / Number(n_species);
        temp_right_[n_species - 1] = 1.;   /* rho */
        temp_right_[n_species] = 0.;       /* u */
        temp_right_[n_species + 1] = 1.e2; /* p */
        this->add_parameter(
            "primitive state right",
            temp_right_,
            "Initial 1d primitive state (Y_0, ..., Y_{n-2}, rho, u, p) on the "
            "right");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();

          const auto primitive_left_ = extend_primitive(temp_left_);
          const auto primitive_middle_ = extend_primitive(temp_middle_);
          const auto primitive_right_ = extend_primitive(temp_right_);

          state_left_ = view.from_initial_state(primitive_left_);
          state_middle_ = view.from_initial_state(primitive_middle_);
          state_right_ = view.from_initial_state(primitive_right_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        return point[0] >= left_length_ + middle_length_ ? state_right_
               : point[0] >= left_length_                ? state_middle_
                                                         : state_left_;
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      Number left_length_;
      Number middle_length_;

      primitive_state_type temp_left_;
      primitive_state_type temp_middle_;
      primitive_state_type temp_right_;

      state_type state_left_;
      state_type state_middle_;
      state_type state_right_;
    };
  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
