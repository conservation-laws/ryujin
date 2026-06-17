//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
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
     * An initial state formed by a contrast of a given "left" and "right"
     * primitive (initial) state.
     *
     * The primitive state format is:
     *   (Y_0, Y_1, ..., Y_{n-2}, rho, u, p)
     * where Y_k are mass fractions for k = 0 to n_species-2, and the last
     * mass fraction Y_{n-1} = 1 - sum(Y_k) is computed automatically.
     *
     * @note The @p t argument is ignored.
     *
     * @ingroup MultiSpeciesEulerEquations
     */
    template <typename Description, int dim, typename Number>
    class Contrast : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;

      /* 1D primitive state: (Y_0, ..., Y_{n-2}, rho, u, p) */
      static constexpr unsigned int primitive_dim = n_species + 2;
      using primitive_state_type = dealii::Tensor<1, primitive_dim, Number>;

      Contrast(const HyperbolicSystem &hyperbolic_system,
               const std::string subsection)
          : InitialState<Description, dim, Number>("contrast", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        /* Default: equal mass fractions */
        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_left_[k] = Number(1.) / Number(n_species);
        temp_left_[n_species - 1] = 1.4; /* rho */
        temp_left_[n_species] = 0.0;     /* u */
        temp_left_[n_species + 1] = 1.0; /* p */
        this->add_parameter(
            "primitive state left",
            temp_left_,
            "Initial 1d primitive state (Y_0, ..., Y_{n-2}, rho, u, p) on the "
            "left");

        for (unsigned int k = 0; k < n_species - 1; ++k)
          temp_right_[k] = Number(1.) / Number(n_species);
        temp_right_[n_species - 1] = 1.4; /* rho */
        temp_right_[n_species] = 0.0;     /* u */
        temp_right_[n_species + 1] = 1.0; /* p */
        this->add_parameter(
            "primitive state right",
            temp_right_,
            "Initial 1d primitive state (Y_0, ..., Y_{n-2}, rho, u, p) on the "
            "right");

        const auto convert_states = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();

          const auto primitive_left_ = extend_primitive(temp_left_);
          const auto primitive_right_ = extend_primitive(temp_right_);

          state_left_ = view.from_initial_state(primitive_left_);
          state_right_ = view.from_initial_state(primitive_right_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        return (point[0] > 0. ? state_right_ : state_left_);
      }

    private:
      const HyperbolicSystem &hyperbolic_system_;

      primitive_state_type temp_left_;
      primitive_state_type temp_right_;

      state_type state_left_;
      state_type state_right_;
    };
  } // namespace MultiSpeciesEulerInitialStates
} // namespace ryujin
