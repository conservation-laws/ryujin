//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <initial_state_library.h>

#include "initial_values.h"
#include "simd.h"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>

#include <random>

namespace ryujin
{
  using namespace dealii;

  template <typename Description, int dim, typename Number>
  InitialValues<Description, dim, Number>::InitialValues(
      const HyperbolicSystem &hyperbolic_system,
      const OfflineData<dim, Number> &offline_data,
      const std::string &subsection)
      : ParameterAcceptor(subsection)
      , hyperbolic_system_(&hyperbolic_system)
      , offline_data_(&offline_data)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(std::bind(
        &InitialValues<Description, dim, Number>::parse_parameters_callback,
        this));

    configuration_ = "uniform";
    add_parameter("configuration",
                  configuration_,
                  "The initial state configuration. Valid names are given by "
                  "any of the subsections defined below.");

    initial_direction_[0] = 1.;
    add_parameter(
        "direction",
        initial_direction_,
        "Initial direction of initial configuration (Galilei transform)");

    initial_position_[0] = 1.;
    add_parameter(
        "position",
        initial_position_,
        "Initial position of initial configuration (Galilei transform)");

    perturbation_ = 0.;
    add_parameter("perturbation",
                  perturbation_,
                  "Add a random perturbation of the specified magnitude to the "
                  "initial state.");

    /*
     * And finally populate the initial state list with all initial state
     * configurations defined in the InitialStateLibrary namespace:
     */
    Description::InitialStateLibrary::
        template populate_initial_state_list<dim, Number>(
            initial_state_list_, *hyperbolic_system_, subsection);
  }

  namespace
  {
    /**
     * An affine transformation:
     */
    template <int dim>
    inline DEAL_II_ALWAYS_INLINE dealii::Point<dim>
    affine_transform(const dealii::Tensor<1, dim> initial_direction,
                     const dealii::Point<dim> initial_position,
                     const dealii::Point<dim> x)
    {
      auto direction = x - initial_position;

      /* Roll third component of initial_direction onto xy-plane: */
      if constexpr (dim == 3) {
        auto n_x = initial_direction[0];
        auto n_z = initial_direction[2];
        const auto norm = std::sqrt(n_x * n_x + n_z * n_z);
        n_x /= norm;
        n_z /= norm;
        auto new_direction = direction;
        if (norm > 1.0e-14) {
          new_direction[0] = n_x * direction[0] + n_z * direction[2];
          new_direction[2] = -n_z * direction[0] + n_x * direction[2];
        }
        direction = new_direction;
      }

      /* Roll second component of initial_direction onto x-axis: */
      if constexpr (dim >= 2) {
        auto n_x = initial_direction[0];
        auto n_y = initial_direction[1];
        const auto norm = std::sqrt(n_x * n_x + n_y * n_y);
        n_x /= norm;
        n_y /= norm;
        auto new_direction = direction;
        if (norm > 1.0e-14) {
          new_direction[0] = n_x * direction[0] + n_y * direction[1];
          new_direction[1] = -n_y * direction[0] + n_x * direction[1];
        }
        direction = new_direction;
      }

      return Point<dim>() + direction;
    }


    /**
     * Transform vector:
     */
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE dealii::Tensor<1, dim, Number>
    affine_transform_vector(const dealii::Tensor<1, dim> initial_direction,
                            dealii::Tensor<1, dim, Number> direction)
    {
      if constexpr (dim >= 2) {
        auto n_x = initial_direction[0];
        auto n_y = initial_direction[1];
        const auto norm = std::sqrt(n_x * n_x + n_y * n_y);
        n_x /= norm;
        n_y /= norm;
        auto new_direction = direction;
        if (norm > 1.0e-14) {
          new_direction[0] = n_x * direction[0] - n_y * direction[1];
          new_direction[1] = n_y * direction[0] + n_x * direction[1];
        }
        direction = new_direction;
      }

      if constexpr (dim == 3) {
        auto n_x = initial_direction[0];
        auto n_z = initial_direction[2];
        const auto norm = std::sqrt(n_x * n_x + n_z * n_z);
        n_x /= norm;
        n_z /= norm;
        auto new_direction = direction;
        if (norm > 1.0e-14) {
          new_direction[0] = n_x * direction[0] - n_z * direction[2];
          new_direction[2] = n_z * direction[0] + n_x * direction[2];
        }
        direction = new_direction;
      }

      return direction;
    }
  } /* namespace */


  template <typename Description, int dim, typename Number>
  void InitialValues<Description, dim, Number>::parse_parameters_callback()
  {
    /* First, let's normalize the direction: */

    AssertThrow(initial_direction_.norm() != 0.,
                ExcMessage("Initial direction is set to the zero vector."));
    initial_direction_ /= initial_direction_.norm();

    /* Populate std::function object: */

    {
      bool initialized = false;
      for (auto &it : initial_state_list_)
        if (it->name() == configuration_) {
          initial_state_ = [this, &it](const dealii::Point<dim> &point,
                                       Number t) {
            const auto transformed_point =
                affine_transform(initial_direction_, initial_position_, point);
            auto state = it->compute(transformed_point, t);
            state = hyperbolic_system_->apply_galilei_transform(
                state, [&](const auto &momentum) {
                  return affine_transform_vector(initial_direction_, momentum);
                });
            return state;
          };

          flux_contributions_ = [this, &it](const dealii::Point<dim> &point) {
            const auto transformed_point =
                affine_transform(initial_direction_, initial_position_, point);
            return it->initial_precomputations(transformed_point);
          };
          initialized = true;
          break;
        }

      AssertThrow(
          initialized,
          ExcMessage(
              "Could not find an initial state description with name \"" +
              configuration_ + "\""));
    }

    /* Add a random perturbation to the original function object: */

    if (perturbation_ != 0.) {
      initial_state_ = [old_state = this->initial_state_,
                        perturbation = this->perturbation_](
                           const dealii::Point<dim> &point, Number t) {
        auto state = old_state(point, t);

        if (t > 0.)
          return state;

        static std::default_random_engine generator =
            std::default_random_engine(std::random_device()());
        static std::uniform_real_distribution<Number> distribution(-1., 1.);
        static auto draw = std::bind(distribution, generator);
        for (unsigned int i = 0; i < problem_dimension; ++i)
          state[i] *= (Number(1.) + perturbation * draw());

        return state;
      };
    }
  }


  template <typename Description, int dim, typename Number>
  auto InitialValues<Description, dim, Number>::interpolate(Number t) const
      -> vector_type
  {
#ifdef DEBUG_OUTPUT
    std::cout << "InitialValues<dim, Number>::interpolate(t = " << t << ")"
              << std::endl;
#endif

    vector_type U;
    U.reinit(offline_data_->vector_partitioner());

    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    const auto callable = [&](const auto &p) { return initial_state(p, t); };

    scalar_type temp;
    const auto scalar_partitioner = offline_data_->scalar_partitioner();
    temp.reinit(scalar_partitioner);

    for (unsigned int d = 0; d < problem_dimension; ++d) {
      VectorTools::interpolate(offline_data_->dof_handler(),
                               to_function<dim, Number>(callable, d),
                               temp);
      U.insert_component(temp, d);
    }

    /*
     * Cosmetic fix up: Ensure that the initial state is compatible with
     * slip and no_slip boundary conditions. This ensures that nothing is
     * ever transported out of slip and no slip boundaries - even if
     * initial conditions happen to be set incorrectly.
     */

    const auto &boundary_map = offline_data_->boundary_map();
    const unsigned int n_owned = offline_data_->n_locally_owned();

    for (auto entry : boundary_map) {
      const auto i = entry.first;
      if (i >= n_owned)
        continue;

      const auto normal = std::get<0>(entry.second);
      const auto id = std::get<3>(entry.second);

      if (id == Boundary::slip || id == Boundary::no_slip) {
        auto U_i = U.get_tensor(i);
        U_i = hyperbolic_system_->apply_boundary_conditions(
            id, U_i, normal, [&]() { return U_i; });
        U.write_tensor(U_i, i);
      }
    }

    U.update_ghost_values();
    return U;
  }


  template <typename Description, int dim, typename Number>
  auto InitialValues<Description, dim, Number>::
      interpolate_precomputed_initial_values() const
      -> MultiComponentVector<Number, n_precomputed_values>
  {
    const auto scalar_partitioner = offline_data_->scalar_partitioner();

    MultiComponentVector<Number, n_precomputed_values> precomputed;
    precomputed.reinit_with_scalar_partitioner(scalar_partitioner);

    if constexpr (n_precomputed_values == 0)
      return precomputed;

    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    const auto callable = [&](const auto &p) { return flux_contributions(p); };

    scalar_type temp;
    temp.reinit(scalar_partitioner);

    for (unsigned int d = 0; d < n_precomputed_values; ++d) {
      VectorTools::interpolate(offline_data_->dof_handler(),
                               to_function<dim, Number>(callable, d),
                               temp);
      precomputed.insert_component(temp, d);
    }

    precomputed.update_ghost_values();
    return precomputed;
  }

} /* namespace ryujin */
