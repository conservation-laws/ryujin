//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>


namespace ryujin
{
  namespace EulerPoisson
  {
    class ParabolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      static inline const std::string problem_name =
          "Electrostatic force model with background magnetic field";

      static constexpr bool is_identity = false;

      ParabolicSystem(const std::string &subsection = "/B - Equation");

      ACCESSOR_READ_ONLY(alpha)

      unsigned int n_parabolic_state_vectors() const
      {
        return parabolic_component_names_.size();
      }

      ACCESSOR_READ_ONLY(parabolic_component_names);

      ACCESSOR_READ_ONLY(background_density);

      ACCESSOR_READ_ONLY(magnetic_field_x);
      ACCESSOR_READ_ONLY(magnetic_field_y);
      ACCESSOR_READ_ONLY(magnetic_field_z);

    private:
      std::string alpha_expression_;
      double alpha_;
      const std::vector<std::string> parabolic_component_names_ = {"phi"};
      std::string background_density_;
      std::string magnetic_field_x_;
      std::string magnetic_field_y_;
      std::string magnetic_field_z_;
    };


    inline ParabolicSystem::ParabolicSystem(const std::string &subsection)
        : ParameterAcceptor(subsection)
    {
      alpha_expression_ = "0.0";
      add_parameter(
          "alpha", alpha_expression_, "The coupling constant expression");

      /*
       * We allow for the coupling term alpha to be given as an expression in
       * the parameter file for convenience, but this should be a constant
       * expression (i.e. not depending on space or time)
       */
      const auto initialize_alpha = [this] {
        auto alpha_function = dealii::FunctionParser<1>(alpha_expression_);
        alpha_ = alpha_function.value(dealii::Point<1>());
      };
      ParameterAcceptor::parse_parameters_call_back.connect(initialize_alpha);

      background_density_ = "0.0";
      add_parameter("background density",
                    background_density_,
                    "Background density function expression");

      magnetic_field_x_ = "0.0";
      add_parameter(
          "magnetic field x", magnetic_field_x_, "Magnetic field x component");

      magnetic_field_y_ = "0.0";
      add_parameter(
          "magnetic field y", magnetic_field_y_, "Magnetic field y component");

      magnetic_field_z_ = "0.0";
      add_parameter(
          "magnetic field z", magnetic_field_z_, "Magnetic field z component");
    }

  } // namespace EulerPoisson
} // namespace ryujin
