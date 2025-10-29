//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "electrostatic_configuration.h"

#include <deal.II/base/function_parser.h>

namespace ryujin
{
  namespace ElectrostaticConfigurationLibrary
  {
    /**
     * A user-specified equation of state
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number = double>
    class Function : public ElectrostaticConfiguration<dim, Number>
    {
    public:
      using curl_type = ElectrostaticConfiguration<dim, Number>::curl_type;

      Function(const std::string &subsection)
          : ElectrostaticConfiguration<dim, Number>("function", subsection)
      {
        background_density_expression_ = "0.";
        this->add_parameter("background density",
                            background_density_expression_,
                            "A function expression for the background density");

        if constexpr (dim >= 2) {
          magnetic_field_z_expression_ = "0.";
          this->add_parameter("magnetic field z",
                              magnetic_field_z_expression_,
                              "A function expression for the z component of "
                              "the magnetic field");
        }

        if constexpr (dim == 3) {
          magnetic_field_x_expression_ = "0.";
          this->add_parameter("magnetic field x",
                              magnetic_field_x_expression_,
                              "A function expression for the x component of "
                              "the magnetic field");

          magnetic_field_y_expression_ = "0.";
          this->add_parameter("magnetic field y",
                              magnetic_field_y_expression_,
                              "A function expression for the y component of "
                              "the magnetic field");
        }

        /* Set up all muparser objects: */

        const auto set_up_muparser = [this] {
          /*
           * This variant of the constructor initializes the function
           * parser with support for a time-dependent description involving
           * a variable »t«:
           */
          using FP = dealii::FunctionParser<dim>;
          background_density_ =
              std::make_unique<FP>(background_density_expression_);

          if constexpr (dim >= 2) {
            magnetic_field_z_ =
                std::make_unique<FP>(magnetic_field_z_expression_);
          }

          if constexpr (dim == 3) {
            magnetic_field_x_ =
                std::make_unique<FP>(magnetic_field_x_expression_);
            magnetic_field_y_ =
                std::make_unique<FP>(magnetic_field_y_expression_);
          }
        };

        set_up_muparser();
        this->parse_parameters_call_back.connect(set_up_muparser);
      }

      virtual double background_density(const dealii::Point<dim> &point,
                                        Number t) const final
      {
        background_density_->set_time(t);
        return static_cast<Number>(background_density_->value(point));
      }

      virtual curl_type magnetic_field(const dealii::Point<dim> &point,
                                       Number t) const final
      {
        if constexpr (dim == 1) {
          return curl_type{};
        }

        if constexpr (dim == 2) {
          magnetic_field_z_->set_time(t);
          return curl_type{
              {static_cast<Number>(magnetic_field_z_->value(point))}};
        }

        if constexpr (dim == 3) {
          magnetic_field_x_->set_time(t);
          magnetic_field_y_->set_time(t);
          magnetic_field_z_->set_time(t);
          return curl_type{
              {static_cast<Number>(magnetic_field_x_->value(point)),
               static_cast<Number>(magnetic_field_y_->value(point)),
               static_cast<Number>(magnetic_field_z_->value(point))}};
        }
      }

    private:
      std::string background_density_expression_;
      std::string magnetic_field_x_expression_;
      std::string magnetic_field_y_expression_;
      std::string magnetic_field_z_expression_;

      std::unique_ptr<dealii::FunctionParser<dim>> background_density_;
      std::unique_ptr<dealii::FunctionParser<dim>> magnetic_field_x_;
      std::unique_ptr<dealii::FunctionParser<dim>> magnetic_field_y_;
      std::unique_ptr<dealii::FunctionParser<dim>> magnetic_field_z_;
    };
  } // namespace ElectrostaticConfigurationLibrary
} // namespace ryujin
