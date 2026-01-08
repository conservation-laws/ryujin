//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "electrostatic_configuration.h"

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
    class Constant : public ElectrostaticConfiguration<dim, Number>
    {
    public:
      using curl_type = ElectrostaticConfiguration<dim, Number>::curl_type;

      Constant(const std::string &subsection)
          : ElectrostaticConfiguration<dim, Number>("constant", subsection)
      {
        background_density_ = 0.;
        this->add_parameter("background density",
                            background_density_,
                            "a constant background (charge) density value");

        magnetic_field_ = curl_type{};
        this->add_parameter("magnetic field",
                            magnetic_field_,
                            "a constant background magnetic field density");
      }

      double background_density(const dealii::Point<dim> & /*point*/,
                                Number /*t*/) const final
      {
        return background_density_;
      }

      curl_type magnetic_field(const dealii::Point<dim> & /*point*/,
                               Number /*t*/) const final
      {
        return magnetic_field_;
      }

    private:
      double background_density_;
      curl_type magnetic_field_;
    };
  } // namespace ElectrostaticConfigurationLibrary
} // namespace ryujin
