//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <discretization.h>

#include "convenience_macros.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <string>

namespace ryujin
{
  namespace ElectrostaticConfigurationLibrary
  {
    /**
     * A small abstract base class to group configuration options for an
     * electrostatic configuration
     *
     * @ingroup EulerPoissonEquations
     */
    template <int dim, typename Number = double>
    class ElectrostaticConfiguration : public dealii::ParameterAcceptor
    {
    public:
      using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, Number>;

      ElectrostaticConfiguration(const std::string &name,
                                 const std::string &subsection)
          : ParameterAcceptor(subsection + "/" + name)
          , name_(name)
      {
        parabolic_boundary_ = Boundary::dirichlet;
        this->add_parameter(
            "boundary condition",
            parabolic_boundary_,
            "Type of boundary condition enforced on the electrostatic "
            "potential. Supported values are dirichlet, neumann, periodic.");
      }

      /**
       * Return a background (charge) density that is added to the (fluid)
       * density when enforcing the Gauß law.
       */
      virtual double background_density(const dealii::Point<dim> &point,
                                        Number t) const = 0;

      /**
       * Return a background magnetic field.
       */
      virtual curl_type magnetic_field(const dealii::Point<dim> &point,
                                       Number t) const = 0;

      /**
       * Return the name of the configuration as a (const reference) std::string
       */
      ACCESSOR_READ_ONLY(name)

      /**
       * Return the name of the configuration as a (const reference) std::string
       */

    private:
      const std::string name_;
      Boundary parabolic_boundary_;
    };
  } // namespace ElectrostaticConfigurationLibrary
} /* namespace ryujin */
