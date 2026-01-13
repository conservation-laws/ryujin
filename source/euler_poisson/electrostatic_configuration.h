//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
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
        dirichlet_boundaries_.insert({Boundary::do_nothing,
                                      Boundary::slip,
                                      Boundary::no_slip,
                                      Boundary::dirichlet,
                                      Boundary::dynamic,
                                      Boundary::dirichlet_momentum,
                                      Boundary::dirichlet_velocity});
        this->add_parameter(
            "dirichlet boundaries",
            dirichlet_boundaries_,
            "A list of hyperbolic boundary types where homogeneous boundary "
            "conditions will be enforced on the potential.");

        is_time_dependent_ = false;
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
       * Return the name of the configuration as a (const reference)
       * std::string.
       */
      ACCESSOR_READ_ONLY(name)

      /**
       * Return the selected boundary type.
       */
      ACCESSOR_READ_ONLY(dirichlet_boundaries)

      /**
       * Return a boolean indicating whether the background fields are time
       * dependent.
       */
      ACCESSOR_READ_ONLY(is_time_dependent)

    protected:
      bool is_time_dependent_;

    private:
      const std::string name_;
      std::set<dealii::types::boundary_id> dirichlet_boundaries_;
    };
  } // namespace ElectrostaticConfigurationLibrary
} /* namespace ryujin */
