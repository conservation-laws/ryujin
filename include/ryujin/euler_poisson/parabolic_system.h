//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/interface/parabolic_system.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>


namespace ryujin
{
  namespace EulerPoisson
  {
    /**
     * A parabolic system describing an electrostatic force model with a
     * background magnetic field. The electrostatic potential is obtained
     * by solving a Poisson problem for the given density distribution.
     *
     * @ingroup EulerPoissonEquations
     */
    class ParabolicSystem final : public Interface::ParabolicSystem
    {
    public:
      ParabolicSystem(const std::string &subsection = "/ParabolicSystem");

      ACCESSOR_READ_ONLY(alpha)

      ACCESSOR_READ_ONLY(magnetic_drift_limit);

      ACCESSOR_READ_ONLY(electrostatic_configuration);

      ACCESSOR_READ_ONLY(subsection);

    private:
      double alpha_;
      bool magnetic_drift_limit_;
      std::string subsection_;

      std::string electrostatic_configuration_;
    };


    inline ParabolicSystem::ParabolicSystem(const std::string &subsection)
        : Interface::ParabolicSystem(
              "Electrostatic force model with background magnetic field",
              /* the electrostatic potential */
              {{{"phi"}, {"phi"}, {"phi"}}},
              {},
              subsection)
        , subsection_(subsection)
    {
      alpha_ = 1.0;
      add_parameter("alpha", alpha_, "The coupling constant alpha");

      magnetic_drift_limit_ = false;
      add_parameter("set up magnetic drift limit",
                    magnetic_drift_limit_,
                    "If set to true, then the velocity field is initialized to "
                    "satisfy the magnetic drift limit.");

      electrostatic_configuration_ = "constant";
      add_parameter(
          "electrostatic configuration",
          electrostatic_configuration_,
          "Valid names are given by any of the subsections defined below");
    }

  } // namespace EulerPoisson
} // namespace ryujin
