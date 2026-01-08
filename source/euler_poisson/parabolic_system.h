//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
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

      ParabolicSystem(const std::string &subsection = "/ParabolicSystem");

      ACCESSOR_READ_ONLY(alpha)

      ACCESSOR_READ_ONLY(magnetic_drift_limit);

      ACCESSOR_READ_ONLY(electrostatic_configuration);

      ACCESSOR_READ_ONLY(subsection);

      unsigned int n_parabolic_state_vectors() const
      {
        return parabolic_component_names_.size();
      }


      ACCESSOR_READ_ONLY(parabolic_component_names);

    private:
      double alpha_;
      bool magnetic_drift_limit_;
      std::string subsection_;

      std::string electrostatic_configuration_;

      static inline const std::vector<std::string> parabolic_component_names_ =
          {"phi"};
    };


    inline ParabolicSystem::ParabolicSystem(const std::string &subsection)
        : ParameterAcceptor(subsection)
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
