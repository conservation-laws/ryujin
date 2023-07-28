//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    /**
     * The parabolic subsystem of the compressible Euler equations. This is
     * just the idenity operator.
     *
     * @ingroup EulerEquations
     */
    class ParabolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the hyperbolic system as a string.
       */
      static inline const std::string problem_name = "Identity";

      /**
       * This parabolic subsystem represents an identity.
       */
      static constexpr bool is_identity = true;

      /**
       * Constructor.
       */
      ParabolicSystem(const std::string &subsection = "/ParabolicSystem");
    }; /* ParabolicSystem */


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    inline ParabolicSystem::ParabolicSystem(const std::string &subsection)
        : ParameterAcceptor(subsection)
    {
    }

  } // namespace EulerAEOS
} // namespace ryujin
