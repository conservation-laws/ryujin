//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    /**
     * The parabolic subsystem. This is just the identity operator.
     *
     * @ingroup ScalarConservation
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

  } // namespace ScalarConservation
} // namespace ryujin
