//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <convenience_macros.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  /**
   * The parabolic subsystem of the compressible Euler equations. This is
   * just the idenity operator.
   *
   * @ingroup ParabolicModule
   */
  class StubParabolicSystem final : public dealii::ParameterAcceptor
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
    StubParabolicSystem(const std::string &subsection = "/ParabolicSystem");

    ACCESSOR_READ_ONLY(parabolic_component_names);

  private:
    const std::vector<std::string> parabolic_component_names_;
  }; /* StubParabolicSystem */


  /*
   * -------------------------------------------------------------------------
   * Inline definitions
   * -------------------------------------------------------------------------
   */


  inline StubParabolicSystem::StubParabolicSystem(const std::string &subsection)
      : ParameterAcceptor(subsection)
  {
  }
} // namespace ryujin
