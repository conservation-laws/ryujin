//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <convenience_macros.h>

#include <deal.II/base/parameter_acceptor.h>

#include <string>
#include <vector>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The description-free part of a parabolic system: its name and the
     * names of its components.
     *
     * All components of a parabolic system are dynamic: they are stored as
     * scalar vectors in StateVector::V (state) and StateVector::W
     * (precomputed values).
     *
     * @ingroup ParabolicModule
     */
    class ParabolicSystem : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the parabolic system as a string.
       */
      ACCESSOR_READ_ONLY(problem_name)

      /**
       * The names of the state components for dimension @p dim. State
       * components are evolved by the time integrator.
       *
       * @note Valid once the run time parameters are parsed.
       */
      virtual std::vector<std::string>
      state_names(const unsigned int dim) const = 0;

      /**
       * The names of the precomputed values for dimension @p dim.
       * Precomputed values are derived from the state.
       *
       * @note Valid once the run time parameters are parsed.
       */
      virtual std::vector<std::string>
      precomputed_names(const unsigned int dim) const = 0;

    protected:
      /**
       * Constructor.
       */
      ParabolicSystem(const std::string &problem_name,
                      const std::string &subsection)
          : ParameterAcceptor(subsection)
          , problem_name_(problem_name)
      {
      }

      /**
       * The name of the parabolic system.
       *
       * @note A derived class may update it once the run time parameters
       * are known.
       */
      std::string problem_name_;
    };
  } // namespace Interface
} // namespace ryujin
