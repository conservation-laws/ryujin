//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>

#include <array>
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
     * scalar vectors in StateVector::dynamic_state and
     * StateVector::dynamic_precomputed.
     *
     * @ingroup Interfaces
     */
    class ParabolicSystem : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the system as a string.
       */
      ACCESSOR_READ_ONLY(problem_name)

      /**
       * A list of component names for every dimension:
       */
      using Names = std::array<std::vector<std::string>, 3>;

      /**
       * The names of the state components for dimension @p dim.
       *
       * @note Valid once the run time parameters are parsed.
       */
      const std::vector<std::string> &state_names(const unsigned int dim) const
      {
        AssertIndexRange(dim - 1, 3);
        return state_names_[dim - 1];
      }

      /**
       * The names of the precomputed values for dimension @p dim.
       *
       * @note Valid once the run time parameters are parsed.
       */
      const std::vector<std::string> &
      precomputed_names(const unsigned int dim) const
      {
        AssertIndexRange(dim - 1, 3);
        return precomputed_names_[dim - 1];
      }

      /**
       * Whether the parabolic system is the identity, i.e., whether the
       * equation is purely hyperbolic.
       */
      ACCESSOR_READ_ONLY(is_identity)

    protected:
      /**
       * Constructor.
       */
      ParabolicSystem(const std::string &problem_name,
                      const Names &state_names,
                      const Names &precomputed_names,
                      const std::string &subsection)
          : ParameterAcceptor(subsection)
          , problem_name_(problem_name)
          , state_names_(state_names)
          , precomputed_names_(precomputed_names)
          , is_identity_(false)
      {
      }

      /**
       * The name of the system.
       *
       * @note A derived class may update it once the run time parameters
       * are known.
       */
      std::string problem_name_;

      /**
       * The names of the state components for every dimension.
       *
       * @note A derived class may update them once the run time
       * parameters are known.
       */
      Names state_names_;

      /**
       * The names of the precomputed values for every dimension.
       *
       * @note A derived class may update them once the run time
       * parameters are known.
       */
      Names precomputed_names_;

      /**
       * Whether the parabolic system is the identity.
       *
       * @note A derived class that models the identity sets it in its
       * constructor.
       */
      bool is_identity_;
    };
  } // namespace Interface
} // namespace ryujin
