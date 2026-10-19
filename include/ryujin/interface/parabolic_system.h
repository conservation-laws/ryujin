//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/interface/system.h>

#include <string>

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
     * @ingroup ParabolicModule
     */
    class ParabolicSystem : public System
    {
    public:
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
          : System(problem_name, state_names, precomputed_names, subsection)
          , is_identity_(false)
      {
      }

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
