//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <ryujin/interface/system.h>

#include <deal.II/base/exceptions.h>

#include <string>
#include <vector>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The description-free part of a hyperbolic system: its name and the
     * names of its components.
     *
     * The first View<dim>::problem_dimension state names and the first
     * View<dim>::n_precomputed_values precomputed names are static: they
     * are stored interleaved in StateVector::state and
     * StateVector::precomputed. All remaining names are dynamic: they are
     * stored as scalar vectors in StateVector::dynamic_state and
     * StateVector::dynamic_precomputed.
     *
     * The primitive names (View<dim>::problem_dimension entries) name the
     * primitive state returned by View::to_primitive_state(), and the
     * initial precomputed names (View<dim>::n_initial_precomputed_values
     * entries) name the time-independent initial precomputed values.
     *
     * @ingroup HyperbolicModule
     */
    class HyperbolicSystem : public System
    {
    public:
      /**
       * The names of the primitive state components for dimension @p dim.
       *
       * @note Valid once the run time parameters are parsed.
       */
      const std::vector<std::string> &
      primitive_names(const unsigned int dim) const
      {
        AssertIndexRange(dim - 1, 3);
        return primitive_names_[dim - 1];
      }

      /**
       * The names of the initial precomputed values for dimension @p dim.
       *
       * Initial precomputed values are time-independent and stored outside
       * of the state vector.
       *
       * @note Valid once the run time parameters are parsed.
       */
      const std::vector<std::string> &
      initial_precomputed_names(const unsigned int dim) const
      {
        AssertIndexRange(dim - 1, 3);
        return initial_precomputed_names_[dim - 1];
      }

    protected:
      /**
       * Constructor.
       */
      HyperbolicSystem(const std::string &problem_name,
                       const Names &state_names,
                       const Names &primitive_names,
                       const Names &precomputed_names,
                       const Names &initial_precomputed_names,
                       const std::string &subsection)
          : System(problem_name, state_names, precomputed_names, subsection)
          , primitive_names_(primitive_names)
          , initial_precomputed_names_(initial_precomputed_names)
      {
      }

      /**
       * The names of the primitive state components for every dimension.
       *
       * @note A derived class may update them once the run time
       * parameters are known.
       */
      Names primitive_names_;

      /**
       * The names of the initial precomputed values for every dimension.
       *
       * @note A derived class may update them once the run time
       * parameters are known.
       */
      Names initial_precomputed_names_;
    };
  } // namespace Interface
} // namespace ryujin
