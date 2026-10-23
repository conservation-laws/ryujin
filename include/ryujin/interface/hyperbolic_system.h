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
     * @ingroup Interfaces
     */
    class HyperbolicSystem : public dealii::ParameterAcceptor
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
          : ParameterAcceptor(subsection)
          , problem_name_(problem_name)
          , state_names_(state_names)
          , precomputed_names_(precomputed_names)
          , primitive_names_(primitive_names)
          , initial_precomputed_names_(initial_precomputed_names)
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
