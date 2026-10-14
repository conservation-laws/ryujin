//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <convenience_macros.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>

#include <iterator>
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
     * are stored interleaved in StateVector::U and
     * StateVector::precomputed and have to equal the names declared in
     * the hyperbolic system view. All remaining names are dynamic: they
     * are stored as scalar vectors in StateVector::V (state) and
     * StateVector::W (precomputed values).
     *
     * @ingroup HyperbolicModule
     */
    class HyperbolicSystem : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the hyperbolic system as a string.
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
      HyperbolicSystem(const std::string &problem_name,
                       const std::string &subsection)
          : ParameterAcceptor(subsection)
          , problem_name_(problem_name)
      {
      }

      /**
       * The name of the hyperbolic system.
       *
       * @note A derived class may update it once the run time parameters
       * are known.
       */
      std::string problem_name_;

      /**
       * A helper for a derived class: return the names of the static state
       * components declared in the hyperbolic system view @p View for
       * dimension @p dim.
       */
      template <template <int, typename...> typename View>
      static std::vector<std::string> view_state_names(const unsigned int dim)
      {
        switch (dim) {
        case 1:
          return to_vector(View<1>::component_names);
        case 2:
          return to_vector(View<2>::component_names);
        case 3:
          return to_vector(View<3>::component_names);
        }
        AssertThrow(false, dealii::ExcNotImplemented());
        return {};
      }

      /**
       * A helper for a derived class: return the names of the static
       * precomputed values declared in the hyperbolic system view @p View
       * for dimension @p dim.
       */
      template <template <int, typename...> typename View>
      static std::vector<std::string>
      view_precomputed_names(const unsigned int dim)
      {
        switch (dim) {
        case 1:
          return to_vector(View<1>::precomputed_names);
        case 2:
          return to_vector(View<2>::precomputed_names);
        case 3:
          return to_vector(View<3>::precomputed_names);
        }
        AssertThrow(false, dealii::ExcNotImplemented());
        return {};
      }

    private:
      template <typename Array>
      static std::vector<std::string> to_vector(const Array &names)
      {
        return std::vector<std::string>(std::begin(names), std::end(names));
      }
    };
  } // namespace Interface
} // namespace ryujin
