//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <set>
#include <string>

namespace ryujin
{
  /**
   * A small abstract base class to group configuration options for a
   * number of initial flow configurations.
   *
   * @note By convention all initial state configurations described by this
   * class shall be centered at the origin (0, 0) and facing in positive x
   * direction. The InitialValues wrapper class already allows to apply an
   * affine translation to the coordinate system; so additional
   * configuration options for location and direction are not needed.
   *
   * @ingroup InitialValues
   */
  template <typename Description, int dim, typename Number = double>
  class InitialState : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystemView
     */
    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    using state_type = typename View::state_type;
    using precomputed_state_type = typename View::precomputed_state_type;

    /**
     * Constructor taking initial state name @p name and a subsection @p
     * subsection as an argument. The dealii::ParameterAcceptor is
     * initialized with the subsubsection `subsection + "/" + name`.
     */
    InitialState(const std::string &name, const std::string &subsection)
        : ParameterAcceptor(subsection + "/" + name)
        , name_(name)
    {
    }

    /**
     * Given a position @p point returns the corresponding (conserved)
     * initial state. The function is used to interpolate initial values
     * and enforce Dirichlet boundary conditions. For the latter, the
     * function signature has an additional parameter @p t denoting the
     * current time to allow for time-dependent (in-flow) Dirichlet data.
     */
    virtual state_type compute(const dealii::Point<dim> &point, Number t) = 0;

    /**
     * Given a position @p point returns a precomputed value used for the
     * flux computation via HyperbolicSystem::flux_contribution().
     *
     * The default implementation of this function simply returns a zero
     * value. In case of the @ref ShallowWaterEquations we precompute the
     * bathymetry. In case of @ref LinearTransport we precompute the
     * advection field.
     */
    virtual precomputed_state_type
    initial_precomputations(const dealii::Point<dim> & /*point*/)
    {
      return precomputed_state_type();
    }

    /**
     * Return the name of the initial state as (const reference) std::string
     */
    ACCESSOR_READ_ONLY(name)

  private:
    const std::string name_;
  };


  /**
   * A "factory" class that is used to populate a list of all possible
   * initial states for a given equation desribed by @p Description.
   *
   * This works by specializing the static member function
   * populate_initial_state_list for all possible equation @p Description.
   *
   * @ingroup InitialValues
   */
  template <typename Description, int dim, typename Number>
  class InitialStateLibrary
  {
  public:
    /**
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    /**
     * The type of the initial state list
     */
    using initial_state_list_type =
        std::set<std::unique_ptr<InitialState<Description, dim, Number>>>;

    /**
     * Populate a given container with all initial states defined for the
     * given equation @p Description and dimension @p dim.
     *
     * @ingroup InitialValues
     */
    static void
    populate_initial_state_list(initial_state_list_type &initial_state_list,
                                const HyperbolicSystem &h,
                                const std::string &s);
  };
} // namespace ryujin
