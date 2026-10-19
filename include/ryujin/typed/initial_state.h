//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/library.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

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
  template <typename HyperbolicDescription, int dim, typename Number = double>
  class InitialState : public dealii::ParameterAcceptor
  {
  public:
    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;
    using View = typename HyperbolicSystem::template View<dim, Number>;

    using state_type = typename View::state_type;

    using initial_precomputed_type = typename View::initial_precomputed_type;

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
     * value. In case of the @ref ShallowWaterEquations we pre-compute the
     * bathymetry. In case of @ref LinearTransport we precompute the
     * advection field.
     */
    virtual initial_precomputed_type
    initial_precomputations(const dealii::Point<dim> & /*point*/)
    {
      return initial_precomputed_type{};
    }

    /**
     * Return the name of the initial state as (const reference) std::string
     */
    ACCESSOR_READ_ONLY(name)

  private:
    const std::string name_;
  };


  /**
   * The library of initial states: a registry of all initial states that
   * InitialValues can select by name, for a given @p HyperbolicDescription,
   * dimension @p dim, and number type @p Number.
   *
   * @ingroup InitialValues
   */
  template <typename HyperbolicDescription, int dim, typename Number = double>
  using InitialStateLibrary =
      Library<InitialState<HyperbolicDescription, dim, Number>,
              const typename HyperbolicDescription::HyperbolicSystem &,
              const std::string &>;
} // namespace ryujin
