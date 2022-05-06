//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

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
  template <int dim,
            typename Number,
            typename state_type,
            int n_precomputed_values = 0>
  class InitialState : public dealii::ParameterAcceptor
  {
  public:
    using PrecomputedValues = std::array<Number, n_precomputed_values>;

    /**
     * Constructor taking geometry name @p name and a subsection @p
     * subsection as an argument. The dealii::ParameterAcceptor is
     * initialized with the subsubsection `subsection + "/" + name`.
     */
    InitialState(const std::string &name,
                 const std::string &subsection)
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
    virtual PrecomputedValues
    compute_flux_contributions(const dealii::Point<dim> & /*point*/)
    {
      return PrecomputedValues();
    }

  private:
    const std::string name_;

    /**
     * Return the name of the geometry as (const reference) std::string
     */
    ACCESSOR_READ_ONLY(name)
  };

} /* namespace ryujin */
