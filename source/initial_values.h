//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "initial_state_library.h"
#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace ryujin
{
  /**
   * A class implementing a number of different initial value
   * configurations.
   *
   * Given a position @p point the member function
   * InitialValues::initial_state() returns the corresponding (conserved)
   * initial state. The function is used to interpolate initial values and
   * enforce Dirichlet boundary conditions. For the latter, the the
   * function signature has an additional parameter @p t denoting the
   * current time to allow for time-dependent (in-flow) Dirichlet data.
   *
   * For validation purposes a number of analytic solutions are implemented
   * as well.
   *
   * @ingroup InitialValues
   */
  template <typename Description, int dim, typename Number = double>
  class InitialValues : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using state_type = typename View::state_type;

    static constexpr auto n_initial_precomputed_values =
        View::n_initial_precomputed_values;

    using initial_precomputed_type = typename View::initial_precomputed_type;

    using StateVector = typename View::StateVector;

    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;

    //@}
    /**
     * @name Interpolate initial states
     */
    //@{

    /**
     * Constructor.
     */
    InitialValues(const HyperbolicSystem &hyperbolic_system,
                  const OfflineData<dim, Number> &offline_data,
                  const std::string &subsection = "/InitialValues");


    /**
     * Callback for ParameterAcceptor::initialize(). After we read in
     * configuration parameters from the parameter file we have to do some
     * (minor) preparatory work in this class to precompute some initial
     * state values. Do this with a callback.
     */
    void parse_parameters_callback();


    /**
     * Given a position @p point returns the corresponding (conserved)
     * initial state. The function is used to interpolate initial values
     * and enforce Dirichlet boundary conditions. For the latter, the the
     * function signature has an additional parameter @p t denoting the
     * current time to allow for time-dependent (in-flow) Dirichlet data.
     */
    DEAL_II_ALWAYS_INLINE inline state_type
    initial_state(const dealii::Point<dim> &point, Number t) const
    {
      return initial_state_(point, t);
    }


    /**
     * Given a position @p point returns the corresponding (conserved)
     * initial state. The function is used to interpolate initial values
     * and enforce Dirichlet boundary conditions. For the latter, the the
     * function signature has an additional parameter @p t denoting the
     * current time to allow for time-dependent (in-flow) Dirichlet data.
     */
    DEAL_II_ALWAYS_INLINE inline initial_precomputed_type
    initial_precomputed(const dealii::Point<dim> &point) const
    {
      return initial_precomputed_(point);
    }


    /**
     * This routine computes and returns a state vector populated with
     * initial values for a specified time @p t.
     */
    StateVector interpolate_state_vector(Number t = 0) const;


    /**
     * This routine computes and returns a state vector populated with
     * initial values for a specified time @p t.
     */
    InitialPrecomputedVector interpolate_initial_precomputed_vector() const;

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    std::string configuration_;

    dealii::Point<dim> initial_position_;

    dealii::Tensor<1, dim> initial_direction_;

    Number perturbation_;

    //@}
    /**
     * @name Internal data:
     */
    //@{

    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;

    typename InitialStateLibrary<Description, dim, Number>::
        initial_state_list_type initial_state_list_;

    std::function<state_type(const dealii::Point<dim> &, Number)>
        initial_state_;

    std::function<initial_precomputed_type(const dealii::Point<dim> &)>
        initial_precomputed_;

    //@}
  };

} /* namespace ryujin */
