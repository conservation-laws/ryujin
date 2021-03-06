//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "initial_state.h"
#include "offline_data.h"
#include "problem_description.h"

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
  template <int dim, typename Number = double>
  class InitialValues : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc ProblemDescription::state_type
     */
    using state_type = ProblemDescription::state_type<dim, Number>;

    /**
     * @copydoc OfflineData::vector_type
     */
    using vector_type = typename OfflineData<dim, Number>::vector_type;

    /**
     * Constructor.
     */
    InitialValues(const ProblemDescription &problem_description,
                  const OfflineData<dim, Number> &offline_data,
                  const std::string &subsection = "InitialValues");


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
     * This routine computes and returns a state vector populated with
     * initial values for a specified time @p t.
     */
    vector_type interpolate(Number t = 0);

  private:
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

    dealii::SmartPointer<const ProblemDescription> problem_description_;
    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;

    std::set<std::unique_ptr<InitialState<dim, Number>>> initial_state_list_;

    std::function<state_type(const dealii::Point<dim> &point, Number t)>
        initial_state_;

    //@}
  };

} /* namespace ryujin */
