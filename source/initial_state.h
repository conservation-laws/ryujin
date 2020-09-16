//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef INITIAL_STATE_H
#define INITIAL_STATE_H

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "problem_description.h"

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
   * direction. The InitialValues wrapper class alread allows to apply an
   * affine translation to the coordinate system; so additional
   * configuration options for location and direction are not needed.
   *
   * @ingroup InitialValues
   */
  template <int dim, typename Number>
  class InitialState : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc ProblemDescription::rank1_type
     */
    using rank1_type = ProblemDescription::rank1_type<dim, Number>;

    /**
     * Constructor taking geometry name @p name and a subsection @p
     * subsection as an argument. The dealii::ParameterAcceptor is
     * initialized with the subsubsection `subsection + "/" + name`.
     */
    InitialState(const ProblemDescription &problem_description,
                 const std::string &name,
                 const std::string &subsection)
        : ParameterAcceptor(subsection + "/" + name)
        , problem_description(problem_description)
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
    virtual rank1_type compute(const dealii::Point<dim> &point, Number t) = 0;

  protected:
    const ProblemDescription &problem_description;

  private:
    const std::string name_;

    /**
     * Return the name of the geometry as (const reference) std::string
     */
    ACCESSOR_READ_ONLY(name)
  };


} /* namespace ryujin */

#endif /* INITIAL_STATE_H */
