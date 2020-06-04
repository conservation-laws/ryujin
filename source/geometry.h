//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/parameter_acceptor.h>

#include <string>

namespace ryujin
{
#ifndef DOXYGEN
  /* forward declaration */
  template <int dim>
  class Discretization;
#endif

  /**
   * A small abstract base class to group configuration options for a
   * number of geometries together.
   *
   * @ingroup Discretization
   */
  template <int dim>
  class Geometry : public dealii::ParameterAcceptor
  {
  public:
    /**
     * A typdef for the deal.II triangulation that is used by this class.
     * Inherited from Discretization.
     */
    using Triangulation = typename Discretization<dim>::Triangulation;

    /**
     * Constructor that initializes the dealii::ParameterAcceptor object
     * with the subsection @p subsection concatenated with `+ "/" +
     * name()`.
     */
    Geometry(std::string &subsection)
        : ParameterAcceptor(subsection + "/" + name())
    {
    }

    /**
     * Return the name of the geometry as std::string.
     */
    virtual const std::string &name() = 0;

    /**
     * Create the triangulation according to the appropriate geometry
     * description.
     */
    virtual void create_triangulation(Triangulation &triangulation) = 0;
  };

  /**
   *
   */
  namespace Geometries
  {
  }

} /* namespace ryujin */

#endif /* GEOMETRY_H */
