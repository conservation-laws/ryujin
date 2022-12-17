//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

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
   * @ingroup Mesh
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
     * Constructor taking geometry name @p name and a subsection @p
     * subsection as an argument. The dealii::ParameterAcceptor is
     * initialized with the subsubsection `subsection + "/" + name`.
     */
    Geometry(const std::string &name, const std::string &subsection)
        : ParameterAcceptor(subsection + "/" + name)
        , name_(name)
    {
    }

    /**
     * Create the triangulation according to the appropriate geometry
     * description.
     */
    virtual void create_triangulation(Triangulation &triangulation) = 0;

    /**
     * Return the name of the geometry as (const reference) std::string
     */
    ACCESSOR_READ_ONLY(name)

  private:
    const std::string name_;
  };

} /* namespace ryujin */
