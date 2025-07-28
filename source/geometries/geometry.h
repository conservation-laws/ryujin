//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>

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
     * Create a triangulation representing the current Geometry. This
     * virtual method needs to be implemented in derived classes.
     */
    virtual void
    create_triangulation(dealii::Triangulation<dim> &triangulation) const = 0;

    /**
     * Set the correct active FE index for each active cell for the given
     * DoFHandler. This method can be left empty for a standard geometry
     * that only uses only one reference element. The method must be
     * reimplemented for geometries that use hp capabilities, such as
     * meshes with mixed finite elements, or meshes with FE_Nothing.
     */
    virtual void
    set_active_fe_index(dealii::DoFHandler<dim> & /*dof_handler*/) const
    {
    }

    /**
     * Return the name of the geometry as (const reference) std::string
     */
    ACCESSOR_READ_ONLY(name)

  private:
    const std::string name_;
  };

} /* namespace ryujin */
