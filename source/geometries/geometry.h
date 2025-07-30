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
     * Create a coarse triangulation representing the current Geometry.
     * This virtual method needs to be implemented in a derived classes.
     */
    virtual void create_coarse_triangulation(
        dealii::Triangulation<dim> &triangulation) const = 0;

    /**
     * This method is called before we distribute dofs and can be used to
     * set the correct active FE index for each active cell for the given
     * DoFHandler, or update material, or manifold ids, etc.
     *
     * This method can be left empty for a standard geometry
     * that only uses only one reference element. The method must be
     * reimplemented for geometries that use hp capabilities, such as
     * meshes with mixed finite elements, or meshes with FE_Nothing.
     */
    virtual void
    update_dof_handler(dealii::DoFHandler<dim> & /*dof_handler*/) const
    {
    }

    /**
     * Populate all hp::*Collection objects for finite elements, mappings,
     * and quadratures. As this is a formidable zoo of different collection
     * objects, we get a writable reference to the discretization object to
     * set them directly.
     */
    virtual bool populate_hp_collections(
        const unsigned int /*fe_degree*/,
        const bool /*have_discontinuous_ansatz*/,
        typename ryujin::Discretization<dim>::Collection & /*collection*/) const
    {
      /*
       * Signal, that we did nothing. In this case the Discretization
       * object will populate all collections with appropriate objects for
       * the cG Qk, dG Qk finite element on purely quadrilateral, or
       * hexahedral meshes.
       */
      return false;
    }

    /**
     * Return the name of the geometry as (const reference) std::string
     */
    ACCESSOR_READ_ONLY(name)

  private:
    const std::string name_;
  };

} /* namespace ryujin */
