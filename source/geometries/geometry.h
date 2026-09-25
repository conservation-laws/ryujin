//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>

#include <functional>
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
     * An enum class describing the hp collection
     */
    enum class HP_Collection {
      /*
       * Instruct the Discretization class to set up standard
       * continuous/discontinuous Q_k spaces for quarilaterals/hexahedra.
       */
      standard_quadrilaterals,
      /*
       * Instruct the Discretization class to set up standard
       * continuous/discontinuous P_k spaces for simplices.
       */
      standard_simplices,
      /*
       * Inform the Discretization class that the hp::*Collection objects
       * have already been populated by the Geometry class.
       */
      populated_by_geometry
    };

    /**
     * Populate all hp::*Collection objects for finite elements, mappings,
     * and quadratures. As this is a formidable zoo of different collection
     * objects, we get a writable reference to the discretization object to
     * set them directly.
     */
    virtual HP_Collection populate_hp_collections(
        const unsigned int /*fe_degree*/,
        typename ryujin::Discretization<dim>::Collection & /*collection*/) const
    {
      /*
       * Signal, that we did nothing. In this case the Discretization
       * object will populate all collections with appropriate objects for
       * the cG Qk, dG Qk finite element on purely quadrilateral, or
       * hexahedral meshes.
       */
      return HP_Collection::standard_quadrilaterals;
    }

    /**
     * Return the (optional) forward transformation that maps points of the
     * undeformed triangulation created by create_coarse_triangulation()
     * (and its refinements) to the actual geometry.
     *
     * If the function is nonempty the Discretization class refines the
     * triangulation without manifolds and realizes the geometry with a
     * MappingQCache whose support points are computed with this
     * transformation. If the function object is empty then a standard
     * MappinQ is used instead.
     */
    ACCESSOR_READ_ONLY(transformation)

    /**
     * Return the name of the geometry as (const reference) std::string
     */
    ACCESSOR_READ_ONLY(name)

  protected:
    /**
     * The transformation function object which should be set in
     * create_coarse_triangulation() of derived classes that need
     * MappingQCache to be set up.
     */
    mutable std::function<dealii::Point<dim>(
        const typename dealii::Triangulation<dim>::cell_iterator &,
        const dealii::Point<dim> &)>
        transformation_;

  private:
    const std::string name_;
  };

} /* namespace ryujin */
