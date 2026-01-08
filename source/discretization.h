//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "geometries/geometry.h"
#include "mpi_ensemble.h"
#include "patterns_conversion.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <memory>
#include <set>

namespace ryujin
{
  /**
   * An enum of type dealii::types::boundary_id that provides an mnemonic
   * for prescribing different boundary conditions on faces.
   *
   * @note In deal.II boundary ids are prescribed on faces. However, in our
   * stencil-based method we need such an information for individual
   * boundary degrees of freedom. Thus, the face boundary indicator has to
   * be translated to individual degrees of freedom which happens in
   * OfflineData::prepare() when constructing the
   * OfflineData::boundary_map_ object.
   *
   * @note OfflineData::boundary_map_ is a std::vector that stores all
   * encountered boundary conditions for an individual degree of freedom.
   * The individual algebraic constraint is applied in no particular order.
   * It is thus important to ensure that neighboring boundary conditions,
   * are compatible. For example, inflow conditions prescribed via a
   * Boundary::dirichlet face neighboring a Boundary::no_slip face have to
   * ensure that they prescribe a state compatible with the no slip
   * condition, etc.
   *
   * @note Data structures in OfflineData are initialized with the ensemble
   * subrange communicator stored in MPIEnsemble.
   *
   * @ingroup Mesh
   */
  enum Boundary : dealii::types::boundary_id {
    /**
     * The "do nothing" outflow boundary condition: no special treatment of
     * the boundary degree of freedom. For stability reasons it is
     * important to ensure that this boundary id is only prescribed on
     * degrees of freedom with a velocity vector pointing outward of the
     * computational domain <b>and</b> coming from the interior of the
     * domain.
     */
    do_nothing = 0,

    /**
     * Prescribe periodic boundary conditions by identifying opposing
     * degrees of freedom. This currently requires a mesh with "standard
     * orientation".
     */
    periodic = 1,

    /**
     * On (free) slip boundary degrees of freedom we enforce a vanishing
     * normal component of the momentum in the Euler module. This is done
     * by explicitly removing the normal component of the momentum for the
     * degree of freedom at the end of TimeStep::euler_step(). In the
     * dissipation module \f$v\cdot n\f$ is enforced strongly which leads
     * to a natural boundary condition on the symmetric stress tensor:
     * \f$\tau\cdot\mathbb{S}(v)\cdot\vec n\f$.
     */
    slip = 2,

    /**
     * On no-slip boundary degrees of freedom we enforce a vanishing normal
     * component of the momentum in the Euler module. This is done by
     * explicitly removing the normal component of the momentum for the
     * degree of freedom at the end of TimeStep::euler_step(). In the
     * dissipation module a vanishing velocity \f$v=0\f$ is enforced
     * strongly.
     */
    no_slip = 3,

    /**
     * On degrees of freedom marked as Dirichlet boundary we reset the
     * state of the degree of freedom to the value of
     * InitialData::initial_state(). Such Dirichlet conditions can only be
     * meaningfully enforced as inflow conditions, i.e., the velocity
     * vector associated with a Dirichlet boundary degree of freedom has to
     * point into the computational domain, and no "backward traveling"
     * shock front or other flow feature must reach a Dirichlet boundary
     * degree of freedom during the computation.
     */
    dirichlet = 4,

    /**
     * On degrees of freedom marked as a "dynamic" boundary we distinguish
     * four cases (for the compressible Euler equations or related PDEs):
     *  - supersonic inflow, where we reset the state of a boundary degree
     *    of freedom to the value returned by InitialData::initial_state().
     *    This is equivalent to "dirichlet" boundary conditions.
     *  - supersonic outflow, where we do nothing, similarly to the "do
     *    nothing" boundary conditions.
     *  - in case of subsonic in-, or outflow the state of a boundary
     *    degree of freedom is translated into "Riemann characteristics"
     *    and the values of all incoming characteristics are replaced by
     *    the corresponding value of the state returned by
     *    InitialData::initial_state().
     */
    dynamic = 5,

    /**
     * For the Shallow Water Equations: On degrees of freedom marked as
     * "dirichlet momentum" boundary, we reset only the momentum of the
     * degree of freedom to the value of InitialData::initial_state(). Such
     * conditions are used in many steady state problems with inflow
     * conditions.
     */
    dirichlet_momentum = 6
  };


  /**
   * An enum class for setting the finite element ansatz.
   *
   * @ingroup Mesh
   */
  enum class Ansatz {
    /** cG Q1: continuous bi- (tri-) linear Lagrange elements */
    cg_q1,

    /** cG Q2: continuous bi- (tri-) quadratic Lagrange elements */
    cg_q2,

    /** cG Q3: continuous bi- (tri-) cubic Lagrange elements */
    cg_q3,

    /** dG Q1: discontinuous bi- (tri-) linear Lagrange elements */
    dg_q1,

    /** dG Q2: discontinuous bi- (tri-) quadratic Lagrange elements */
    dg_q2,

    /** dG Q3: discontinuous bi- (tri-) cubic Lagrange elements */
    dg_q3
  };

  /**
   * An enum class for setting the type of Triangulation that should be
   * constructed.
   *
   * @ingroup Mesh
   */
  enum class MeshType {
    /** Use serial dealii::Triangulation<dim> */
    serial,
    /** Use parallel dealii::parallel::shared::Triangulation<dim> */
    parallel_shared,
    /** Use parallel dealii::parallel::distributed::Triangulation<dim> */
    parallel_distributed,
    /** Use parallel dealii::parallel::fullydistributed::Triangulation<dim> */
    parallel_fullydistributed
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::Boundary,
             LIST({ryujin::Boundary::do_nothing, "do nothing"},
                  {ryujin::Boundary::periodic, "periodic"},
                  {ryujin::Boundary::slip, "slip"},
                  {ryujin::Boundary::no_slip, "no slip"},
                  {ryujin::Boundary::dirichlet, "dirichlet"},
                  {ryujin::Boundary::dynamic, "dynamic"},
                  {ryujin::Boundary::dirichlet_momentum,
                   "dirichlet momentum"}));

DECLARE_ENUM(ryujin::Ansatz,
             LIST({ryujin::Ansatz::cg_q1, "cG Q1"},
                  {ryujin::Ansatz::cg_q2, "cG Q2"},
                  {ryujin::Ansatz::cg_q3, "cG Q3"},
                  {ryujin::Ansatz::dg_q1, "dG Q1"},
                  {ryujin::Ansatz::dg_q2, "dG Q2"},
                  {ryujin::Ansatz::dg_q3, "dG Q3"}));

DECLARE_ENUM(ryujin::MeshType,
             LIST({ryujin::MeshType::serial, "serial"},
                  {ryujin::MeshType::parallel_shared, "parallel shared"},
                  {ryujin::MeshType::parallel_distributed,
                   "parallel distributed"},
                  {ryujin::MeshType::parallel_fullydistributed,
                   "parallel fullydistributed"}));
#endif

namespace ryujin
{
  /**
   * This class is as a container for data related to the discretization,
   * this includes the triangulation, finite element, mapping, and
   * quadrature. After prepare() is called, the getter functions
   * Discretization::triangulation(), Discretization::finite_element(),
   * Discretization::mapping(), and Discretization::quadrature() return
   * valid const references to the mentioned objects.
   *
   * The class uses dealii::ParameterAcceptor to handle a multitude of
   * parameters to control the creation of meshes for a variety of
   * benchmark configurations and to read in meshes in one of the formats
   * supported by the deal.II library.
   *
   * @ingroup Mesh
   */
  template <int dim>
  class Discretization : public dealii::ParameterAcceptor
  {
  public:
    /**
     * Constructor.
     */
    Discretization(const MPIEnsemble &mpi_ensemble,
                   const std::string &subsection = "/Discretization");

    /**
     * Create the triangulation and set up the finite element, mapping and
     * quadrature objects.
     */
    void prepare(const std::string &base_name);

    /**
     * A collection of mappings, finite elements, and quadratures that are
     * set up by the Discretization class. We create a dedicated struct
     * with all unique_ptr to keep the interface to
     * Geometry::populate_hp_collections() sane.
     */
    struct Collection {
      std::unique_ptr<const dealii::hp::MappingCollection<dim>> mapping;
      std::unique_ptr<const dealii::hp::FECollection<dim>> finite_element_cg;
      std::unique_ptr<const dealii::hp::FECollection<dim>> finite_element_dg;
      std::unique_ptr<const dealii::hp::QCollection<dim>> quadrature;
      std::unique_ptr<const dealii::hp::QCollection<dim>> quadrature_high_order;
      std::unique_ptr<const dealii::hp::QCollection<dim>> nodal_quadrature;
      std::unique_ptr<const dealii::hp::QCollection<1>> quadrature_1d;
      std::unique_ptr<const dealii::hp::QCollection<1>> nodal_quadrature_1d;
      std::unique_ptr<const dealii::hp::QCollection<dim - 1>> face_quadrature;
      std::unique_ptr<const dealii::hp::QCollection<dim - 1>>
          face_nodal_quadrature;
    };

    /**
     * @name Accessors to data structures managed by this class.
     */
    //@{

    /**
     * Return a read-only const reference to the selected geometry.
     */
    ACCESSOR_READ_ONLY(selected_geometry)

    /**
     * Return a read-only const reference to the finite element ansatz.
     */
    ACCESSOR_READ_ONLY(ansatz)

    /**
     * Return a boolean indicating whether the chosen Ansatz space is
     * discontinuous.
     */
    bool have_discontinuous_ansatz() const
    {
      switch (ansatz_) {
        /* Continuous Ansatz: */
      case Ansatz::cg_q1:
        [[fallthrough]];
      case Ansatz::cg_q2:
        [[fallthrough]];
      case Ansatz::cg_q3:
        return false;

        /* Discontinuous Ansatz: */
      case Ansatz::dg_q1:
        [[fallthrough]];
      case Ansatz::dg_q2:
        [[fallthrough]];
      case Ansatz::dg_q3:
        return true;
      }

      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    /**
     * Return the polynomial degree of the chosen finite element ansatz.
     */
    unsigned int polynomial_degree() const
    {
      switch (ansatz_) {
      case Ansatz::cg_q1:
        [[fallthrough]];
      case Ansatz::dg_q1:
        return 1;
      case Ansatz::cg_q2:
        [[fallthrough]];
      case Ansatz::dg_q2:
        return 2;
      case Ansatz::cg_q3:
        [[fallthrough]];
      case Ansatz::dg_q3:
        return 3;
      }

      AssertThrow(false, dealii::ExcInternalError());
      __builtin_trap();
    }

    /**
     * Return a mutable reference to the refinement variable.
     */
    ACCESSOR(refinement)

    /**
     * Return a mutable reference to the triangulation.
     */
    ACCESSOR(triangulation)

    /**
     * Return a read-only const reference to the triangulation.
     */
    ACCESSOR_READ_ONLY(triangulation)
    /**
     * Return a read-only const reference to the mapping.
     *
     * @note The accessor returns an MappingCollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, mapping)

    /**
     * Return a read-only const reference to a continuous ("cG") variant of
     * the selected finite element space.
     *
     * @note If the selected finite element space is continuous then this
     * method simply returns the same object as finite_element().
     *
     * @note The accessor returns an FECollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, finite_element_cg)

    /**
     * Return a read-only const reference to a discontinuous ("dG") variant
     * of the selected finite element space.
     *
     * @note If the selected finite element space is discontinuous then
     * this method simply returns the same object as finite_element().
     *
     * @note The accessor returns an FECollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, finite_element_dg)

    /**
     * Return a read-only const reference to the selected finite element.
     *
     * @note The accessor returns an FECollection object.
     */
    const dealii::hp::FECollection<dim> &finite_element() const
    {
      if (have_discontinuous_ansatz())
        return *collection_.finite_element_dg;
      else
        return *collection_.finite_element_cg;
    }

    /**
     * Return a read-only const reference to the quadrature rule.
     *
     * @note The accessor returns an QCollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, quadrature)

    /**
     * Return a read-only const reference to a highe order quadrature rule
     * used for computing errors.
     *
     * @note The accessor returns an QCollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, quadrature_high_order)

    /**
     * Return a read-only const reference to the nodal quadrature rule
     * (Gauß Lobatto).
     *
     * @note The accessor returns an QCollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, nodal_quadrature)

    /**
     * Return a read-only const reference to the 1D quadrature rule.
     *
     * @note The accessor returns an QCollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, quadrature_1d)

    /**
     * Return a read-only const reference to the 1D nodal quadrature rule
     * (Gauß Lobatto).
     *
     * @note The accessor returns an QCollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, nodal_quadrature_1d)

    /**
     * Return a read-only const reference to the face quadrature rule.
     *
     * @note The accessor returns an QCollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, face_quadrature)

    /**
     * Return a read-only const reference to the nodal face quadrature rule
     * (Gauß Lobatto).
     *
     * @note The accessor returns an QCollection object.
     */
    ACCESSOR_CONTAINER_READ_ONLY(collection_, face_nodal_quadrature)

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    Ansatz ansatz_;
    MeshType mesh_type_;

    std::string geometry_;

    unsigned int refinement_;

    bool mesh_writeout_;
    double mesh_distortion_;

    //@}
    /**
     * @name Internal data:
     */
    //@{
    //
    const MPIEnsemble &mpi_ensemble_;

    std::unique_ptr<dealii::Triangulation<dim>> triangulation_;

    Collection collection_;

    std::set<std::shared_ptr<Geometry<dim>>> geometry_list_;
    std::shared_ptr<Geometry<dim>> selected_geometry_;

    //@}

    /**
     * In the SolutionTransfer class we need writable access to the
     * triangulation object in order to prepare data for mesh adaptation
     * and checkpointing / restart. Work around this issue by declaring the
     * solution transfer class to be a friend rather than changing the
     * constructor, or augmenting the methods in SolutionTransfer.
     */
    template <typename Discretization, int dim_, typename Number_>
    friend class SolutionTransfer;

    /**
     * For complex geometries with mixed finite elements (or when using
     * FE_Nothing) we need to defer the setup of the hp::*Collection
     * objects to the selected geometry. Thus, declare the Geometry class
     * to be a friend so that it can set all the collection objects
     * directly.
     */
    template <int dim_>
    friend class Geometry;
  };
} /* namespace ryujin */
