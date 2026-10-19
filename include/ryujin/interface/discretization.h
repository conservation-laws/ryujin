//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/patterns_conversion.h>
#include <ryujin/geometries/geometry.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <memory>
#include <string>
#include <vector>

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
     * For the Euler and shallow Water equations: On degrees of freedom
     * marked as "dirichlet momentum" boundary, we reset only the momentum
     * of the degree of freedom to the value of
     * InitialData::initial_state(). Where appropriate, we keep water
     * height h, density rho, and internal energy e of the boundary state.
     * Such conditions are used, for example, in many steady state problems
     * for the shallow water equations with inflow conditions.
     */
    dirichlet_momentum = 6,

    /**
     * For the Euler and shallow Water equations: On degrees of freedom
     * marked as "dirichlet velocity" boundary, we reset only the velocity
     * of the degree of freedom to the value of
     * InitialData::initial_state(). Where appropriate, we keep water
     * height h, density rho, and internal energy e of the boundary state.
     */
    dirichlet_velocity = 7
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
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::Boundary,
             LIST({ryujin::Boundary::do_nothing, "do nothing"},
                  {ryujin::Boundary::periodic, "periodic"},
                  {ryujin::Boundary::slip, "slip"},
                  {ryujin::Boundary::no_slip, "no slip"},
                  {ryujin::Boundary::dirichlet, "dirichlet"},
                  {ryujin::Boundary::dynamic, "dynamic"},
                  {ryujin::Boundary::dirichlet_momentum, "dirichlet momentum"},
                  {ryujin::Boundary::dirichlet_velocity,
                   "dirichlet velocity"}));

DECLARE_ENUM(ryujin::Ansatz,
             LIST({ryujin::Ansatz::cg_q1, "cG Q1"},
                  {ryujin::Ansatz::cg_q2, "cG Q2"},
                  {ryujin::Ansatz::cg_q3, "cG Q3"},
                  {ryujin::Ansatz::dg_q1, "dG Q1"},
                  {ryujin::Ansatz::dg_q2, "dG Q2"},
                  {ryujin::Ansatz::dg_q3, "dG Q3"}));
#endif

namespace ryujin
{
  namespace Interface
  {
    /**
     * The interface of a Discretization: a container for data related to
     * the discretization, this includes the triangulation, finite element,
     * mapping, and quadrature. After prepare() is called, the getter
     * functions triangulation(), finite_element(), mapping(), and
     * quadrature() return valid const references to the mentioned
     * objects.
     *
     * The interface holds all state in protected fields with a
     * non-virtual accessor each. A derived class fills the fields in
     * prepare() and update_mapping().
     *
     * @ingroup Mesh
     */
    template <int dim>
    class Discretization : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

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
        std::unique_ptr<const dealii::hp::QCollection<dim>>
            quadrature_high_order;
        std::unique_ptr<const dealii::hp::QCollection<dim>> nodal_quadrature;
        std::unique_ptr<const dealii::hp::QCollection<1>> quadrature_1d;
        std::unique_ptr<const dealii::hp::QCollection<1>> nodal_quadrature_1d;
        std::unique_ptr<const std::vector<dealii::hp::QCollection<dim - 1>>>
            face_quadrature;
        std::unique_ptr<const std::vector<dealii::hp::QCollection<dim - 1>>>
            face_nodal_quadrature;
      };

      //@}
      /**
       * @name Setup
       */
      //@{

      /**
       * Create the triangulation and set up the finite element, mapping
       * and quadrature objects.
       */
      virtual void prepare(const std::string &base_name) = 0;

      /**
       * Recompute the cached mapping after the triangulation has changed.
       */
      virtual void update_mapping() = 0;

      //@}
      /**
       * @name Information and statistics
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
       * Return a read-only const reference to a continuous ("cG") variant
       * of the selected finite element space.
       *
       * @note If the selected finite element space is continuous then this
       * method simply returns the same object as finite_element().
       *
       * @note The accessor returns an FECollection object.
       */
      ACCESSOR_CONTAINER_READ_ONLY(collection_, finite_element_cg)

      /**
       * Return a read-only const reference to a discontinuous ("dG")
       * variant of the selected finite element space.
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
       * Return a read-only const reference to a higher order quadrature
       * rule used for computing errors.
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
       * Return a read-only const reference to the face quadrature rules.
       *
       * @note The accessor returns a vector of QCollection objects, one
       * per finite element of the collection, that holds one quadrature per
       * face of the reference cell (or a single quadrature used on all
       * faces).
       */
      ACCESSOR_CONTAINER_READ_ONLY(collection_, face_quadrature)

      /**
       * Return a read-only const reference to the nodal face quadrature
       * rules (Gauß Lobatto).
       *
       * @note The accessor returns a vector of QCollection objects, see
       * face_quadrature().
       */
      ACCESSOR_CONTAINER_READ_ONLY(collection_, face_nodal_quadrature)

      //@}

    protected:
      /**
       * Constructor.
       */
      explicit Discretization(const std::string &subsection)
          : ParameterAcceptor(subsection)
          , ansatz_(Ansatz::cg_q1)
          , refinement_(0)
      {
      }

      /**
       * @name Internal data
       */
      //@{

      Ansatz ansatz_;

      unsigned int refinement_;

      std::unique_ptr<dealii::Triangulation<dim>> triangulation_;

      Collection collection_;
      std::shared_ptr<dealii::MappingQCache<dim>> mapping_cache_;

      std::shared_ptr<Geometry<dim>> selected_geometry_;

      //@}
    };
  } // namespace Interface
} // namespace ryujin
