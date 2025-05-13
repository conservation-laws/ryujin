//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "geometries/geometry.h"
#include "mpi_ensemble.h"
#include "patterns_conversion.h"

#include <deal.II/base/parameter_acceptor.h>
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
#endif

namespace ryujin
{
  namespace
  {
    template <int dim>
    struct Proxy {
      using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    };

    template <>
    struct Proxy<1> {
      using Triangulation = dealii::parallel::shared::Triangulation<1>;
    };

  } // namespace


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
     * A type alias denoting the Triangulation we are using:
     *
     * In one spatial dimensions we use a
     * dealii::parallel::shared::Triangulation and for two and three
     * dimensions a dealii::parallel::distributed::Triangulation.
     */
    using Triangulation = typename Proxy<dim>::Triangulation;

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
     * @name Accessors to data structures managed by this class.
     */
    //@{

    /**
     * Return a read-only const reference to the finite element ansatz.
     */
    ACCESSOR_READ_ONLY(ansatz)

  public:
    /**
     * Return a boolean indicating  whether the chosen Ansatz space is
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
     */
    ACCESSOR_READ_ONLY(mapping)

    /**
     * Return a read-only const reference to the finite element.
     */
    ACCESSOR_READ_ONLY(finite_element)

    /**
     * Return a read-only const reference to a continuous ("cG") variant of
     * the selected discontinuous finite element space.
     *
     * @note This object is unavailable for the dG Q0 discretization.
     */
    ACCESSOR_READ_ONLY(finite_element_cg)

    /**
     * Return a read-only const reference to the quadrature rule.
     */
    ACCESSOR_READ_ONLY(quadrature)

    /**
     * Return a read-only const reference to the nodal quadrature rule
     * (Gauß Lobatto).
     */
    ACCESSOR_READ_ONLY(nodal_quadrature)

    /**
     * Return a read-only const reference to the 1D quadrature rule.
     */
    ACCESSOR_READ_ONLY(quadrature_1d)

    /**
     * Return a read-only const reference to the face quadrature rule.
     */
    ACCESSOR_READ_ONLY(face_quadrature)

    /**
     * Return a read-only const reference to the nodal face quadrature rule
     * (Gauß Lobatto).
     */
    ACCESSOR_READ_ONLY(face_nodal_quadrature)

  protected:
    const MPIEnsemble &mpi_ensemble_;

    std::unique_ptr<Triangulation> triangulation_;
    std::unique_ptr<const dealii::hp::MappingCollection<dim>> mapping_;
    std::unique_ptr<const dealii::hp::FECollection<dim>> finite_element_;
    std::unique_ptr<const dealii::hp::FECollection<dim>> finite_element_cg_;
    std::unique_ptr<const dealii::hp::QCollection<dim>> quadrature_;
    std::unique_ptr<const dealii::hp::QCollection<dim>> nodal_quadrature_;
    std::unique_ptr<const dealii::hp::QCollection<1>> quadrature_1d_;
    std::unique_ptr<const dealii::hp::QCollection<dim - 1>> face_quadrature_;
    std::unique_ptr<const dealii::hp::QCollection<dim - 1>>
        face_nodal_quadrature_;

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    Ansatz ansatz_;

    std::string geometry_;

    unsigned int refinement_;

    bool mesh_writeout_;
    double mesh_distortion_;

    //@}
    /**
     * @name Internal data:
     */
    //@{

    std::set<std::unique_ptr<Geometry<dim>>> geometry_list_;

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
  };


  /**
   * A templated constexpr boolean that is true if we use a parallel
   * distributed triangulation (for the specified dimension).
   */
  template <int dim>
  constexpr bool have_distributed_triangulation =
      std::is_same<typename Discretization<dim>::Triangulation,
                   dealii::parallel::distributed::Triangulation<dim>>::value;
} /* namespace ryujin */
