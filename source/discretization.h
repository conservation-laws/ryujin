//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "geometry.h"
#include "patterns_conversion.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

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
   * @note OfflineData::boundary_map_ is a std::multimap that stores all
   * encountered boundary conditions for an individual degree of freedom.
   * The individual algebraic constraint is applied in no particular order.
   * It is thus important to ensure that neighboring boundary conditions,
   * are compatible. For example, inflow conditions prescribed via a
   * Boundary::dirichlet face neighboring a Boundary::no_slip face have to
   * ensure that they prescribe a state compatible with the no slip
   * condition, etc.
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
     * point into the computational domain, and no "backward travelling"
     * shock front or other flow feature must reach a Dirichlet boundary
     * degree of freedom during the computation.
     */
    dirichlet = 4,

    /**
     * On degrees of freedom marked as a "dynamic" Dirichlet boundary we
     * reset the state of the degree of freedom to the value of
     * InitialData::initial_state() if the state of the degree of freedom
     * is inflow. Otherwise we do nothing.
     */
    dynamic = 5,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::Boundary,
             LIST({ryujin::Boundary::do_nothing, "do_nothing"},
                  {ryujin::Boundary::periodic, "periodic"},
                  {ryujin::Boundary::slip, "slip"},
                  {ryujin::Boundary::no_slip, "no_slip"},
                  {ryujin::Boundary::dirichlet, "dirichlet"},
                  {ryujin::Boundary::dynamic, "dynamic"}));
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

#ifndef DOXYGEN
  /* forward declaration */
  template <int dim>
  class Discretization;
#endif

  /**
   * A templated constexpr boolean that is true if we use a parallel
   * distributed triangulation (for the specified dimension).
   */
  template <int dim>
  constexpr bool have_distributed_triangulation =
      std::is_same<typename Discretization<dim>::Triangulation,
                   dealii::parallel::distributed::Triangulation<dim>>::value;


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

    static_assert(dim == 1 || have_distributed_triangulation<dim>);

    /**
     * Constructor.
     */
    Discretization(const MPI_Comm &mpi_communicator,
                   const std::string &subsection = "/Discretization");

    /**
     * Create the triangulation and set up the finite element, mapping and
     * quadrature objects.
     */
    void prepare();

    /**
     * @name Discretization compile time options
     */
    //@{

    static constexpr unsigned int order_finite_element = ORDER_FINITE_ELEMENT;
    static constexpr unsigned int order_mapping = ORDER_MAPPING;
    static constexpr unsigned int order_quadrature = ORDER_QUADRATURE;

    //@}
    /**
     * @name
     */
    //@{

  protected:
    const MPI_Comm &mpi_communicator_;

    std::unique_ptr<Triangulation> triangulation_;
    std::unique_ptr<const dealii::Mapping<dim>> mapping_;
    std::unique_ptr<const dealii::FiniteElement<dim>> finite_element_;
    std::unique_ptr<const dealii::Quadrature<dim>> quadrature_;
    std::unique_ptr<const dealii::Quadrature<1>> quadrature_1d_;

  public:
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
     * Return a read-only const reference to the quadrature rule.
     */
    ACCESSOR_READ_ONLY(quadrature)

    /**
     * Return a read-only const reference to the 1D quadrature rule.
     */
    ACCESSOR_READ_ONLY(quadrature_1d)

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    std::string geometry_;

    double mesh_distortion_;

    unsigned int refinement_;

    bool repartitioning_;

    //@}
    /**
     * @name Internal data:
     */
    //@{

    std::set<std::unique_ptr<Geometry<dim>>> geometry_list_;

    //@}
  };
} /* namespace ryujin */
