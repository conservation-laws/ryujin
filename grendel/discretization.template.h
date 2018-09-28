#ifndef DISCRETIZATION_TEMPLATE_H
#define DISCRETIZATION_TEMPLATE_H

#include "discretization.h"
#include "geometry_helper.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

namespace grendel
{
  using namespace dealii;

  template <int dim>
  Discretization<dim>::Discretization(const MPI_Comm &mpi_communicator,
                                      dealii::TimerOutput &computing_timer,
                                      const std::string &subsection)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
  {
    geometry_ = "shard";
    add_parameter(
        "geometry", geometry_, "Geometry. Valid names are \"shard\".");

    length_ = 3.;
    add_parameter("geometry length",
                  length_,
                  "Length of geometry (interpretation depends on geometry)");

    height_ = 3.;
    add_parameter("geometry height",
                  height_,
                  "Height of geometry (interpretation depends on geometry)");

    object_height_ = 1.;
    add_parameter(
        "object height",
        object_height_,
        "Height of immersed object (interpretation depends on geometry)");

    refinement_ = 5;
    add_parameter("initial refinement",
                  refinement_,
                  "Initial refinement of the geometry");

    order_mapping_ = 1;
    add_parameter("order mapping", order_mapping_, "Order of the mapping");

    order_finite_element_ = 1;
    add_parameter("order finite element",
                  order_finite_element_,
                  "Polynomial order of the finite element space");

    order_quadrature_ = 3;
    add_parameter(
        "order quadrature", order_quadrature_, "Order of the quadrature rule");
  }


  template <int dim>
  void Discretization<dim>::create_triangulation()
  {
    deallog << "Discretization<dim>::create_triangulation()" << std::endl;
    TimerOutput::Scope t(computing_timer_,
                         "discretization - create_triangulation");

    if (!triangulation_)
      triangulation_.reset(
          new parallel::distributed::Triangulation<dim>(mpi_communicator_));

    auto &triangulation = *triangulation_;
    triangulation.clear();

    if (geometry_ == "shard") {
      create_coarse_grid_shard(triangulation, length_, height_, object_height_);
    } else {
      AssertThrow(false, dealii::ExcMessage("Unknown geometry name."));
    }

    triangulation.refine_global(refinement_);

    mapping_.reset(new MappingQ<dim>(order_mapping_));

    finite_element_.reset(new FE_Q<dim>(order_finite_element_));

    quadrature_.reset(new QGauss<dim>(order_quadrature_));
  }

} /* namespace grendel */

#endif /* DISCRETIZATION_TEMPLATE_H */
