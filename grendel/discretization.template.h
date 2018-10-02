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
    geometry_ = "immersed triangle";
    add_parameter("geometry",
                  geometry_,
                  "Geometry. Valid names are \"immersed triangle\", \"tube\".");

    immersed_triangle_length_ = 3.;
    add_parameter("immersed triangle - length",
                  immersed_triangle_length_,
                  "Immersed Triangle: length of computational domain");

    immersed_triangle_height_ = 3.;
    add_parameter("immersed triangle - height",
                  immersed_triangle_height_,
                  "Immersed Triangle: height of computational domain");

    immersed_triangle_object_height_ = 1.;
    add_parameter("immersed triangle - object height",
                  immersed_triangle_object_height_,
                  "Immersed Triangle: height of immersed triangle");

    tube_length_ = 1.;
    add_parameter("tube - length",
                  tube_length_,
                  "Shock tube: length of computational domain");

    tube_diameter_ = 1.;
    add_parameter("tube - diameter",
                  tube_diameter_,
                  "Shock tube: diameter of tube (ignored in 1D)");

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
  void Discretization<dim>::prepare()
  {
    deallog << "Discretization<dim>::prepare()" << std::endl;
    TimerOutput::Scope t(computing_timer_,
                         "discretization - prepare");

    if (!triangulation_)
      triangulation_.reset(
          new parallel::distributed::Triangulation<dim>(mpi_communicator_));

    auto &triangulation = *triangulation_;
    triangulation.clear();

    if (geometry_ == "immersed triangle") {
      create_coarse_grid_triangle(triangulation,
                                  immersed_triangle_length_,
                                  immersed_triangle_height_,
                                  immersed_triangle_object_height_);

    } else if (geometry_ == "tube") {

      create_coarse_grid_tube(triangulation, tube_length_, tube_diameter_);

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
