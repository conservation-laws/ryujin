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

#include <fstream>

namespace grendel
{
  using namespace dealii;

  template <int dim>
  Discretization<dim>::Discretization(const MPI_Comm &mpi_communicator,
                                      const std::string &subsection)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
  {
    geometry_ = "cylinder";
    add_parameter("geometry",
                  geometry_,
                  "Geometry. Valid names are \"file\", \"triangle\", \"tube\", "
                  "\"tube analytical\", \"tube periodic\", \"step\", "
                  "\"cylinder\", or \"wall\".");

    grid_file_ = "wall.msh";
    add_parameter("grid file",
                  grid_file_,
                  "Grid file (in gmsh msh format) that is read in when "
                  "geometry is set to file");

    /* Immersed triangle: */

    immersed_triangle_length_ = 3.;
    add_parameter("immersed triangle - length",
                  immersed_triangle_length_,
                  "Immersed triangle: length of computational domain");

    immersed_triangle_height_ = 3.;
    add_parameter("immersed triangle - height",
                  immersed_triangle_height_,
                  "Immersed triangle: height of computational domain");

    immersed_triangle_object_height_ = 1.;
    add_parameter("immersed triangle - object height",
                  immersed_triangle_object_height_,
                  "Immersed triangle: height of immersed triangle");

    /* Shock tube: */

    tube_length_ = 1.;
    add_parameter("tube - length",
                  tube_length_,
                  "Shock tube: length of computational domain");

    tube_diameter_ = 1.;
    add_parameter("tube - diameter",
                  tube_diameter_,
                  "Shock tube: diameter of tube (ignored in 1D)");

    /* Mach step: */

    mach_step_length_ = 3.;
    add_parameter("mach step - length",
                  mach_step_length_,
                  "Mach step : length of computational domain");

    mach_step_height_ = 1.;
    add_parameter("mach step - height",
                  mach_step_height_,
                  "Mach step : height of computational domain");

    mach_step_step_position_ = 0.6;
    add_parameter("mach step - step position",
                  mach_step_step_position_,
                  "Mach step : position of step ");

    mach_step_step_height_ = 0.2;
    add_parameter("mach step - step height",
                  mach_step_step_height_,
                  "Mach step : height of step ");

    /* Immersed cylinder: */

    immersed_cylinder_length_ = 4.;
    add_parameter("immersed cylinder - length",
                  immersed_cylinder_length_,
                  "Immersed cylinder: length of computational domain");

    immersed_cylinder_height_ = 2.;
    add_parameter("immersed cylinder - height",
                  immersed_cylinder_height_,
                  "Immersed cylinder: height of computational domain");

    immersed_cylinder_object_position_ = 0.6;
    add_parameter(
        "immersed cylinder - object position",
        immersed_cylinder_object_position_,
        "Immersed cylinder: x position of immersed cylinder center point");

    immersed_cylinder_object_diameter_ = 0.5;
    add_parameter("immersed cylinder - object diameter",
                  immersed_cylinder_object_diameter_,
                  "Immersed cylinder: diameter of immersed cylinder");

    /* Wall: */

    wall_length_ = 3.2;
    add_parameter(
        "wall - length", wall_length_, "Wall: length of computational domain");

    wall_height_ = 1.0;
    add_parameter(
        "wall - height", wall_height_, "Wall: height of computational domain");

    wall_position_ = 1. / 6.;
    add_parameter(
        "wall - wall position", wall_position_, "Wall: x position of wall");

    /* Options: */

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
#ifdef DEBUG_OUTPUT
    std::cout << "Discretization<dim>::prepare()" << std::endl;
#endif

    if (!triangulation_)
      triangulation_.reset(new parallel::distributed::Triangulation<dim>(
          mpi_communicator_,
          Triangulation<dim>::limit_level_difference_at_vertices,
          dealii::parallel::distributed::Triangulation<
              dim>::construct_multigrid_hierarchy));

    auto &triangulation = *triangulation_;

    triangulation.clear();

    if (geometry_ == "file") {

      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);

#ifdef DEBUG_OUTPUT
      std::cout << "        reading in \"" << grid_file_ << "\"" << std::endl;
#endif

      std::ifstream file(grid_file_);
      grid_in.read_msh(file);

    } else if (geometry_ == "triangle") {

      create_coarse_grid_triangle(triangulation,
                                  immersed_triangle_length_,
                                  immersed_triangle_height_,
                                  immersed_triangle_object_height_);

    } else if (geometry_ == "tube") {

      create_coarse_grid_tube(triangulation,
                              tube_length_,
                              tube_diameter_,
                              /*prescribe*/ false,
                              /*periodic*/ false);

    } else if (geometry_ == "tube analytical") {

      create_coarse_grid_tube(triangulation,
                              tube_length_,
                              tube_diameter_,
                              /*prescribe*/ true,
                              /*periodic*/ false);

    } else if (geometry_ == "tube periodic") {

      create_coarse_grid_tube(triangulation,
                              tube_length_,
                              tube_diameter_,
                              /*prescribe*/ false,
                              /*periodic*/ true);

    } else if (geometry_ == "step") {

      AssertThrow(refinement_ >= 4,
                  dealii::ExcMessage("The mach step geometry requires at least "
                                     "4 levels of refinement"));

      create_coarse_grid_step(triangulation,
                              mach_step_length_,
                              mach_step_height_,
                              mach_step_step_position_,
                              mach_step_step_height_);

    } else if (geometry_ == "cylinder") {

      create_coarse_grid_cylinder(triangulation,
                                  immersed_cylinder_length_,
                                  immersed_cylinder_height_,
                                  immersed_cylinder_object_position_,
                                  immersed_cylinder_object_diameter_);

    } else if (geometry_ == "wall") {

      create_coarse_grid_wall(
          triangulation, wall_length_, wall_height_, wall_position_);

    } else {

      AssertThrow(false, dealii::ExcMessage("Unknown geometry name."));
    }

    /* Handle periodic faces: */

    const auto bdy_ids = triangulation.get_boundary_ids();
    if constexpr (dim != 1)
      if (std::find(bdy_ids.begin(), bdy_ids.end(), Boundary::periodic) !=
          bdy_ids.end()) {

#ifdef DEBUG_OUTPUT
        std::cout << "        collecting periodic faces" << std::endl;
#endif

        std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>>
            periodic_faces;

        for (int i = 1; i < dim; ++i) /* omit x direction! */
          GridTools::collect_periodic_faces(triangulation,
                                            /*b_id */ Boundary::periodic,
                                            /*direction*/ i,
                                            periodic_faces);

        triangulation.add_periodicity(periodic_faces);
      }


    if (geometry_ == "step") {
      triangulation.refine_global(refinement_ - 4);
    } else {
      triangulation.refine_global(refinement_);
    }

    mapping_.reset(new MappingQ<dim>(order_mapping_));

    finite_element_.reset(new FE_Q<dim>(order_finite_element_));

    quadrature_.reset(new QGauss<dim>(order_quadrature_));
  }

} /* namespace grendel */

#endif /* DISCRETIZATION_TEMPLATE_H */
