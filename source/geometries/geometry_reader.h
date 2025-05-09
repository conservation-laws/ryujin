#pragma once

#include "geometry_common_includes.h"

#include <deal.II/grid/grid_in.h>

#include <fstream>

namespace ryujin
{
  namespace Geometries
  {
    /**
     * This class imports a triangulation from various supported mesh files
     * via the dealii::GridIn reader. See
     * https://www.dealii.org/current/doxygen/deal.II/classGridIn.html
     * for more details on supported file types and extensions.
     *
     * @note The mesh format must support setting boundary IDs in the mesh
     * file. Supported boundary IDs and their meaning are collected in the
     * Boundary enum.
     *
     * @ingroup Mesh
     */
    template <int dim>
    class Reader : public Geometry<dim>
    {
    public:
      Reader(const std::string subsection)
          : Geometry<dim>("reader", subsection)
      {
        filename_ = "ryujin.msh";
        this->add_parameter("filename",
                            filename_,
                            "The mesh file to read in via dealii::GridIn. This "
                            "class supports, among others, reading in Gmsh "
                            "*.msh files, and the *.ucd file format.");
      }

      void create_triangulation(
          typename Geometry<dim>::Triangulation &triangulation) final
      {
        dealii::GridIn<dim> gridin;
        gridin.attach_triangulation(triangulation);
        gridin.read(filename_);
      }

    private:
      std::string filename_;
    };
  } // namespace Geometries
} // namespace ryujin
