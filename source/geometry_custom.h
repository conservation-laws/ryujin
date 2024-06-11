#pragma once

#include "geometry_common_includes.h"

namespace ryujin
{
  namespace GridIn
  {
    /**
     * Import 2D Mesh from GMSH (.msh).
     *
     * @ingroup Mesh
     */
    template <int dim, template <int, int> class Triangulation>
    void custom(Triangulation<dim, dim> &)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
    }
    #ifndef DOXYGEN
        template <template <int, int> class Triangulation>
        void custom(Triangulation<2, 2> &triangulation)
        {
            dealii::GridIn<2, 2> gridin;
            gridin.attach_triangulation(triangulation);
            std::ifstream f("/prm/meshes/nozzle_jet.msh");
            gridin.read_msh(f);
        }
    /* Boundary IDs are set in .geo mesh file */
    #endif
  }

  namespace Geometries
  {
    template <int dim>
    class Custom: public Geometry<dim>
    {
        public:
        Custom(const std::string subsection)
            : Geometry<dim>("custom", subsection)
            {}
        void create_triangulation(
            typename Geometry<dim>::Triangulation &triangulation) final
            {
                GridIn::custom(triangulation);
            }

    };
  }
}