#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

#include "boilerplate.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

namespace grendel
{
  template <int dim>
  class Discretization : public dealii::ParameterAcceptor
  {
  public:
    Discretization(const MPI_Comm &mpi_communicator,
                   dealii::TimerOutput &computing_timer,
                   const std::string &subsection = "Discretization");
    virtual ~Discretization() final = default;

    void create_triangulation();

  protected:

    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    std::string geometry_;
    double length_;
    double height_;
    double object_height_;

    unsigned int refinement_;

    unsigned int order_finite_element_;
    unsigned int order_mapping_;
    unsigned int order_quadrature_;

    std::unique_ptr<dealii::parallel::distributed::Triangulation<dim>>
        triangulation_;
    A_RO(triangulation)

    std::unique_ptr<const dealii::Mapping<dim>> mapping_;
    A_RO(mapping)

    std::unique_ptr<const dealii::FiniteElement<dim>> finite_element_;
    A_RO(finite_element)

    std::unique_ptr<const dealii::Quadrature<dim>> quadrature_;
    A_RO(quadrature)
  };

} /* namespace grendel */

#endif /* DISCRETIZATION_H */
