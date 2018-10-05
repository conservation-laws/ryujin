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
  /**
   * This class serves as a container for data related to the
   * discretization.This includes the triangulation, finite element,
   * mapping, and quadrature.
   *
   * This class uses dealii::ParameterAcceptor to handle parameters. After
   * @p prepare() was called, the getter functions
   * Discretization::triangulation(), Discretization::finite_element(),
   * Discretization::mapping(), and Discretization::quadrature() return
   * valid const references.
   */
  template <int dim>
  class Discretization : public dealii::ParameterAcceptor
  {
  public:

    /**
     * Constructor.
     */
    Discretization(const MPI_Comm &mpi_communicator,
                   dealii::TimerOutput &computing_timer,
                   const std::string &subsection = "Discretization");

    /**
     * Destructor. We prevent the creation of derived classes by declaring
     * the destructor to be <code>final</code>.
     */
    virtual ~Discretization() final = default;

    /**
     * Create the triangulation and set up the finite element, mapping and
     * quadrature objects.
     */
    void prepare();

  protected:

    /**
     * The triangulation and a getter function returning a const reference.
     */
    std::unique_ptr<dealii::parallel::distributed::Triangulation<dim>>
        triangulation_;
    ACCESSOR_READ_ONLY(triangulation)

    /**
     * The mapping and a getter function returning a const reference.
     */
    std::unique_ptr<const dealii::Mapping<dim>> mapping_;
    ACCESSOR_READ_ONLY(mapping)

    /**
     * The underlying finite element space.
     */
    std::unique_ptr<const dealii::FiniteElement<dim>> finite_element_;
    ACCESSOR_READ_ONLY(finite_element)

    /*
     * The quadrature used to for assembly.
     */
    std::unique_ptr<const dealii::Quadrature<dim>> quadrature_;
    ACCESSOR_READ_ONLY(quadrature)

  private:

    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    /*
     * Configuration data used to create the triangulation and set up
     * finite element, mapping and quadrature.
     */

    std::string geometry_;
    double immersed_triangle_length_;
    double immersed_triangle_height_;
    double immersed_triangle_object_height_;
    double tube_length_;
    double tube_diameter_;

    unsigned int refinement_;

    unsigned int order_finite_element_;
    unsigned int order_mapping_;
    unsigned int order_quadrature_;
  };

} /* namespace grendel */

#endif /* DISCRETIZATION_H */
