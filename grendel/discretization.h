#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

#include "boilerplate.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/tria.h>

namespace grendel
{
  template <int dim>
  class Discretization : public dealii::ParameterAcceptor
  {
  public:
    Discretization(const std::string &subsection = "Discretization");
    virtual ~Discretization() final = default;

    A_RO(triangulation)
    A_RO(mapping)
    A_RO(finite_element)
    A_RO(quadrature)

    void parse_parameters_callback();

  protected:

    unsigned int refinement_;

    unsigned int order_finite_element_;
    unsigned int order_mapping_;
    unsigned int order_quadrature_;

    std::unique_ptr<dealii::Triangulation<dim>> triangulation_;
    std::unique_ptr<const dealii::Mapping<dim>> mapping_;
    std::unique_ptr<const dealii::FiniteElement<dim>> finite_element_;
    std::unique_ptr<const dealii::Quadrature<dim>> quadrature_;
  };

} /* namespace grendel */

#endif /* DISCRETIZATION_H */
