#ifndef Model_H
#define Model_H

#include "boilerplate.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace grendel
{
  template <int dim>
  class Model : public dealii::ParameterAcceptor
  {
  public:
    Model(const std::string &subsection = "Model");
    virtual ~Model() final = default;

    void parse_parameters_callback();

    //     std::function<double(const dealii::Point<dim> &,
    //                          const dealii::types::material_id &)>
    //         epsilon;
  };

} /* namespace grendel */

#endif /* Model_H */
