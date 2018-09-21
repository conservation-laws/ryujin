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
    static constexpr unsigned int problem_dimension = 2 + dim;
    typedef dealii::Tensor<1, problem_dimension, double> rank1_type;

    Model(const std::string &subsection = "Model");
    virtual ~Model() final = default;

    void parse_parameters_callback();

    static constexpr auto lambda =
        [](rank1_type, rank1_type, dealii::Tensor<1, dim>) { return 0.; };

  private:
  };

} /* namespace grendel */

#endif /* Model_H */
