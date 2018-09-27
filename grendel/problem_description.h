#ifndef PROBLEM_DESCRIPTION_H
#define PROBLEM_DESCRIPTION_H

#include "boilerplate.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <functional>

namespace grendel
{
  /*
   *
   * We have a (2 + n) dimensional state space [rho, m_1, ..., m_n, E],
   * where rho denotes the pressure, [m_1, ..., m_n] is the momentum vector
   * field, and E is the total Energy.
   */
  template <int dim>
  class ProblemDescription : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension = 2 + dim;
    typedef dealii::Tensor<1, problem_dimension, double> rank1_type;

    ProblemDescription(const std::string &subsection = "ProblemDescription");
    virtual ~ProblemDescription() final = default;

  private:
    double gamma_;
    A_RO(gamma)

    double b_;
    A_RO(b)
  };

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_H */
