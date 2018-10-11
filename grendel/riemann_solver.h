#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include "helper.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <functional>

namespace grendel
{

  /**
   * A fast approximative Riemann problem solver for the nD compressible
   * Euler problem.
   *
   * FIXME: Desciption
   */
  template <int dim>
  class RiemannSolver : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim>::rank1_type;

    RiemannSolver(const grendel::ProblemDescription<dim> &problem_description,
                  const std::string &subsection = "RiemannSolver");

    virtual ~RiemannSolver() final = default;


    /**
     * FIXME: Description
     *
     * For two given states U_i a U_j and a (normalized) "direction" n_ij
     * compute an estimation of an upper bound for lambda.
     *
     * See [1], page 915, Algorithm 1
     *
     * Returns a tuple consisting of lambda max and the number of Newton
     * iterations used in the solver to find it.
     *
     * References:
     *   [1] J.-L. Guermond, B. Popov. Fast estimation from above fo the
     *       maximum wave speed in the Riemann problem for the Euler equations.
     */
    std::tuple<double /*lambda_max*/,
               double /*p_star*/,
               unsigned int /*iteration*/>
    compute(const rank1_type &U_i,
            const rank1_type &U_j,
            const dealii::Tensor<1, dim> &n_ij) const;


    /**
     * Variant of above function that takes two arrays as input describing
     * the Riemann data instead of two nD states.
     */
    std::tuple<double /*lambda_max*/,
               double /*p_star*/,
               unsigned int /*iteration*/>
    compute(const std::array<double, 6> &riemann_data_i,
            const std::array<double, 6> &riemann_data_j) const;


  protected:
    dealii::SmartPointer<const grendel::ProblemDescription<dim>>
        problem_description_;
    ACCESSOR_READ_ONLY(problem_description)

  private:
    double eps_;
    unsigned int max_iter_;
  };

} /* namespace grendel */

#endif /* RIEMANN_SOLVER_H */
