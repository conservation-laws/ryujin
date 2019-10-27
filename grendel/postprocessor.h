#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.templates.h>

namespace grendel
{

  template <int dim, typename Number = double>
  class Postprocessor : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    using vector_type =
        std::array<dealii::LinearAlgebra::distributed::Vector<Number>,
                   problem_dimension>;

    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, Number>;

    Postprocessor(
        const MPI_Comm &mpi_communicator,
        dealii::TimerOutput &computing_timer,
        const grendel::OfflineData<dim, Number> &offline_data,
        const std::string &subsection = "Postprocessor");

    virtual ~Postprocessor() final = default;

    void prepare();

    void compute(const vector_type &U);

  protected:
    dealii::LinearAlgebra::distributed::Vector<Number> schlieren_;
    ACCESSOR_READ_ONLY(schlieren)

    std::array<dealii::LinearAlgebra::distributed::Vector<Number>,
               dim == 2 ? 1 : dim>
        vorticity_;
    ACCESSOR_READ_ONLY(vorticity)

  private:
    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim, Number>> offline_data_;

    /* Options: */

    Number schlieren_beta_;
  };

} /* namespace grendel */

#endif /* POSTPROCESSOR_H */
