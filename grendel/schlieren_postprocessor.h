#ifndef SCHLIEREN_POSTPROCESSOR_H
#define SCHLIEREN_POSTPROCESSOR_H

#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.templates.h>

namespace grendel
{

  template <int dim, typename Number = double>
  class SchlierenPostprocessor : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    using vector_type =
        std::array<dealii::LinearAlgebra::distributed::Vector<Number>,
                   problem_dimension>;

    SchlierenPostprocessor(
        const MPI_Comm &mpi_communicator,
        dealii::TimerOutput &computing_timer,
        const grendel::OfflineData<dim, Number> &offline_data,
        const std::string &subsection = "SchlierenPostprocessor");

    virtual ~SchlierenPostprocessor() final = default;

    void prepare();

    void compute_schlieren(const vector_type &U);

  protected:
    dealii::LinearAlgebra::distributed::Vector<Number> schlieren_;
    ACCESSOR_READ_ONLY(schlieren)

  private:
    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim, Number>> offline_data_;

    /* Scratch data: */

    dealii::Vector<Number> r_i_;

    /* Options: */

    unsigned int schlieren_index_;
    Number schlieren_beta_;
  };

} /* namespace grendel */

#endif /* SCHLIEREN_POSTPROCESSOR_H */
