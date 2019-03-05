#ifndef SCHLIEREN_POSTPROCESSOR_H
#define SCHLIEREN_POSTPROCESSOR_H

#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_vector.templates.h>

namespace grendel
{

  template <int dim>
  class SchlierenPostprocessor : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim>::rank1_type;

    using vector_type =
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,
                   problem_dimension>;

    SchlierenPostprocessor(const MPI_Comm &mpi_communicator,
             dealii::TimerOutput &computing_timer,
             const grendel::OfflineData<dim> &offline_data,
             const std::string &subsection = "SchlierenPostprocessor");

    virtual ~SchlierenPostprocessor() final = default;

    void prepare();

    void compute_schlieren(const vector_type &U);

  protected:
    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim>> offline_data_;
    ACCESSOR_READ_ONLY(offline_data)

  private:
    /* Scratch data: */

    dealii::LinearAlgebra::distributed::Vector<double> schlieren_;
    ACCESSOR_READ_ONLY(schlieren)

    /* Options: */

    unsigned int schlieren_index_;
    double schlieren_beta_;
  };

} /* namespace grendel */

#endif /* SCHLIEREN_POSTPROCESSOR_H */
