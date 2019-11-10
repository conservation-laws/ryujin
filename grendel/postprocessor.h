#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <compile_time_options.h>
#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/intergrid_map.h>
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

    using scalar_type = dealii::LinearAlgebra::distributed::Vector<Number>;
    using vector_type = std::array<scalar_type, problem_dimension>;

    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, Number>;

    /**
     * The number of postprocessed quantities:
     */
    static constexpr unsigned int n_quantities =
        (dim == 1) ? 2 : (dim == 2) ? 3 : 5;

    /**
     * An array holding all component names as a string.
     */
    const static std::array<std::string, n_quantities> component_names;

    Postprocessor(const MPI_Comm &mpi_communicator,
                  dealii::TimerOutput &computing_timer,
                  const grendel::OfflineData<dim, Number> &offline_data,
                  const std::string &subsection = "Postprocessor");

    virtual ~Postprocessor() final = default;

    void prepare();

    void compute(const vector_type &U, const scalar_type &alpha);

    void write_out_vtu(std::string name, Number t, unsigned int cycle);

  protected:

    std::array<scalar_type, problem_dimension> U_;
    ACCESSOR_READ_ONLY(U)

    std::array<scalar_type, n_quantities> quantities_;
    ACCESSOR_READ_ONLY(quantities)

    std::array<dealii::Vector<Number>, problem_dimension> output_U_;
    ACCESSOR_READ_ONLY(output_U)

    std::array<dealii::Vector<Number>, n_quantities> output_quantities_;
    ACCESSOR_READ_ONLY(output_quantities)

  private:
    const MPI_Comm &mpi_communicator_;
    dealii::TimerOutput &computing_timer_;

    dealii::SmartPointer<const grendel::OfflineData<dim, Number>> offline_data_;

    /* Options: */

    Number schlieren_beta_;

    unsigned int coarsening_level_;
  };

} /* namespace grendel */

#endif /* POSTPROCESSOR_H */
