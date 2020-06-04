//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <compile_time_options.h>

#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/lac/la_parallel_vector.templates.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <future>

namespace ryujin
{

  template <int dim, typename Number = double>
  class Postprocessor final : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;
    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, Number>;

    using vector_type = dealii::LinearAlgebra::distributed::Vector<Number>;

    /**
     * The number of postprocessed quantities:
     */
    static constexpr unsigned int n_quantities = (dim == 1) ? 2 : 3;

    /**
     * An array holding all component names as a string.
     */
    const static std::array<std::string, n_quantities> component_names;

    Postprocessor(const MPI_Comm &mpi_communicator,
                  const ryujin::OfflineData<dim, Number> &offline_data,
                  const std::string &subsection = "Postprocessor");

    void prepare();

    void schedule_output(const vector_type &U,
                         const vector_type &alpha,
                         std::string name,
                         Number t,
                         unsigned int cycle,
                         bool output_full = true,
                         bool output_cutplanes = true);

    bool is_active();
    void wait();

  protected:
    std::array<vector_type, n_quantities> quantities_;

  private:
    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;

    std::future<void> background_thread_status;

    /* Options: */

    bool use_mpi_io_;
    ACCESSOR_READ_ONLY(use_mpi_io)

    Number schlieren_beta_;
    Number vorticity_beta_;

    using plane_description = std::tuple<dealii::Point<dim> /*origin*/,
                                         dealii::Tensor<1, dim> /*normal*/,
                                         double /*tolerance*/>;
    std::vector<plane_description> output_planes_;
  };

} /* namespace ryujin */

#endif /* POSTPROCESSOR_H */
