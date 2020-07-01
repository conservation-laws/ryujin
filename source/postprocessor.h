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

  /**
   * @todo Write documentation.
   *
   * @ingroup TimeLoop
   */
  template <int dim, typename Number = double>
  class Postprocessor final : public dealii::ParameterAcceptor
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    /**
     * @copydoc ProblemDescription::rank1_type
     */
    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    /**
     * @todo Write documentation.
     */
    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, Number>;

    /**
     * Shorthand typedef for
     * dealii::LinearAlgebra::distributed::Vector<Number>.
     */
    using scalar_type = dealii::LinearAlgebra::distributed::Vector<Number>;

    /**
     * Shorthand typedef for a MultiComponentVector storing the current
     * simulation state.
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * The number of postprocessed quantities:
     */
    static constexpr unsigned int n_quantities = (dim == 1) ? 2 : 3;

    /**
     * An array of strings for all component names.
     */
    const static std::array<std::string, n_quantities> component_names;

    /**
     * Constructor.
     */
    Postprocessor(const MPI_Comm &mpi_communicator,
                  const ryujin::OfflineData<dim, Number> &offline_data,
                  const std::string &subsection = "Postprocessor");

    /**
     * Prepare postprocessor. A call to @ref prepare() allocates temporary
     * storage and is necessary before schedule_output() can be called.
     */
    void prepare();

    /**
     * @todo Write documentation
     */
    void schedule_output(const vector_type &U,
                         const scalar_type &alpha,
                         std::string name,
                         Number t,
                         unsigned int cycle,
                         bool output_full = true,
                         bool output_cutplanes = true);

    /**
     * @todo Write documentation
     */
    bool is_active();

    /**
     * @todo Write documentation
     */
    void wait();

  protected:
    std::array<scalar_type, n_quantities> quantities_;

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
