//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "offline_data.h"
#include "problem_description.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <future>

namespace ryujin
{

  /**
   * the VTUOutput class implements a number of postprocessing
   * primitives in particular for a scaled and normalized Schlieren-like
   * plot, and a scaled and normalized magnitude of the vorticity. The
   * normalization is computed as follows:
   * \f[
   *   \text{quantity}[i] = \exp\left(-\beta \frac{ |\mathbf q_i| - \min_k
   * |\mathbf q_k|}
   *   {\max_k |\mathbf q_k| - \min_k |\mathbf q_k|}\right),
   * \f]
   * where \f$\mathbf q_i\f$ is either
   *  - the gradient of the density postprocessed as follows,
   *    \f[
   *       \mathbf q_i =  \frac{1}{m_i}\;\sum_{j\in \mathcal{J}(i)}
   * \mathbf{c}_{ij} \rho_j; \f]
   *  - the vorticity of the velocity field postprocessed as follows,
   *    \f[
   *       \mathbf q_i =  \frac{1}{m_i}\;\sum_{j\in \mathcal{J}(i)}
   * \mathbf{c}_{ij} \times \mathbf{m}_j / \rho_j. \f]
   *
   * In addition, the generated VTU output also contains the full state
   * vector, and a local estimate of the effective residual viscosity
   * \f$\mu_{\text{res}}\f$ caused by the graph viscosity stabilization.
   *
   * @ingroup TimeLoop
   */
  template <int dim, typename Number = double>
  class VTUOutput final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc ProblemDescription::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = ProblemDescription::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc ProblemDescription::state_type
     */
    using state_type = ProblemDescription::state_type<dim, Number>;

    /**
     * Type used to store a curl of an 2D/3D vector field. Departing from
     * mathematical rigor, in 2D this is a number (stored as
     * `Tensor<1,1>`), in 3D this is a rank 1 tensor.
     */
    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, Number>;

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * @copydoc OfflineData::vector_type
     */
    using vector_type = typename OfflineData<dim, Number>::vector_type;

    /**
     * The number of postprocessed quantities:
     */
    static constexpr unsigned int n_quantities = (dim == 1) ? 1 : 2;

    /**
     * An array of strings for all component names.
     */
    const static std::array<std::string, n_quantities> component_names;

    /**
     * Constructor.
     */
    VTUOutput(const MPI_Comm &mpi_communicator,
              const ryujin::OfflineData<dim, Number> &offline_data,
              const std::string &subsection = "VTUOutput");

    /**
     * Prepare VTU output. A call to @ref prepare() allocates temporary
     * storage and is necessary before schedule_output() can be called.
     *
     * Calling prepare() allocates temporary storage for additional (dim +
     * 5) scalar vectors of type OfflineData::scalar_type.
     */
    void prepare();

    /**
     * Given a state vector @p U and a file name prefix @p name, the
     * current time @p t, and the current output cycle @p cycle) schedule a
     * solution output.
     *
     * The function post-processes quantities synchronously, and (depending
     * on configuration options) schedules the write-out asynchronously
     * onto a background worker thread. This implies that @p U can again be
     * modified once schedule_output() returned.
     *
     * The booleans @p output_full controls whether the full vector field
     * is written out. Correspondingly, @p output_cutplanes controls
     * whether cells in the vicinity of predefined cutplanes are written
     * out.
     *
     * The function requires MPI communication and is not reentrant.
     */
    void schedule_output(const vector_type &U,
                         std::string name,
                         Number t,
                         unsigned int cycle,
                         bool output_full = true,
                         bool output_cutplanes = true);

    /**
     * Returns true if at least one background thread is active writing out
     * the solution to disk.
     */
    bool is_active();

    /**
     * Wait for all background threads to finish writing out the solution
     * to disk.
     */
    void wait();

  private:
    /**
     * @name Run time options
     */
    //@{

    bool use_mpi_io_;
    ACCESSOR_READ_ONLY(use_mpi_io)

    Number schlieren_beta_;
    Number vorticity_beta_;

    std::vector<std::string> manifolds_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;

    std::future<void> background_thread_status;

    std::array<scalar_type, problem_dimension> state_vector_;
    std::array<scalar_type, n_quantities> quantities_;

    //@}
  };

} /* namespace ryujin */
