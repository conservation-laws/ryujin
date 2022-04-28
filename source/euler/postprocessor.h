//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <offline_data.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  /**
   * The Postprocessor class implements a number of postprocessing
   * primitives in particular for a scaled and normalized schlieren like
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
  class Postprocessor final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    // clang-format off
    static constexpr unsigned int problem_dimension = HyperbolicSystem::problem_dimension<dim>;
    // clang-format on

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type = HyperbolicSystem::state_type<dim, Number>;

    /**
     * Type used to store a curl of an 2D/3D vector field. Departing from
     * mathematical rigor, in 2D this is a number (stored as
     * `Tensor<1,1>`), in 3D this is a rank 1 tensor.
     */
    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, Number>;

    /**
     * The number of postprocessed quantities:
     */
    static constexpr unsigned int n_quantities = (dim == 1) ? 1 : 2;

    /**
     * An array of strings for all component names.
     */
    const static std::array<std::string, n_quantities> component_names;

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * @copydoc OfflineData::vector_type
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * Constructor.
     */
    Postprocessor(const MPI_Comm &mpi_communicator,
                  const ryujin::HyperbolicSystem &hyperbolic_system,
                  const ryujin::OfflineData<dim, Number> &offline_data,
                  const std::string &subsection = "Postprocessor");

    /**
     * Prepare Postprocessor. A call to @ref prepare() allocates temporary
     * storage and is necessary before schedule_output() can be called.
     *
     * Calling prepare() allocates temporary storage for two additional
     * scalar vectors of type OfflineData::scalar_type.
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
     * The function requires MPI communication and is not reentrant.
     */
    void compute(const vector_type &U) const;

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

    Number schlieren_beta_;
    Number vorticity_beta_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;

    mutable std::array<scalar_type, n_quantities> quantities_;
    ACCESSOR_READ_ONLY(quantities)

    //@}
  };

} // namespace ryujin
