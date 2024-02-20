//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

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
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number = double>
  class Postprocessor final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    /**
     * @copydoc HyperbolicSystemView
     */
    using HyperbolicSystemView =
        typename Description::HyperbolicSystem::template View<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    static constexpr unsigned int problem_dimension =
        HyperbolicSystemView::problem_dimension;

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type = typename HyperbolicSystemView::state_type;

    /**
     * The type used to store the gradient of a scalar quantitty;
     */
    template <typename T>
    using grad_type = dealii::Tensor<1, dim, T>;

    /**
     * Type used to store the curl of an 2D/3D vector field. Departing from
     * mathematical rigor, in 2D this is a number (stored as
     * `Tensor<1,1>`), in 3D this is a rank 1 tensor.
     */
    template <typename T>
    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, T>;

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * @copydoc HyperbolicSystemView::vector_type
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * Constructor.
     */
    Postprocessor(const MPI_Comm &mpi_communicator,
                  const HyperbolicSystem &hyperbolic_system,
                  const OfflineData<dim, Number> &offline_data,
                  const std::string &subsection = "/Postprocessor");

    /**
     * Prepare Postprocessor. A call to @ref prepare() allocates temporary
     * storage and is necessary before schedule_output() can be called.
     *
     * Calling prepare() allocates temporary storage for two additional
     * scalar vectors of type OfflineData::scalar_type.
     */
    void prepare();

    /**
     * Returns the number of computed quantities.
     */
    unsigned int n_quantities() const
    {
      return quantities_.size();
    }

    /**
     * A vector of strings for all component names.
     */
    const std::vector<std::string> component_names() const
    {
      return component_names_;
    }

    /**
     * Reset computed normalization bounds. Calling this function will
     * force a recomputation of the normalization bounds during the next
     * call to compute().
     */
    void reset_bounds() const
    {
      bounds_.clear();
    }

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
     * Returns a reference to the quantities_ vector that has been filled
     * by the compute() function.
     */
    ACCESSOR_READ_ONLY(quantities)


  private:
    /**
     * @name Run time options
     */
    //@{

    bool recompute_bounds_;
    Number beta_;

    std::vector<std::string> schlieren_quantities_;
    std::vector<std::string> vorticity_quantities_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;

    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;

    std::vector<std::string> component_names_;
    std::vector<std::pair<bool /*primitive*/, unsigned int>> schlieren_indices_;
    std::vector<std::pair<bool /*primitive*/, unsigned int>> vorticity_indices_;

    mutable std::vector<std::pair<Number, Number>> bounds_;
    mutable std::vector<scalar_type> quantities_;
    //@}
  };

} // namespace ryujin
