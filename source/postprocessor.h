//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "mpi_ensemble.h"
#include "observer_pointer.h"
#include "offline_data.h"

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
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using state_type = typename View::state_type;

    template <typename T>
    using grad_type = dealii::Tensor<1, dim, T>;

    template <typename T>
    using curl_type = dealii::Tensor<1, dim == 2 ? 1 : dim, T>;

    using StateVector = typename View::StateVector;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    Postprocessor(const MPIEnsemble &mpi_ensemble,
                  const OfflineData<dim, Number> &offline_data,
                  const HyperbolicSystem &hyperbolic_system,
                  const ParabolicSystem &parabolic_system,
                  const std::string &subsection = "/Postprocessor");

    /**
     * Prepare Postprocessor. A call to @ref prepare() allocates temporary
     * storage and is necessary before compute() can be called.
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
     * Given a state vector @p state_vector compute all configured
     * postprocessed quantities and store the result in the vector returned
     * by quantities().
     *
     * The normalization bounds are computed on the first invocation and
     * then reused for all subsequent invocations, unless the
     * "recompute bounds" run time option is set, or reset_bounds() has been
     * called in the meantime.
     *
     * The function requires MPI communication and is not reentrant.
     */
    void compute(const StateVector &state_vector) const;

    /**
     * Returns a reference to the quantities_ vector that has been filled
     * by the compute() function.
     */
    ACCESSOR_READ_ONLY(quantities)

  private:
    //@}
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

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<const ParabolicSystem> parabolic_system_;

    std::vector<std::string> component_names_;
    std::vector<std::pair<bool /*primitive*/, unsigned int>> schlieren_indices_;
    std::vector<std::pair<bool /*primitive*/, unsigned int>> vorticity_indices_;

    mutable std::vector<std::pair<Number, Number>> bounds_;
    using ScalarHostVector = Vectors::ScalarHostVector<Number>;
    mutable std::vector<ScalarHostVector> quantities_;
    //@}
  };

} // namespace ryujin
