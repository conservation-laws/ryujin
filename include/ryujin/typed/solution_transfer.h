//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/interface/offline_data.h>
#include <ryujin/interface/parabolic_system.h>
#include <ryujin/interface/solution_transfer.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>

#include <utility>
#include <vector>

namespace ryujin
{
  /**
   * The SolutionTransfer class is an adaptation of the
   * parallel::distributed::SolutionTransfer class and method implemented
   * in deal.II. This class is, in contrast to the deal.II version,
   * specifically tailored to our needs:
   *  - it uses the MultiComponentVector directly,
   *  - it implements a local mass matrix projection with convex limiting
   *    for the hyperbolic state,
   *  - it interpolates the dynamic state, without limiting, on the
   *    continuous finite element space of Discretization::dof_handler_cg().
   *
   * Mass redistribution, cf. Section 4.3 of @cite ryujin-2025-8. For a
   * constraint \f$(j,\{c^j_i\})\f$ with \f$\sum_i c^j_i=1\f$ and intermediate
   * masses and states \f$\tilde m_i\f$, \f$\tilde U_i\f$:
   * \f{align*}
   *   D_j &:= \tilde U_j - \sum_{k} c^j_k \tilde U_k,
   *   \\
   *   m_i &:= \tilde m_i + \sum_{j\in\mathcal{I}^{AC}(i)} c^j_i \tilde m_j,
   *   \\
   *   W^j_i(l) &:= \tilde U_j
   *     + l \Big(\tilde U_i - \sum_{k} c^j_k \tilde U_k\Big),
   *   \\
   *   U_i &:= \frac{1}{m_i}\Big(\tilde m_i \tilde U_i
   *     + \sum_{j\in\mathcal{I}^{AC}(i)} c^j_i \tilde m_j W^j_i(l_j)\Big),
   *   \quad i\notin AC,
   *   \\
   *   U_j &:= 0, \quad j\in AC.
   * \f}
   * With \f$l_j\f$ and \f$\mathcal{B}_j\f$ the relaxed bounds of
   * \f$\{\tilde U_j\}\cup\{\tilde U_k: c^j_k\neq 0\}\f$:
   * \f{align*}
   *   l_j &:= 1 \quad\text{if}\quad |(D_j)_\nu| \le 100\,\varepsilon\,
   *     \max\big(|(\tilde U_j)_\nu|,\max_{k}|(\tilde U_k)_\nu|\big)
   *     \;\forall\nu,
   *   \\
   *   l_j &:= \min_{i:\,c^j_i\neq 0}\,
   *     \max\big\{l\in[0,1]: W^j_i(l)\in\mathcal{B}_j\big\}
   *     \quad\text{otherwise}.
   * \f}
   * Moreover,
   * \f{align*}
   *   l_j = 0 &: \; W^j_i = \tilde U_j \;\Rightarrow\; U_i = U_i^L,
   *   \\
   *   l_j = 1 &: \; W^j_i = \tilde U_i + D_j \;\Rightarrow\; U_i = U_i^H,
   *   \\
   *   D_j = 0,\; l_j = 1 &: \; U_i = \tilde U_i,
   *   \\
   *   \sum_{i\notin AC} m_i U_i &= \sum_{i} \tilde m_i \tilde U_i
   *     \quad\forall\, l_j.
   * \f}
   *
   * @todo generalize the transfer of the dynamic state to support
   * Discretization::dof_handler_cg() and Discretization::dof_handler_dg()
   *
   * @ingroup Typed
   */
  template <typename HD, int structdim, int dim, typename Number = double>
  class SolutionTransfer final : public Interface::SolutionTransfer<dim, Number>
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename HD::HyperbolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using Limiter = typename HD::template Limiter<Number>;

    using LimiterView =
        decltype(std::declval<const Limiter &>().template view<dim, Number>());

    static constexpr auto problem_dimension = View::problem_dimension;

    static constexpr auto n_precomputed_values = View::n_precomputed_values;

    using state_type = typename View::state_type;

    /**
     * The number of stored entries in the bounds array.
     */
    static constexpr unsigned int n_bounds = LimiterView::n_bounds;

    /**
     * Array type used to store accumulated bounds.
     */
    using Bounds = typename LimiterView::Bounds;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    SolutionTransfer(const MPIEnsemble &mpi_ensemble,
                     Discretization<structdim, dim> &discretization, /*mutable*/
                     const Interface::OfflineData<dim, Number> &offline_data,
                     const HyperbolicSystem &hyperbolic_system,
                     const Interface::ParabolicSystem &parabolic_system,
                     const std::string &subsection = "/SolutionTransfer");

    /**
     * Destructor
     */
    ~SolutionTransfer() override = default;

    //@}
    /**
     * @name Functions for solution transfer after mesh adaptation and for
     * serialization/deserialization
     */
    //@{

    /**
     * Prepare projection (and serialization) by registering a call back to
     * all cells of the distributed triangulation. The call back attaches
     * all necessary data to the triangulation to perform a projection
     * after mesh adaptation, or a Triangulation::store()/load() operation.
     */
    void
    prepare_projection(const StateVector<Number> &old_state_vector) override;

    /**
     * Project the data stored in the triangulation into a new state vector
     * @p new_state_vector. The attached data either comes from a previous
     * call to prepare_projection() prior to mesh adaptation or has been
     * read back in from a checkpoint after a call to
     * Triangulation::load().
     *
     * @note After mesh refinement all internal data structures stored in
     * the OfflineData object must be reinitialized with a call to
     * prepare() and the @p new_state_vector must be appropriately
     * initialized with the new Partitioner.
     */
    void project(StateVector<Number> &new_state_vector) override;

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    Limiter limiter_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<Discretization<structdim, dim>> discretization_;
    dealii::ObserverPointer<const Interface::OfflineData<dim, Number>>
        offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<const Interface::ParabolicSystem> parabolic_system_;

    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * Helper function reading in a state from the hyperbolic state vector.
     * The function translates the global index @p global_i into the local
     * range (operated on by this MPI rank) and automatically distributes
     * values to constrained degrees of freedom.
     *
     * @note Ordinarily distribute()ing constrained degrees of freedom
     * would happen with AffineConstraints<>::distribute() - but this
     * function is incompatible with our MultiComponentVector.
     */
    state_type read_tensor(const MultiComponentVector<Number> &U,
                           const dealii::types::global_dof_index global_i);

    /**
     * Helper function adding a supplied state to the hyperbolic state
     * vector. The function translates the global index @p global_i into
     * the local range (operated on by this MPI rank).
     */
    void add_tensor(MultiComponentVector<Number> &U,
                    const state_type &U_i,
                    const dealii::types::global_dof_index global_i);

    //@}
  };
} // namespace ryujin
