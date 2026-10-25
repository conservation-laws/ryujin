//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/simd.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/hyperbolic_system.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <array>
#include <concepts>
#include <string>
#include <utility>

namespace ryujin
{
  /**
   * The concepts that a hyperbolic and a parabolic description, and their
   * members, have to satisfy.
   *
   * @ingroup Concepts
   */
  namespace Concept
  {
    /**
     * The type of the view that an object hands out for the given
     * dimension and number type.
     */
    template <typename T, int dim, typename Number>
    using view_type =
        decltype(std::declval<const T &>().template view<dim, Number>());


    /**
     * The scalar type underlying the (possibly vectorized) number type.
     */
    template <typename Number>
    using scalar_type = typename get_value_type<Number>::type;


    /**
     * The view of a hyperbolic system: sizes and types of the state and
     * the kernels that the generic modules call. All names are stored in
     * Interface::HyperbolicSystem.
     *
     * @ingroup Concepts
     */
    template <typename V, int dim, typename Number>
    concept HyperbolicSystemView =
        requires {
          typename V::ScalarNumber;
          typename V::flux_contribution_type;
          typename V::PrecomputedVectorView;
          typename V::InitialPrecomputedVectorView;
          {
            V::have_high_order_flux
          } -> std::convertible_to<bool>;
          {
            V::have_source_terms
          } -> std::convertible_to<bool>;
        } &&
        std::same_as<typename V::state_type,
                     dealii::Tensor<1, V::problem_dimension, Number>> &&
        std::same_as<typename V::precomputed_type,
                     std::array<Number, V::n_precomputed_values>> &&
        std::same_as<typename V::initial_precomputed_type,
                     std::array<Number, V::n_initial_precomputed_values>> &&
        requires(const V &view,
                 const typename V::state_type &U,
                 const typename V::flux_contribution_type &flux,
                 const typename V::PrecomputedVectorView &precomputed,
                 const typename V::InitialPrecomputedVectorView &initial,
                 const dealii::Tensor<1, dim, Number> &c_ij,
                 const dealii::types::boundary_id id,
                 typename V::state_type (&get_dirichlet_data)(),
                 dealii::Tensor<1, dim, Number> (&transform)(
                     const dealii::Tensor<1, dim, Number> &),
                 const unsigned int i,
                 const unsigned int *js) {
          {
            view.is_admissible(U)
          } -> std::convertible_to<bool>;
          {
            view.apply_boundary_conditions(id, U, c_ij, get_dirichlet_data)
          } -> std::same_as<typename V::state_type>;
          {
            view.flux_contribution(precomputed, initial, i, U)
          } -> std::same_as<typename V::flux_contribution_type>;
          {
            view.flux_contribution(precomputed, initial, js, U)
          } -> std::same_as<typename V::flux_contribution_type>;
          {
            view.flux_divergence(flux, flux, c_ij)
          } -> std::same_as<typename V::state_type>;
          {
            view.to_primitive_state(U)
          } -> std::same_as<typename V::state_type>;
          {
            view.from_primitive_state(U)
          } -> std::same_as<typename V::state_type>;
          {
            view.apply_galilei_transform(U, transform)
          } -> std::same_as<typename V::state_type>;
        } &&
        (!V::have_high_order_flux ||
         requires(const V &view,
                  const typename V::flux_contribution_type &flux,
                  const dealii::Tensor<1, dim, Number> &c_ij) {
           {
             view.high_order_flux_divergence(flux, flux, c_ij)
           } -> std::same_as<typename V::state_type>;
         }) &&
        (!V::have_source_terms ||
         requires(const V &view,
                  const typename V::state_type &U,
                  const typename V::PrecomputedVectorView &precomputed,
                  const typename V::ScalarNumber tau,
                  const unsigned int i,
                  const unsigned int *js) {
           {
             view.nodal_source(precomputed, i, U, tau)
           } -> std::same_as<typename V::state_type>;
           {
             view.nodal_source(precomputed, js, U, tau)
           } -> std::same_as<typename V::state_type>;
         });


    /**
     * The view of a hyperbolic system that forms equilibrated states
     * \f$U_i^{\ast,j}\f$ and \f$U_j^{\ast,i}\f$ for the low-order update,
     * the graph viscosity, and the limiter bounds. The latter are
     * additionally shifted by an affine shift.
     *
     * @ingroup Concepts
     */
    template <typename V, int dim, typename Number>
    concept HyperbolicSystemViewEquilibratesStates =
        HyperbolicSystemView<V, dim, Number> &&
        requires(const V &view,
                 const typename V::flux_contribution_type &flux,
                 const dealii::Tensor<1, dim, Number> &c_ij,
                 const Number &d_ij) {
          {
            view.equilibrated_states(flux, flux)
          } -> std::same_as<std::array<typename V::state_type, 2>>;
          {
            view.affine_shift(flux, flux, c_ij, d_ij)
          } -> std::same_as<typename V::state_type>;
        };


    /**
     * A hyperbolic system fills the precomputed values for the scalar
     * number type Number.
     *
     * @ingroup Concepts
     */
    template <typename T, int dim, typename Number>
    concept HyperbolicSystemFillsPrecomputedValues =
        requires(const T &system,
                 const OfflineData<dim, Number> &offline_data,
                 StateVector<Number> &state_vector) {
          system.fill_precomputed_values(offline_data, state_vector);
        };


    /**
     * A hyperbolic system: constructed from the subsection of its run
     * time parameters, it hands out views and fills the precomputed
     * values.
     *
     * Not expressible as a constraint: state_names(dim) and
     * primitive_names(dim) have exactly problem_dimension entries,
     * precomputed_names(dim) exactly n_precomputed_values, and
     * initial_precomputed_names(dim) exactly n_initial_precomputed_values
     * entries of the view. HyperbolicModule::prepare() asserts this.
     *
     * @ingroup Concepts
     */
    template <typename T, int dim, typename Number>
    concept HyperbolicSystem =
        std::derived_from<T, Interface::HyperbolicSystem> &&
        std::constructible_from<T, const std::string &> &&
        HyperbolicSystemView<view_type<T, dim, Number>, dim, Number> &&
        HyperbolicSystemFillsPrecomputedValues<T, dim, scalar_type<Number>>;
  } // namespace Concept
} // namespace ryujin
