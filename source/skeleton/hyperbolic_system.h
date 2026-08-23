//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <convenience_macros.h>
#include <discretization.h>
#include <loop.h>
#include <multicomponent_vector.h>
#include <patterns_conversion.h>
#include <simd.h>
#include <state_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>

namespace ryujin
{
  namespace Skeleton
  {
    template <int dim, typename Number>
    class HyperbolicSystemView;

    /**
     * Description of a hyperbolic conservation law.
     *
     * @ingroup SkeletonEquations
     */
    class HyperbolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the hyperbolic system as a string.
       */
      static inline const std::string problem_name =
          "Skeleton Hyperbolic System";

      /**
       * Constructor.
       */
      HyperbolicSystem(const std::string &subsection = "/HyperbolicSystem")
          : ParameterAcceptor(subsection)
      {
      }

      /**
       * Alias for the view on the hyperbolic system for a given dimension @p
       * dim and choice of number type @p Number.
       */
      template <int dim, typename Number = double>
      using View = HyperbolicSystemView<dim, Number>;

      /**
       * Return a view on the Hyperbolic System for a given dimension @p
       * dim and choice of number type @p Number (which can be a scalar
       * float, or double, as well as a VectorizedArray holding packed
       * scalars.
       */
      template <int dim, typename Number>
      auto view() const
      {
        return View<dim, Number>{*this};
      }

      /**
       * Part of step 1 of the hyperbolic update step: Compute "precomputed
       * values" and fill into the state vector.
       *
       * @note The method does not update the ghost range of the state
       * vector. The precomputed part has to be synchronized by explicitly
       * calling the update ghost values function.
       */
      template <int dim, typename ScalarNumber>
      void fill_precomputed_values(
          const OfflineData<dim, ScalarNumber> & /*offline_data*/,
          typename HyperbolicSystemView<dim, ScalarNumber>::StateVector
              & /*state_vector*/,
          const bool /*skip_constrained_dofs*/ = true) const
      {
        // nothing to do
      }

    private:
      /**
       * @name Internal data
       */
      //@{

      template <int dim, typename Number>
      friend class HyperbolicSystemView;

      //@}
    }; /* HyperbolicSystem */


    /**
     * A view on the HyperbolicSystem for a given dimension @p dim and
     * choice of number type @p Number (which can be a scalar float, or
     * double, as well as a VectorizedArray holding packed scalars.
     *
     * @ingroup SkeletonEquations
     */
    template <int dim, typename Number>
    class HyperbolicSystemView
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      /**
       * The underlying scalar number type.
       */
      using ScalarNumber = typename get_value_type<Number>::type;

      /**
       * The dimension of the state space.
       */
      static constexpr unsigned int problem_dimension = 1;

      /**
       * Storage type for a (conserved) state vector \f$\boldsymbol U\f$.
       */
      using state_type = dealii::Tensor<1, problem_dimension, Number>;

      /**
       * Storage type for the flux \f$\mathbf{f}\f$.
       */
      using flux_type =
          dealii::Tensor<1, problem_dimension, dealii::Tensor<1, dim, Number>>;

      /**
       * The storage type used for flux contributions.
       */
      using flux_contribution_type = flux_type;

      /**
       * An array holding all component names of the conserved state as a
       * string.
       */
      static inline const auto component_names =
          []() -> std::array<std::string, problem_dimension> {
        if constexpr (dim == 1)
          return {"u"};
        else if constexpr (dim == 2)
          return {"u"};
        else if constexpr (dim == 3)
          return {"u"};
        __builtin_trap();
      }();

      /**
       * An array holding all component names of the primitive state as a
       * string.
       */
      static inline const auto primitive_component_names =
          []() -> std::array<std::string, problem_dimension> {
        if constexpr (dim == 1)
          return {"u"};
        else if constexpr (dim == 2)
          return {"u"};
        else if constexpr (dim == 3)
          return {"u"};
        __builtin_trap();
      }();

      /**
       * The number of precomputed values.
       */
      static constexpr unsigned int n_precomputed_values = 0;

      /**
       * Array type used for precomputed values.
       */
      using precomputed_type = std::array<Number, n_precomputed_values>;

      /**
       * An array holding all component names of the precomputed values.
       */
      static inline const auto precomputed_names =
          std::array<std::string, n_precomputed_values>{};

      /**
       * The number of precomputed initial values.
       */
      static constexpr unsigned int n_initial_precomputed_values = 0;

      /**
       * Array type used for precomputed initial values.
       */
      using initial_precomputed_type =
          std::array<Number, n_initial_precomputed_values>;

      /**
       * An array holding all component names of the precomputed values.
       */
      static inline const auto initial_precomputed_names =
          std::array<std::string, n_initial_precomputed_values>{};

      /**
       * A compound state vector.
       */
      using StateVector = Vectors::
          StateVector<ScalarNumber, problem_dimension, n_precomputed_values>;

      /**
       * MulticomponentVector for storing the hyperbolic state vector:
       */
      using HyperbolicVector =
          Vectors::MultiComponentVector<ScalarNumber, problem_dimension>;

      /**
       * MulticomponentVector for storing a vector of precomputed states:
       */
      using PrecomputedVector =
          Vectors::MultiComponentVector<ScalarNumber, n_precomputed_values>;

      /**
       * MulticomponentVectorView for accessing a vector of precomputed
       * states:
       */
      using PrecomputedVectorView = Vectors::MultiComponentVectorView<
          ScalarNumber,
          n_precomputed_values,
          dealii::VectorizedArray<ScalarNumber>::size(),
          dealii::MemorySpace::Host,
          /*writable=*/false>;

      /**
       * MulticomponentVector for storing a vector of precomputed initial
       * states:
       */
      using InitialPrecomputedVector =
          Vectors::MultiComponentVector<ScalarNumber,
                                        n_initial_precomputed_values>;

      /**
       * MulticomponentVectorView for accessing a vector of precomputed
       * initial states:
       */
      using InitialPrecomputedVectorView = Vectors::MultiComponentVectorView<
          ScalarNumber,
          n_initial_precomputed_values,
          dealii::VectorizedArray<ScalarNumber>::size(),
          dealii::MemorySpace::Host,
          /*writable=*/false>;

      //@}
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor taking a reference to the underlying
       * HyperbolicSystem
       */
      HyperbolicSystemView(const HyperbolicSystem &hyperbolic_system)
          : hyperbolic_system_(hyperbolic_system)
      {
      }

      //@}
      /**
       * @name Computing derived physical quantities
       */
      //@{

      /**
       * Returns whether the state @p U is admissible. If @p U is a
       * vectorized state then @p U is admissible if all vectorized
       * values are admissible.
       */
      bool is_admissible(const state_type & /*U*/) const
      {
        return true;
      }

      //@}
      /**
       * @name Special functions for boundary states
       */
      //@{

      /**
       * Apply boundary conditions.
       */
      template <typename Lambda>
      state_type apply_boundary_conditions(
          const dealii::types::boundary_id /*id*/,
          const state_type &U,
          const dealii::Tensor<1, dim, Number> & /*normal*/,
          const Lambda & /*get_dirichlet_data*/) const
      {
        return U;
      }

      //@}
      /**
       * @name Flux computations
       */
      //@{

      /**
       * Given a state @p U_i and an index @p i compute flux contributions.
       *
       * Intended usage:
       * ```
       * IndicatorView<dim, Number> indicator_view;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   const auto flux_i = flux_contribution(precomputed..., i, U_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     const auto flux_j = flux_contribution(precomputed..., js, U_j);
       *     const auto flux_ij = flux_divergence(flux_i, flux_j, c_ij);
       *   }
       * }
       * ```
       *
       * For the Euler equations we simply compute <code>f(U_i)</code>.
       */
      flux_contribution_type
      flux_contribution(const PrecomputedVectorView & /*pv*/,
                        const InitialPrecomputedVectorView & /*piv*/,
                        const unsigned int /*i*/,
                        const state_type & /*U_i*/) const
      {
        return flux_contribution_type{};
      }

      flux_contribution_type
      flux_contribution(const PrecomputedVectorView & /*pv*/,
                        const InitialPrecomputedVectorView & /*piv*/,
                        const unsigned int * /*js*/,
                        const state_type & /*U_j*/) const
      {
        return flux_contribution_type{};
      }

      /**
       * Given flux contributions @p flux_i and @p flux_j compute the flux
       * <code>(-f(U_i) - f(U_j) * c_ij</code>
       */
      state_type
      flux_divergence(const flux_contribution_type &flux_i,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &c_ij) const
      {
        return -contract(add(flux_i, flux_j), c_ij);
      }

      /**
       * The low-order and high-order fluxes are the same:
       */
      static constexpr bool have_high_order_flux = false;

      state_type high_order_flux_divergence(
          const flux_contribution_type &,
          const flux_contribution_type &,
          const dealii::Tensor<1, dim, Number> &) const = delete;

      //@}
      /**
       * @name Computing stencil source terms
       */
      //@{

      /** We do not have source terms */
      static constexpr bool have_source_terms = false;

      state_type nodal_source(const PrecomputedVectorView & /*pv*/,
                              const unsigned int /*i*/,
                              const state_type & /*U_i*/,
                              const ScalarNumber /*tau*/) const = delete;

      state_type nodal_source(const PrecomputedVectorView & /*pv*/,
                              const unsigned int * /*js*/,
                              const state_type & /*U_j*/,
                              const ScalarNumber /*tau*/) const = delete;

      //@}
      /**
       * @name State transformations
       */
      //@{

      /**
       * Given a state vector associated with a different spatial
       * dimensions than the current one, return an "expanded" version of
       * the state vector associated with @a dim spatial dimensions where
       * the momentum vector of the conserved state @p state is expaned
       * with zeros to a total length of @a dim entries.
       *
       * @note @a dim has to be larger or equal than the dimension of the
       * @a ST vector.
       */
      template <typename ST>
      state_type expand_state(const ST &state) const
      {
        return state;
      }

      /**
       * Given a primitive state [rho, u_1, ..., u_d, p] return a conserved
       * state
       */
      state_type from_primitive_state(const state_type &primitive_state) const
      {
        return primitive_state;
      }

      /**
       * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
       * p]
       */
      state_type to_primitive_state(const state_type &state) const
      {
        return state;
      }

      /**
       * Transform the current state according to a  given operator
       * @p lambda acting on a @a dim dimensional momentum (or velocity)
       * vector.
       */
      template <typename Lambda>
      state_type apply_galilei_transform(const state_type &state,
                                         const Lambda & /*lambda*/) const
      {
        return state;
      }

    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const HyperbolicSystem &hyperbolic_system_;

      //@}
    }; /* HyperbolicSystemView */
  } // namespace Skeleton
} // namespace ryujin
