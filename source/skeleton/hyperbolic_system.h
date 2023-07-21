//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <convenience_macros.h>
#include <discretization.h>
#include <multicomponent_vector.h>
#include <patterns_conversion.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <functional>

namespace ryujin
{
  namespace Skeleton
  {
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

    public:
      /**
       * A view on the HyperbolicSystem for a given dimension @p dim and
       * choice of number type @p Number (which can be a scalar float, or
       * double, as well as a VectorizedArray holding packed scalars.
       */
      template <int dim, typename Number>
      class View
      {
      public:
        /**
         * Constructor taking a reference to the underlying
         * HyperbolicSystem
         */
        View(const HyperbolicSystem &hyperbolic_system)
            : hyperbolic_system_(hyperbolic_system)
        {
        }

        /**
         * Create a modified view from the current one:
         */
        template <int dim2, typename Number2>
        auto view() const
        {
          return View<dim2, Number2>{hyperbolic_system_};
        }

        /**
         * The underlying scalar number type.
         */
        using ScalarNumber = typename get_value_type<Number>::type;


      private:
        const HyperbolicSystem &hyperbolic_system_;


      public:
        /**
         * @name Types and compile time constants
         */
        //@{

        /**
         * The dimension of the state space.
         */
        static constexpr unsigned int problem_dimension = 1;

        /**
         * The storage type used for a (conserved) state vector \f$\boldsymbol
         * U\f$.
         */
        using state_type = dealii::Tensor<1, problem_dimension, Number>;

        /**
         * MulticomponentVector for storing a vector of conserved states:
         */
        using vector_type =
            MultiComponentVector<ScalarNumber, problem_dimension>;

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
         * The storage type used for a primitive state vector.
         */
        using primitive_state_type =
            dealii::Tensor<1, problem_dimension, Number>;

        /**
         * An array holding all component names of the primitive state as a
         * string.
         */
        static inline const auto primitive_component_names =
            std::array<std::string, problem_dimension>{"u"};

        /**
         * The storage type used for the flux \f$\mathbf{f}\f$.
         */
        using flux_type = dealii::
            Tensor<1, problem_dimension, dealii::Tensor<1, dim, Number>>;

        /**
         * The storage type used for flux contributions.
         */
        using flux_contribution_type = flux_type;

        //@}
        /**
         * @name Precomputed quantities
         */
        //@{

        /**
         * The number of precomputed initial values.
         */
        static constexpr unsigned int n_precomputed_initial_values = 0;

        /**
         * Array type used for precomputed initial values.
         */
        using precomputed_initial_state_type =
            std::array<Number, n_precomputed_initial_values>;

        /**
         * MulticomponentVector for storing a vector of precomputed initial
         * states:
         */
        using precomputed_initial_vector_type =
            MultiComponentVector<ScalarNumber, n_precomputed_initial_values>;

        /**
         * An array holding all component names of the precomputed values.
         */
        static inline const auto precomputed_initial_names =
            std::array<std::string, n_precomputed_initial_values>{};

        /**
         * The number of precomputed values.
         */
        static constexpr unsigned int n_precomputed_values = 0;

        /**
         * Array type used for precomputed values.
         */
        using precomputed_state_type = std::array<Number, n_precomputed_values>;

        /**
         * MulticomponentVector for storing a vector of precomputed states:
         */
        using precomputed_vector_type =
            MultiComponentVector<ScalarNumber, n_precomputed_values>;

        /**
         * An array holding all component names of the precomputed values.
         */
        static inline const auto precomputed_names =
            std::array<std::string, n_precomputed_values>{};

        /**
         * The number of precomputation cycles.
         */
        static constexpr unsigned int n_precomputation_cycles = 0;

        /**
         * Precomputed values for a given state.
         */
        template <unsigned int cycle, typename SPARSITY>
        void precomputation(precomputed_vector_type &precomputed_values,
                            const vector_type &U,
                            const SPARSITY &sparsity_simd,
                            unsigned int i) const = delete;

        //@}
        /**
         * @name Computing derived physical quantities
         */
        //@{

        /**
         * Returns whether the state @ref U is admissible. If @ref U is a
         * vectorized state then @ref U is admissible if all vectorized values
         * are admissible.
         */
        bool is_admissible(const state_type &/*U*/) const
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
         * Indicator<dim, Number> indicator;
         * for (unsigned int i = n_internal; i < n_owned; ++i) {
         *   // ...
         *   const auto flux_i = flux_contribution(precomputed..., i, U_i);
         *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
         *     // ...
         *     const auto flux_j = flux_contribution(precomputed..., js, U_j);
         *     const auto flux_ij = flux(flux_i, flux_j);
         *   }
         * }
         * ```
         *
         * For the Euler equations we simply compute <code>f(U_i)</code>.
         */
        flux_contribution_type
        flux_contribution(const precomputed_vector_type & /*pv*/,
                          const precomputed_initial_vector_type & /*piv*/,
                          const unsigned int /*i*/,
                          const state_type & /*U_i*/) const
        {
          return flux_contribution_type{};
        }

        flux_contribution_type
        flux_contribution(const precomputed_vector_type & /*pv*/,
                          const precomputed_initial_vector_type & /*piv*/,
                          const unsigned int * /*js*/,
                          const state_type & /*U_j*/) const
        {
          return flux_contribution_type{};
        }

        /**
         * Given flux contributions @p flux_i and @p flux_j compute the flux
         * <code>(-f(U_i) - f(U_j)</code>
         */
        flux_type flux(const flux_contribution_type & /*flux_i*/,
                       const flux_contribution_type & /*flux_j*/) const
        {
          return flux_type{};
        }

        /**
         * The low-order and high-order fluxes are the same:
         */
        static constexpr bool have_high_order_flux = false;

        flux_type
        high_order_flux(const flux_contribution_type &,
                        const flux_contribution_type &) const = delete;

        /** We do not perform state equilibration */
        static constexpr bool have_equilibrated_states = false;

        std::array<state_type, 2>
        equilibrated_states(const flux_contribution_type &,
                            const flux_contribution_type &) const = delete;

        //@}
        /**
         * @name Computing stencil source terms
         */
        //@{

        /** We do not have source terms */
        static constexpr bool have_source_terms = false;

        state_type low_order_nodal_source(const precomputed_vector_type &,
                                          const unsigned int,
                                          const state_type &) const = delete;

        state_type high_order_nodal_source(const precomputed_vector_type &,
                                           const unsigned int,
                                           const state_type &) const = delete;

        state_type low_order_stencil_source(
            const flux_contribution_type &,
            const flux_contribution_type &,
            const Number,
            const dealii::Tensor<1, dim, Number> &) const = delete;

        state_type high_order_stencil_source(
            const flux_contribution_type &,
            const flux_contribution_type &,
            const Number,
            const dealii::Tensor<1, dim, Number> &) const = delete;

        state_type affine_shift_stencil_source(
            const flux_contribution_type &,
            const flux_contribution_type &,
            const Number,
            const dealii::Tensor<1, dim, Number> &) const = delete;

        //@}
        /**
         * @name State transformations
         */
        //@{

        /*
         * Given a state vector associated with @ref dim2 spatial dimensions
         * return an "expanded" version of the state vector associated with
         * @ref dim1 spatial dimensions where the momentum vector is projected
         * onto the first @ref dim2 unit directions of the @ref dim dimensional
         * euclidean space.
         *
         * @precondition dim has to be larger or equal than dim2.
         */
        template <typename ST>
        state_type expand_state(const ST &state) const
        {
          return state;
        }

        /*
         * Given a primitive state [rho, u_1, ..., u_d, p] return a conserved
         * state
         */
        state_type
        from_primitive_state(const primitive_state_type &primitive_state) const
        {
          return primitive_state;
        }

        /*
         * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
         * p]
         */
        primitive_state_type to_primitive_state(const state_type &state) const
        {
          return state;
        }

        /*
         * Transform the current state according to a  given operator @ref
         * momentum_transform acting on a @p dim dimensional momentum (or
         * velocity) vector.
         */
        template <typename Lambda>
        state_type apply_galilei_transform(const state_type &state,
                                           const Lambda & /*lambda*/) const
        {
          return state;
        }

      }; /* HyperbolicSystem::View */

      template <int dim, typename Number>
      friend class View;

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
    }; /* HyperbolicSystem */
  } // namespace Skeleton
} // namespace ryujin
