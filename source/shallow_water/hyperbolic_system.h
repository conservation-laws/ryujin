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

namespace ryujin
{
  namespace ShallowWater
  {
    /**
     * Description of a @p dim dimensional hyperbolic conservation law
     * modeling the shallow water equations.
     *
     * We have a (1 + dim) dimensional state space \f$[h, \textbf m]\f$, where
     * \f$h\f$ denotes the water depth, abd \f$\textbf m\f$ is the momentum.
     *
     * @ingroup ShallowWaterEquations
     */
    class HyperbolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the hyperbolic system as a string.
       */
      static inline const std::string problem_name = "Shallow water equations";

      /**
       * Constructor.
       */
      HyperbolicSystem(const std::string &subsection = "/HyperbolicSystem")
          : ParameterAcceptor(subsection)
      {
      }

    private:
      /**
       * @name Runtime parameters, internal fields and methods
       */
      //@{
      double gravity_;
      double mannings_;

      double reference_water_depth_;
      double dry_state_relaxation_sharp_;
      double dry_state_relaxation_mollified_;
      //@}

    public:
      /**
       * A view of the HyperbolicSystem that makes methods available for a
       * given dimension @p dim and choice of number type @p Number (which
       * can be a scalar float, or double, as well as a VectorizedArray
       * holding packed scalars.
       *
       * Intended usage:
       * ```
       * HyperbolicSystem hyperbolic_system;
       * const auto view = hyperbolic_system.template view<dim, Number>();
       * const auto flux_i = view.flux_contribution(...);
       * const auto flux_j = view.flux_contribution(...);
       * const auto flux_ij = view.flux(flux_i, flux_j);
       * // etc.
       * ```
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

        /**
         * @name Access to runtime parameters
         */
        //@{

        DEAL_II_ALWAYS_INLINE inline ScalarNumber gravity() const
        {
          return ScalarNumber(hyperbolic_system_.gravity_);
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber mannings() const
        {
          return ScalarNumber(hyperbolic_system_.mannings_);
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber reference_water_depth() const
        {
          return ScalarNumber(hyperbolic_system_.reference_water_depth_);
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber
        dry_state_relaxation_sharp() const
        {
          return ScalarNumber(hyperbolic_system_.dry_state_relaxation_sharp_);
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber
        dry_state_relaxation_mollified() const
        {
          return ScalarNumber(
              hyperbolic_system_.dry_state_relaxation_mollified_);
        }

        //@}
        /**
         * @name Internal data
         */
        //@{

      private:
        const HyperbolicSystem &hyperbolic_system_;

      public:
        //@}
        /**
         * @name Types and compile time constants
         */
        //@{

        /**
         * The dimension of the state space.
         */
        static constexpr unsigned int problem_dimension = dim + 1;

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
            return {"h", "m"};
          else if constexpr (dim == 2)
            return {"h", "m_1", "m_2"};
          else if constexpr (dim == 3)
            return {"h", "m_1", "m_2", "m_3"};
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
            []() -> std::array<std::string, problem_dimension> {
          if constexpr (dim == 1)
            return {"h", "v"};
          else if constexpr (dim == 2)
            return {"h", "v_1", "v_2"};
          else if constexpr (dim == 3)
            return {"h", "v_1", "v_2", "v_3"};
          __builtin_trap();
        }();

        /**
         * The storage type used for the flux \f$\mathbf{f}\f$.
         */
        using flux_type = dealii::
            Tensor<1, problem_dimension, dealii::Tensor<1, dim, Number>>;

        /**
         * The storage type used for flux contributions.
         */
        using flux_contribution_type = std::tuple<state_type, Number>;

        //@}
        /**
         * @name Precomputed quantities
         */
        //@{

        /**
         * The number of precomputed initial values.
         */
        static constexpr unsigned int n_precomputed_initial_values = 1;

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
            std::array<std::string, n_precomputed_initial_values>{"bathymetry"};

        /**
         * The number of precomputed values.
         */
        static constexpr unsigned int n_precomputed_values = 1;

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
            std::array<std::string, n_precomputed_values>{"eta_m"};

        /**
         * The number of precomputation cycles.
         */
        static constexpr unsigned int n_precomputation_cycles = 1;

        /**
         * Step 0: precompute values for hyperbolic update. This routine is
         * called within our usual loop() idiom in HyperbolicModule
         */
        template <typename DISPATCH, typename SPARSITY>
        void
        precomputation_loop(unsigned int /*cycle*/,
                            const DISPATCH &/*dispatch_check*/,
                            precomputed_vector_type & /*precomputed_values*/,
                            const SPARSITY & /*sparsity_simd*/,
                            const vector_type & /*U*/,
                            unsigned int /*left*/,
                            unsigned int /*right*/) const
        {
          // FIXME
        }

        //@}
        /**
         * @name Computing derived physical quantities
         */
        //@{

        /**
         * For a given (1+dim dimensional) state vector <code>U</code>, return
         * the water depth <code>U[0]</code>
         */
        static Number water_depth(const state_type &U);

        /**
         * For a given (1+dim dimensional) state vector <code>U</code>,
         * return a regularized inverse of the water depth. This function
         * returns 2h / (h^2+max(h, h_cutoff)^2), where h_cutoff is the
         * reference water depth multiplied by eps.
         */
        Number inverse_water_depth_mollified(const state_type &U) const;

        /**
         * For a given (1+dim dimensional) state vector <code>U</code>, return
         * the regularized water depth <code>U[0]</code> This function returns
         * max(h, h_cutoff), where h_cutoff is the reference water depth
         * multiplied by eps.
         */
        Number water_depth_sharp(const state_type &U) const;

        /**
         * For a given (1+dim dimensional) state vector <code>U</code>, return
         * a regularized inverse of the water depth. This function returns 1 /
         * max(h, h_cutoff), where h_cutoff is the reference water depth
         * multiplied by eps.
         */
        Number inverse_water_depth_sharp(const state_type &U) const;

        /**
         * Given a water depth @ref h this function returns 0 if h is in the
         * interval [-relaxation * h_cutoff, relaxation * h_cutoff], otherwise
         * h is returned unmodified. Here, h_cutoff is the reference water
         * depth multiplied by eps.
         */
        Number filter_dry_water_depth(const Number &h) const;

        /**
         * For a given (1+dim dimensional) state vector <code>U</code>, return
         * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
         */
        static dealii::Tensor<1, dim, Number> momentum(const state_type &U);

        /**
         * For a given (1+dim dimensional) state vector <code>U</code>, compute
         * and return the kinetic energy.
         * \f[
         *   KE = 1/2 |m|^2 / h
         * \f]
         */
        Number kinetic_energy(const state_type &U) const;

        /**
         * For a given (state dimensional) state vector <code>U</code>, compute
         * and return the hydrostatic pressure \f$p\f$:
         * \f[
         *   p = 1/2 g h^2
         * \f]
         */
        Number pressure(const state_type &U) const;

        /**
         * For a given (1+dim dimensional) state vector <code>U</code>, compute
         * the (physical) speed of sound:
         * \f[
         *   c^2 = g * h
         * \f]
         */
        Number speed_of_sound(const state_type &U) const;

        /**
         * For a given (1+dim dimensional) state vector <code>U</code>, compute
         * and return the entropy \f$\eta = 1/2 g h^2 + 1/2 |m|^2 / h\f$.
         */
        Number mathematical_entropy(const state_type &U) const;

        /**
         * For a given (1+dim dimensional) state vector <code>U</code>, compute
         * and return the derivative \f$\eta'\f$ of the entropy defined above.
         */
        state_type mathematical_entropy_derivative(const state_type &U) const;

        /**
         * Returns whether the state @p U is admissible. If @p U is a
         * vectorized state then @p U is admissible if all vectorized
         * values are admissible.
         */
        bool is_admissible(const state_type &U) const;

        //@}
        /**
         * @name Special functions for boundary states
         */
        //@{

        /**
         * Decomposes a given state @p U into Riemann invariants and then
         * replaces the first or second Riemann characteristic from the one
         * taken from @p U_bar state.
         */
        template <int component>
        state_type prescribe_riemann_characteristic(
            const state_type &U,
            const state_type &U_bar,
            const dealii::Tensor<1, dim, Number> &normal) const;

        /**
         * Apply boundary conditions.
         */
        template <typename Lambda>
        state_type
        apply_boundary_conditions(const dealii::types::boundary_id id,
                                  const state_type &U,
                                  const dealii::Tensor<1, dim, Number> &normal,
                                  const Lambda &get_dirichlet_data) const;

        //@}
        /**
         * @name Flux computations
         */
        //@{

        /**
         * For a given (1+dim dimensional) state vector <code>U</code> and
         * left/right topography states <code>Z_left</code> and
         * <code>Z_right</code>, return the star_state <code>U_star</code>
         */
        state_type star_state(const state_type &U,
                              const Number &Z_left,
                              const Number &Z_right) const;

        /**
         * Given a state @p U compute the flux
         * \f[
         * \begin{pmatrix}
         *   \textbf m \\
         *   \textbf v\otimes \textbf m + p\mathbb{I}_d \\
         * \end{pmatrix},
         * \f]
         */
        flux_type f(const state_type &U) const;

        /**
         * Given a state @p U compute the flux
         * \g[
         * \begin{pmatrix}
         *   \textbf m \\
         *   \textbf v\otimes \textbf m \\
         * \end{pmatrix},
         * \f]
         */
        flux_type g(const state_type &U) const;

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
         * For the Shallow water equations we simply retrieve the
         * bathymetry and return, both, state and bathymetry.
         */
        flux_contribution_type
        flux_contribution(const precomputed_vector_type &pv,
                          const precomputed_initial_vector_type &piv,
                          const unsigned int i,
                          const state_type &U_i) const;

        flux_contribution_type
        flux_contribution(const precomputed_vector_type &pv,
                          const precomputed_initial_vector_type &piv,
                          const unsigned int *js,
                          const state_type &U_j) const;

        /**
         * Given precomputed flux contributions @p prec_i and @p prec_j
         * compute the equilibrated, low-order flux \f$(f(U_i^{\ast,j}) +
         * f(U_j^{\ast,i})\f$
         */
        flux_type flux(const flux_contribution_type &flux_i,
                       const flux_contribution_type &flux_j) const;

        /**
         * The low-order and high-order fluxes differ:
         */
        static constexpr bool have_high_order_flux = true;

        /**
         * Given precomputed flux contributions @p prec_i and @p prec_j
         * compute the high-order flux \f$(f(U_i^{\ast,j}) +
         * f(U_j^{\ast,i})\f$
         */
        flux_type high_order_flux(const flux_contribution_type &flux_i,
                                  const flux_contribution_type &flux_j) const;

        /**
         * We need to perform state equilibration: */
        static constexpr bool have_equilibrated_states = true;

        /**
         * Given precomputed flux contributions @p prec_i and @p prec_j
         * compute the equilibrated states \f$U_i^{\ast,j}\f$ and
         * \f$U_j^{\ast,i}\f$.
         */
        std::array<state_type, 2>
        equilibrated_states(const flux_contribution_type &,
                            const flux_contribution_type &) const;

        //@}
        /**
         * @name Computing stencil source terms
         */
        //@{

        /**
         * We do have source terms
         */
        static constexpr bool have_source_terms = true;

        /**
         * FIXME
         */
        state_type low_order_nodal_source(const precomputed_vector_type &pv,
                                          const unsigned int i,
                                          const state_type &U_i) const;

        /**
         * FIXME
         */
        state_type high_order_nodal_source(const precomputed_vector_type &pv,
                                           const unsigned int i,
                                           const state_type &U_i) const;

        /**
         * Given precomputed flux contributions @p prec_i and @p prec_j compute
         * the equilibrated, low-order source term
         * \f$-g(H^{\ast,j}_i)^2c_ij\f$.
         */
        state_type low_order_stencil_source(
            const flux_contribution_type &prec_i,
            const flux_contribution_type &prec_j,
            const Number &d_ij,
            const dealii::Tensor<1, dim, Number> &c_ij) const;

        /**
         * Given precomputed flux contributions @p prec_i and @p prec_j compute
         * the high-order source term \f$ g H_i Z_j c_ij\f$.
         */
        state_type high_order_stencil_source(
            const flux_contribution_type &prec_i,
            const flux_contribution_type &prec_j,
            const Number &d_ij,
            const dealii::Tensor<1, dim, Number> &c_ij) const;

        /**
         * Given precomputed flux contributions @p prec_i and @p prec_j compute
         * the equilibrated, low-order affine shift
         * \f$ B_{ij} = -2d_ij(U^{\ast,j}_i)-2f((U^{\ast,j}_i))c_ij\f$.
         */
        state_type affine_shift_stencil_source(
            const flux_contribution_type &prec_i,
            const flux_contribution_type &prec_j,
            const Number &d_ij,
            const dealii::Tensor<1, dim, Number> &c_ij) const;

        //@}
        /**
         * @name State transformations (primitive states, expanding
         * dimensionality, Galilei transform, etc.)
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
        state_type expand_state(const ST &state) const;

        /**
         * Given a primitive state [rho, u_1, ..., u_d, p] return a conserved
         * state
         */
        state_type
        from_primitive_state(const primitive_state_type &primitive_state) const;

        /**
         * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
         * p]
         */
        primitive_state_type to_primitive_state(const state_type &state) const;

        /**
         * Transform the current state according to a  given operator
         * @p lambda acting on a @a dim dimensional momentum (or velocity)
         * vector.
         */
        template <typename Lambda>
        state_type apply_galilei_transform(const state_type &state,
                                           const Lambda &lambda) const;

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
  }    // namespace ShallowWater
} // namespace ryujin
