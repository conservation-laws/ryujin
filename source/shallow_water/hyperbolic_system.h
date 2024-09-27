//
// SPDX-License-Identifier: Apache-2.0
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2024 by the ryujin authors
// Copyright (C) 2023 - 2024 by Triad National Security, LLC
//

#pragma once

#include <convenience_macros.h>
#include <deal.II/base/config.h>
#include <discretization.h>
#include <multicomponent_vector.h>
#include <openmp.h>
#include <patterns_conversion.h>
#include <simd.h>
#include <state_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>

namespace ryujin
{
  namespace ShallowWater
  {
    template <int dim, typename Number>
    class HyperbolicSystemView;

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
      HyperbolicSystem(const std::string &subsection = "/HyperbolicSystem");

      /**
       * Return a view on the Hyperbolic System for a given dimension @p
       * dim and choice of number type @p Number (which can be a scalar
       * float, or double, as well as a VectorizedArray holding packed
       * scalars.
       */
      template <int dim, typename Number>
      auto view() const
      {
        return HyperbolicSystemView<dim, Number>{*this};
      }

      unsigned int n_auxiliary_state_vectors() const
      {
        return auxiliary_component_names_.size();
      }

      ACCESSOR_READ_ONLY(auxiliary_component_names);

    private:
      /**
       * @name Runtime parameters, internal fields, methods, and friends
       */
      //@{
      double gravity_;
      double manning_friction_coefficient_;

      double reference_water_depth_;
      double dry_state_relaxation_factor_;
      double dry_state_relaxation_small_;
      double dry_state_relaxation_large_;

      const std::vector<std::string> auxiliary_component_names_;

      template <int dim, typename Number>
      friend class HyperbolicSystemView;
      //@}
    }; /* HyperbolicSystem */


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
     * const auto flux_ij = view.flux_divergence(flux_i, flux_j, c_ij);
     * // etc.
     * ```
     */
    template <int dim, typename Number>
    class HyperbolicSystemView
    {
    public:
      /**
       * Constructor taking a reference to the underlying
       * HyperbolicSystem
       */
      HyperbolicSystemView(const HyperbolicSystem &hyperbolic_system)
          : hyperbolic_system_(hyperbolic_system)
      {
      }

      /**
       * Create a modified view from the current one:
       */
      template <int dim2, typename Number2>
      auto view() const
      {
        return HyperbolicSystemView<dim2, Number2>{hyperbolic_system_};
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
        return hyperbolic_system_.gravity_;
      }

      DEAL_II_ALWAYS_INLINE inline ScalarNumber
      manning_friction_coefficient() const
      {
        return hyperbolic_system_.manning_friction_coefficient_;
      }

      DEAL_II_ALWAYS_INLINE inline ScalarNumber reference_water_depth() const
      {
        return hyperbolic_system_.reference_water_depth_;
      }

      DEAL_II_ALWAYS_INLINE inline ScalarNumber
      dry_state_relaxation_factor() const
      {
        return hyperbolic_system_.dry_state_relaxation_factor_;
      }

      DEAL_II_ALWAYS_INLINE inline ScalarNumber
      dry_state_relaxation_small() const
      {
        return hyperbolic_system_.dry_state_relaxation_small_;
      }

      DEAL_II_ALWAYS_INLINE inline ScalarNumber
      dry_state_relaxation_large() const
      {
        return hyperbolic_system_.dry_state_relaxation_large_;
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
       * @name Types and constexpr constants
       */
      //@{

      /**
       * The dimension of the state space.
       */
      static constexpr unsigned int problem_dimension = 1 + dim;

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
      using flux_contribution_type = std::tuple<state_type, Number>;

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
       * The number of precomputed values.
       */
      static constexpr unsigned int n_precomputed_values = 2;

      /**
       * Array type used for precomputed values.
       */
      using precomputed_type = std::array<Number, n_precomputed_values>;

      /**
       * An array holding all component names of the precomputed values.
       */
      static inline const auto precomputed_names =
          std::array<std::string, n_precomputed_values>{"eta_m", "h_star"};

      /**
       * The number of precomputed initial values.
       */
      static constexpr unsigned int n_initial_precomputed_values = 1;

      /**
       * Array type used for precomputed initial values.
       */
      using initial_precomputed_type =
          std::array<Number, n_initial_precomputed_values>;

      /**
       * An array holding all component names of the precomputed values.
       */
      static inline const auto initial_precomputed_names =
          std::array<std::string, n_initial_precomputed_values>{"bathymetry"};

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
       * MulticomponentVector for storing a vector of precomputed initial
       * states:
       */
      using InitialPrecomputedVector =
          Vectors::MultiComponentVector<ScalarNumber,
                                        n_initial_precomputed_values>;

      //@}
      /**
       * @name Computing precomputed quantities
       */
      //@{

      /**
       * The number of precomputation cycles.
       */
      static constexpr unsigned int n_precomputation_cycles = 1;

      /**
       * Step 0: precompute values for hyperbolic update. This routine is
       * called within our usual loop() idiom in HyperbolicModule
       */
      template <typename DISPATCH, typename SPARSITY>
      void precomputation_loop(unsigned int cycle,
                               const DISPATCH &dispatch_check,
                               const SPARSITY &sparsity_simd,
                               StateVector &state_vector,
                               unsigned int left,
                               unsigned int right) const;

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
       * For a given (1+dim dimensional) state vector <code>U</code> and
       * left/right topography states <code>Z_left</code> and
       * <code>Z_right</code>, return the star_state <code>U_star</code>
       */
      state_type star_state(const state_type &U,
                            const Number &Z_left,
                            const Number &Z_right) const;

      /**
       * Given precomputed flux contributions @p prec_i and @p prec_j
       * compute the equilibrated states \f$U_i^{\ast,j}\f$ and
       * \f$U_j^{\ast,i}\f$.
       */
      std::array<state_type, 2>
      equilibrated_states(const flux_contribution_type &,
                          const flux_contribution_type &) const;

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
       *     const auto flux_ij = flux_divergence(flux_i, flux_j, c_ij);
       *   }
       * }
       * ```
       *
       * For the Shallow water equations we simply retrieve the
       * bathymetry and return, both, state and bathymetry.
       */
      flux_contribution_type
      flux_contribution(const PrecomputedVector &pv,
                        const InitialPrecomputedVector &piv,
                        const unsigned int i,
                        const state_type &U_i) const;

      flux_contribution_type
      flux_contribution(const PrecomputedVector &pv,
                        const InitialPrecomputedVector &piv,
                        const unsigned int *js,
                        const state_type &U_j) const;

      /**
       * Given precomputed flux contributions @p prec_i and @p prec_j
       * compute the equilibrated, low-order flux \f$(f(U_i^{\ast,j}) +
       * f(U_j^{\ast,i})\f$
       */
      state_type
      flux_divergence(const flux_contribution_type &flux_i,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &c_ij) const;

      /**
       * The low-order and high-order fluxes differ:
       */
      static constexpr bool have_high_order_flux = true;

      /**
       * Given precomputed flux contributions @p prec_i and @p prec_j
       * compute the high-order flux \f$(f(U_i^{\ast,j}) +
       * f(U_j^{\ast,i})\f$
       */
      state_type high_order_flux_divergence(
          const flux_contribution_type &flux_i,
          const flux_contribution_type &flux_j,
          const dealii::Tensor<1, dim, Number> &c_ij) const;

      /**
       * Given precomputed flux contributions @p prec_i and @p prec_j compute
       * the equilibrated, low-order affine shift
       * \f$ B_{ij} = -2d_ij(U^{\ast,j}_i)-2f((U^{\ast,j}_i))c_ij\f$.
       */
      state_type affine_shift(const flux_contribution_type &flux_i,
                              const flux_contribution_type &flux_j,
                              const dealii::Tensor<1, dim, Number> &c_ij,
                              const Number &d_ij) const;

      //@}
      /**
       * @name Computing source terms
       */
      //@{

      /** We do have source terms */
      static constexpr bool have_source_terms = true;

      /**
       * A regularized Gauckler-Manning friction coefficient.
       *
       * FIXME: details
       */
      state_type manning_friction(const state_type &U,
                                  const Number &h_star,
                                  const ScalarNumber tau) const;

      state_type nodal_source(const PrecomputedVector &pv,
                              const unsigned int i,
                              const state_type &U_i,
                              const ScalarNumber tau) const;

      state_type nodal_source(const PrecomputedVector &pv,
                              const unsigned int *js,
                              const state_type &U_j,
                              const ScalarNumber tau) const;

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
       * Given an initial state [h, u_1, ..., u_d] return a
       * conserved state [h, m_1, ..., m_d].
       *
       * This function simply calls from_primitive_state() and
       * expand_state().
       *
       * @note This function is used to conveniently convert (user
       * provided) primitive initial states to a conserved state in the
       * ShallowWaterInitialStateLibrary.
       */
      template <typename ST>
      state_type from_initial_state(const ST &initial_state) const;

      /**
       * Given a primitive state [h, u_1, ..., u_d] return a conserved
       * state
       */
      state_type from_primitive_state(const state_type &primitive_state) const;

      /**
       * Given a conserved state return a primitive state [h, u_1, ..., u_d]
       */
      state_type to_primitive_state(const state_type &state) const;

      /**
       * Transform the current state according to a  given operator
       * @p lambda acting on a @a dim dimensional momentum (or velocity)
       * vector.
       */
      template <typename Lambda>
      state_type apply_galilei_transform(const state_type &state,
                                         const Lambda &lambda) const;

    }; /* HyperbolicSystemView */


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    inline HyperbolicSystem::HyperbolicSystem(const std::string &subsection)
        : ParameterAcceptor(subsection)
    {
      gravity_ = 9.81;
      add_parameter("gravity", gravity_, "Gravitational constant [m/s^2]");

      manning_friction_coefficient_ = 0.;
      add_parameter("manning friction coefficient",
                    manning_friction_coefficient_,
                    "Roughness coefficient for friction source");

      reference_water_depth_ = 1.;
      add_parameter("reference water depth",
                    reference_water_depth_,
                    "Problem specific water depth reference");

      dry_state_relaxation_factor_ = 2.e-1;
      add_parameter("dry state relaxation factor",
                    dry_state_relaxation_factor_,
                    "Problem specific dry-state relaxation parameter");

      dry_state_relaxation_small_ = 1.e2;
      add_parameter("dry state relaxation small",
                    dry_state_relaxation_small_,
                    "Problem specific dry-state relaxation parameter");

      dry_state_relaxation_large_ = 1.e4;
      add_parameter("dry state relaxation large",
                    dry_state_relaxation_large_,
                    "Problem specific dry-state relaxation parameter");
    }


    template <int dim, typename Number>
    template <typename DISPATCH, typename SPARSITY>
    DEAL_II_ALWAYS_INLINE inline void
    HyperbolicSystemView<dim, Number>::precomputation_loop(
        unsigned int cycle [[maybe_unused]],
        const DISPATCH &dispatch_check,
        const SPARSITY &sparsity_simd,
        StateVector &state_vector,
        unsigned int left,
        unsigned int right) const
    {
      Assert(cycle == 0, dealii::ExcInternalError());

      const auto &U = std::get<0>(state_vector);
      auto &precomputed = std::get<1>(state_vector);

      /* We are inside a thread parallel context */

      unsigned int stride_size = get_stride_size<Number>;

      RYUJIN_OMP_FOR
      for (unsigned int i = left; i < right; i += stride_size) {

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        dispatch_check(i);

        const auto U_i = U.template get_tensor<Number>(i);

        const auto eta_m = mathematical_entropy(U_i);

        const auto h_sharp = water_depth_sharp(U_i);
        const auto h_star = ryujin::pow(h_sharp, ScalarNumber(4. / 3.));

        const precomputed_type prec_i{eta_m, h_star};
        precomputed.template write_tensor<Number>(prec_i, i);
      }
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::water_depth(const state_type &U)
    {
      return U[0];
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::inverse_water_depth_mollified(
        const state_type &U) const
    {
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const Number h_cutoff_mollified =
          reference_water_depth() * dry_state_relaxation_large() * Number(eps);

      const Number h = water_depth(U);
      const Number h_pos = positive_part(water_depth(U));
      const Number h_max = std::max(h, h_cutoff_mollified);
      const Number denom = h * h + h_max * h_max;
      return ScalarNumber(2.) * h_pos / denom;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::water_depth_sharp(
        const state_type &U) const
    {
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const Number h_cutoff_small =
          reference_water_depth() * dry_state_relaxation_small() * Number(eps);

      const Number h = water_depth(U);
      const Number h_max = std::max(h, h_cutoff_small);
      return h_max;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::inverse_water_depth_sharp(
        const state_type &U) const
    {
      return ScalarNumber(1.) / water_depth_sharp(U);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::filter_dry_water_depth(
        const Number &h) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const Number h_cutoff_large =
          reference_water_depth() * dry_state_relaxation_large() * Number(eps);

      return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          std::abs(h), h_cutoff_large, Number(0.), h);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
    HyperbolicSystemView<dim, Number>::momentum(const state_type &U)
    {
      dealii::Tensor<1, dim, Number> result;

      for (unsigned int i = 0; i < dim; ++i)
        result[i] = U[1 + i];
      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::kinetic_energy(const state_type &U) const
    {
      const auto h = water_depth(U);
      const auto vel = momentum(U) * inverse_water_depth_sharp(U);

      /* KE = 1/2 h |v|^2 */
      return ScalarNumber(0.5) * h * vel.norm_square();
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::pressure(const state_type &U) const
    {
      const Number h_sqd = U[0] * U[0];

      /* p = 1/2 g h^2 */
      return ScalarNumber(0.5) * gravity() * h_sqd;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::speed_of_sound(const state_type &U) const
    {
      /* c^2 = g * h */
      return std::sqrt(gravity() * U[0]);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::mathematical_entropy(
        const state_type &U) const
    {
      const auto p = pressure(U);
      const auto k_e = kinetic_energy(U);
      return p + k_e;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::mathematical_entropy_derivative(
        const state_type &U) const -> state_type
    {
      /*
       * With
       *   eta = 1/2 g h^2 + 1/2 |m|^2 / h
       *
       * we get
       *
       *   eta' = (g h - 1/2 |vel|^2, vel)
       *
       * where vel = m / h
       */

      state_type result;

      const Number &h = U[0];
      const auto vel = momentum(U) * inverse_water_depth_sharp(U);

      // water depth component
      result[0] = gravity() * h - ScalarNumber(0.5) * vel.norm_square();

      // momentum components
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = vel[i];
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline bool
    HyperbolicSystemView<dim, Number>::is_admissible(const state_type &U) const
    {
      const auto h = filter_dry_water_depth(water_depth(U));

      constexpr auto gte = dealii::SIMDComparison::greater_than_or_equal;
      const auto test = dealii::compare_and_apply_mask<gte>(
          h, Number(0.), Number(0.), Number(-1.));

#ifdef DEBUG_OUTPUT
      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: Negative state [h] detected!\n";
        std::cout << "\t\th: " << h << "\n" << std::endl;
        __builtin_trap();
      }
#endif

      return (test == Number(0.));
    }


    template <int dim, typename Number>
    template <int component>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::prescribe_riemann_characteristic(
        const state_type &U,
        const state_type &U_bar,
        const dealii::Tensor<1, dim, Number> &normal) const -> state_type
    {
      /* Note that U_bar are the dirichlet values that are prescribed */
      static_assert(component == 1 || component == 2,
                    "component has to be 1 or 2");

      using ScalarNumber = typename get_value_type<Number>::type;

      const auto m = momentum(U);
      const auto a = speed_of_sound(U);
      const auto vn = m * normal * inverse_water_depth_sharp(U);

      const auto m_bar = momentum(U_bar);
      const auto a_bar = speed_of_sound(U_bar);
      const auto vn_bar = m_bar * normal * inverse_water_depth_sharp(U_bar);

      /* First Riemann characteristic: v * n - 2 * a */

      const auto R_1 = component == 1 ? vn_bar - ScalarNumber(2.) * a_bar
                                      : vn - ScalarNumber(2.) * a;

      /* Second Riemann characteristic: v * n + 2 * a */

      const auto R_2 = component == 2 ? vn_bar + ScalarNumber(2.) * a_bar
                                      : vn + ScalarNumber(2.) * a;

      const auto vperp = m * inverse_water_depth_sharp(U) - vn * normal;

      const auto vn_new = ScalarNumber(0.5) * (R_1 + R_2);

      const auto h_new =
          ryujin::fixed_power<2>((R_2 - R_1) / ScalarNumber(4.)) / gravity();

      state_type U_new;
      U_new[0] = h_new;
      for (unsigned int d = 0; d < dim; ++d) {
        U_new[1 + d] = h_new * (vn_new * normal + vperp)[d];
      }

      return U_new;
    }


    template <int dim, typename Number>
    template <typename Lambda>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::apply_boundary_conditions(
        const dealii::types::boundary_id id,
        const state_type &U,
        const dealii::Tensor<1, dim, Number> &normal,
        const Lambda &get_dirichlet_data) const -> state_type
    {
      state_type result = U;

      if (id == Boundary::dirichlet) {
        result = get_dirichlet_data();

      } else if (id == Boundary::dirichlet_momentum) {
        /*
         * For some benchmarks for the Shallow Water Equations, only the
         * momentum is enforced. We "do nothing" for the water
         * depth.
         */
        auto m_dir = momentum(get_dirichlet_data());
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = m_dir[k];

      } else if (id == Boundary::slip) {
        auto m = momentum(U);
        m -= 1. * (m * normal) * normal;
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = m[k];

      } else if (id == Boundary::no_slip) {
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = Number(0.);

      } else if (id == Boundary::dynamic) {
        /*
         * On dynamic boundary conditions, we distinguish four cases:
         *
         *  - supersonic inflow: prescribe full state
         *  - subsonic inflow:
         *      decompose into Riemann invariants and leave R_2
         *      characteristic untouched.
         *  - supersonic outflow: do nothing
         *  - subsonic outflow:
         *      decompose into Riemann invariants and prescribe incoming
         *      R_1 characteristic.
         */
        const auto m = momentum(U);
        const auto h_inverse = inverse_water_depth_sharp(U);
        const auto a = speed_of_sound(U);
        const auto vn = m * normal * h_inverse;

        /* Supersonic inflow: */
        if (vn < -a) {
          result = get_dirichlet_data();
        }

        /* Subsonic inflow: */
        if (vn >= -a && vn <= 0.) {
          const auto U_dirichlet = get_dirichlet_data();
          result = prescribe_riemann_characteristic<2>(U_dirichlet, U, normal);
        }

        /* Subsonic outflow: */
        if (vn > 0. && vn <= a) {
          const auto U_dirichlet = get_dirichlet_data();
          result = prescribe_riemann_characteristic<1>(U, U_dirichlet, normal);
        }

        /* Supersonic outflow: do nothing, i.e., keep U as is */
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::f(const state_type &U) const -> flux_type
    {
      const auto h_inverse = inverse_water_depth_sharp(U);
      const auto m = momentum(U);
      const auto p = pressure(U);

      flux_type result;

      result[0] = (m * h_inverse) * U[0];
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = (m * h_inverse) * m[i];
        result[1 + i][i] += p;
      }
      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::g(const state_type &U) const -> flux_type
    {
      const auto h_inverse = inverse_water_depth_sharp(U);
      const auto m = momentum(U);

      flux_type result;

      result[0] = (m * h_inverse) * U[0];
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = (m * h_inverse) * m[i];
      }
      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::star_state(const state_type &U,
                                                  const Number &Z_left,
                                                  const Number &Z_right) const
        -> state_type
    {
      const Number Z_max = std::max(Z_left, Z_right);
      const Number h = water_depth(U);
      const Number H_star = std::max(Number(0.), h + Z_left - Z_max);

      return U * H_star * inverse_water_depth_mollified(U);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::equilibrated_states(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j) const -> std::array<state_type, 2>
    {
      const auto &[U_i, Z_i] = flux_i;
      const auto &[U_j, Z_j] = flux_j;

      const auto U_star_ij = star_state(U_i, Z_i, Z_j);
      const auto U_star_ji = star_state(U_j, Z_j, Z_i);

      return {U_star_ij, U_star_ji};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_contribution(
        const PrecomputedVector & /*pv*/,
        const InitialPrecomputedVector &piv,
        const unsigned int i,
        const state_type &U_i) const -> flux_contribution_type
    {
      const auto Z_i = piv.template get_tensor<Number>(i)[0];
      return {U_i, Z_i};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_contribution(
        const PrecomputedVector & /*pv*/,
        const InitialPrecomputedVector &piv,
        const unsigned int *js,
        const state_type &U_j) const -> flux_contribution_type
    {
      const auto Z_j = piv.template get_tensor<Number>(js)[0];
      return {U_j, Z_j};
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_divergence(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij) const -> state_type
    {
      const auto &[U_i, Z_i] = flux_i;
      const auto &[U_star_ij, U_star_ji] = equilibrated_states(flux_i, flux_j);

      const auto H_i = water_depth(U_i);
      const auto H_star_ij = water_depth(U_star_ij);
      const auto H_star_ji = water_depth(U_star_ji);

      const auto g_i = g(U_star_ij);
      const auto g_j = g(U_star_ji);

      auto result = -add(g_i, g_j);

      const auto factor =
          (ScalarNumber(0.5) * (H_star_ji * H_star_ji - H_star_ij * H_star_ij) +
           H_i * H_i) *
          gravity();

      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i][i] -= factor;
      }

      return contract(result, c_ij);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::high_order_flux_divergence(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij) const -> state_type
    {
      const auto &[U_i, Z_i] = flux_i;
      const auto &[U_j, Z_j] = flux_j;

      const auto H_i = water_depth(U_i);
      const auto H_j = water_depth(U_j);

      const auto g_i = g(U_i);
      const auto g_j = g(U_j);

      auto result = -add(g_i, g_j);

      const auto factor = gravity() * H_i * (H_j + Z_j - Z_i);
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i][i] -= factor;
      }

      return contract(result, c_ij);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::affine_shift(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij,
        const Number &d_ij) const -> state_type
    {
      const auto &[U_i, Z_i] = flux_i;
      const auto &[U_j, Z_j] = flux_j;
      const auto U_star_ij = star_state(U_i, Z_i, Z_j);

      const auto h_inverse = inverse_water_depth_sharp(U_i);
      const auto m = momentum(U_i);
      const auto factor = ScalarNumber(2.) * (d_ij + h_inverse * (m * c_ij));

      return -factor * (U_star_ij - U_i);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::manning_friction(
        const state_type &U, const Number &h_star, const ScalarNumber tau) const
        -> state_type
    {
      state_type result;

      const auto g = gravity();
      const auto n = manning_friction_coefficient();

      const auto h_inverse = inverse_water_depth_mollified(U);

      const auto m = momentum(U);
      const auto v_norm = (m * h_inverse).norm();
      const auto factor = ScalarNumber(2.) * g * n * n * v_norm;

      const auto denominator = h_star + std::max(h_star, tau * factor);
      const auto denominator_inverse = ScalarNumber(1.) / denominator;

      for (unsigned int d = 0; d < dim; ++d)
        result[d + 1] = -factor * denominator_inverse * m[d];

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::nodal_source(
        const PrecomputedVector &pv,
        const unsigned int i,
        const state_type &U_i,
        const ScalarNumber tau) const -> state_type
    {
      const auto &[eta_m, h_star] =
          pv.template get_tensor<Number, precomputed_type>(i);

      return manning_friction(U_i, h_star, tau);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::nodal_source(
        const PrecomputedVector &pv,
        const unsigned int *js,
        const state_type &U_j,
        const ScalarNumber tau) const -> state_type
    {
      const auto &[eta_m, h_star] =
          pv.template get_tensor<Number, precomputed_type>(js);

      return manning_friction(U_j, h_star, tau);
    }


    template <int dim, typename Number>
    template <typename ST>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::expand_state(const ST &state) const
        -> state_type
    {
      using T = typename ST::value_type;
      static_assert(std::is_same_v<Number, T>, "template mismatch");

      constexpr auto dim2 = ST::dimension - 1;
      static_assert(dim >= dim2,
                    "the space dimension of the argument state must not be "
                    "larger than the one of the target state");

      state_type result;
      result[0] = state[0];
      for (unsigned int i = 1; i < dim2 + 1; ++i)
        result[i] = state[i];

      return result;
    }

    template <int dim, typename Number>
    template <typename ST>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::from_initial_state(
        const ST &initial_state) const -> state_type
    {
      const auto primitive_state = expand_state(initial_state);
      return from_primitive_state(primitive_state);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::from_primitive_state(
        const state_type &primitive_state) const -> state_type
    {
      const auto &h = primitive_state[0];

      auto state = primitive_state;
      /* Fix up momentum: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        state[i] *= h;

      return state;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::to_primitive_state(
        const state_type &state) const -> state_type
    {
      const auto h_inverse = inverse_water_depth_sharp(state);

      auto primitive_state = state;
      /* Fix up velocity: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        primitive_state[i] *= h_inverse;

      return primitive_state;
    }


    template <int dim, typename Number>
    template <typename Lambda>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::apply_galilei_transform(
        const state_type &state, const Lambda &lambda) const -> state_type
    {
      auto result = state;
      auto M = lambda(momentum(state));
      for (unsigned int d = 0; d < dim; ++d)
        result[1 + d] = M[d];
      return result;
    }

  } // namespace ShallowWater
} // namespace ryujin
