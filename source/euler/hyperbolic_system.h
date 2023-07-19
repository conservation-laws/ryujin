//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
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
  namespace Euler
  {
    /**
     * Description of a @p dim dimensional hyperbolic conservation law.
     *
     * We have a (2 + dim) dimensional state space \f$[\rho, \textbf m,
     * E]\f$, where \f$\rho\f$ denotes the density, \f$\textbf m\f$ is the
     * momentum, and \f$E\f$ is the total energy.
     *
     * @ingroup EulerEquations
     */
    class HyperbolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the hyperbolic system as a string.
       */
      static inline const std::string problem_name =
          "Compressible Euler equations (polytropic gas EOS)";

      /**
       * Constructor.
       */
      HyperbolicSystem(const std::string &subsection = "/HyperbolicSystem");

    private:
      double gamma_;
      double reference_density_;
      double vacuum_state_relaxation_;

      double gamma_inverse_;
      double gamma_minus_one_inverse_;
      double gamma_minus_one_over_gamma_plus_one_;
      double gamma_plus_one_inverse_;

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

        DEAL_II_ALWAYS_INLINE inline ScalarNumber gamma() const
        {
          return ScalarNumber(hyperbolic_system_.gamma_);
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber reference_density() const
        {
          return ScalarNumber(hyperbolic_system_.reference_density_);
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber
        vacuum_state_relaxation() const
        {
          return ScalarNumber(hyperbolic_system_.vacuum_state_relaxation_);
        }

        //@}
        /**
         * @name Access to cached inverses
         *
         * A collection of commonly used expressions with gamma that would
         * otherwise need to be recomputed many times putting unnecessary
         * pressure on the div/sqrt ALU unit.
         */
        //@{

        DEAL_II_ALWAYS_INLINE inline ScalarNumber gamma_inverse() const
        {
          return ScalarNumber(hyperbolic_system_.gamma_inverse_);
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber gamma_plus_one_inverse() const
        {
          return ScalarNumber(hyperbolic_system_.gamma_plus_one_inverse_);
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber
        gamma_minus_one_inverse() const
        {
          return ScalarNumber(hyperbolic_system_.gamma_minus_one_inverse_);
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber
        gamma_minus_one_over_gamma_plus_one() const
        {
          return ScalarNumber(
              hyperbolic_system_.gamma_minus_one_over_gamma_plus_one_);
        }

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
        static constexpr unsigned int problem_dimension = 2 + dim;

        /**
         * The storage type used for a (conserved) state vector \f$\boldsymbol
         * U\f$.
         */
        using state_type = dealii::Tensor<1, problem_dimension, Number>;

        /**
         * An array holding all component names of the conserved state as a
         * string.
         */
        static inline const auto component_names =
            []() -> std::array<std::string, problem_dimension> {
          if constexpr (dim == 1)
            return {"rho", "m", "E"};
          else if constexpr (dim == 2)
            return {"rho", "m_1", "m_2", "E"};
          else if constexpr (dim == 3)
            return {"rho", "m_1", "m_2", "m_3", "E"};
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
            return {"rho", "v", "p"};
          else if constexpr (dim == 2)
            return {"rho", "v_1", "v_2", "p"};
          else if constexpr (dim == 3)
            return {"rho", "v_1", "v_2", "v_3", "p"};
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
         * An array holding all component names of the precomputed values.
         */
        static inline const auto precomputed_initial_names =
            std::array<std::string, n_precomputed_initial_values>{};

        /**
         * The number of precomputed values.
         */
        static constexpr unsigned int n_precomputed_values = 2;

        /**
         * Array type used for precomputed values.
         */
        using precomputed_state_type = std::array<Number, n_precomputed_values>;

        /**
         * An array holding all component names of the precomputed values.
         */
        static inline const auto precomputed_names =
            std::array<std::string, n_precomputed_values>{"s", "eta_h"};

        /**
         * The number of precomputation cycles.
         */
        static constexpr unsigned int n_precomputation_cycles = 1;

        /**
         * Precomputed values for a given state.
         */
        template <unsigned int cycle, typename MCV, typename SPARSITY>
        void precomputation(
            MCV &precomputed_values,
            const MultiComponentVector<ScalarNumber, problem_dimension> &U,
            const SPARSITY &sparsity_simd,
            unsigned int i) const;

        //@}
        /**
         * @name Computing derived physical quantities
         */
        //@{

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, return
         * the density <code>U[0]</code>
         */
        static Number density(const state_type &U);

        /**
         * Given a density @ref rho this function returns 0 if rho is in the
         * interval [-relaxation * rho_cutoff, relaxation * rho_cutoff],
         * otherwise rho is returned unmodified. Here, rho_cutoff is the
         * reference density multiplied by eps.
         */
        Number filter_vacuum_density(const Number &rho) const;

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, return
         * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
         */
        static dealii::Tensor<1, dim, Number> momentum(const state_type &U);

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, return
         * the total energy <code>U[1+dim]</code>
         */
        static Number total_energy(const state_type &U);

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, compute
         * and return the internal energy \f$\varepsilon = (\rho e)\f$.
         */
        static Number internal_energy(const state_type &U);

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, compute
         * and return the derivative of the internal energy
         * \f$\varepsilon = (\rho e)\f$.
         */
        static state_type internal_energy_derivative(const state_type &U);

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, compute
         * and return the pressure \f$p\f$.
         *
         * We assume that the pressure is given by a polytropic equation of
         * state, i.e.,
         * \f[
         *   p = \frac{\gamma - 1}{1 - b*\rho}\; (\rho e)
         * \f]
         *
         * @note If you want to set the covolume paramete @ref b_ to nonzero
         * you have to enable the @ref covolume_ compile-time option.
         */
        Number pressure(const state_type &U) const;

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, compute
         * the (physical) speed of sound:
         * \f[
         *   c^2 = \frac{\gamma * p}{\rho\;(1 - b * \rho)}
         * \f]
         */
        Number speed_of_sound(const state_type &U) const;

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, compute
         * and return the (scaled) specific entropy
         * \f[
         *   e^{(\gamma-1)s} = \frac{\rho\,e}{\rho^\gamma}
         *   (1 - b * \rho)^(\gamma -1).
         * \f]
         */
        Number specific_entropy(const state_type &U) const;

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, compute
         * and return the Harten-type entropy
         * \f[
         *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
         * \f]
         */
        Number harten_entropy(const state_type &U) const;

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, compute
         * and return the derivative \f$\eta'\f$ of the Harten-type entropy
         * \f[
         *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
         * \f]
         */
        state_type harten_entropy_derivative(const state_type &U) const;

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, compute
         * and return the entropy \f$\eta = p^{1/\gamma}\f$.
         */
        Number mathematical_entropy(const state_type &U) const;

        /**
         * For a given (2+dim dimensional) state vector <code>U</code>, compute
         * and return the derivative \f$\eta'\f$ of the entropy \f$\eta =
         * p^{1/\gamma}\f$.
         */
        state_type mathematical_entropy_derivative(const state_type &U) const;

        /**
         * Returns whether the state @ref U is admissible. If @ref U is a
         * vectorized state then @ref U is admissible if all vectorized values
         * are admissible.
         */
        bool is_admissible(const state_type &U) const;

        //@}
        /**
         * @name Special functions for boundary states
         */
        //@{

        /**
         * For a given state @p U and normal direction @p normal returns the
         * n-th pair of left and right eigenvectors of the linearized normal
         * flux.
         */
        template <int component>
        std::array<state_type, 2> linearized_eigenvector(
            const state_type &U,
            const dealii::Tensor<1, dim, Number> &normal) const;

        /**
         * Decomposes a given state @p U into Riemann invariants and then
         * replaces the first or second Riemann characteristic from the one
         * taken from @p U_bar state. Note that the @p U_bar state is just the
         * prescribed dirichlet values.
         */
        template <int component>
        state_type prescribe_riemann_characteristic(
            const state_type &U,
            const state_type &U_bar,
            const dealii::Tensor<1, dim, Number> &normal) const;

        /**
         * Apply boundary conditions.
         *
         * For the compressible Euler equations we have:
         *
         *  - Dirichlet boundary conditions by prescribing the return value of
         *    get_dirichlet_data() as is.
         *
         *  - Slip boundary conditions where we remove the normal component of
         *    the momentum.
         *
         *  - No slip boundary conditions where we set the momentum to 0.
         *
         *  - "Dynamic boundary" conditions that prescribe different Riemann
         *    invariants from the return value of get_dirichlet_data()
         *    depending on the flow state (supersonic versus subsonic, outflow
         *    versus inflow).
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
         *   \textbf v(E+p)
         * \end{pmatrix},
         * \f]
         */
        flux_type f(const state_type &U) const;

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
        template <typename MCV1, typename MCV2>
        flux_contribution_type
        flux_contribution(const MCV1 &precomputed_values,
                          const MCV2 & /*precomputed_initial_values*/,
                          const unsigned int i,
                          const state_type &U_i) const;

        template <typename MCV1, typename MCV2>
        flux_contribution_type
        flux_contribution(const MCV1 &precomputed_values,
                          const MCV2 & /*precomputed_initial_values*/,
                          const unsigned int *js,
                          const state_type &U_j) const;

        /**
         * Given flux contributions @p flux_i and @p flux_j compute the flux
         * <code>(-f(U_i) - f(U_j)</code>
         */
        flux_type flux(const flux_contribution_type &flux_i,
                       const flux_contribution_type &flux_j) const;

        /**
         * The low-order and high-order fluxes are the same:
         */
        static constexpr bool have_high_order_flux = false;

        flux_type
        high_order_flux(const flux_contribution_type &flux_i,
                        const flux_contribution_type &flux_j) const = delete;

        /** We do not perform state equilibration */
        static constexpr bool have_equilibrated_states = false;

        std::array<state_type, 2>
        equilibrated_states(const flux_contribution_type &flux_i,
                            const flux_contribution_type &flux_j) = delete;

        //@}
        /**
         * @name Computing stencil source terms
         */
        //@{

        /** We do not have source terms */
        static constexpr bool have_source_terms = false;

        template <typename MultiComponentVector>
        state_type low_order_nodal_source(const MultiComponentVector &,
                                          const unsigned int,
                                          const state_type &) const = delete;

        template <typename MultiComponentVector>
        state_type high_order_nodal_source(const MultiComponentVector &,
                                           const unsigned int i,
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
        state_type expand_state(const ST &state) const;

        /*
         * Given a primitive state [rho, u_1, ..., u_d, p] return a conserved
         * state
         */
        state_type
        from_primitive_state(const primitive_state_type &primitive_state) const;

        /*
         * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
         * p]
         */
        primitive_state_type to_primitive_state(const state_type &state) const;

        /*
         * Transform the current state according to a  given operator @ref
         * momentum_transform acting on a @p dim dimensional momentum (or
         * velocity) vector.
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


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    inline HyperbolicSystem::HyperbolicSystem(const std::string &subsection)
        : ParameterAcceptor(subsection)
    {
      gamma_ = 7. / 5.;
      add_parameter("gamma", gamma_, "The ratio of specific heats");

      reference_density_ = 1.;
      add_parameter("reference density",
                    reference_density_,
                    "Problem specific density reference");

      vacuum_state_relaxation_ = 10000.;
      add_parameter("vacuum state relaxation",
                    vacuum_state_relaxation_,
                    "Problem specific vacuum relaxation parameter");

      /*
       * Precompute a number of derived gamma coefficients that contain
       * divisions:
       */
      const auto compute_inverses = [this] {
        gamma_inverse_ = 1. / gamma_;
        gamma_plus_one_inverse_ = 1. / (gamma_ + 1.);
        gamma_minus_one_inverse_ = 1. / (gamma_ - 1.);
        gamma_minus_one_over_gamma_plus_one_ = (gamma_ - 1.) / (gamma_ + 1.);
      };

      compute_inverses();
      ParameterAcceptor::parse_parameters_call_back.connect(compute_inverses);
    }


    template <int dim, typename Number>
    template <unsigned int cycle, typename MCV, typename SPARSITY>
    DEAL_II_ALWAYS_INLINE inline void
    HyperbolicSystem::View<dim, Number>::precomputation(
        MCV &precomputed_values,
        const MultiComponentVector<ScalarNumber, problem_dimension> &U,
        const SPARSITY & /*sparsity_simd*/,
        unsigned int i) const
    {
      static_assert(cycle == 0, "internal error");

      const auto U_i = U.template get_tensor<Number>(i);
      const precomputed_state_type prec_i{specific_entropy(U_i),
                                          harten_entropy(U_i)};
      precomputed_values.template write_tensor<Number>(prec_i, i);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::density(const state_type &U)
    {
      return U[0];
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::filter_vacuum_density(
        const Number &rho) const
    {
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const Number rho_cutoff_big =
          reference_density() * vacuum_state_relaxation() * eps;

      return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          std::abs(rho), rho_cutoff_big, Number(0.), rho);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
    HyperbolicSystem::View<dim, Number>::momentum(const state_type &U)
    {
      dealii::Tensor<1, dim, Number> result;
      for (unsigned int i = 0; i < dim; ++i)
        result[i] = U[1 + i];
      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::total_energy(const state_type &U)
    {
      return U[1 + dim];
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::internal_energy(const state_type &U)
    {
      /*
       * rho e = (E - 1/2*m^2/rho)
       */
      const Number rho_inverse = ScalarNumber(1.) / U[0];
      const auto m = momentum(U);
      const Number E = U[dim + 1];
      return E - ScalarNumber(0.5) * m.norm_square() * rho_inverse;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::internal_energy_derivative(
        const state_type &U) -> state_type
    {
      /*
       * With
       *   rho e = E - 1/2 |m|^2 / rho
       * we get
       *   (rho e)' = (1/2m^2/rho^2, -m/rho , 1 )^T
       */

      const Number rho_inverse = ScalarNumber(1.) / U[0];
      const auto u = momentum(U) * rho_inverse;

      state_type result;

      result[0] = ScalarNumber(0.5) * u.norm_square();
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = -u[i];
      }
      result[dim + 1] = ScalarNumber(1.);

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::pressure(const state_type &U) const
    {
      /* p = (gamma - 1) / (1 - b * rho) * (rho e) */
      return (gamma() - ScalarNumber(1.)) * internal_energy(U);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::speed_of_sound(
        const state_type &U) const
    {
      /* c^2 = gamma * p / rho / (1 - b * rho) */
      const Number rho_inverse = ScalarNumber(1.) / U[0];
      const Number p = pressure(U);
      return std::sqrt(gamma() * p * rho_inverse);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::specific_entropy(
        const state_type &U) const
    {
      /* exp((gamma - 1)s) = (rho e) / rho ^ gamma */
      const auto rho_inverse = ScalarNumber(1.) / U[0];
      return internal_energy(U) * ryujin::pow(rho_inverse, gamma());
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::harten_entropy(
        const state_type &U) const
    {
      /* rho^2 e = \rho E - 1/2*m^2 */

      const Number rho = U[0];
      const auto m = momentum(U);
      const Number E = U[dim + 1];

      const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();
      return ryujin::pow(rho_rho_e, gamma_plus_one_inverse());
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::harten_entropy_derivative(
        const state_type &U) const -> state_type
    {
      /*
       * With
       *   eta = (rho^2 e) ^ 1/(gamma+1)
       *   rho^2 e = rho * E - 1/2 |m|^2
       *
       * we get
       *
       *   eta' = 1/(gamma+1) * (rho^2 e) ^ -gamma/(gamma+1) * (E,-m,rho)^T
       *
       */

      const Number rho = U[0];
      const auto m = momentum(U);
      const Number E = U[dim + 1];

      const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

      const auto factor =
          gamma_plus_one_inverse() *
          ryujin::pow(rho_rho_e, -gamma() * gamma_plus_one_inverse());

      state_type result;

      result[0] = factor * E;
      for (unsigned int i = 0; i < dim; ++i)
        result[1 + i] = -factor * m[i];
      result[dim + 1] = factor * rho;

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::mathematical_entropy(
        const state_type &U) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;
      const auto p = pressure(U);
      return ryujin::pow(p, gamma_inverse());
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::mathematical_entropy_derivative(
        const state_type &U) const -> state_type
    {
      /*
       * With
       *   eta = p ^ (1/gamma)
       *   p = (gamma - 1) * (rho e)
       *   rho e = E - 1/2 |m|^2 / rho
       *
       * we get
       *
       *   eta' = (gamma - 1)/gamma p ^(1/gamma - 1) *
       *
       *     (1/2m^2/rho^2 , -m/rho , 1 )^T
       */
      const Number &rho = U[0];
      const Number rho_inverse = ScalarNumber(1.) / rho;
      const auto u = momentum(U) * rho_inverse;
      const auto p = pressure(U);

      const auto factor = (gamma() - ScalarNumber(1.0)) * gamma_inverse() *
                          ryujin::pow(p, gamma_inverse() - ScalarNumber(1.));

      state_type result;

      result[0] = factor * ScalarNumber(0.5) * u.norm_square();
      result[dim + 1] = factor;
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = -factor * u[i];
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline bool
    HyperbolicSystem::View<dim, Number>::is_admissible(
        const state_type &U) const
    {
      const auto rho_new = density(U);
      const auto e_new = internal_energy(U);
      const auto s_new = specific_entropy(U);

      constexpr auto gt = dealii::SIMDComparison::greater_than;
      using T = Number;
      const auto test =
          dealii::compare_and_apply_mask<gt>(rho_new, T(0.), T(0.), T(-1.)) + //
          dealii::compare_and_apply_mask<gt>(e_new, T(0.), T(0.), T(-1.)) +   //
          dealii::compare_and_apply_mask<gt>(s_new, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: Negative state [rho, e, s] detected!\n";
        std::cout << "\t\trho: " << rho_new << "\n";
        std::cout << "\t\tint: " << e_new << "\n";
        std::cout << "\t\tent: " << s_new << "\n" << std::endl;
      }
#endif

      return (test == Number(0.));
    }


    template <int dim, typename Number>
    template <int component>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::linearized_eigenvector(
        const state_type &U, const dealii::Tensor<1, dim, Number> &normal) const
        -> std::array<state_type, 2>
    {
      static_assert(component == 1 || component == problem_dimension,
                    "Only first and last eigenvectors implemented");

      const auto rho = density(U);
      const auto m = momentum(U);
      const auto v = m / rho;
      const auto a = speed_of_sound(U);
      const auto gamma = this->gamma();

      state_type b;
      state_type c;

      const auto e_k = 0.5 * v.norm_square();

      switch (component) {
      case 1:
        b[0] = (gamma - 1.) * e_k + a * v * normal;
        for (unsigned int i = 0; i < dim; ++i)
          b[1 + i] = (1. - gamma) * v[i] - a * normal[i];
        b[dim + 1] = gamma - 1.;
        b /= 2. * a * a;

        c[0] = 1.;
        for (unsigned int i = 0; i < dim; ++i)
          c[1 + i] = v[i] - a * normal[i];
        c[dim + 1] = a * a / (gamma - 1) + e_k - a * (v * normal);

        return {b, c};

      case problem_dimension:
        b[0] = (gamma - 1.) * e_k - a * v * normal;
        for (unsigned int i = 0; i < dim; ++i)
          b[1 + i] = (1. - gamma) * v[i] + a * normal[i];
        b[dim + 1] = gamma - 1.;
        b /= 2. * a * a;

        c[0] = 1.;
        for (unsigned int i = 0; i < dim; ++i)
          c[1 + i] = v[i] + a * normal[i];
        c[dim + 1] = a * a / (gamma - 1) + e_k + a * (v * normal);

        return {b, c};
      }

      __builtin_unreachable();
    }


    template <int dim, typename Number>
    template <int component>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::prescribe_riemann_characteristic(
        const state_type &U,
        const state_type &U_bar,
        const dealii::Tensor<1, dim, Number> &normal) const -> state_type
    {
      static_assert(component == 1 || component == 2,
                    "component has to be 1 or 2");

      const auto m = momentum(U);
      const auto rho = density(U);
      const auto a = speed_of_sound(U);
      const auto vn = m * normal / rho;

      const auto m_bar = momentum(U_bar);
      const auto rho_bar = density(U_bar);
      const auto a_bar = speed_of_sound(U_bar);
      const auto vn_bar = m_bar * normal / rho_bar;

      /* First Riemann characteristic: v* n - 2 / (gamma - 1) * a */

      const auto R_1 = component == 1
                           ? vn_bar - 2. * a_bar / (gamma() - ScalarNumber(1.))
                           : vn - 2. * a / (gamma() - ScalarNumber(1.));

      /* Second Riemann characteristic: v* n + 2 / (gamma() - 1) * a */

      const auto R_2 = component == 2
                           ? vn_bar + 2. * a_bar / (gamma() - ScalarNumber(1.))
                           : vn + 2. * a / (gamma() - ScalarNumber(1.));

      const auto p = pressure(U);
      const auto s = p / ryujin::pow(rho, gamma());

      const auto vperp = m / rho - vn * normal;

      const auto vn_new = 0.5 * (R_1 + R_2);

      auto rho_new = 1. / (gamma() * s) *
                     ryujin::fixed_power<2>(ScalarNumber((gamma() - 1.) / 4.) *
                                            (R_2 - R_1));
      rho_new = ryujin::pow(rho_new, 1. / (gamma() - 1.));

      const auto p_new = s * std::pow(rho_new, gamma());

      state_type U_new;
      U_new[0] = rho_new;
      for (unsigned int d = 0; d < dim; ++d) {
        U_new[1 + d] = rho_new * (vn_new * normal + vperp)[d];
      }
      U_new[1 + dim] = p_new / ScalarNumber(gamma() - 1.) +
                       0.5 * rho_new * (vn_new * vn_new + vperp.norm_square());

      return U_new;
    }


    template <int dim, typename Number>
    template <typename Lambda>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::apply_boundary_conditions(
        dealii::types::boundary_id id,
        const state_type &U,
        const dealii::Tensor<1, dim, Number> &normal,
        const Lambda &get_dirichlet_data) const -> state_type
    {
      state_type result = U;

      if (id == Boundary::dirichlet) {
        result = get_dirichlet_data();

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
        const auto rho = density(U);
        const auto a = speed_of_sound(U);
        const auto vn = m * normal / rho;

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
    HyperbolicSystem::View<dim, Number>::f(const state_type &U) const
        -> flux_type
    {
      const auto rho_inverse = ScalarNumber(1.) / U[0];
      const auto m = momentum(U);
      const auto p = pressure(U);
      const auto E = U[dim + 1];

      flux_type result;

      result[0] = m;
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = m * (m[i] * rho_inverse);
        result[1 + i][i] += p;
      }
      result[dim + 1] = m * (rho_inverse * (E + p));

      return result;
    }


    template <int dim, typename Number>
    template <typename MCV1, typename MCV2>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::flux_contribution(
        const MCV1 & /*precomputed_values*/,
        const MCV2 & /*precomputed_initial_values*/,
        const unsigned int /*i*/,
        const state_type &U_i) const -> flux_contribution_type
    {
      return f(U_i);
    }


    template <int dim, typename Number>
    template <typename MCV1, typename MCV2>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::flux_contribution(
        const MCV1 & /*precomputed_values*/,
        const MCV2 & /*precomputed_initial_values*/,
        const unsigned int * /*js*/,
        const state_type &U_j) const -> flux_contribution_type
    {
      return f(U_j);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto HyperbolicSystem::View<dim, Number>::flux(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j) const -> flux_type
    {
      return -add(flux_i, flux_j);
    }


    template <int dim, typename Number>
    template <typename ST>
    auto
    HyperbolicSystem::View<dim, Number>::expand_state(const ST &state) const
        -> state_type
    {
      using T = typename ST::value_type;
      static_assert(std::is_same_v<Number, T>, "template mismatch");

      constexpr auto dim2 = ST::dimension - 2;
      static_assert(dim >= dim2,
                    "the space dimension of the argument state must not be "
                    "larger than the one of the target state");

      state_type result;
      result[0] = state[0];
      result[dim + 1] = state[dim2 + 1];
      for (unsigned int i = 1; i < dim2 + 1; ++i)
        result[i] = state[i];

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::from_primitive_state(
        const primitive_state_type &primitive_state) const -> state_type
    {
      const auto &rho = primitive_state[0];
      /* extract velocity: */
      const auto u = /*SIC!*/ momentum(primitive_state);
      const auto &p = primitive_state[dim + 1];

      auto state = primitive_state;
      /* Fix up momentum: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        state[i] *= rho;
      /* Compute total energy: */
      state[dim + 1] =
          p / (ScalarNumber(gamma() - 1.)) + Number(0.5) * rho * u * u;

      return state;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::to_primitive_state(
        const state_type &state) const -> primitive_state_type
    {
      const auto &rho = state[0];
      const auto rho_inverse = Number(1.) / rho;
      const auto p = pressure(state);

      auto primitive_state = state;
      /* Fix up velocity: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        primitive_state[i] *= rho_inverse;
      /* Set pressure: */
      primitive_state[dim + 1] = p;

      return primitive_state;
    }


    template <int dim, typename Number>
    template <typename Lambda>
    auto HyperbolicSystem::View<dim, Number>::apply_galilei_transform(
        const state_type &state, const Lambda &lambda) const -> state_type
    {
      auto result = state;
      const auto M = lambda(momentum(state));
      for (unsigned int d = 0; d < dim; ++d)
        result[1 + d] = M[d];
      return result;
    }

  } // namespace Euler
} // namespace ryujin
