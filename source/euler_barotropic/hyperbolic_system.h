//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "barotropic_equation_of_state_library.h"

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
  namespace EulerBarotropic
  {
    /*
     * For various divisions in the barotropic equation of state module we
     * have a mathematical guarantee that the numerator and denominator are
     * nonnegative and the limit (of zero numerator and denominator) must
     * converge to zero. The following function takes care of rounding
     * issues when computing such quotients by (a) avoiding division by
     * zero and (b) ensuring non-negativity of the result.
     */
    template <typename Number>
    DEAL_II_ALWAYS_INLINE inline Number safe_division(const Number &numerator,
                                                      const Number &denominator)
    {
      using ScalarNumber = typename get_value_type<Number>::type;
      constexpr ScalarNumber min = std::numeric_limits<ScalarNumber>::min();

      return std::max(numerator, Number(0.)) /
             std::max(denominator, Number(min));
    }


    template <int dim, typename Number>
    class HyperbolicSystemView;

    /**
     * The compressible Euler equations of gas dynamics. Specialized
     * implementation for a subclass of barotropic equations of state where
     * the pressure, internal energy and entropies are a function of the
     * density. We use a specialied Riemann solver, entropy viscosity
     * commutator, and limiter for this class of equations.
     *
     * We have a (1 + dim) dimensional state space \f$[\rho, \textbf m]\f$,
     * where \f$\rho\f$ denotes the density, \f$\textbf m\f$ is the
     * momentum.
     *
     * @ingroup EulerEquations
     */
    class HyperbolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the hyperbolic system as a string.
       */
      static inline std::string problem_name =
          "Compressible Euler equations (barotropic EOS, optimized barotropic)";

      /**
       * Constructor.
       */
      HyperbolicSystem(const std::string &subsection = "/HyperbolicSystem");

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
          const OfflineData<dim, ScalarNumber> &offline_data,
          typename HyperbolicSystemView<dim, ScalarNumber>::StateVector
              &state_vector,
          const bool skip_constrained_dofs = true) const;

    private:
      /**
       * @name Runtime parameters, internal fields, methods, and friends
       */
      //@{

      std::string barotropic_equation_of_state_;
      double reference_density_;
      double vacuum_state_relaxation_small_;
      double vacuum_state_relaxation_large_;

      BarotropicEquationOfStateLibrary::equation_of_state_list_type
          barotropic_equation_of_state_list_;

      using BarotropicEquationOfState =
          BarotropicEquationOfStateLibrary::BarotropicEquationOfState;
      std::shared_ptr<BarotropicEquationOfState>
          selected_barotropic_equation_of_state_;

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

      DEAL_II_ALWAYS_INLINE inline const std::string &
      barotropic_equation_of_state() const
      {
        return hyperbolic_system_.barotropic_equation_of_state_;
      }

      DEAL_II_ALWAYS_INLINE inline ScalarNumber reference_density() const
      {
        return hyperbolic_system_.reference_density_;
      }

      DEAL_II_ALWAYS_INLINE inline ScalarNumber
      vacuum_state_relaxation_small() const
      {
        return hyperbolic_system_.vacuum_state_relaxation_small_;
      }

      DEAL_II_ALWAYS_INLINE inline ScalarNumber
      vacuum_state_relaxation_large() const
      {
        return hyperbolic_system_.vacuum_state_relaxation_large_;
      }

      //@}
      /**
       * @name Low-level access to the selected equation of state
       */
      //@{

      /**
       * For a given density \f$\rho\f$ return the
       * <i>specific</i> internal energy \f$e\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number
      beos_specific_internal_energy(const Number &rho) const
      {
        const auto &beos =
            hyperbolic_system_.selected_barotropic_equation_of_state_;

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          return ScalarNumber(beos->specific_internal_energy(rho));
        } else {
          Number e;
          for (unsigned int k = 0; k < Number::size(); ++k) {
            e[k] = ScalarNumber(beos->specific_internal_energy(rho[k]));
          }
          return e;
        }
      }

      /**
       * For a given density \f$\rho\f$ return the pressure \f$p\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number beos_pressure(const Number &rho) const
      {
        const auto &beos =
            hyperbolic_system_.selected_barotropic_equation_of_state_;

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          return ScalarNumber(beos->pressure(rho));
        } else {
          Number p;
          for (unsigned int k = 0; k < Number::size(); ++k) {
            p[k] = ScalarNumber(beos->pressure(rho[k]));
          }
          return p;
        }
      }

      /**
       * For a given density \f$\rho\f$ and <i>specific</i> internal
       * energy \f$e\f$ return the sound speed \f$a\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number
      beos_speed_of_sound(const Number &rho) const
      {
        const auto &beos =
            hyperbolic_system_.selected_barotropic_equation_of_state_;

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          return ScalarNumber(beos->speed_of_sound(rho));
        } else {
          Number a;
          for (unsigned int k = 0; k < Number::size(); ++k) {
            a[k] = ScalarNumber(beos->speed_of_sound(rho[k]));
          }
          return a;
        }
      }

      //@}
      /**
       * @name Constexpr booleans used in the EulerInitialStates namespace
       */
      //@{

      static constexpr bool have_gamma = false;
      static constexpr bool have_covolume_constant = false;
      static constexpr bool have_energy_equation = false;

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
       * @name Typedefs and constexpr constants
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
      using flux_contribution_type = flux_type;

      /**
       * An array holding all component names of the conserved state as a
       * string.
       */
      static inline const auto component_names =
          []() -> std::array<std::string, problem_dimension> {
        if constexpr (dim == 1)
          return {"rho", "m"};
        else if constexpr (dim == 2)
          return {"rho", "m_1", "m_2"};
        else if constexpr (dim == 3)
          return {"rho", "m_1", "m_2", "m_3"};
        __builtin_trap();
      }();

      /**
       * An array holding all component names of the primitive state as a
       * string.
       */
      static inline const auto primitive_component_names =
          []() -> std::array<std::string, problem_dimension> {
        if constexpr (dim == 1)
          return {"rho", "v"};
        else if constexpr (dim == 2)
          return {"rho", "v_1", "v_2"};
        else if constexpr (dim == 3)
          return {"rho", "v_1", "v_2", "v_3"};
        __builtin_trap();
      }();

      /**
       * The number of precomputed values.
       */
      static constexpr unsigned int n_precomputed_values = 3;

      /**
       * Array type used for precomputed values.
       */
      using precomputed_type = std::array<Number, n_precomputed_values>;

      /**
       * An array holding all component names of the precomputed values.
       */
      static inline const auto precomputed_names =
          std::array<std::string, n_precomputed_values>{{"e", "p", "a"}};

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
      using PrecomputedVectorView =
          Vectors::MultiComponentVectorView<ScalarNumber, n_precomputed_values>;

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
      using InitialPrecomputedVectorView =
          Vectors::MultiComponentVectorView<ScalarNumber,
                                            n_initial_precomputed_values>;

      //@}
      /**
       * @name Computing derived physical quantities
       */
      //@{

      /**
       * For a given (1+dim dimensional) state vector <code>U</code>, return
       * the density <code>U[0]</code>
       */
      static Number density(const state_type &U);

      /**
       * Given a density @p rho this function returns 0 if the magnitude
       * of rho is smaller or equal than relaxation_large * rho_cutoff.
       * Otherwise rho is returned unmodified. Here, rho_cutoff is the
       * reference density multiplied by eps.
       */
      Number filter_vacuum_density(const Number &rho) const;

      /**
       * For a given (1+dim dimensional) state vector <code>U</code>, return
       * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
       */
      static dealii::Tensor<1, dim, Number> momentum(const state_type &U);

      /**
       * For a given (1+dim dimensional) barotropic state vector
       * <code>U</code>, compute and return the total energy of the system.
       * \f[
       *   \eta = \rho e(\rho) + \frac12\rho&{-1}|\vec m|^2
       * \f]
       */
      Number total_energy(const state_type &U,
                          const Number &specific_internal_energy) const;

      /**
       * For a given (1+dim dimensional) barotropic state vector
       * <code>U</code>, compute and return the derivative \f$\eta'\f$ of
       * the total energy.
       */
      state_type total_energy_derivative(const state_type &U,
                                         const Number &specific_internal_energy,
                                         const Number &pressure) const;

      /**
       * Returns whether the state @p U is admissible. If @p U is a
       * vectorized state then @p U is admissible if all vectorized values
       * are admissible.
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
       * taken from @p U_bar state. Note that the @p U_bar state is just the
       * prescribed dirichlet values.
       */
      template <int component>
      state_type prescribe_riemann_characteristic(
          const state_type &U,
          const Number &p,
          const state_type &U_bar,
          const Number &p_bar,
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
       * Given a state @p U and a pressure @p p compute the flux
       * \f[
       * \begin{pmatrix}
       *   \textbf m \\
       *   \textbf v\otimes \textbf m + p\mathbb{I}_d
       * \end{pmatrix},
       * \f]
       */
      flux_type f(const state_type &U, const Number &p) const;

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
      flux_contribution(const PrecomputedVectorView &pv,
                        const InitialPrecomputedVectorView &piv,
                        const unsigned int i,
                        const state_type &U_i) const;

      flux_contribution_type
      flux_contribution(const PrecomputedVectorView &pv,
                        const InitialPrecomputedVectorView &piv,
                        const unsigned int *js,
                        const state_type &U_j) const;

      /**
       * Given flux contributions @p flux_i and @p flux_j compute the flux
       * <code>(-f(U_i) - f(U_j)</code>
       */
      state_type
      flux_divergence(const flux_contribution_type &flux_i,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &c_ij) const;

      /**
       * The low-order and high-order fluxes are the same:
       */
      static constexpr bool have_high_order_flux = false;

      state_type high_order_flux_divergence(
          const flux_contribution_type &flux_i,
          const flux_contribution_type &flux_j,
          const dealii::Tensor<1, dim, Number> &c_ij) const = delete;

      //@}
      /**
       * @name Computing stencil source terms
       */
      //@{

      /** We do not have source terms */
      static constexpr bool have_source_terms = false;

      state_type nodal_source(const PrecomputedVectorView &pv,
                              const unsigned int i,
                              const state_type &U_i,
                              const ScalarNumber tau) const = delete;

      state_type nodal_source(const PrecomputedVectorView &pv,
                              const unsigned int *js,
                              const state_type &U_j,
                              const ScalarNumber tau) const = delete;

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
      state_type expand_state(const ST &state) const;

      /**
       * Given an initial state [rho, u_1, ..., u_d, p] return a
       * conserved state [rho, m_1, ..., m_d, E]. Most notably, the
       * specific equation of state oracle is queried to convert the
       * pressure value into a specific internal energy.
       *
       * @note This function is used to conveniently convert (user
       * provided) primitive initial states with pressure values to a
       * conserved state in the EulerInitialStateLibrary. As such, this
       * function is implemented in the Euler::HyperbolicSystem and
       * EulerBarotropic::HyperbolicSystem classes.
       */
      template <typename ST>
      state_type from_initial_state(const ST &initial_state) const;

      /**
       * Given a primitive state [rho, v_1, ..., v_d] return a conserved
       * state.
       */
      state_type from_primitive_state(const state_type &primitive_state) const;

      /**
       * Given a conserved state return a primitive state [rho, v_1, ..., v_d]
       */
      state_type to_primitive_state(const state_type &state) const;

      /**
       * Transform the current state according to a given operator
       * @p lambda acting on a @a dim dimensional momentum (or velocity)
       * vector.
       */
      template <typename Lambda>
      state_type apply_galilei_transform(const state_type &state,
                                         const Lambda &lambda) const;
      //@}
    }; /* HyperbolicSystemView */


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    inline HyperbolicSystem::HyperbolicSystem(
        const std::string &subsection /*= "HyperbolicSystem"*/)
        : ParameterAcceptor(subsection)
    {
      barotropic_equation_of_state_ = "isothermal";
      add_parameter("barotropic equation of state",
                    barotropic_equation_of_state_,
                    "The barotropic equation of state. Valid names are given "
                    "by any of the subsections defined below");

      reference_density_ = 1.;
      add_parameter("reference density",
                    reference_density_,
                    "Problem specific density reference");

      vacuum_state_relaxation_small_ = 1.e2;
      add_parameter("vacuum state relaxation small",
                    vacuum_state_relaxation_small_,
                    "Problem specific vacuum relaxation parameter");

      vacuum_state_relaxation_large_ = 1.e4;
      add_parameter("vacuum state relaxation large",
                    vacuum_state_relaxation_large_,
                    "Problem specific vacuum relaxation parameter");

      /*
       * And finally populate the equation of state list with all equation of
       * state configurations defined in the EquationOfState namespace:
       */
      BarotropicEquationOfStateLibrary::populate_equation_of_state_list(
          barotropic_equation_of_state_list_, subsection);

      const auto populate_functions = [this]() {
        bool initialized = false;
        for (auto &it : barotropic_equation_of_state_list_)

          /* Populate EOS-specific quantities and functions */
          if (it->name() == barotropic_equation_of_state_) {
            selected_barotropic_equation_of_state_ = it;
            problem_name = "Compressible Euler equations (" + it->name() +
                           " EOS, optimized barotropic)";
            initialized = true;
            break;
          }

        AssertThrow(initialized,
                    dealii::ExcMessage("Could not find a barotropic equation "
                                       "of state description with name \"" +
                                       barotropic_equation_of_state_ + "\""));
      };

      ParameterAcceptor::parse_parameters_call_back.connect(populate_functions);
      populate_functions();
    }


    template <int dim, typename ScalarNumber>
    inline void HyperbolicSystem::fill_precomputed_values(
        const OfflineData<dim, ScalarNumber> &offline_data,
        typename HyperbolicSystemView<dim, ScalarNumber>::StateVector
            &state_vector,
        const bool skip_constrained_dofs) const
    {
      const unsigned int n_internal = offline_data.n_locally_internal();
      const unsigned int n_owned = offline_data.n_locally_owned();
      const auto &sparsity_simd = offline_data.sparsity_pattern_simd();
      using VA = dealii::VectorizedArray<ScalarNumber>;

      const auto &U = std::get<0>(state_vector);
      auto &precomputed = std::get<1>(state_vector);

      const auto body = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);
        using View = HyperbolicSystemView<dim, T>;
        using precomputed_type = typename View::precomputed_type;

        const unsigned int row_length = sparsity_simd.row_length(i);
        if (skip_constrained_dofs && row_length == 1)
          return;

        const auto U_i = U.template read_tensor<T>(i);
        const auto view = this->view<dim, T>();
        const auto rho_i = view.density(U_i);

        const auto e_i = view.beos_specific_internal_energy(rho_i);
        const auto p_i = view.beos_pressure(rho_i);
        const auto a_i = view.beos_speed_of_sound(rho_i);

        const precomputed_type prec_i{e_i, p_i, a_i};
        precomputed.template write_tensor<T>(prec_i, i);
      };

      cpu_simd_loop<ScalarNumber>("time_step_1", body, 0, n_internal, n_owned);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::density(const state_type &U)
    {
      return U[0];
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::filter_vacuum_density(
        const Number &rho) const
    {
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const Number rho_cutoff_large =
          reference_density() * vacuum_state_relaxation_large() * eps;

      return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          std::abs(rho), rho_cutoff_large, Number(0.), rho);
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
    HyperbolicSystemView<dim, Number>::total_energy(
        const state_type &U, const Number &specific_internal_energy) const
    {
      const auto rho = density(U);
      const auto rho_inverse = ScalarNumber(1.) / rho;
      const auto m = momentum(U);

      return rho * specific_internal_energy +
             ScalarNumber(0.5) * rho_inverse * m.norm_square();
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::total_energy_derivative(
        const state_type &U,
        const Number &specific_internal_energy,
        const Number &pressure) const -> state_type
    {
      const auto rho = density(U);
      const auto rho_inverse = ScalarNumber(1.) / rho;
      const auto m = momentum(U);

      state_type result;

      result[0] =
          specific_internal_energy + rho_inverse * pressure -
          ScalarNumber(0.5) * rho_inverse * rho_inverse * m.norm_square();
      for (unsigned int i = 0; i < dim; ++i)
        result[1 + i] = rho_inverse * m[i];

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline bool
    HyperbolicSystemView<dim, Number>::is_admissible(const state_type &U) const
    {
      const auto rho = density(U);
      constexpr auto gt = dealii::SIMDComparison::greater_than;
      using T = Number;
      const auto test =
          dealii::compare_and_apply_mask<gt>(rho, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: Negative state [rho, e] detected!\n";
        std::cout << "\t\trho:           " << rho << "\n";
      }
#endif

      return (test == Number(0.));
    }


    template <int dim, typename Number>
    template <int component>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::prescribe_riemann_characteristic(
        const state_type & /*U*/,
        const Number & /*p*/,
        const state_type & /*U_bar*/,
        const Number & /*p_bar*/,
        const dealii::Tensor<1, dim, Number> & /*normal*/) const -> state_type
    {
      // FIXME
      AssertThrow(false, dealii::ExcNotImplemented());
      __builtin_trap();
      return state_type{};
    }


    template <int dim, typename Number>
    template <typename Lambda>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::apply_boundary_conditions(
        dealii::types::boundary_id id,
        const state_type &U,
        const dealii::Tensor<1, dim, Number> &normal,
        const Lambda &get_dirichlet_data) const -> state_type
    {
      state_type result = U;

      if (id == Boundary::dirichlet) {
        result = get_dirichlet_data();

      } else if (id == Boundary::dirichlet_momentum) {
        /* Only enforce Dirichlet conditions on the momentum: */
        auto m_dirichlet = momentum(get_dirichlet_data());
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = m_dirichlet[k];

      } else if (id == Boundary::dirichlet_velocity) {
        /* Only enforce Dirichlet conditions on the velocity: */
        const auto U_dirichlet = get_dirichlet_data();
        const auto rho_dirichlet = density(U_dirichlet);
        const auto v_dirichlet = momentum(U_dirichlet) / rho_dirichlet;
        const auto rho = density(result);
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = rho * v_dirichlet[k];

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

        /*
         * We do not have precomputed values available. Thus, simply query
         * the pressure and speed of sound oracle:
         */
        const auto p = beos_pressure(rho);
        const auto a = beos_speed_of_sound(rho);
        const auto vn = m * normal / rho;

        /* Supersonic inflow: */
        if (vn < -a) {
          result = get_dirichlet_data();
        }

        /* Subsonic inflow: */
        if (vn >= -a && vn <= 0.) {
          const auto U_dirichlet = get_dirichlet_data();
          const auto rho_dirichlet = density(U_dirichlet);
          const auto p_dirichlet = beos_pressure(rho_dirichlet);

          result = prescribe_riemann_characteristic<2>(
              U_dirichlet, p_dirichlet, U, p, normal);
        }

        /* Subsonic outflow: */
        if (vn > 0. && vn <= a) {
          const auto U_dirichlet = get_dirichlet_data();
          const auto rho_dirichlet = density(U_dirichlet);
          const auto p_dirichlet = beos_pressure(rho_dirichlet);

          result = prescribe_riemann_characteristic<1>(
              U, p, U_dirichlet, p_dirichlet, normal);
        }
        /* Supersonic outflow: do nothing, i.e., keep U as is */

      } else {
        AssertThrow(false, dealii::ExcNotImplemented());
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::f(const state_type &U,
                                         const Number &p) const -> flux_type
    {
      const auto rho_inverse = ScalarNumber(1.) / density(U);
      const auto m = momentum(U);

      flux_type result;

      result[0] = m;
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = m * (m[i] * rho_inverse);
        result[1 + i][i] += p;
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_contribution(
        const PrecomputedVectorView &pv,
        const InitialPrecomputedVectorView & /*piv*/,
        const unsigned int i,
        const state_type &U_i) const -> flux_contribution_type
    {
      const auto &[e_i, p_i, a_i] =
          pv.template read_tensor<Number, precomputed_type>(i);
      return f(U_i, p_i);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_contribution(
        const PrecomputedVectorView &pv,
        const InitialPrecomputedVectorView & /*piv*/,
        const unsigned int *js,
        const state_type &U_j) const -> flux_contribution_type
    {
      const auto &[e_j, p_j, a_j] =
          pv.template read_tensor<Number, precomputed_type>(js);
      return f(U_j, p_j);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_divergence(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij) const -> state_type
    {
      return -contract(add(flux_i, flux_j), c_ij);
    }


    template <int dim, typename Number>
    template <typename ST>
    auto HyperbolicSystemView<dim, Number>::expand_state(const ST &state) const
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
      const auto rho = density(primitive_state);

      auto state = primitive_state;
      /* Fix up momentum: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        state[i] *= rho;

      return state;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::to_primitive_state(
        const state_type &state) const -> state_type
    {
      const auto rho = density(state);
      const auto rho_inverse = Number(1.) / rho;

      auto primitive_state = state;
      /* Fix up velocity: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        primitive_state[i] *= rho_inverse;

      return primitive_state;
    }


    template <int dim, typename Number>
    template <typename Lambda>
    auto HyperbolicSystemView<dim, Number>::apply_galilei_transform(
        const state_type &state, const Lambda &lambda) const -> state_type
    {
      auto result = state;
      const auto M = lambda(momentum(state));
      for (unsigned int d = 0; d < dim; ++d)
        result[1 + d] = M[d];
      return result;
    }
  } // namespace EulerBarotropic
} // namespace ryujin
