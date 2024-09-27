//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "equation_of_state_library.h"

#include <compile_time_options.h>
#include <convenience_macros.h>
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
  namespace EulerAEOS
  {
    template <int dim, typename Number>
    class HyperbolicSystemView;

    /**
     * The compressible Euler equations of gas dynamics. Generalized
     * implementation with a modified approximative Riemann solver,
     * indicator, and limiter suitable for arbitrary equations of state.
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
      static inline std::string problem_name =
          "Compressible Euler equations (arbitrary EOS)";

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

      std::string equation_of_state_;
      double reference_density_;
      double vacuum_state_relaxation_small_;
      double vacuum_state_relaxation_large_;
      bool compute_strict_bounds_;

      EquationOfStateLibrary::equation_of_state_list_type
          equation_of_state_list_;

      using EquationOfState = EquationOfStateLibrary::EquationOfState;
      std::shared_ptr<EquationOfState> selected_equation_of_state_;

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

      DEAL_II_ALWAYS_INLINE inline const std::string &equation_of_state() const
      {
        return hyperbolic_system_.equation_of_state_;
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

      DEAL_II_ALWAYS_INLINE inline bool compute_strict_bounds() const
      {
        return hyperbolic_system_.compute_strict_bounds_;
      }

      //@}
      /**
       * @name Low-level access to the selected equation of state.
       */
      //@{

      /**
       * For a given density \f$\rho\f$ and <i>specific</i> internal
       * energy \f$e\f$ return the pressure \f$p\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number eos_pressure(const Number &rho,
                                                       const Number &e) const
      {
        const auto &eos = hyperbolic_system_.selected_equation_of_state_;

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          return ScalarNumber(eos->pressure(rho, e));
        } else {
          Number p;
          for (unsigned int k = 0; k < Number::size(); ++k) {
            p[k] = ScalarNumber(eos->pressure(rho[k], e[k]));
          }
          return p;
        }
      }

      /**
       * For a given density \f$\rho\f$ and pressure \f$p\f$ return the
       * <i>specific</i> internal energy \f$e\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number
      eos_specific_internal_energy(const Number &rho, const Number &p) const
      {
        const auto &eos = hyperbolic_system_.selected_equation_of_state_;

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          return ScalarNumber(eos->specific_internal_energy(rho, p));
        } else {
          Number e;
          for (unsigned int k = 0; k < Number::size(); ++k) {
            e[k] = ScalarNumber(eos->specific_internal_energy(rho[k], p[k]));
          }
          return e;
        }
      }

      /**
       * For a given density \f$\rho\f$ and specific internal energy \f$e\f$
       * return the temperature \f$T\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number eos_temperature(const Number &rho,
                                                          const Number &e) const
      {
        const auto &eos = hyperbolic_system_.selected_equation_of_state_;

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          return ScalarNumber(eos->temperature(rho, e));
        } else {
          Number temp;
          for (unsigned int k = 0; k < Number::size(); ++k) {
            temp[k] = ScalarNumber(eos->temperature(rho[k], e[k]));
          }
          return temp;
        }
      }

      /**
       * For a given density \f$\rho\f$ and <i>specific</i> internal
       * energy \f$e\f$ return the sound speed \f$a\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number
      eos_speed_of_sound(const Number &rho, const Number &e) const
      {
        const auto &eos = hyperbolic_system_.selected_equation_of_state_;

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          return ScalarNumber(eos->speed_of_sound(rho, e));
        } else {
          Number c;
          for (unsigned int k = 0; k < Number::size(); ++k) {
            c[k] = ScalarNumber(eos->speed_of_sound(rho[k], e[k]));
          }
          return c;
        }
      }

      /**
       * Return the interpolatory co-volume \f$b_{\text{interp}}\f$.
       */
      DEAL_II_ALWAYS_INLINE inline ScalarNumber eos_interpolation_b() const
      {
        const auto &eos = hyperbolic_system_.selected_equation_of_state_;
        return ScalarNumber(eos->interpolation_b());
      }

      /**
       * constexpr boolean used in the EulerInitialStates namespace
       */
      static constexpr bool have_gamma = false;

      /**
       * constexpr boolean used in the EulerInitialStates namespace
       */
      static constexpr bool have_eos_interpolation_b = true;

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
      static constexpr unsigned int problem_dimension = 2 + dim;

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
          return {"rho", "m", "E"};
        else if constexpr (dim == 2)
          return {"rho", "m_1", "m_2", "E"};
        else if constexpr (dim == 3)
          return {"rho", "m_1", "m_2", "m_3", "E"};
        __builtin_trap();
      }();

      /**
       * An array holding all component names of the primitive state as a
       * string.
       */
      static inline const auto primitive_component_names =
          []() -> std::array<std::string, problem_dimension> {
        if constexpr (dim == 1)
          return {"rho", "v", "e"};
        else if constexpr (dim == 2)
          return {"rho", "v_1", "v_2", "e"};
        else if constexpr (dim == 3)
          return {"rho", "v_1", "v_2", "v_3", "e"};
        __builtin_trap();
      }();

      /**
       * The number of precomputed values.
       */
      static constexpr unsigned int n_precomputed_values = 4;

      /**
       * Array type used for precomputed values.
       */
      using precomputed_type = std::array<Number, n_precomputed_values>;

      /**
       * An array holding all component names of the precomputed values.
       */
      static inline const auto precomputed_names =
          std::array<std::string, n_precomputed_values>{
              {"p",
               "surrogate_gamma",
               "surrogate_specific_entropy",
               "surrogate_harten_entropy"}};

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
      static constexpr unsigned int n_precomputation_cycles = 2;

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
       * For a given (2+dim dimensional) state vector <code>U</code>, return
       * the density <code>U[0]</code>
       */
      static Number density(const state_type &U);

      /**
       * Given a density @p rho this function returns 0 if the magniatude
       * of rho is smaller or equal than relaxation_large * rho_cutoff.
       * Otherwise rho is returned unmodified. Here, rho_cutoff is the
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

      //@}
      /**
       * @name Surrogate functions for computing various interpolatory
       * physical quantities that are needed for Riemann solver,
       * indicator and limiter.
       */
      //@{

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return a (scaled) surrogate specific entropy
       * \f[
       *   e^{(\gamma_{\text{min} - 1)s} =
       *   \frac{\rho\,e}{\rho^\gamma_{\text{min}}
       *   (1 - b * \rho)^(\gamma_{\text{min}} -1).
       * \f]
       */
      Number surrogate_specific_entropy(const state_type &U,
                                        const Number &gamma_min) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return a surrogate Harten-type entropy
       * \f[
       *   \eta = (\rho^2 e \cdot (1 - interpolation_b \rho)
       *   ^{\gamma_{text}min} - 1})^{1 / (\gamma_{\text{min}} + 1)}.
       * \f]
       */
      Number surrogate_harten_entropy(const state_type &U,
                                      const Number &gamma_min) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the derivative \f$\eta'\f$ of the Harten-type entropy
       * \f[
       *   \eta = (\rho^2 e \cdot (1 - interpolation_b \rho)
       *   ^{\gamma_{text}min} - 1})^{1 / (\gamma_{\text{min}} + 1)}.
       * \f]
       */
      state_type
      surrogate_harten_entropy_derivative(const state_type &U,
                                          const Number &eta,
                                          const Number &gamma_min) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code> and
       * pressure <code>p</code>, compute a surrogate gamma:
       * \f[
       *   \gamma(\rho, p, e) = 1 + \frac{p * (1 - b * \rho)}{\rho * e}
       * \f]
       *
       * This function is used in various places to create interpolations
       * of the pressure.
       */
      Number surrogate_gamma(const state_type &U, const Number &p) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code> and
       * gamma <code>gamma</code>, compute a surrogate pressure:
       * \f[
       *   p(\rho, \gamma, e) = \frac{(\gamma - 1) * \rho * e}{1 - b * \rho}
       * \f]
       *
       * This function is the complementary function to surrogate_gamma(),
       * meaning the following property holds true:
       * ```
       *   surrogate_gamma(U, surrogate_pressure(U, gamma)) == gamma
       *       surrogate_pressure(U, surrogate_gamma(U, p)) == p
       * ```
       */
      Number surrogate_pressure(const state_type &U, const Number &gamma) const;

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
       * Given a state @p U and a pressure @p p compute the flux
       * \f[
       * \begin{pmatrix}
       *   \textbf m \\
       *   \textbf v\otimes \textbf m + p\mathbb{I}_d \\
       *   \textbf v(E+p)
       * \end{pmatrix},
       * \f]
       */
      flux_type f(const state_type &U, const Number &p) const;

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
       * For the Euler equations we simply compute <code>f(U_i)</code>.
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

      /**
       * @name Computing stencil source terms
       */
      //@{

      /** We do not have source terms */
      static constexpr bool have_source_terms = false;

      state_type nodal_source(const PrecomputedVector &pv,
                              const unsigned int i,
                              const state_type &U_i,
                              const ScalarNumber tau) const = delete;

      state_type nodal_source(const PrecomputedVector &pv,
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
       * Given an initial state [rho, u_1, ..., u_?, p] return a
       * conserved state [rho, m_1, ..., m_d, E]. Most notably, the
       * specific equation of state oracle is queried to convert the
       * pressure value into a specific internal energy.
       *
       * @note This function is used to conveniently convert (user
       * provided) primitive initial states with pressure values to a
       * conserved state in the EulerInitialStateLibrary. As such, this
       * function is implemented in the Euler::HyperbolicSystem and
       * EulerAEOS::HyperbolicSystem classes.
       */
      template <typename ST>
      state_type from_initial_state(const ST &initial_state) const;

      /**
       * Given a primitive state [rho, u_1, ..., u_d, e] return a conserved
       * state.
       */
      state_type from_primitive_state(const state_type &primitive_state) const;

      /**
       * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
       * e]
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
      equation_of_state_ = "polytropic gas";
      add_parameter(
          "equation of state",
          equation_of_state_,
          "The equation of state. Valid names are given by any of the "
          "subsections defined below");

      compute_strict_bounds_ = true;
      add_parameter(
          "compute strict bounds",
          compute_strict_bounds_,
          "Compute strict, but significantly more expensive bounds at various "
          "places: (a) an expensive, but better upper wavespeed estimate in "
          "the approximate RiemannSolver; (b) entropy viscosity-commutator "
          "with correct gamma_min over the stencil; (c) mathematically correct "
          "surrogate specific entropy minimum with gamma_min over the "
          "stencil.");

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
      EquationOfStateLibrary::populate_equation_of_state_list(
          equation_of_state_list_, subsection);

      const auto populate_functions = [this]() {
        bool initialized = false;
        for (auto &it : equation_of_state_list_)

          /* Populate EOS-specific quantities and functions */
          if (it->name() == equation_of_state_) {
            selected_equation_of_state_ = it;
            problem_name =
                "Compressible Euler equations (" + it->name() + " EOS)";
            initialized = true;
            break;
          }

        AssertThrow(
            initialized,
            dealii::ExcMessage(
                "Could not find an equation of state description with name \"" +
                equation_of_state_ + "\""));
      };

      ParameterAcceptor::parse_parameters_call_back.connect(populate_functions);
      populate_functions();
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
      Assert(cycle == 0 || cycle == 1, dealii::ExcInternalError());

      const auto &U = std::get<0>(state_vector);
      auto &precomputed = std::get<1>(state_vector);

      /* We are inside a thread parallel context */

      const auto &eos = hyperbolic_system_.selected_equation_of_state_;
      unsigned int stride_size = get_stride_size<Number>;

      if (cycle == 0) {
        if (eos->prefer_vector_interface()) {
          /*
           * Set up temporary storage for p, rho, e and make two calls into
           * the eos library.
           */
          const auto offset = left;
          const auto size = right - left;

          static /* shared */ std::vector<double> p;
          static /* shared */ std::vector<double> rho;
          static /* shared */ std::vector<double> e;
          RYUJIN_OMP_SINGLE
          {
            p.resize(size);
            rho.resize(size);
            e.resize(size);
          }

          RYUJIN_OMP_FOR
          for (unsigned int i = 0; i < size; i += stride_size) {
            const auto U_i = U.template get_tensor<Number>(offset + i);
            const auto rho_i = density(U_i);
            const auto e_i = internal_energy(U_i) / rho_i;
            /*
             * Populate rho and e also for interpolated values from
             * constrainted degrees of freedom so that the vectors contain
             * physically admissible entries throughout.
             */
            write_entry<Number>(rho, rho_i, i);
            write_entry<Number>(e, e_i, i);
          }

          /* Make sure the call into eospac (and others) is single threaded. */
          RYUJIN_OMP_SINGLE
          {
            eos->pressure(p, rho, e);
          }

          RYUJIN_OMP_FOR
          for (unsigned int i = 0; i < size; i += stride_size) {
            /* Skip constrained degrees of freedom: */
            const unsigned int row_length = sparsity_simd.row_length(i);
            if (row_length == 1)
              continue;

            dispatch_check(i);

            using PT = precomputed_type;
            const auto U_i = U.template get_tensor<Number>(offset + i);
            const auto p_i = get_entry<Number>(p, i);
            const auto gamma_i = surrogate_gamma(U_i, p_i);
            const PT prec_i{p_i, gamma_i, Number(0.), Number(0.)};
            precomputed.template write_tensor<Number>(prec_i, offset + i);
          }
        } else {
          /*
           * This is the variant with slightly better performance provided
           * that a call to the eos is not too expensive. This variant
           * calls into the eos library for every single degree of freedom.
           */
          RYUJIN_OMP_FOR
          for (unsigned int i = left; i < right; i += stride_size) {
            /* Skip constrained degrees of freedom: */
            const unsigned int row_length = sparsity_simd.row_length(i);
            if (row_length == 1)
              continue;

            dispatch_check(i);

            const auto U_i = U.template get_tensor<Number>(i);
            const auto rho_i = density(U_i);
            const auto e_i = internal_energy(U_i) / rho_i;
            const auto p_i = eos_pressure(rho_i, e_i);

            const auto gamma_i = surrogate_gamma(U_i, p_i);
            using PT = precomputed_type;
            const PT prec_i{p_i, gamma_i, Number(0.), Number(0.)};
            precomputed.template write_tensor<Number>(prec_i, i);
          }
        } /* prefer_vector_interface */
      }   /* cycle == 0 */

      if (cycle == 1) {
        RYUJIN_OMP_FOR
        for (unsigned int i = left; i < right; i += stride_size) {
          using PT = precomputed_type;

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          dispatch_check(i);

          const auto U_i = U.template get_tensor<Number>(i);
          auto prec_i = precomputed.template get_tensor<Number, PT>(i);
          auto &[p_i, gamma_min_i, s_i, eta_i] = prec_i;

          const unsigned int *js = sparsity_simd.columns(i) + stride_size;
          for (unsigned int col_idx = 1; col_idx < row_length;
               ++col_idx, js += stride_size) {

            const auto U_j = U.template get_tensor<Number>(js);
            const auto prec_j = precomputed.template get_tensor<Number, PT>(js);
            auto &[p_j, gamma_min_j, s_j, eta_j] = prec_j;
            const auto gamma_j = surrogate_gamma(U_j, p_j);
            gamma_min_i = std::min(gamma_min_i, gamma_j);
          }

          s_i = surrogate_specific_entropy(U_i, gamma_min_i);
          eta_i = surrogate_harten_entropy(U_i, gamma_min_i);
          precomputed.template write_tensor<Number>(prec_i, i);
        }
      }
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
    HyperbolicSystemView<dim, Number>::total_energy(const state_type &U)
    {
      return U[1 + dim];
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::internal_energy(const state_type &U)
    {
      /*
       * rho e = (E - 1/2*m^2/rho)
       */
      const Number rho_inverse = ScalarNumber(1.) / density(U);
      const auto m = momentum(U);
      const Number E = total_energy(U);
      return E - ScalarNumber(0.5) * m.norm_square() * rho_inverse;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::internal_energy_derivative(
        const state_type &U) -> state_type
    {
      /*
       * With
       *   rho e = E - 1/2 |m|^2 / rho
       * we get
       *   (rho e)' = (1/2m^2/rho^2, -m/rho , 1 )^T
       */

      const Number rho_inverse = ScalarNumber(1.) / density(U);
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
    HyperbolicSystemView<dim, Number>::surrogate_specific_entropy(
        const state_type &U, const Number &gamma_min) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const auto rho = density(U);
      const auto rho_inverse = ScalarNumber(1.) / rho;
      const auto interpolation_b = Number(eos_interpolation_b());
      const auto covolume = Number(1.) - interpolation_b * rho;

      return internal_energy(U) *
             ryujin::pow(rho_inverse - interpolation_b, gamma_min) / covolume;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::surrogate_harten_entropy(
        const state_type &U, const Number &gamma_min) const
    {
      const auto rho = density(U);
      const auto m = momentum(U);
      const auto E = total_energy(U);
      const auto rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

      const auto exponent = ScalarNumber(1.) / (gamma_min + Number(1.));

      const auto interpolation_b = Number(eos_interpolation_b());
      const auto covolume = Number(1.) - interpolation_b * rho;
      const auto covolume_term = ryujin::pow(covolume, gamma_min - Number(1.));

      return ryujin::pow(rho_rho_e * covolume_term, exponent);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::surrogate_harten_entropy_derivative(
        const state_type &U, const Number &eta, const Number &gamma_min) const
        -> state_type
    {
      /*
       * With
       *   eta = (rho^2 e * (1 - interpolation_b * rho)) ^ {1 / (gamma + 1)},
       *   rho^2 e = rho * E - 1/2 |m|^2,
       *
       * we get
       *
       *   eta' = factor * (1 - interpolation_b rho) * (E,-m,rho)^T +
       *          factor * rho^2 e * (gamma - 1) * b * (1,0,0)^T
       *
       *   factor = 1/(gamma+1) * (eta/(1-interpolation_b rho)^-gamma
       *                        / (1-interpolation_b rho)^2
       */

      const auto rho = density(U);
      const auto m = momentum(U);
      const auto E = total_energy(U);
      const auto rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

      const auto interpolation_b = Number(eos_interpolation_b());
      const auto covolume = Number(1.) - interpolation_b * rho;
      const auto covolume_inverse = Number(1.) / covolume;

      const auto factor = ryujin::pow(eta * covolume_inverse, -gamma_min) *
                          fixed_power<2>(covolume_inverse) /
                          (gamma_min + Number(1.));

      state_type result;

      result[0] = factor * (covolume * E - (gamma_min - Number(1.)) *
                                               rho_rho_e * interpolation_b);
      for (unsigned int i = 0; i < dim; ++i)
        result[1 + i] = -factor * covolume * m[i];
      result[dim + 1] = factor * covolume * rho;

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::surrogate_gamma(const state_type &U,
                                                       const Number &p) const
    {
      const auto rho = density(U);
      const auto rho_e = internal_energy(U);
      const auto interpolation_b = Number(eos_interpolation_b());
      const auto covolume = Number(1.) - interpolation_b * rho;

      return Number(1.) + p * covolume / rho_e;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::surrogate_pressure(
        const state_type &U, const Number &gamma) const
    {
      const auto rho = density(U);
      const auto rho_e = internal_energy(U);
      const auto interpolation_b = Number(eos_interpolation_b());
      const auto covolume = Number(1.) - interpolation_b * rho;

      return (gamma - Number(1.)) * rho_e / covolume;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline bool
    HyperbolicSystemView<dim, Number>::is_admissible(const state_type &U) const
    {
      const auto rho = density(U);
      const auto e = internal_energy(U);

      constexpr auto gt = dealii::SIMDComparison::greater_than;
      using T = Number;
      const auto test =
          dealii::compare_and_apply_mask<gt>(rho, T(0.), T(0.), T(-1.)) + //
          dealii::compare_and_apply_mask<gt>(e, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: Negative state [rho, e] detected!\n";
        std::cout << "\t\trho: " << rho << "\n";
        std::cout << "\t\tint: " << e << "\n";
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
      __builtin_trap(); // untested and likely needs to be refactored

      static_assert(component == 1 || component == 2,
                    "component has to be 1 or 2");

      const auto m = momentum(U);
      const auto rho = density(U);
      const auto vn = m * normal / rho;

      const auto p = surrogate_pressure(U); // FIXME: discuss
      const auto gamma = surrogate_gamma(U, p);
      const auto interpolation_b = ScalarNumber(eos_interpolation_b());
      const auto x = Number(1.) - interpolation_b * rho;
      const auto a = std::sqrt(gamma * p / (rho * x)); // local speed of sound


      const auto m_bar = momentum(U_bar);
      const auto rho_bar = density(U_bar);
      const auto vn_bar = m_bar * normal / rho_bar;

      const auto p_bar = surrogate_pressure(U_bar); // FIXME: discuss
      const auto gamma_bar = surrogate_gamma(U_bar, p_bar);
      const auto x_bar = Number(1.) - interpolation_b * rho_bar;
      const auto a_bar = std::sqrt(gamma_bar * p_bar / (rho_bar * x_bar));

      /* First Riemann characteristic: v* n - 2 / (gamma - 1) * a */

      const auto R_1 = component == 1
                           ? vn_bar - 2. * a_bar / (gamma_bar - 1.) * x_bar
                           : vn - 2. * a / (gamma - 1.) * x;

      /* Second Riemann characteristic: v* n + 2 / (gamma - 1) * a */

      const auto R_2 = component == 2
                           ? vn_bar + 2. * a_bar / (gamma_bar - 1.) * x_bar
                           : vn + 2. * a / (gamma - 1.) * x;

      const auto s = p / ryujin::pow(rho, gamma) * ryujin::pow(x, gamma);

      const auto vperp = m / rho - vn * normal;

      const auto vn_new = 0.5 * (R_1 + R_2);

      auto rho_new =
          1. / (gamma * s) * ryujin::pow((gamma - 1.) / 4. * (R_2 - R_1), 2.);
      rho_new =
          1. / (ryujin::pow(rho_new, 1. / (1. - gamma)) + interpolation_b);

      const auto x_new = 1. - interpolation_b * rho_new;

      const auto p_new =
          s * std::pow(rho_new, gamma) / ryujin::pow(x_new, gamma);

      state_type U_new;
      U_new[0] = rho_new;
      for (unsigned int d = 0; d < dim; ++d) {
        U_new[1 + d] = rho_new * (vn_new * normal + vperp)[d];
      }
      U_new[1 + dim] = p_new / (gamma - 1.) +
                       0.5 * rho_new * (vn_new * vn_new + vperp.norm_square());

      return U_new;
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

      } else if (id == Boundary::slip) {
        auto m = momentum(U);
        m -= 1. * (m * normal) * normal;
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = m[k];

      } else if (id == Boundary::no_slip) {
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = Number(0.);

      } else if (id == Boundary::dynamic) {
        __builtin_trap(); // untested and likely needs to be refactored
#if 0
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
#endif
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
      const auto E = total_energy(U);

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
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_contribution(
        const PrecomputedVector &pv,
        const InitialPrecomputedVector & /*piv*/,
        const unsigned int i,
        const state_type &U_i) const -> flux_contribution_type
    {
      const auto &[p_i, gamma_min_i, s_i, eta_i] =
          pv.template get_tensor<Number, precomputed_type>(i);
      return f(U_i, p_i);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_contribution(
        const PrecomputedVector &pv,
        const InitialPrecomputedVector & /*piv*/,
        const unsigned int *js,
        const state_type &U_j) const -> flux_contribution_type
    {
      const auto &[p_j, gamma_min_j, s_j, eta_j] =
          pv.template get_tensor<Number, precomputed_type>(js);
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
    template <typename ST>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::from_initial_state(
        const ST &initial_state) const -> state_type
    {
      auto primitive_state = expand_state(initial_state);

      /* pressure into specific internal energy: */
      const auto rho = density(primitive_state);
      const auto p = /*SIC!*/ total_energy(primitive_state);
      const auto e = eos_specific_internal_energy(rho, p);
      primitive_state[dim + 1] = e;

      return from_primitive_state(primitive_state);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::from_primitive_state(
        const state_type &primitive_state) const -> state_type
    {
      const auto rho = density(primitive_state);
      /* extract velocity: */
      const auto u = /*SIC!*/ momentum(primitive_state);
      /* extract specific internal energy: */
      const auto &e = /*SIC!*/ total_energy(primitive_state);

      auto state = primitive_state;
      /* Fix up momentum: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        state[i] *= rho;

      /* Compute total energy: */
      state[dim + 1] = rho * e + Number(0.5) * rho * u * u;

      return state;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::to_primitive_state(
        const state_type &state) const -> state_type
    {
      const auto rho = density(state);
      const auto rho_inverse = Number(1.) / rho;
      const auto rho_e = internal_energy(state);

      auto primitive_state = state;
      /* Fix up velocity: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        primitive_state[i] *= rho_inverse;
      /* Set specific internal energy: */
      primitive_state[dim + 1] = rho_e * rho_inverse;

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
  } // namespace EulerAEOS
} // namespace ryujin
