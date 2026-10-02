//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "equation_of_state_library.h"

#include <convenience_macros.h>
#include <discretization.h>
#include <gpu.h>
#include <loop.h>
#include <multicomponent_vector.h>
#include <patterns_conversion.h>
#include <simd.h>
#include <state_vector.h>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>

namespace ryujin
{
  namespace EulerAEOS
  {
    template <int dim,
              typename Number,
              typename MemorySpace = dealii::MemorySpace::Host>
    class HyperbolicSystemView;

    /**
     * The compressible Euler equations of gas dynamics. Generalized
     * implementation with a modified approximate Riemann solver for
     * finding max wave speed, indicator, and limiter suitable for
     * arbitrary equations of state.
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
       * A structure holding all runtime parameters of the hyperbolic
       * system, as well as the interpolatory NASG parameters of the
       * selected equation of state.
       */
      struct Parameters {
        /**
         * @name Run time parameters
         */
        //@{

        double reference_density;
        double vacuum_state_relaxation_small;
        double vacuum_state_relaxation_large;

        bool compute_strict_bounds;

        //@}
        /**
         * @name Cached equation of state parameters
         *
         * The interpolatory NASG parameters of the selected equation of
         * state. These are recomputed in update_parameters().
         */
        //@{

        double covolume_constant;
        double interpolation_pinfty;
        double interpolation_q;

        //@}
      };

      /**
       * Constructor.
       */
      HyperbolicSystem(const std::string &subsection = "/HyperbolicSystem");

      /**
       * Alias for the view on the hyperbolic system for a given dimension @p
       * dim, choice of number type @p Number, and memory space @p
       * MemorySpace.
       */
      template <int dim,
                typename Number = double,
                typename MemorySpace = dealii::MemorySpace::Host>
      using View = HyperbolicSystemView<dim, Number, MemorySpace>;

      /**
       * Return a view on the Hyperbolic System for a given dimension @p
       * dim and choice of number type @p Number (which can be a scalar
       * float, or double, as well as a VectorizedArray holding packed
       * scalars. The optional @p MemorySpace template parameter selects
       * whether the view is intended for the host or device memory space.
       */
      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return View<dim, Number, MemorySpace>{*this};
      }

      /**
       * Part of step 1 of the hyperbolic update step: Compute "precomputed
       * values" and fill into the state vector.
       *
       * @note The method does not update the ghost range of the state
       * vector. The precomputed part has to be synchronized by explicitly
       * calling the update ghost values function.
       *
       * @note The pressure is computed with the selected equation of state
       * on the host memory space. If @p MemorySpace is the default (device)
       * memory space the state vector is thus temporarily transferred to
       * the host memory space.
       */
      template <typename MemorySpace = dealii::MemorySpace::Host,
                int dim,
                typename ScalarNumber>
      void fill_precomputed_values(
          const OfflineData<dim, ScalarNumber> &offline_data,
          typename HyperbolicSystemView<dim, ScalarNumber>::StateVector
              &state_vector,
          const bool skip_constrained_dofs = true) const;

    private:
      /**
       * @name Internal fields, methods, and friends
       */
      //@{

      /**
       * Select the equation of state, update the problem name, and copy
       * the interpolatory NASG parameters of the selected equation of
       * state into the Parameters structure. This function is called from
       * the constructor and connected to the parse_parameters_call_back
       * signal of this class and of all equation of state classes.
       */
      void update_parameters();

      Mirrored<Parameters> parameters_;

      std::string equation_of_state_;

      EquationOfStateLibrary::equation_of_state_list_type
          equation_of_state_list_;

      using EquationOfState = EquationOfStateLibrary::EquationOfState;
      std::shared_ptr<EquationOfState> selected_equation_of_state_;

      template <int, typename, typename>
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
     *
     * @note This class is designed to be copied by value into computation
     * loops with access to either the host or device memory space. As such
     * we do not store a reference to the underlying HyperbolicSystem but
     * copy all runtime parameters into the view when it is created.
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number, typename MemorySpace>
    class HyperbolicSystemView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

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
               "surrogate_gamma_min",
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
       * MulticomponentVectorView for accessing a vector of precomputed
       * states:
       */
      using PrecomputedVectorView = Vectors::MultiComponentVectorView<
          ScalarNumber,
          n_precomputed_values,
          dealii::VectorizedArray<ScalarNumber>::size(),
          MemorySpace,
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
          MemorySpace,
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
          : parameters_(
                hyperbolic_system.parameters_.template view<MemorySpace>())
          , equation_of_state_(
                hyperbolic_system.selected_equation_of_state_.get())
      {
      }

      //@}
      /**
       * @name Access to runtime parameters
       */
      //@{

      /**
       * Return the name of the selected equation of state.
       *
       * @note This function is only available on the host.
       */
      DEAL_II_ALWAYS_INLINE inline const std::string &equation_of_state() const
      {
        return equation_of_state_->name();
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber reference_density() const
      {
        return ScalarNumber(parameters_->reference_density);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      vacuum_state_relaxation_small() const
      {
        return ScalarNumber(parameters_->vacuum_state_relaxation_small);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      vacuum_state_relaxation_large() const
      {
        return ScalarNumber(parameters_->vacuum_state_relaxation_large);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE bool compute_strict_bounds() const
      {
        return parameters_->compute_strict_bounds;
      }

      //@}
      /**
       * @name Low-level access to the selected equation of state
       *
       * @note The oracle functions eos_pressure(),
       * eos_specific_internal_energy(), eos_specific_entropy(),
       * eos_cold_curve_bound(), eos_temperature(), and eos_speed_of_sound()
       * call into the selected equation of state and are thus only
       * available on the host.
       */
      //@{

      /**
       * For a given density \f$\rho\f$ and <i>specific</i> internal
       * energy \f$e\f$ return the pressure \f$p\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number eos_pressure(const Number &rho,
                                                       const Number &e) const
      {
        const auto &eos = equation_of_state_;

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
        const auto &eos = equation_of_state_;

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
       * For a given density \f$\rho\f$ and <i>specific</i> internal
       * energy \f$e\f$ return the specific entropy \f$s\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number
      eos_specific_entropy(const Number &rho, const Number &e) const
      {
        const auto &eos = equation_of_state_;

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          return ScalarNumber(eos->specific_entropy(rho, e));
        } else {
          Number p;
          for (unsigned int k = 0; k < Number::size(); ++k) {
            p[k] = ScalarNumber(eos->specific_entropy(rho[k], e[k]));
          }
          return p;
        }
      }

      /**
       * For a given density \f$\rho\f$ return the cold curve bound \f$e\ge
       * e_{s_0}(\rho)\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number
      eos_cold_curve_bound(const Number &rho) const
      {
        const auto &eos = equation_of_state_;

        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          return ScalarNumber(eos->cold_curve_bound(rho));
        } else {
          Number p;
          for (unsigned int k = 0; k < Number::size(); ++k) {
            p[k] = ScalarNumber(eos->cold_curve_bound(rho[k]));
          }
          return p;
        }
      }

      /**
       * For a given density \f$\rho\f$ and specific internal energy \f$e\f$
       * return the temperature \f$T\f$.
       */
      DEAL_II_ALWAYS_INLINE inline Number eos_temperature(const Number &rho,
                                                          const Number &e) const
      {
        const auto &eos = equation_of_state_;

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
        const auto &eos = equation_of_state_;

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
       * Return the interpolatory covolume \f$b_{\text{interp}}\f$.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      eos_covolume_constant() const
      {
        return ScalarNumber(parameters_->covolume_constant);
      }

      /**
       * Return the interpolatory reference pressure \f$p_{\infty}\f$.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      eos_interpolation_pinfty() const
      {
        return ScalarNumber(parameters_->interpolation_pinfty);
      }

      /**
       * Return the interpolatory reference specific internal energy
       * \f$q\f$.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber eos_interpolation_q() const
      {
        return ScalarNumber(parameters_->interpolation_q);
      }

      //@}
      /**
       * @name Constexpr booleans used in the EulerInitialStates namespace
       */
      //@{

      static constexpr bool have_gamma = false;
      static constexpr bool have_covolume_constant = true;
      static constexpr bool have_energy_equation = true;

      //@}
      /**
       * @name Computing derived physical quantities
       */
      //@{

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, return
       * the density <code>U[0]</code>
       */
      static DEAL_II_HOST_DEVICE Number density(const state_type &U);

      /**
       * Given a density @p rho this function returns 0 if the magniatude
       * of rho is smaller or equal than relaxation_large * rho_cutoff.
       * Otherwise rho is returned unmodified. Here, rho_cutoff is the
       * reference density multiplied by eps.
       */
      DEAL_II_HOST_DEVICE Number filter_vacuum_density(const Number &rho) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, return
       * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
       */
      static DEAL_II_HOST_DEVICE dealii::Tensor<1, dim, Number>
      momentum(const state_type &U);

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, return
       * the total energy <code>U[1+dim]</code>
       */
      static DEAL_II_HOST_DEVICE Number total_energy(const state_type &U);

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the internal energy \f$\varepsilon = (\rho e)\f$.
       */
      static DEAL_II_HOST_DEVICE Number internal_energy(const state_type &U);

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the derivative of the internal energy
       * \f$\varepsilon = (\rho e)\f$.
       */
      static DEAL_II_HOST_DEVICE state_type
      internal_energy_derivative(const state_type &U);

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
       *   e^{(\gamma_{\text{min}} - 1)s} =
       *   \frac{\rho\,(e-q)-p_{\infty}(1-b\rho)}{\rho^\gamma_{\text{min}}}
       *   (1 - b\,\rho)^{\gamma_{\text{min}} -1}.
       * \f]
       */
      DEAL_II_HOST_DEVICE Number surrogate_specific_entropy(
          const state_type &U, const Number &gamma_min) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return a surrogate Harten-type entropy
       * \f[
       *   \eta =
       *   (1-b\,\rho)^{\frac{\gamma_{\text{min}-1}}{\gamma_{\text{min}}+1}}
       *   \big(\rho^2 (e-q) - \rho p_{\infty}(1-b\,\rho)\big)
       *   ^{1/(\gamma_{\text{min}}+1)}
       * \f]
       */
      DEAL_II_HOST_DEVICE Number surrogate_harten_entropy(
          const state_type &U, const Number &gamma_min) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the derivative \f$\eta'\f$ of the Harten-type entropy
       * \f[
       *   \eta =
       *   (1-b\,\rho)^{\frac{\gamma_{\text{min}-1}}{\gamma_{\text{min}}+1}}
       *   \big(\rho^2 (e-q) - \rho p_{\infty}(1-b\,\rho)\big)
       *   ^{1/(\gamma_{\text{min}}+1)}
       * \f]
       */
      DEAL_II_HOST_DEVICE state_type
      surrogate_harten_entropy_derivative(const state_type &U,
                                          const Number &eta,
                                          const Number &gamma_min) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code> and
       * pressure <code>p</code>, compute a surrogate gamma:
       * \f[
       *   \gamma(\rho, e, p) = 1 + \frac{(p + p_{\infty})(1 - b \rho)}
       *   {\rho (e-q) - p_{\infty}(1-b \rho)}
       * \f]
       *
       * This function is used in various places to create interpolations
       * of the pressure.
       */
      DEAL_II_HOST_DEVICE Number surrogate_gamma(const state_type &U,
                                                 const Number &p) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code> and
       * gamma <code>gamma</code>, compute a surrogate pressure:
       * \f[
       *   p(\rho, e, \gamma) = (\gamma - 1) \frac{\rho (e - q)}{1 - b \rho}
       *   -\gamma\,p_{\infty}
       * \f]
       *
       * This function is the complementary function to surrogate_gamma(),
       * meaning the following property holds true:
       * ```
       *   surrogate_gamma(U, surrogate_pressure(U, gamma)) == gamma
       *       surrogate_pressure(U, surrogate_gamma(U, p)) == p
       * ```
       */
      DEAL_II_HOST_DEVICE Number surrogate_pressure(const state_type &U,
                                                    const Number &gamma) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code> and
       * gamma <code>gamma</code>, compute a surrogate speed of sound:
       * \f{align}
       *   c^2(\rho, e, \gamma) = \frac{\gamma (p + p_\infty)}{\rho X}
       *       = \frac{\gamma (\gamma -1)[\rho (e - q) - p_\infty X]}{\rho X^2}
       * \f}
       */
      DEAL_II_HOST_DEVICE Number
      surrogate_speed_of_sound(const state_type &U, const Number &gamma) const;

      /**
       * Returns whether the state @p U is admissible. If @p U is a
       * vectorized state then @p U is admissible if all vectorized values
       * are admissible.
       */
      DEAL_II_HOST_DEVICE bool is_admissible(const state_type &U) const;

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
      DEAL_II_HOST_DEVICE state_type prescribe_riemann_characteristic(
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
       *
       * @note This function calls into the selected equation of state for
       * dynamic boundary conditions and is thus only available on the
       * host.
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
      DEAL_II_HOST_DEVICE flux_type f(const state_type &U,
                                      const Number &p) const;

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
      DEAL_II_HOST_DEVICE
      flux_contribution_type
      flux_contribution(const PrecomputedVectorView &pv,
                        const InitialPrecomputedVectorView &piv,
                        const unsigned int i,
                        const state_type &U_i) const;

      DEAL_II_HOST_DEVICE
      flux_contribution_type
      flux_contribution(const PrecomputedVectorView &pv,
                        const InitialPrecomputedVectorView &piv,
                        const unsigned int *js,
                        const state_type &U_j) const;

      /**
       * Given flux contributions @p flux_i and @p flux_j compute the flux
       * <code>(-f(U_i) - f(U_j)</code>
       */
      DEAL_II_HOST_DEVICE
      state_type
      flux_divergence(const flux_contribution_type &flux_i,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &c_ij) const;

      /**
       * The low-order and high-order fluxes are the same:
       */
      static constexpr bool have_high_order_flux = false;

      DEAL_II_HOST_DEVICE
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

      DEAL_II_HOST_DEVICE
      state_type nodal_source(const PrecomputedVectorView &pv,
                              const unsigned int i,
                              const state_type &U_i,
                              const ScalarNumber tau) const = delete;

      DEAL_II_HOST_DEVICE
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
      DEAL_II_HOST_DEVICE state_type expand_state(const ST &state) const;

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
       * EulerAEOS::HyperbolicSystem classes.
       *
       * @note This function calls into the selected equation of state and
       * is thus only available on the host.
       */
      template <typename ST>
      state_type from_initial_state(const ST &initial_state) const;

      /**
       * Given a primitive state [rho, u_1, ..., u_d, e] return a conserved
       * state.
       */
      DEAL_II_HOST_DEVICE
      state_type from_primitive_state(const state_type &primitive_state) const;

      /**
       * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
       * e]
       */
      DEAL_II_HOST_DEVICE
      state_type to_primitive_state(const state_type &state) const;

      /**
       * Transform the current state according to a  given operator
       * @p lambda acting on a @a dim dimensional momentum (or velocity)
       * vector.
       */
      template <typename Lambda>
      DEAL_II_HOST_DEVICE state_type apply_galilei_transform(
          const state_type &state, const Lambda &lambda) const;

    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const HyperbolicSystem::Parameters *const parameters_;

      /* Only valid (and dereferenced) on the host: */
      const EquationOfStateLibrary::EquationOfState *const equation_of_state_;

      //@}
    }; /* HyperbolicSystemView */


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    inline HyperbolicSystem::HyperbolicSystem(const std::string &subsection)
        : ParameterAcceptor(subsection)
        , parameters_("euler_aeos_hyperbolic_system_parameters",
                      TransferPolicy::implicit_transfers_host_resident)
    {
      /* reference remains valid due to implicit_transfers_host_resident */
      auto &parameters = *parameters_.view();

      equation_of_state_ = "polytropic gas";
      add_parameter(
          "equation of state",
          equation_of_state_,
          "The equation of state. Valid names are given by any of the "
          "subsections defined below");

      parameters.compute_strict_bounds = true;
      add_parameter(
          "compute strict bounds",
          parameters.compute_strict_bounds,
          "Compute strict, but significantly more expensive bounds at various "
          "places: (a) an expensive, but better upper wavespeed estimate in "
          "the approximate WaveSpeedEstimator; (b) entropy "
          "viscosity-commutator "
          "with correct gamma_min over the stencil; (c) mathematically correct "
          "surrogate specific entropy minimum with gamma_min over the "
          "stencil.");

      parameters.reference_density = 1.;
      add_parameter("reference density",
                    parameters.reference_density,
                    "Problem specific density reference");

      parameters.vacuum_state_relaxation_small = 1.e2;
      add_parameter("vacuum state relaxation small",
                    parameters.vacuum_state_relaxation_small,
                    "Problem specific vacuum relaxation parameter");

      parameters.vacuum_state_relaxation_large = 1.e4;
      add_parameter("vacuum state relaxation large",
                    parameters.vacuum_state_relaxation_large,
                    "Problem specific vacuum relaxation parameter");

      /*
       * And finally populate the equation of state list with all equation of
       * state configurations defined in the EquationOfState namespace:
       */
      EquationOfStateLibrary::populate_equation_of_state_list(
          equation_of_state_list_, subsection);

      /*
       * The equation of state classes update their interpolatory NASG
       * parameters in their own parse_parameters_call_back signal, which
       * is invoked after ours. Thus, also connect to theirs:
       */
      ParameterAcceptor::parse_parameters_call_back.connect(
          [this] { update_parameters(); });
      for (auto &it : equation_of_state_list_)
        it->parse_parameters_call_back.connect([this] { update_parameters(); });

      update_parameters();
    }


    inline void HyperbolicSystem::update_parameters()
    {
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

      auto &parameters = *parameters_.view();

      parameters.covolume_constant =
          selected_equation_of_state_->covolume_constant();
      parameters.interpolation_pinfty =
          selected_equation_of_state_->interpolation_pinfty();
      parameters.interpolation_q =
          selected_equation_of_state_->interpolation_q();
    }


    template <typename MemorySpace, int dim, typename ScalarNumber>
    inline void HyperbolicSystem::fill_precomputed_values(
        const OfflineData<dim, ScalarNumber> &offline_data,
        typename HyperbolicSystemView<dim, ScalarNumber>::StateVector
            &state_vector,
        const bool skip_constrained_dofs) const
    {
      using HostSpace = dealii::MemorySpace::Host;

      const unsigned int n_internal = offline_data.n_locally_internal();
      const unsigned int n_owned = offline_data.n_locally_owned();

      auto &U = std::get<0>(state_vector);
      auto &precomputed = std::get<1>(state_vector);

      /*
       * Compute the pressure (and an initial surrogate gamma_min) over the
       * diagonal. This requires calling into the selected equation of
       * state, which is only possible on the host memory space. Thus,
       * temporarily transfer the state vector and the precomputed values
       * to the host memory space:
       */

      constexpr bool transfer =
          have_separate_memory_spaces &&
          !std::is_same_v<MemorySpace, dealii::MemorySpace::Host>;

      [[maybe_unused]] const bool host_resident =
          U.template is_resident<HostSpace>();

      if constexpr (transfer) {
        U.template copy_to_memory_space<HostSpace>();
        precomputed.template move_to_memory_space<HostSpace>();
      }

      {
        const auto sparsity_simd_view =
            offline_data.sparsity_pattern_simd().template view<HostSpace>();

        /* We only read the hyperbolic state vector: */
        const auto U_view = std::as_const(U).template view<HostSpace>();
        const auto precomputed_view = precomputed.template view<HostSpace>();

        const auto hyperbolic_system_views =
            make_select_view<dim, ScalarNumber, HostSpace>(*this);

        const auto body = [=](auto sentinel, unsigned int i) {
          using T = decltype(sentinel);
          using View = HyperbolicSystemView<dim, T, HostSpace>;
          using precomputed_type = typename View::precomputed_type;

          const unsigned int row_length = sparsity_simd_view.row_length(i);
          if (skip_constrained_dofs && row_length == 1)
            return;

          const auto view = hyperbolic_system_views.template view<T>();

          const auto U_i = U_view.template read_tensor<T>(i);
          const auto rho_i = view.density(U_i);
          const auto e_i = view.internal_energy(U_i) / rho_i;

          /* Calls into the selected equation of state: */
          const auto p_i = view.eos_pressure(rho_i, e_i);

          const auto gamma_i = view.surrogate_gamma(U_i, p_i);
          const precomputed_type prec_i{p_i, gamma_i, T(0.), T(0.)};
          precomputed_view.template write_tensor<T>(prec_i, i);
        };

        loop<HostSpace, ScalarNumber>(
            "hyperbolic_kernel_01b", body, 0, n_internal, n_owned);
      }

      if constexpr (transfer) {
        /* Drop the (now stale) host mirror unless it was resident before: */
        if (!host_resident)
          U.template move_to_memory_space<MemorySpace>();
        precomputed.template move_to_memory_space<MemorySpace>();
      }

      precomputed.template view<MemorySpace>().update_ghost_values();

      /*
       * Compute gamma_min over the stencil:
       */

      {
        const auto sparsity_simd_view =
            offline_data.sparsity_pattern_simd().template view<MemorySpace>();

        /* We only read the hyperbolic state vector: */
        const auto U_view = std::as_const(U).template view<MemorySpace>();
        const auto precomputed_view = precomputed.template view<MemorySpace>();

        const auto hyperbolic_system_views =
            make_select_view<dim, ScalarNumber, MemorySpace>(*this);

        const auto body = [=](auto sentinel, unsigned int i) {
          using T = decltype(sentinel);
          using View = HyperbolicSystemView<dim, T, MemorySpace>;
          using PT = typename View::precomputed_type;

          const unsigned int row_length = sparsity_simd_view.row_length(i);
          if (skip_constrained_dofs && row_length == 1)
            return;

          const auto view = hyperbolic_system_views.template view<T>();

          const auto U_i = U_view.template read_tensor<T>(i);
          auto prec_i = precomputed_view.template read_tensor<T, PT>(i);
          /* Previous loop: gamma_min_i == gamma_i, s_i == 0, eta_i == 0 */
          auto &[p_i, gamma_min_i, s_i, eta_i] = prec_i;

          const unsigned int stride_size = sparsity_simd_view.stride_of_row(i);
          const unsigned int *js = sparsity_simd_view.columns(i) + stride_size;
          for (unsigned int col_idx = 1; col_idx < row_length;
               ++col_idx, js += stride_size) {

            const auto U_j = U_view.template read_tensor<T>(js);
            const auto prec_j =
                precomputed_view.template read_tensor<T, PT>(js);
            const auto p_j = std::get<0>(prec_j);
            const auto gamma_j = view.surrogate_gamma(U_j, p_j);
            gamma_min_i = std::min(gamma_min_i, gamma_j);
          }

          s_i = view.surrogate_specific_entropy(U_i, gamma_min_i);
          eta_i = view.surrogate_harten_entropy(U_i, gamma_min_i);
          precomputed_view.template write_tensor<T>(prec_i, i);
        };

        loop<MemorySpace, ScalarNumber>(
            "hyperbolic_kernel_01c", body, 0, n_internal, n_owned);
      }
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::density(const state_type &U)
    {
      return U[0];
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::filter_vacuum_density(
        const Number &rho) const
    {
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const Number rho_cutoff_large =
          reference_density() * vacuum_state_relaxation_large() * eps;

      return ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          std::abs(rho), rho_cutoff_large, Number(0.), rho);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE dealii::Tensor<1, dim, Number>
    HyperbolicSystemView<dim, Number, MemorySpace>::momentum(
        const state_type &U)
    {
      dealii::Tensor<1, dim, Number> result;
      for (unsigned int i = 0; i < dim; ++i)
        result[i] = U[1 + i];
      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::total_energy(
        const state_type &U)
    {
      return U[1 + dim];
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::internal_energy(
        const state_type &U)
    {
      /*
       * rho e = (E - 1/2*m^2/rho)
       */
      const Number rho_inverse = ScalarNumber(1.) / density(U);
      const auto m = momentum(U);
      const Number E = total_energy(U);
      return E - ScalarNumber(0.5) * m.norm_square() * rho_inverse;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::internal_energy_derivative(
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::surrogate_specific_entropy(
        const state_type &U, const Number &gamma_min) const
    {
      const auto b = Number(eos_covolume_constant());
      const auto pinf = Number(eos_interpolation_pinfty());
      const auto q = Number(eos_interpolation_q());

      const auto rho = density(U);
      const auto rho_inverse = ScalarNumber(1.) / rho;

      const auto covolume = Number(1.) - b * rho;

      const auto shift = internal_energy(U) - rho * q - pinf * covolume;

      return shift * ryujin::pow(rho_inverse - b, gamma_min) / covolume;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::surrogate_harten_entropy(
        const state_type &U, const Number &gamma_min) const
    {
      const auto b = Number(eos_covolume_constant());
      const auto pinf = Number(eos_interpolation_pinfty());
      const auto q = Number(eos_interpolation_q());

      const auto rho = density(U);
      const auto m = momentum(U);
      const auto E = total_energy(U);
      const auto rho_rho_e_q =
          rho * E - ScalarNumber(0.5) * m.norm_square() - rho * rho * q;

      const auto exponent = ScalarNumber(1.) / (gamma_min + Number(1.));

      const auto covolume = Number(1.) - b * rho;
      const auto covolume_term = ryujin::pow(covolume, gamma_min - Number(1.));

      const auto rho_pinfcov = rho * pinf * covolume;

      return ryujin::pow(
          positive_part(rho_rho_e_q - rho_pinfcov) * covolume_term, exponent);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::
        surrogate_harten_entropy_derivative(const state_type &U,
                                            const Number &eta,
                                            const Number &gamma_min) const
        -> state_type
    {
      /*
       * With
       *   eta = (shift * (1-b*rho)^{gamma-1}) ^ {1/(gamma+1)},
       *   shift = rho * E - 1/2 |m|^2 - rho^2 * q - p_infty * rho * (1 - b rho)
       *
       *   shift' = [E - 2 * rho * q - p_infty * (1 - 2 b rho), -m, rho]^T
       *   factor = 1/(gamma+1) * (eta/(1-b rho))^-gamma / (1-b rho)^2
       *
       * we get
       *
       *   eta' = factor * (1-b*rho) * shift' -
       *          factor * shift * (gamma - 1) * b * [1, 0, 0]^T
       *
       */
      const auto b = Number(eos_covolume_constant());
      const auto pinf = Number(eos_interpolation_pinfty());
      const auto q = Number(eos_interpolation_q());

      const auto rho = density(U);
      const auto m = momentum(U);
      const auto E = total_energy(U);

      const auto covolume = Number(1.) - b * rho;
      const auto covolume_inverse = ScalarNumber(1.) / covolume;

      const auto shift = rho * E - ScalarNumber(0.5) * m.norm_square() -
                         rho * rho * q - rho * pinf * covolume;

      constexpr auto eps = std::numeric_limits<ScalarNumber>::epsilon();
      const auto regularization = m.norm() * eps;
      const auto max_val = ryujin::pow(
          std::max(regularization, eta * covolume_inverse), gamma_min);
      auto factor = safe_division(Number(1.0), max_val);
      factor *= fixed_power<2>(covolume_inverse) / (gamma_min + Number(1.));

      state_type result;

      const auto first_term = E - ScalarNumber(2.) * rho * q -
                              pinf * (Number(1.) - ScalarNumber(2.) * b * rho);
      const auto second_term = -(gamma_min - Number(1.)) * shift * b;

      result[0] = factor * (covolume * first_term + second_term);
      for (unsigned int i = 0; i < dim; ++i)
        result[1 + i] = -factor * covolume * m[i];
      result[dim + 1] = factor * covolume * rho;

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::surrogate_gamma(
        const state_type &U, const Number &p) const
    {
      const auto b = Number(eos_covolume_constant());
      const auto pinf = Number(eos_interpolation_pinfty());
      const auto q = Number(eos_interpolation_q());

      const auto rho = density(U);
      const auto rho_e = internal_energy(U);
      const auto covolume = Number(1.) - b * rho;

      const auto numerator = (p + pinf) * covolume;
      const auto denominator = rho_e - rho * q - covolume * pinf;
      return Number(1.) + safe_division(numerator, denominator);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::surrogate_pressure(
        const state_type &U, const Number &gamma) const
    {
      const auto b = Number(eos_covolume_constant());
      const auto pinf = Number(eos_interpolation_pinfty());
      const auto q = Number(eos_interpolation_q());

      const auto rho = density(U);
      const auto rho_e = internal_energy(U);
      const auto covolume = Number(1.) - b * rho;

      return positive_part(gamma - Number(1.)) *
                 safe_division(rho_e - rho * q, covolume) -
             gamma * pinf;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::surrogate_speed_of_sound(
        const state_type &U, const Number &gamma) const
    {
      const auto b = Number(eos_covolume_constant());
      const auto pinf = Number(eos_interpolation_pinfty());
      const auto q = Number(eos_interpolation_q());

      const auto rho = density(U);
      const auto rho_e = internal_energy(U);
      const auto covolume = Number(1.) - b * rho;

      auto radicand =
          (rho_e - rho * q - pinf * covolume) / (covolume * covolume * rho);
      radicand *= gamma * (gamma - 1.);
      return std::sqrt(positive_part(radicand));
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE bool
    HyperbolicSystemView<dim, Number, MemorySpace>::is_admissible(
        const state_type &U) const
    {
      const auto b = Number(eos_covolume_constant());
      const auto pinf = Number(eos_interpolation_pinfty());
      const auto q = Number(eos_interpolation_q());

      const auto rho = density(U);
      const auto rho_e = internal_energy(U);
      const auto covolume = Number(1.) - b * rho;

      const auto shift = rho_e - rho * q - pinf * covolume;

      constexpr auto gt = dealii::SIMDComparison::greater_than;
      using T = Number;
      const auto test =
          ryujin::compare_and_apply_mask<gt>(rho, T(0.), T(0.), T(-1.)) + //
          ryujin::compare_and_apply_mask<gt>(shift, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: Negative state [rho, e] detected!\n";
        std::cout << "\t\trho:           " << rho << "\n";
        std::cout << "\t\tint (shifted): " << shift << "\n";
      }
#endif

      return (test == Number(0.));
    }


    template <int dim, typename Number, typename MemorySpace>
    template <int component>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::
        prescribe_riemann_characteristic(
            const state_type &U,
            const Number &p,
            const state_type &U_bar,
            const Number &p_bar,
            const dealii::Tensor<1, dim, Number> &normal) const -> state_type
    {
      static_assert(component == 1 || component == 2,
                    "component has to be 1 or 2");

      const auto b = Number(eos_covolume_constant());
      const auto pinf = Number(eos_interpolation_pinfty());
      const auto q = Number(eos_interpolation_q());

      /*
       * The "four" Riemann characteristics are formed under the assumption
       * of a locally isentropic flow. For this, we first transform both
       * states into {rho, vn, vperp, gamma, a}, where we use the NASG EOS
       * interpolation to derive a surrogate gamma and speed of sound a.
       *
       * See, e.g., https://arxiv.org/pdf/2004.08750, "Compressible flow in
       * a NOble-Abel Stiffened-Gas fluid", M. I. Radulescu.
       */

      const auto m = momentum(U);
      const auto rho = density(U);
      const auto vn = m * normal / rho;

      const auto gamma = surrogate_gamma(U, p);
      const auto a = surrogate_speed_of_sound(U, gamma);
      const auto covolume = 1. - b * rho;

      const auto m_bar = momentum(U_bar);
      const auto rho_bar = density(U_bar);
      const auto vn_bar = m_bar * normal / rho_bar;

      const auto gamma_bar = surrogate_gamma(U_bar, p_bar);
      const auto a_bar = surrogate_speed_of_sound(U_bar, gamma_bar);
      const auto covolume_bar = 1. - b * rho_bar;

      /*
       * Now compute the Riemann characteristics {R_1, R_2, vperp, s}:
       *   R_1 = v * n - 2 / (gamma - 1) * a * (1 - b * rho)
       *   R_2 = v * n + 2 / (gamma - 1) * a * (1 - b * rho)
       *   vperp
       *   S = (p + p_infty) / rho^gamma * (1 - b * rho)^gamma
       *
       * Here, we replace either R_1, or R_2 with values coming from U_bar:
       */

      const auto R_1 =
          component == 1 ? vn_bar - 2. * a_bar / (gamma_bar - 1.) * covolume_bar
                         : vn - 2. * a / (gamma - 1.) * covolume;

      const auto R_2 =
          component == 2 ? vn_bar + 2. * a_bar / (gamma_bar - 1.) * covolume_bar
                         : vn + 2. * a / (gamma - 1.) * covolume;

      /*
       * Note that we are really hoping for the best here... We require
       * that R_2 >= R_1 so that we can extract a valid sound speed...
       */

      Assert(
          R_2 >= R_1,
          dealii::ExcMessage("Encountered R_2 < R_1 in dynamic boundary value "
                             "enforcement. This implies that the interpolation "
                             "with Riemann characteristics failed."));

      const auto vperp = m / rho - vn * normal;

      const auto S = (p + pinf) * ryujin::pow(Number(1.) / rho - b, gamma);

      /*
       * Now, we have to reconstruct the actual conserved state U from the
       * Riemann characteristics R_1, R_2, vperp, and s. We first set up
       * {vn_new, vperp_new, a_new, S} and then solve for {rho_new, p_new}
       * with the help of the NASG EOS surrogate formulas:
       *
       *   S = (p + p_infty) / rho^gamma * (1 - b * rho)^gamma
       *
       *   a^2 = gamma * (p + p_infty) / (rho * cov)
       *
       *   This implies:
       *
       *   a^2 / (gamma * S) = rho^{gamma - 1} / (1 - b * rho)^{1 + gamma}
       */

      const auto vn_new = Number(0.5) * (R_1 + R_2);

      /*
       * Technically, we would need to solve for rho subject to a number of
       * nonlinear relationships:
       *
       *   a   = (gamma - 1) * (R_2 - R_1) / (4. * (1 - b * rho))
       *
       *   a^2 / (gamma * S) = rho^{gamma - 1} / (1 - b * rho)^{gamma + 1}
       *
       * This seems to be a bit expensive for the fact that our dynamic
       * boundary conditions are already terribly heuristic...
       *
       * So instead, we rewrite this system as:
       *
       *   a * (1 - b * rho) = (gamma - 1) * (R_2 - R_1) / 4.
       *
       *   a^2 / (gamma * S) (1 - b * rho)^2
       *                           = (rho / (1 - b * rho))^{gamma - 1}
       *
       * And compute the terms on the left simply with the old covolume and
       * solving an easier easier nonlinear equation for the density. The
       * resulting system reads:
       *
       *   a = (gamma - 1) * (R_2 - R_1) / (4. * (1 - b * rho_old))
       *   A = {a^2 / (gamma * S) (1 - b * rho_old)^{2 gamma}}^{1/(gamma - 1)}
       *
       *   rho = A / (1 + b * A)
       */

      const auto a_new_square =
          ryujin::fixed_power<2>((gamma - 1.) * (R_2 - R_1) / (4. * covolume));

      auto term = ryujin::pow(a_new_square / (gamma * S), 1. / (gamma - 1.));
      if (b != ScalarNumber(0.)) {
        term *= ryujin::pow(covolume, 2. / (gamma - 1.));
      }

      const auto rho_new = term / (1. + b * term);

      const auto covolume_new = (1. - b * rho_new);
      const auto p_new = a_new_square / gamma * rho_new * covolume_new - pinf;

      /*
       * And translate back into conserved quantities:
       */

      const auto rho_e_new =
          rho_new * q + (p_new + gamma * pinf) * covolume_new / (gamma - 1.);

      state_type U_new;
      U_new[0] = rho_new;
      for (unsigned int d = 0; d < dim; ++d) {
        U_new[1 + d] = rho_new * (vn_new * normal + vperp)[d];
      }
      U_new[1 + dim] =
          rho_e_new + 0.5 * rho_new * (vn_new * vn_new + vperp.norm_square());

      return U_new;
    }


    template <int dim, typename Number, typename MemorySpace>
    template <typename Lambda>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number, MemorySpace>::apply_boundary_conditions(
        dealii::types::boundary_id id,
        const state_type &U,
        const dealii::Tensor<1, dim, Number> &normal,
        const Lambda &get_dirichlet_data) const -> state_type
    {
      state_type result = U;

      if (id == Boundary::dirichlet) {
        result = get_dirichlet_data();

      } else if (id == Boundary::dirichlet_momentum) {
        /*
         * Only enforce Dirichlet conditions on the momentum, and keep the
         * internal energy constant:
         */
        const auto m_dirichlet = momentum(get_dirichlet_data());
        const auto rho = density(result);
        const auto m = momentum(result);

        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = m_dirichlet[k];
        result[dim + 1] +=
            Number(0.5) / rho * (m_dirichlet.norm_square() - m.norm_square());

      } else if (id == Boundary::dirichlet_velocity) {
        /*
         * Only enforce Dirichlet conditions on the velocity, and keep the
         * internal energy constant:
         */
        const auto U_dirichlet = get_dirichlet_data();
        const auto rho_dirichlet = density(U_dirichlet);
        const auto v_dirichlet = momentum(U_dirichlet) / rho_dirichlet;
        const auto rho = density(result);
        const auto v = momentum(result) / rho;

        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = rho * v_dirichlet[k];
        result[dim + 1] +=
            Number(0.5) * rho * (v_dirichlet.norm_square() - v.norm_square());

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
        const auto rho_e = internal_energy(U);

        /*
         * We do not have precomputed values available. Thus, simply query
         * the pressure oracle and compute a surrogate speed of sound from
         * there:
         */
        const auto p = eos_pressure(rho, rho_e / rho);
        const auto gamma = surrogate_gamma(U, p);
        const auto a = surrogate_speed_of_sound(U, gamma);
        const auto vn = m * normal / rho;

        /* Supersonic inflow: */
        if (vn < -a) {
          result = get_dirichlet_data();
        }

        /* Subsonic inflow: */
        if (vn >= -a && vn <= 0.) {
          const auto U_dirichlet = get_dirichlet_data();
          const auto rho_dirichlet = density(U_dirichlet);
          const auto rho_e_dirichlet = internal_energy(U_dirichlet);
          const auto p_dirichlet =
              eos_pressure(rho_dirichlet, rho_e_dirichlet / rho_dirichlet);

          result = prescribe_riemann_characteristic<2>(
              U_dirichlet, p_dirichlet, U, p, normal);
        }

        /* Subsonic outflow: */
        if (vn > 0. && vn <= a) {
          const auto U_dirichlet = get_dirichlet_data();
          const auto rho_dirichlet = density(U_dirichlet);
          const auto rho_e_dirichlet = internal_energy(U_dirichlet);
          const auto p_dirichlet =
              eos_pressure(rho_dirichlet, rho_e_dirichlet / rho_dirichlet);

          result = prescribe_riemann_characteristic<1>(
              U, p, U_dirichlet, p_dirichlet, normal);
        }
        /* Supersonic outflow: do nothing, i.e., keep U as is */

      } else {
        AssertThrow(false, dealii::ExcNotImplemented());
      }

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::f(const state_type &U,
                                                      const Number &p) const
        -> flux_type
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::flux_contribution(
        const PrecomputedVectorView &pv,
        const InitialPrecomputedVectorView & /*piv*/,
        const unsigned int i,
        const state_type &U_i) const -> flux_contribution_type
    {
      const auto &[p_i, gamma_min_i, s_i, eta_i] =
          pv.template read_tensor<Number, precomputed_type>(i);
      return f(U_i, p_i);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::flux_contribution(
        const PrecomputedVectorView &pv,
        const InitialPrecomputedVectorView & /*piv*/,
        const unsigned int *js,
        const state_type &U_j) const -> flux_contribution_type
    {
      const auto &[p_j, gamma_min_j, s_j, eta_j] =
          pv.template read_tensor<Number, precomputed_type>(js);
      return f(U_j, p_j);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::flux_divergence(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij) const -> state_type
    {
      return -contract(add(flux_i, flux_j), c_ij);
    }


    template <int dim, typename Number, typename MemorySpace>
    template <typename ST>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::expand_state(
        const ST &state) const -> state_type
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


    template <int dim, typename Number, typename MemorySpace>
    template <typename ST>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number, MemorySpace>::from_initial_state(
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::from_primitive_state(
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


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::to_primitive_state(
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


    template <int dim, typename Number, typename MemorySpace>
    template <typename Lambda>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::apply_galilei_transform(
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
