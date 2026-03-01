//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
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
  namespace MultiSpeciesEuler
  {
    /**
     * Compile-time parameter specifying the number of species in the
     * multi-species Euler equations. Change this value and recompile to
     * use a different number of species.
     *
     * The state vector dimension will be: n_species + dim + 1
     * (n_species partial densities + dim momentum components + 1 total energy)
     */
    constexpr unsigned int n_species = 2;
    static_assert(n_species >= 1 && n_species <= 3,
                  "n_species must be between 1 and 3");
    /*
     * For various divisions in the multi-species module we have a
     * mathematical guarantee that the numerator and denominator are
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


    /**
     * Convert from user-friendly primitive format
     *   (Y_0, ..., Y_{n-2}, rho, u, p)
     * to internal primitive format
     *   (alpha_0*rho_0, ..., alpha_{n-1}*rho_{n-1}, u, p)
     *
     * The last species' partial density is computed from the constraint
     * that mass fractions sum to one.
     */
    template <typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, n_species + 2, Number>
    extend_primitive(
        const dealii::Tensor<1, n_species + 2, Number> &primitive_in)
    {
      dealii::Tensor<1, n_species + 2, Number> result;

      const auto rho = primitive_in[n_species - 1];
      Number Y_sum = Number(0.);

      /* Convert mass fractions to partial densities */
      for (unsigned int k = 0; k < n_species - 1; ++k) {
        result[k] = primitive_in[k] * rho;
        Y_sum += primitive_in[k];
      }
      result[n_species - 1] = (Number(1.) - Y_sum) * rho;

      /* Copy velocity and pressure */
      result[n_species] = primitive_in[n_species];
      result[n_species + 1] = primitive_in[n_species + 1];

      return result;
    }


    template <int dim, typename Number>
    class HyperbolicSystemView;

    /**
     * The compressible multi-species Euler equations for a gas mixture
     * under the assumption of thermal-mechanical equilibrium. Each species
     * \f$k\f$ is modeled by an ideal gas with specific heat capacities
     * \f$c_{p,k}\f$ and \f$c_{v,k}\f$ at constant pressure and volume,
     * respectively.
     *
     * We have a \f$(n_s + 1 + d)\f$ dimensional state space
     * \f[
     *   \mathbf{u} = [(\alpha_k \rho_k)_{k=1}^{n_s}, \mathbf{m}, E]^T
     *   \in \mathbb{R}^{n_s + 1 + d},
     * \f]
     * where \f$\alpha_k \rho_k\f$ denotes the partial densities,
     * \f$\mathbf{m} = \rho \mathbf{v}\f$ is the mixture momentum, and
     * \f$E\f$ is the mixture total energy.
     *
     * The mixture density is \f$\rho = \sum_k \alpha_k \rho_k\f$ and the
     * mass fractions are \f$Y_k = \alpha_k \rho_k / \rho\f$. The internal
     * energy density is \f$\varepsilon = E - \frac{1}{2}\rho|\mathbf{v}|^2\f$.
     *
     * The mixture pressure is given by an ideal mixture equation of state
     * (see Eq. (2.4) in [ClaytonDzanicTovar-2025]):
     * \f[
     *   p = (\bar{\gamma}(\mathbf{Y}) - 1) \varepsilon,
     * \f]
     * where \f$\bar{\gamma} = \bar{c}_p / \bar{c}_v\f$ with
     * \f$\bar{c}_p = \sum_k Y_k c_{p,k}\f$ and
     * \f$\bar{c}_v = \sum_k Y_k c_{v,k}\f$.
     *
     * The mixture specific entropy is (see Eq. (2.8) in @cite
     * ClaytonDzanicTovar-2025):
     * \f[
     *   s(\mathbf{u}) = \bar{c}_v \log\left(\frac{\rho e}{\rho^{\bar{\gamma}}}
     *   \right) + K(\mathbf{Y}),
     * \f]
     * where \f$e = \varepsilon / \rho\f$ is the specific internal energy and
     * \f$K(\mathbf{Y})\f$ is a mixing term depending only on mass fractions.
     *
     * The invariant domain preserved by the numerical scheme is (see Eq.
     * (2.11) in [ClaytonDzanicTovar-2025]):
     * \f[
     *   \mathcal{A} = \{ \mathbf{u} : \alpha_k \rho_k \geq 0 \;\forall k,\;
     *   \varepsilon(\mathbf{u}) > 0,\; s(\mathbf{u}) \geq s_{\min} \}.
     * \f]
     *
     *
     * @ingroup MultiSpeciesEulerEquations
     */
    class HyperbolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the hyperbolic system as a string.
       */
      static inline std::string problem_name =
          "Compressible multi species Euler equations (ideal mixture)";

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

      dealii::Tensor<1, n_species, double> cp_for_each_species_;
      dealii::Tensor<1, n_species, double> cv_for_each_species_;
      dealii::Tensor<1, n_species, double> r_for_each_species_;
      dealii::Tensor<1, n_species, double> gamma_for_each_species_;
      double reference_density_;
      double vacuum_state_relaxation_small_;
      double vacuum_state_relaxation_large_;
      bool compute_strict_bounds_;

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

      DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, n_species, double>
      cp_for_each_species() const
      {
        return hyperbolic_system_.cp_for_each_species_;
      }

      DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, n_species, double>
      cv_for_each_species() const
      {
        return hyperbolic_system_.cv_for_each_species_;
      }

      DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, n_species, double>
      r_for_each_species() const
      {
        return hyperbolic_system_.r_for_each_species_;
      }

      DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, n_species, double>
      gamma_for_each_species() const
      {
        return hyperbolic_system_.gamma_for_each_species_;
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

      /**
       * constexpr boolean used in the MultiSpeciesEulerInitialStates namespace
       */
      static constexpr bool have_gamma = false;

      /**
       * constexpr boolean used in the MultiSpeciesEulerInitialStates namespace
       */
      static constexpr bool have_eos_interpolation_b = false;

      /**
       * constexpr boolean for energy equation presence
       */
      static constexpr bool have_energy_equation = true;


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
       * The dimension of the state space: n_species partial densities +
       * dim momentum components + 1 total energy.
       */
      static constexpr unsigned int problem_dimension = n_species + 1 + dim;

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
       * Storage type for the "total mixture" state vector \f$\boldsymbol U\f$.
       */
      using mixture_state_type = dealii::Tensor<1, 2 + dim, Number>;

      /**
       * Storage type for the "total mixture" flux \f$\mathbf{f^{mix}}\f$.
       */
      using mixture_flux_type =
          dealii::Tensor<1, 2 + dim, dealii::Tensor<1, dim, Number>>;

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
        std::array<std::string, problem_dimension> names;
        for (unsigned int k = 0; k < n_species; ++k)
          names[k] = "alpha_rho_" + std::to_string(k);
        if constexpr (dim == 1) {
          names[n_species] = "m";
        } else {
          for (unsigned int d = 0; d < dim; ++d)
            names[n_species + d] = "m_" + std::to_string(d + 1);
        }
        names[n_species + dim] = "E";
        return names;
      }();

      /**
       * An array holding all component names of the primitive state as a
       * string.
       */
      static inline const auto primitive_component_names =
          []() -> std::array<std::string, problem_dimension> {
        std::array<std::string, problem_dimension> names;
        for (unsigned int k = 0; k < n_species; ++k)
          names[k] = "alpha_rho_" + std::to_string(k);
        if constexpr (dim == 1) {
          names[n_species] = "v";
        } else {
          for (unsigned int d = 0; d < dim; ++d)
            names[n_species + d] = "v_" + std::to_string(d + 1);
        }
        names[n_species + dim] = "p";
        return names;
      }();

      /**
       * The number of precomputed values.
       */
      static constexpr unsigned int n_precomputed_values = 5;

      /**
       * Array type used for precomputed values.
       */
      using precomputed_type = std::array<Number, n_precomputed_values>;

      /**
       * An array holding all component names of the precomputed values.
       */
      static inline const auto precomputed_names =
          std::array<std::string, n_precomputed_values>{
              {"rho",
               "p",
               "surrogate_gamma_min",
               "specific_entropy",
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
       * @name Computing derived physical quantities
       */
      //@{

      /**
       * For a given state vector <code>U</code>, return the partial density
       * for species <code>k</code>, i.e., <code>U[k]</code>.
       */
      static Number partial_density(const state_type &U, unsigned int k);

      /**
       * For a given state vector <code>U</code>, return the total mixture
       * density obtained by summing all partial densities.
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
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>,
       * return the momentum vector
       * <code>[U[n_species], ..., U[n_species+dim-1]]</code>.
       */
      static dealii::Tensor<1, dim, Number> momentum(const state_type &U);

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>, return
       * the total energy <code>U[n_species+dim]</code>
       */
      static Number total_energy(const state_type &U);

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>, compute
       * and return the internal energy \f$\varepsilon = (\rho e)\f$.
       */
      static Number internal_energy(const state_type &U);

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>, compute
       * and return the mixture gamma for the mixture EOS:
       * \f[
       *  \overline{\gamma} = (sum of (alpha_k rho_k) * c_{p, k}) / (sum of
       * (alpha_k rho_k) * c_{v, k})
       * \f]
       */
      Number gamma_mixture(const state_type &U) const;

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>, compute
       * and return the mixture specific heat capacity at constant pressure:
       * \f[
       *  \overline{c}_p = (sum of (alpha_k rho_k) / rho * c_{p, k})
       * \f]
       */
      Number cp_mixture(const state_type &U) const;

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>, compute
       * and return the mixture specific heat capacity at constant volume:
       * \f[
       *  \overline{c}_v = (sum of (alpha_k rho_k) / rho * c_{v, k})
       * \f]
       */
      Number cv_mixture(const state_type &U) const;

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>, compute
       * and return the pressure \f$p\f$.
       *
       * We assume that the pressure is given by a mixture ideal EOS
       * \f[
       *   p = (\overline{\gamma} - 1)\;(\rho e)
       * \f]
       */
      Number pressure(const state_type &U) const;

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>,
       * compute and return the speed of sound \f$c\f$:
       *
       * We assume that the pressure is given by a mixture ideal EOS
       * \f[
       *   c = sqrt(\overline{\gamma} p / rho)
       * \f]
       */
      Number speed_of_sound(const state_type &U) const;

      //@}
      /**
       * @name Surrogate functions for computing various interpolatory
       * physical quantities that are needed for Riemann solver,
       * indicator and limiter.
       */
      //@{

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>, compute
       * and return the physical specific entropy. Following Eq. (2.8) in
       * [ClaytonDzanicTovar-2025]:
       * \f[
       *   s(\mathbf{u}) = \bar{r} \log(\rho^{-1}) + \bar{c}_v \log(e)
       *   + K(\mathbf{Y}),
       * \f]
       * where \f$\bar{r} = \sum_k Y_k r_k\f$, \f$\bar{c}_v = \sum_k Y_k
       * c_{v,k}\f$, \f$e = \varepsilon / \rho\f$ is the specific internal
       * energy, and \f$K(\mathbf{Y})\f$ is a mixing term (Eq. 2.9).
       */
      Number specific_entropy(const state_type &U) const;

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>, compute
       * and return a surrogate Harten-type entropy. Following Section 4 of
       * [ClaytonDzanicTovar-2025], we use:
       * \f[
       *   \eta(\mathbf{u}; \gamma_{\min}) = (\rho
       * \varepsilon)^{1/(\gamma_{\min}+1)}, \f] where \f$\gamma_{\min}\f$ is
       * the minimum surrogate gamma over the stencil. This entropy is chosen to
       * ensure convexity properties required by the entropy-viscosity
       * indicator.
       */
      Number surrogate_harten_entropy(const state_type &U,
                                      const Number &gamma_min) const;

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code>, compute
       * and return the derivative \f$\eta'\f$ of the Harten-type entropy.
       */
      mixture_state_type
      surrogate_harten_entropy_derivative(const state_type &U,
                                          const Number &eta,
                                          const Number &gamma_min) const;

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code> and
       * pressure <code>p</code>, compute a surrogate gamma. Following
       * Section 3 of [ClaytonDzanicTovar-2025]:
       * \f[
       *   \gamma(\mathbf{u}, p) = 1 + \frac{p}{\varepsilon(\mathbf{u})}.
       * \f]
       * This surrogate gamma allows us to use the single-species Riemann
       * solver machinery with a locally linearized equation of state. It
       * is computed point-wise and the minimum over the stencil is used
       * for the Riemann solver and limiter.
       */
      Number surrogate_gamma(const state_type &U, const Number &p) const;

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code> and
       * gamma <code>gamma</code>, compute a surrogate pressure:
       * \f[
       *   p(\rho, e, \gamma) = (\gamma - 1) (\rho e)
       * \f]
       *
       * This function is the complementary function to surrogate_gamma().
       */
      Number surrogate_pressure(const state_type &U, const Number &gamma) const;

      /**
       * For a given (n_species+1+dim dimensional) state vector <code>U</code> and
       * gamma <code>gamma</code>, compute a surrogate speed of sound.
       */
      Number surrogate_speed_of_sound(const state_type &U,
                                      const Number &gamma) const;

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
       *   \textbf v \alpha_1 \rho_1 \\
       *   \textbf v \alpha_2 \rho_2
       *   \textbf v\otimes \textbf m + p\mathbb{I}_d \\
       *   \textbf v(E+p)
       * \end{pmatrix},
       * \f]
       */
      flux_type f(const state_type &U, const Number &p) const;

      /**
       * Given a state @p U and a pressure @p p compute the "summed system"
       * mixture flux needed for indicator:
       * \f[
       * \begin{pmatrix}
       *   \textbf m \\
       *   \textbf v\otimes \textbf m + p\mathbb{I}_d \\
       *   \textbf v(E+p)
       * \end{pmatrix},
       * \f]
       */
      mixture_flux_type mixture_f(const state_type &U, const Number &p) const;

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
       * Given an initial state
       * [alpha_0 rho_0, ..., alpha_{n-1} rho_{n-1}, u_1, ..., u_d, p]
       * return a conserved state
       * [alpha_0 rho_0, ..., alpha_{n-1} rho_{n-1}, m_1, ..., m_d, E].
       *
       * @note This function is used to conveniently convert (user
       * provided) primitive initial states with pressure values to a
       * conserved state in the MultiSpeciesEulerInitialStateLibrary.
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

      /* Set default values for species parameters */
      for (unsigned int k = 0; k < n_species; ++k) {
        cp_for_each_species_[k] = 1.4 + 0.27 * k; /* Default: 1.4, 1.67, ... */
        cv_for_each_species_[k] = 1.0;
      }

      add_parameter(
          "c_p for each species",
          cp_for_each_species_,
          "Specific heat capacity at constant pressure for each species");

      add_parameter(
          "c_v for each species",
          cv_for_each_species_,
          "Specific heat capacity at constant volume for each species");

      vacuum_state_relaxation_small_ = 1.e2;
      add_parameter("vacuum state relaxation small",
                    vacuum_state_relaxation_small_,
                    "Problem specific vacuum relaxation parameter");

      vacuum_state_relaxation_large_ = 1.e4;
      add_parameter("vacuum state relaxation large",
                    vacuum_state_relaxation_large_,
                    "Problem specific vacuum relaxation parameter");

      /*
       * And finally populate the r and gamma values.
       */
      const auto populate_values = [this]() {
        for (unsigned int k = 0; k < n_species; ++k) {
          r_for_each_species_[k] =
              cp_for_each_species_[k] - cv_for_each_species_[k];
          gamma_for_each_species_[k] =
              cp_for_each_species_[k] / cv_for_each_species_[k];
        }
      };

      ParameterAcceptor::parse_parameters_call_back.connect(populate_values);
      populate_values();
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

      const auto &U = std::get<0>(state_vector);
      auto &precomputed = std::get<1>(state_vector);

      /* Compute values over the diagonal: */

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
        const auto p_i = view.pressure(U_i);
        const auto s_i = view.specific_entropy(U_i);

        const auto gamma_i = view.surrogate_gamma(U_i, p_i);
        using PT = precomputed_type;
        const PT prec_i{rho_i, p_i, gamma_i, s_i, T(0.)};
        precomputed.template write_tensor<T>(prec_i, i);
      };

      cpu_simd_loop<ScalarNumber>("time_step_1", body, 0, n_internal, n_owned);
      precomputed.update_ghost_values();

      /* Compute gamma_min over the stencil: */

      const auto body_stencil = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);
        using View = HyperbolicSystemView<dim, T>;
        using PT = typename View::precomputed_type;

        const unsigned int row_length = sparsity_simd.row_length(i);
        if (skip_constrained_dofs && row_length == 1)
          return;

        const auto U_i = U.template read_tensor<T>(i);
        auto prec_i = precomputed.template read_tensor<T, PT>(i);
        auto &[rho_i, p_i, gamma_min_i, s_i, harten_i] = prec_i;

        const auto view = this->view<dim, T>();

        constexpr unsigned int stride_size = get_stride_size<T>;
        const unsigned int *js = sparsity_simd.columns(i) + stride_size;
        for (unsigned int col_idx = 1; col_idx < row_length;
             ++col_idx, js += stride_size) {

          const auto U_j = U.template read_tensor<T>(js);
          const auto prec_j = precomputed.template read_tensor<T, PT>(js);
          const auto p_j = std::get<1>(prec_j);
          const auto gamma_j = view.surrogate_gamma(U_j, p_j);
          gamma_min_i = std::min(gamma_min_i, gamma_j);
        }

        harten_i = view.surrogate_harten_entropy(U_i, gamma_min_i);
        precomputed.template write_tensor<T>(prec_i, i);
      };

      cpu_simd_loop<ScalarNumber>(
          "time_step_1", body_stencil, 0, n_internal, n_owned);
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::partial_density(const state_type &U,
                                                       unsigned int k)
    {
      AssertIndexRange(k, n_species);
      return U[k];
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::density(const state_type &U)
    {
      auto result = Number(0.);
      for (unsigned int k = 0; k < n_species; ++k)
        result += partial_density(U, k);
      return result;
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
        result[i] = U[n_species + i];
      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::total_energy(const state_type &U)
    {
      return U[n_species + dim];
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
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::cp_mixture(const state_type &U) const
    {
      Number result = 0.;
      const Number rho = density(U);

      for (unsigned int k = 0; k < n_species; ++k) {
        result += U[k] / rho * ScalarNumber(cp_for_each_species()[k]);
      }
      return result;
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::cv_mixture(const state_type &U) const
    {
      Number result = 0.;
      const Number rho = density(U);

      for (unsigned int k = 0; k < n_species; ++k) {
        result += U[k] / rho * ScalarNumber(cv_for_each_species()[k]);
      }
      return result;
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::gamma_mixture(const state_type &U) const
    {
      return cp_mixture(U) / cv_mixture(U);
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::pressure(const state_type &U) const
    {
      /* p = (\overline{gamma} - 1) * (rho e) */
      return (gamma_mixture(U) - Number(1.)) * internal_energy(U);
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::speed_of_sound(const state_type &U) const
    {
      /* c = sqrt(\overline{gamma} * p / rho) */
      const auto gamma = gamma_mixture(U);
      const auto p = pressure(U);
      const auto rho = density(U);
      return std::sqrt(gamma * p / rho);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::specific_entropy(
        const state_type &U) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const auto rho = density(U);
      const auto rho_inverse = ScalarNumber(1.) / rho;
      const auto e = internal_energy(U) * rho_inverse;

      Number cv_bar = 0.;
      Number r_bar = 0.;

      for (unsigned int k = 0; k < n_species; ++k) {
        const auto Y_k = U[k] * rho_inverse;
        r_bar += Y_k * Number(r_for_each_species()[k]);
        cv_bar += Y_k * Number(cv_for_each_species()[k]);
      }

      Number K_factor = 0.;
      for (unsigned int k = 0; k < n_species; ++k) {
        const auto Y_k = U[k] * rho_inverse;
        const auto cv_k = Number(cv_for_each_species()[k]);
        const auto r_k = Number(r_for_each_species()[k]);
        const auto gm1 = Number(gamma_for_each_species()[k] - 1.);

        const auto K_k =
            cv_k * std::log(cv_k / cv_bar * ryujin::pow(r_k / r_bar, gm1));
        K_factor += Y_k * K_k;
      }

      const auto s_bar =
          r_bar * std::log(rho_inverse) + cv_bar * std::log(e) + K_factor;

      return s_bar;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::surrogate_harten_entropy(
        const state_type &U, const Number &gamma_min) const
    {
      const Number rho = density(U);
      const auto m = momentum(U);
      const Number E = total_energy(U);
      const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

      const Number exponent = ScalarNumber(1.) / (gamma_min + Number(1.));
      return ryujin::pow(rho_rho_e, exponent);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::surrogate_harten_entropy_derivative(
        const state_type &U, const Number &eta, const Number &gamma_min) const
        -> mixture_state_type
    {
      const Number rho = density(U);
      const auto m = momentum(U);
      const Number E = total_energy(U);

      const auto factor =
          ryujin::pow(eta, -gamma_min) * Number(1.) / (gamma_min + Number(1.));

      mixture_state_type result;

      result[0] = factor * E;

      for (unsigned int i = 0; i < dim; ++i)
        result[1 + i] = -factor * m[i];

      result[dim + 1] = factor * rho;

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::surrogate_gamma(const state_type &U,
                                                       const Number &p) const
    {
      return Number(1.) + p / internal_energy(U);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::surrogate_pressure(
        const state_type &U, const Number &gamma) const
    {
      return (gamma - Number(1.)) * internal_energy(U);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::surrogate_speed_of_sound(
        const state_type &U, const Number &gamma) const
    {
      const auto rho = density(U);
      const auto rho_e = internal_energy(U);

      auto radicand = gamma * (gamma - Number(1.)) * rho_e / rho;
      return std::sqrt(positive_part(radicand));
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline bool
    HyperbolicSystemView<dim, Number>::is_admissible(const state_type &U) const
    {
      const auto rho = density(U);
      const auto rho_e = internal_energy(U);

      constexpr auto gt = dealii::SIMDComparison::greater_than;
      using T = Number;
      const auto test =
          dealii::compare_and_apply_mask<gt>(rho, T(0.), T(0.), T(-1.)) +   //
          dealii::compare_and_apply_mask<gt>(rho_e, T(0.), T(0.), T(-1.)) + //
          dealii::compare_and_apply_mask<gt>(pressure(U), T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: Negative state [rho, e] detected!\n";
        std::cout << "\t\trho: " << rho << "\n";
        std::cout << "\t\tint: " << rho_e << "\n";
        std::cout << "\t\tp    " << pressure(U) << "\n";
        std::cout << "\t\tent: " << specific_entropy(U) << std::endl;
      }
#endif

      return (test == Number(0.));
    }


    template <int dim, typename Number>
    template <int component>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::prescribe_riemann_characteristic(
        const state_type &U,
        const Number &p,
        const state_type &U_bar,
        const Number &p_bar,
        const dealii::Tensor<1, dim, Number> &normal) const -> state_type
    {
      static_assert(component == 1 || component == 2,
                    "component has to be 1 or 2");

      const auto b = Number(0.);
      const auto pinf = Number(0.);
      const auto q = Number(0.);

      /*
       * The "four" Riemann characteristics are formed under the assumption
       * of a locally isentropic flow. We further assume that the mass fractions
       * are constant along the locally isentropic flow. For this, we first
       * transform both states into {rho, vn, vperp, gamma, a}, where we use the
       * NASG EOS interpolation to derive a surrogate gamma and speed of sound
       * a.
       *
       * See, e.g., https://arxiv.org/pdf/2004.08750, "Compressible flow in
       * a NOble-Abel Stiffened-Gas fluid", M. I. Radulescu.
       */

      const auto rho = density(U);
      /* Store mass fractions Y_k = (alpha_k rho_k) / rho */
      dealii::Tensor<1, n_species, Number> Y;
      for (unsigned int k = 0; k < n_species; ++k)
        Y[k] = partial_density(U, k) / rho;

      const auto m = momentum(U);
      const auto vn = m * normal / rho;

      const auto gamma = surrogate_gamma(U, p);
      const auto a = surrogate_speed_of_sound(U, gamma);
      const auto covolume = 1. - b * rho;

      const auto rho_bar = density(U_bar);
      const auto m_bar = momentum(U_bar);
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
        term *= std::pow(covolume, 2. / (gamma - 1.));
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

      for (unsigned int k = 0; k < n_species; ++k)
        U_new[k] = Y[k] * rho_new;
      for (unsigned int d = 0; d < dim; ++d) {
        U_new[n_species + d] = rho_new * (vn_new * normal + vperp)[d];
      }

      U_new[n_species + dim] =
          rho_e_new + 0.5 * rho_new * (vn_new * vn_new + vperp.norm_square());

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

      } else if (id == Boundary::dirichlet_momentum) {
        /* Only enforce Dirichlet conditions on the momentum: */
        auto m_dirichlet = momentum(get_dirichlet_data());
        for (unsigned int d = 0; d < dim; ++d)
          result[n_species + d] = m_dirichlet[d];

      } else if (id == Boundary::slip) {
        auto m = momentum(U);
        m -= 1. * (m * normal) * normal;
        for (unsigned int d = 0; d < dim; ++d)
          result[n_species + d] = m[d];

      } else if (id == Boundary::no_slip) {
        for (unsigned int d = 0; d < dim; ++d)
          result[n_species + d] = Number(0.);

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
         * the pressure oracle and compute a surrogate speed of sound from
         * there:
         */
        const auto p = pressure(U);
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
          const auto p_dirichlet = pressure(U_dirichlet);

          result = prescribe_riemann_characteristic<2>(
              U_dirichlet, p_dirichlet, U, p, normal);
        }

        /* Subsonic outflow: */
        if (vn > 0. && vn <= a) {
          const auto U_dirichlet = get_dirichlet_data();
          const auto p_dirichlet = pressure(U_dirichlet);

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
      const auto E = total_energy(U);

      flux_type result;

      for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int k = 0; k < n_species; ++k)
          result[k][i] = U[k] * (m[i] * rho_inverse);
        result[n_species + i] = m * (m[i] * rho_inverse);
        result[n_species + i][i] += p;
      }
      result[n_species + dim] = m * (rho_inverse * (E + p));

      return result;
    }

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::mixture_f(const state_type &U,
                                                 const Number &p) const
        -> mixture_flux_type
    {
      const auto rho_inverse = ScalarNumber(1.) / density(U);
      const auto m = momentum(U);
      const auto E = total_energy(U);

      mixture_flux_type result;

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
      const auto &[rho_i, p_i, surrogate_gamma_i, s_i, surrogate_harten_i] =
          pv.template read_tensor<Number, precomputed_type>(i);
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
      const auto &[rho_j, p_j, surrogate_gamma_j, s_j, surrogate_harten_j] =
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

      constexpr auto dim2 = ST::dimension - n_species - 1;
      static_assert(dim >= dim2,
                    "the space dimension of the argument state must not be "
                    "larger than the one of the target state");

      state_type result;
      /* Copy partial densities */
      for (unsigned int k = 0; k < n_species; ++k)
        result[k] = state[k];
      /* Copy total energy */
      result[n_species + dim] = state[n_species + dim2];
      /* Copy momentum components (and zero-fill extra dimensions) */
      for (unsigned int i = 0; i < dim2; ++i)
        result[n_species + i] = state[n_species + i];

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
      /* extract velocity: */
      const auto u = /*SIC!*/ momentum(primitive_state);
      /* extract pressure: */
      const auto &p = primitive_state[n_species + dim];

      auto state = primitive_state;
      /* Fix up momentum: */
      for (unsigned int i = n_species; i < n_species + dim; ++i)
        state[i] *= rho;

      /* Compute total energy: */
      const Number gamma_bar = gamma_mixture(primitive_state);
      state[n_species + dim] =
          p / (gamma_bar - Number(1.)) + ScalarNumber(0.5) * rho * u * u;

      return state;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::to_primitive_state(
        const state_type &state) const -> state_type
    {
      const auto rho = density(state);
      const auto rho_inverse = Number(1.) / rho;
      const auto p = pressure(state);

      auto primitive_state = state;
      /* Fix up velocity: */
      for (unsigned int i = n_species; i < n_species + dim; ++i)
        primitive_state[i] *= rho_inverse;
      /* Set pressure: */
      primitive_state[n_species + dim] = p;

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
        result[n_species + d] = M[d];
      return result;
    }
  } // namespace MultiSpeciesEuler
} // namespace ryujin
