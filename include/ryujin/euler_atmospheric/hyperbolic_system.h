//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// [LANL Copyright Statement]
// Copyright (C) 2020 - 2026 by the ryujin authors
// Copyright (C) 2025 - 2026 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/gpu.h>
#include <ryujin/base/loop.h>
#include <ryujin/base/patterns_conversion.h>
#include <ryujin/base/simd.h>
#include <ryujin/discretization/discretization.h>
#include <ryujin/interface/hyperbolic_system.h>
#include <ryujin/linear_algebra/multicomponent_vector.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>

namespace ryujin
{
  namespace EulerAtmospheric
  {
    namespace internal
    {
      /**
       * Return an approximation of the ratio \f$A(x,y) / L(x,y)\f$ of the
       * arithmetic mean \f$A(x,y) = (x + y) / 2\f$ and the logarithmic
       * mean \f$L(x,y) = (x - y) / \ln(x / y)\f$ of two positive numbers
       * @p x and @p y.
       *
       * @note The ratio is exact up to roundoff for \f$|x - y| / (x + y) <
       * 0.02\f$ and lies between 1 and \f$A(x,y) / L(x,y)\f$ otherwise. It
       * is invariant under \f$x \to 1/x, y \to 1/y\f$.
       */
      template <typename Number>
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
      arithmetic_over_logarithmic_mean(const Number &x, const Number &y)
      {
        using ScalarNumber = typename get_value_type<Number>::type;

        /*
         * With f = (x - y) / (x + y) we have ln(x / y) = 2 atanh(f), and
         * thus A(x,y) / L(x,y) = atanh(f) / f = 1 + f^2/3 + f^4/5 + ....
         * We truncate the series after f^8:
         */
        const auto f = (x - y) / (x + y);
        const auto u = f * f;
        return ScalarNumber(1.) + u * (ScalarNumber(1. / 3.) +
                                       u * (ScalarNumber(1. / 5.) +
                                            u * (ScalarNumber(1. / 7.) +
                                                 u * ScalarNumber(1. / 9.))));
      }
    } // namespace internal


    template <int dim,
              typename Number,
              typename MemorySpace = dealii::MemorySpace::Host>
    class HyperbolicSystemView;

    /**
     * The compressible atmospheric Euler equations with ideal gas equation
     * of state. Well-balanced implementation.
     *
     * We have a (2 + dim) dimensional state space \f$[\rho, \textbf m,
     * \rho\theta]\f$, where \f$\rho\f$ denotes the density, \f$\textbf m\f$ is
     * the momentum, and \f$\rho\theta\f$ is the volumetric potential
     * temperature.
     *
     * @ingroup EulerAtmosphericEquations
     */
    class HyperbolicSystem : public Interface::HyperbolicSystem
    {
    public:
      /**
       * A structure holding all runtime parameters of the hyperbolic
       * system, as well as a number of derived quantities that would
       * otherwise need to be recomputed over and over again.
       */
      struct Parameters {
        /**
         * @name Run time parameters
         */
        //@{

        double c_v;
        double gamma;

        double gravity;
        double reference_pressure;
        double reference_density;
        double vacuum_state_relaxation_small;
        double vacuum_state_relaxation_large;

        //@}
        /**
         * @name Cached derived quantities
         *
         * The equation of state constant and a collection of commonly used
         * expressions with gamma that would otherwise need to be
         * recomputed many times putting unnecessary pressure on the
         * div/sqrt ALU unit. These are recomputed in update_parameters().
         */
        //@{

        double eos_constant;
        double gamma_minus_one;
        double gamma_inverse;
        double gamma_minus_one_inverse;

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
       */
      template <typename MemorySpace = dealii::MemorySpace::Host,
                int dim,
                typename ScalarNumber>
      void fill_precomputed_values(
          const OfflineData<dim, ScalarNumber> &offline_data,
          StateVector<ScalarNumber> &state_vector,
          const bool skip_constrained_dofs = true) const;

      /**
       * Return the initial precomputed values at @p point for the initial
       * (background) state @p U: the vertical coordinate, the damping
       * coefficient of the sponge layers, and the background velocity and
       * potential temperature that the sponge layers relax to.
       */
      template <int dim, typename Number>
      auto
      initial_precomputations(const dealii::Point<dim> &point,
                              const dealii::Tensor<1, dim + 2, Number> &U) const
          -> typename View<dim, Number>::initial_precomputed_type;

    private:
      /**
       * @name Internal fields, methods, and friends
       */
      //@{

      /**
       * Recompute all derived quantities of the Parameters structure and
       * copy the result over to the default memory space.
       */
      void update_parameters();

      Mirrored<Parameters> parameters_;

      /*
       * The damping coefficient of the sponge layers is only evaluated on
       * the host when setting up the initial precomputed values and is
       * thus not part of Parameters. We always use a three dimensional
       * function parser and pad points of lower dimension with zeros:
       */
      std::string sponge_damping_expression_;
      std::unique_ptr<dealii::FunctionParser<3>> sponge_damping_function_;

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
     * loops with access to either the host or device memory space.
     *
     * @ingroup EulerAtmosphericEquations
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
      using flux_contribution_type =
          std::tuple<state_type, Number, Number, Number, Number>;

      /**
       * The number of precomputed values.
       */
      static constexpr unsigned int n_precomputed_values = 4;

      /**
       * Array type used for precomputed values.
       */
      using precomputed_type = std::array<Number, n_precomputed_values>;

      /**
       * The number of precomputed initial values: the vertical coordinate,
       * the damping coefficient of the sponge layers, and the background
       * velocity and potential temperature.
       */
      static constexpr unsigned int n_initial_precomputed_values = dim + 3;

      /**
       * Array type used for precomputed initial values.
       */
      using initial_precomputed_type =
          std::array<Number, n_initial_precomputed_values>;

      template <int n_comp>
      using VectorView =
          MultiComponentVectorView<ScalarNumber, n_comp, MemorySpace, false>;

      /**
       * MulticomponentVectorView for accessing a vector of precomputed
       * states:
       */
      using PrecomputedVectorView = VectorView<n_precomputed_values>;

      /**
       * MulticomponentVectorView for accessing a vector of precomputed
       * initial states:
       */
      using InitialPrecomputedVectorView =
          VectorView<n_initial_precomputed_values>;

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
      {
      }

      //@}
      /**
       * @name Access to runtime parameters
       */
      //@{

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber c_v() const
      {
        return ScalarNumber(parameters_->c_v);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber gamma() const
      {
        return ScalarNumber(parameters_->gamma);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber gravity() const
      {
        return ScalarNumber(parameters_->gravity);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber reference_density() const
      {
        return ScalarNumber(parameters_->reference_density);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber reference_pressure() const
      {
        return ScalarNumber(parameters_->reference_pressure);
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

      //@}
      /**
       * @name Access to cached derived quantities
       *
       * A collection of commonly used expressions with gamma that would
       * otherwise need to be recomputed many times putting unnecessary
       * pressure on the div/sqrt ALU unit.
       */
      //@{

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber eos_constant() const
      {
        return ScalarNumber(parameters_->eos_constant);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber gamma_minus_one() const
      {
        return ScalarNumber(parameters_->gamma_minus_one);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber gamma_inverse() const
      {
        return ScalarNumber(parameters_->gamma_inverse);
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber
      gamma_minus_one_inverse() const
      {
        return ScalarNumber(parameters_->gamma_minus_one_inverse);
      }

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
       * the density times the potential temperature \f$\rho\theta\f$
       * <code>U[1+dim]</code>
       */
      static DEAL_II_HOST_DEVICE Number
      potential_temperature(const state_type &U);

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the pressure \f$p\f$.
       *
       * We assume that the pressure is given by an ideal gas equation of
       * state expressed in terms of the potential temperature, i.e.,
       * \f[
       *   p = K\,(\rho\theta)^\gamma,
       * \f]
       * where \f$K\f$ is the constant returned by eos_constant().
       */
      DEAL_II_HOST_DEVICE Number pressure(const state_type &U) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * the (physical) speed of sound:
       * \f[
       *   c^2 = \frac{\gamma\,p}{\rho}
       * \f]
       */
      DEAL_II_HOST_DEVICE Number speed_of_sound(const state_type &U) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the entropy
       * \f[
       *   \eta = \frac{p}{\gamma - 1} + \frac{|\textbf m|^2}{2\rho}.
       * \f]
       */
      DEAL_II_HOST_DEVICE Number
      mathematical_entropy(const state_type &U) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the derivative \f$\eta'\f$ of the entropy \f$\eta\f$.
       */
      DEAL_II_HOST_DEVICE state_type
      mathematical_entropy_derivative(const state_type &U) const;

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
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE state_type
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
       *   \textbf v\,\rho\theta
       * \end{pmatrix},
       * \f]
       */
      DEAL_II_HOST_DEVICE flux_type f(const state_type &U) const;

      /**
       * Variant of f() that takes the (precomputed) pressure @p p of the
       * state @p U as an additional argument.
       */
      DEAL_II_HOST_DEVICE flux_type f(const state_type &U,
                                      const Number &p) const;

      /**
       * Given a state @p U compute the flux without pressure contribution
       * \f[
       * \begin{pmatrix}
       *   \textbf m \\
       *   \textbf v\otimes \textbf m \\
       *   \textbf v\,\rho\theta
       * \end{pmatrix},
       * \f]
       */
      DEAL_II_HOST_DEVICE flux_type g(const state_type &U) const;

      /**
       * Return the logarithmic mean of \f$1/\theta_i\f$ and
       * \f$1/\theta_j\f$, i.e., \f$\int_{z_i}^{z_j} 1/\theta\,dz /
       * (z_j-z_i)\f$ for an exponential profile through \f$\theta_i\f$
       * and \f$\theta_j\f$ (constant Brunt-Vaisala frequency).
       */
      DEAL_II_HOST_DEVICE Number
      inverse_theta_mean(const Number &theta_i, const Number &theta_j) const;

      /**
       * For a given flux contribution @p flux (consisting of the state
       * \f$U\f$, its vertical coordinate, the precomputed powers
       * \f$\rho^{\gamma-1}\f$ and \f$\theta^{\gamma-1}\f$, and the
       * pressure), its potential temperature @p theta, the potential
       * temperature @p theta_other and the vertical coordinate @p
       * xd_other of the neighboring state, return
       * the hydrostatically reconstructed state <code>U_star</code>
       * together with its pressure.
       */
      DEAL_II_HOST_DEVICE std::tuple<state_type, Number>
      star_state(const flux_contribution_type &flux,
                 const Number &theta,
                 const Number &theta_other,
                 const Number &xd_other) const;

      /**
       * Given flux contributions @p flux_i and @p flux_j compute the
       * equilibrated states \f$U_i^{\ast,j}\f$ and
       * \f$U_j^{\ast,i}\f$.
       */
      DEAL_II_HOST_DEVICE
      std::array<state_type, 2>
      equilibrated_states(const flux_contribution_type &,
                          const flux_contribution_type &) const;

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
       * For the atmospheric Euler equations we return the state together
       * with the vertical coordinate and the precomputed powers of the
       * density and the potential temperature.
       */
      DEAL_II_HOST_DEVICE
      flux_contribution_type
      flux_contribution(const PrecomputedVectorView &pv,
                        const InitialPrecomputedVectorView &ipv,
                        const unsigned int i,
                        const state_type &U_i) const;

      DEAL_II_HOST_DEVICE
      flux_contribution_type
      flux_contribution(const PrecomputedVectorView &pv,
                        const InitialPrecomputedVectorView &ipv,
                        const unsigned int *js,
                        const state_type &U_j) const;

      /**
       * Given flux contributions @p flux_i and @p flux_j compute the
       * well-balanced low-order flux from the equilibrated states.
       */
      DEAL_II_HOST_DEVICE
      state_type
      flux_divergence(const flux_contribution_type &flux_i,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &c_ij) const;

      /** The high-order flux differs from the low-order flux: */
      static constexpr bool have_high_order_flux = true;

      /**
       * Given flux contributions @p flux_i and @p flux_j compute the
       * well-balanced high-order flux.
       */
      DEAL_II_HOST_DEVICE
      state_type high_order_flux_divergence(
          const flux_contribution_type &flux_i,
          const flux_contribution_type &flux_j,
          const dealii::Tensor<1, dim, Number> &c_ij) const;

      /**
       * Given flux contributions @p flux_i and @p flux_j compute the
       * equilibrated, low-order affine shift \f$B_{ij}\f$.
       */
      DEAL_II_HOST_DEVICE
      state_type affine_shift(const flux_contribution_type &flux_i,
                              const flux_contribution_type &flux_j,
                              const dealii::Tensor<1, dim, Number> &c_ij,
                              const Number &d_ij) const;

      //@}
      /**
       * @name Computing stencil source terms
       */
      //@{

      /** We have source terms, the Rayleigh damping of sponge layers: */
      static constexpr bool have_source_terms = true;

      /**
       * Return the Rayleigh damping source term of the sponge layers
       * \f[
       *   \begin{pmatrix}
       *   0 \\
       *   -\sigma(\textbf m - \rho\textbf v_0) \\
       *   -\sigma(\rho\theta - \rho\theta_0)
       *   \end{pmatrix},
       * \f]
       * for the state @p U and initial precomputed values @p initial
       * (holding the damping coefficient \f$\sigma\f$ and the background
       * velocity \f$\textbf v_0\f$ and potential temperature
       * \f$\theta_0\f$).
       */
      DEAL_II_HOST_DEVICE
      state_type
      rayleigh_damping(const state_type &U,
                       const initial_precomputed_type &initial) const;

      DEAL_II_HOST_DEVICE
      state_type nodal_source(const PrecomputedVectorView &pv,
                              const InitialPrecomputedVectorView &ipv,
                              const unsigned int i,
                              const state_type &U_i,
                              const ScalarNumber tau) const;

      DEAL_II_HOST_DEVICE
      state_type nodal_source(const PrecomputedVectorView &pv,
                              const InitialPrecomputedVectorView &ipv,
                              const unsigned int *js,
                              const state_type &U_j,
                              const ScalarNumber tau) const;

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
       * Given an initial state [rho, u_1, ..., u_d, theta] return a
       * conserved state [rho, m_1, ..., m_d, rho_theta].
       *
       * This function simply calls from_primitive_state() and
       * expand_state().
       *
       * @note This function is used to conveniently convert (user
       * provided) primitive initial states to a conserved state in the
       * initial states of the atmospheric Euler equations.
       */
      template <typename ST>
      DEAL_II_HOST_DEVICE state_type
      from_initial_state(const ST &initial_state) const;

      /**
       * Given a primitive state [rho, u_1, ..., u_d, theta] return a conserved
       * state
       */
      DEAL_II_HOST_DEVICE
      state_type from_primitive_state(const state_type &primitive_state) const;

      /**
       * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
       * theta]
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

      //@}
    }; /* HyperbolicSystemView */


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */

    inline HyperbolicSystem::HyperbolicSystem(const std::string &subsection)
        : Interface::HyperbolicSystem(
              "Compressible atmospheric Euler equations",
              {{{"rho", "m", "rho_theta"},
                {"rho", "m_1", "m_2", "rho_theta"},
                {"rho", "m_1", "m_2", "m_3", "rho_theta"}}},
              {{{"rho", "v", "theta"},
                {"rho", "v_1", "v_2", "theta"},
                {"rho", "v_1", "v_2", "v_3", "theta"}}},
              {{{"p", "eta_h", "rho_pow", "theta_pow"},
                {"p", "eta_h", "rho_pow", "theta_pow"},
                {"p", "eta_h", "rho_pow", "theta_pow"}}},
              {{{"x_d", "sigma", "v_0", "theta_0"},
                {"x_d", "sigma", "v_0_1", "v_0_2", "theta_0"},
                {"x_d", "sigma", "v_0_1", "v_0_2", "v_0_3", "theta_0"}}},
              subsection)
        , parameters_("euler_atmospheric_hyperbolic_system_parameters",
                      TransferPolicy::implicit_transfers_host_resident)
    {
      /* reference remains valid due to implicit_transfers_host_resident */
      auto &parameters = *parameters_.view();

      parameters.c_v = 817.;
      add_parameter("c_v", parameters.c_v, "specific heat at constant volume");

      parameters.gamma = 7. / 5.;
      add_parameter("gamma", parameters.gamma, "The ratio of specific heats");

      parameters.gravity = 9.81;
      add_parameter("gravity",
                    parameters.gravity,
                    "Gravitational constant [proper units]");

      parameters.reference_pressure = 101325;
      add_parameter("reference pressure",
                    parameters.reference_pressure,
                    "Problem specific pressure reference");

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

      sponge_damping_expression_ = "0";
      add_parameter("sponge damping",
                    sponge_damping_expression_,
                    "A function expression in x, y, z describing the "
                    "Rayleigh damping coefficient of the sponge layers "
                    "[1/time]. Set to 0 to disable the sponge layers.");

      const auto set_up_muparser = [this] {
        sponge_damping_function_ = std::make_unique<dealii::FunctionParser<3>>(
            sponge_damping_expression_);
      };

      set_up_muparser();
      ParameterAcceptor::parse_parameters_call_back.connect(set_up_muparser);

      ParameterAcceptor::parse_parameters_call_back.connect(
          [this] { update_parameters(); });

      update_parameters();
    }


    inline void HyperbolicSystem::update_parameters()
    {
      auto &parameters = *parameters_.view();

      /*
       * Precompute a number of derived gamma coefficients that contain
       * divisions, and the EOS constant that goes in front of the
       * pressure:
       */
      const auto gamma = parameters.gamma;
      const auto p_0 = parameters.reference_pressure;
      const auto gas_constant = (gamma - 1.) * parameters.c_v;
      parameters.eos_constant = p_0 * std::pow(gas_constant / p_0, gamma);
      parameters.gamma_minus_one = gamma - 1.;
      parameters.gamma_inverse = 1. / gamma;
      parameters.gamma_minus_one_inverse = 1. / (gamma - 1.);
    }


    template <int dim, typename Number>
    auto HyperbolicSystem::initial_precomputations(
        const dealii::Point<dim> &point,
        const dealii::Tensor<1, dim + 2, Number> &U) const ->
        typename View<dim, Number>::initial_precomputed_type
    {
      dealii::Point<3> padded_point;
      for (unsigned int k = 0; k < dim; ++k)
        padded_point[k] = point[k];
      const auto sigma = sponge_damping_function_->value(padded_point);

      const auto view = this->template view<dim, Number>();
      const auto rho = view.density(U);

      typename View<dim, Number>::initial_precomputed_type result;
      result[0] = Number(point[dim - 1]);
      result[1] = Number(sigma);
      for (unsigned int k = 0; k < dim; ++k)
        result[2 + k] = view.momentum(U)[k] / rho;
      result[2 + dim] = view.potential_temperature(U) / rho;
      return result;
    }


    template <typename MemorySpace, int dim, typename ScalarNumber>
    inline void HyperbolicSystem::fill_precomputed_values(
        const OfflineData<dim, ScalarNumber> &offline_data,
        StateVector<ScalarNumber> &state_vector,
        const bool skip_constrained_dofs) const
    {
      const unsigned int n_internal = offline_data.n_locally_internal();
      const unsigned int n_owned = offline_data.n_locally_owned();

      using ScalarView = HyperbolicSystemView<dim, ScalarNumber>;

      const auto sparsity_simd_view =
          offline_data.sparsity_pattern_simd().template view<MemorySpace>();

      /* We only read the hyperbolic state vector: */
      const auto U_view =
          std::as_const(state_vector)
              .state
              .template view<ScalarView::problem_dimension, MemorySpace>();
      const auto precomputed_view =
          state_vector.precomputed
              .template view<ScalarView::n_precomputed_values, MemorySpace>();

      const auto hyperbolic_system_views =
          make_select_view<dim, ScalarNumber, MemorySpace>(*this);

      const auto body = [=](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);
        using View = HyperbolicSystemView<dim, T, MemorySpace>;
        using precomputed_type = typename View::precomputed_type;

        const unsigned int row_length = sparsity_simd_view.row_length(i);
        if (skip_constrained_dofs && row_length == 1)
          return;

        const auto view = hyperbolic_system_views.template view<T>();

        const auto U_i = U_view.template read_tensor<T>(i);
        const auto rho_i = view.density(U_i);
        const auto rho_pow = ryujin::pow(rho_i, view.gamma_minus_one());
        const auto theta_pow = ryujin::pow(
            view.potential_temperature(U_i) / rho_i, view.gamma_minus_one());
        const auto p_i = view.pressure(U_i);
        const auto KE_i =
            ScalarNumber(0.5) * view.momentum(U_i).norm_square() / rho_i;
        const auto eta_i = p_i * view.gamma_minus_one_inverse() + KE_i;
        const precomputed_type prec_i{p_i, eta_i, rho_pow, theta_pow};
        precomputed_view.template write_tensor<T>(prec_i, i);
      };

      loop<MemorySpace, ScalarNumber>(
          "hyperbolic_kernel_01b", body, 0, n_internal, n_owned);
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
    HyperbolicSystemView<dim, Number, MemorySpace>::potential_temperature(
        const state_type &U)
    {
      return U[1 + dim];
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::pressure(
        const state_type &U) const
    {
      /* p = (rho*theta)^gamma */
      return eos_constant() * ryujin::pow(potential_temperature(U), gamma());
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::speed_of_sound(
        const state_type &U) const
    {
      /* c^2 = gamma * p / rho */
      const Number rho_inverse = ScalarNumber(1.) / density(U);
      const Number p = pressure(U);
      return std::sqrt(gamma() * p * rho_inverse);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::mathematical_entropy(
        const state_type &U) const
    {
      const auto KE =
          ScalarNumber(0.5) * momentum(U).norm_square() / density(U);
      const auto p = pressure(U) / gamma_minus_one();
      return p + KE;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::
        mathematical_entropy_derivative(const state_type &U) const -> state_type
    {
      /*
       * With
       *   eta = p + KE
       */
      const Number rho = density(U);
      const Number rho_inverse = ScalarNumber(1.) / rho;
      const auto v = momentum(U) * rho_inverse;
      const auto rho_theta = potential_temperature(U);

      state_type result;

      // density component
      result[0] = -ScalarNumber(0.5) * v.norm_square();

      // momentum component
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = v[i];
      }

      // rho_theta component
      result[dim + 1] =
          gamma() * gamma_minus_one_inverse() * pressure(U) / rho_theta;

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE bool
    HyperbolicSystemView<dim, Number, MemorySpace>::is_admissible(
        const state_type &U) const
    {
      const auto rho_new = density(U);
      const auto theta_new = potential_temperature(U);

      constexpr auto gt = dealii::SIMDComparison::greater_than;
      using T = Number;
      const auto test =
          ryujin::compare_and_apply_mask<gt>(rho_new, T(0.), T(0.), T(-1.)) + //
          ryujin::compare_and_apply_mask<gt>(theta_new, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout
            << "Bounds violation: Negative state [rho, theta] detected!\n";
        std::cout << "\t\trho: " << rho_new << "\n";
        std::cout << "\t\ttheta_new " << theta_new << "\n";
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

      const auto R_1 =
          component == 1 ? vn_bar - ScalarNumber(2.) * a_bar / gamma_minus_one()
                         : vn - ScalarNumber(2.) * a / gamma_minus_one();

      /* Second Riemann characteristic: v* n + 2 / (gamma() - 1) * a */

      const auto R_2 =
          component == 2 ? vn_bar + ScalarNumber(2.) * a_bar / gamma_minus_one()
                         : vn + ScalarNumber(2.) * a / gamma_minus_one();

      const auto p = pressure(U);
      const auto s = p / ryujin::pow(rho, gamma());

      const auto vperp = m / rho - vn * normal;

      const auto vn_new = ScalarNumber(0.5) * (R_1 + R_2);

      auto rho_new = ScalarNumber(1.) / (gamma() * s) *
                     ryujin::fixed_power<2>(ScalarNumber((gamma() - 1.) / 4.) *
                                            (R_2 - R_1));
      rho_new = ryujin::pow(rho_new, ScalarNumber(1.) / gamma_minus_one());

      const auto p_new = s * ryujin::pow(rho_new, gamma());

      state_type U_new;
      U_new[0] = rho_new;
      for (unsigned int d = 0; d < dim; ++d) {
        U_new[1 + d] = rho_new * (vn_new * normal + vperp)[d];
      }
      U_new[1 + dim] = ryujin::pow(p_new / eos_constant(), gamma_inverse());

      return U_new;
    }


    template <int dim, typename Number, typename MemorySpace>
    template <typename Lambda>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
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
        /* Only enforce Dirichlet conditions on the momentum: */
        auto m_dirichlet = momentum(get_dirichlet_data());
        for (unsigned int k = 0; k < dim; ++k)
          result[k + 1] = m_dirichlet[k];

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

      } else {
        Assert(false, dealii::ExcNotImplemented());
      }

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::f(const state_type &U) const
        -> flux_type
    {
      return f(U, pressure(U));
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::f(const state_type &U,
                                                      const Number &p) const
        -> flux_type
    {
      const auto rho_inverse = ScalarNumber(1.) / density(U);
      const auto m = momentum(U);
      const auto rho_theta = potential_temperature(U);

      flux_type result;

      result[0] = m;
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = m * (m[i] * rho_inverse);
        result[1 + i][i] += p;
      }
      result[dim + 1] = m * (rho_inverse * rho_theta);

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::g(const state_type &U) const
        -> flux_type
    {
      /* The star state might be a vacuum state: */
      const auto rho_inverse = safe_division(Number(1.), density(U));
      const auto m = momentum(U);
      const auto rho_theta = potential_temperature(U);

      flux_type result;

      result[0] = m;
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = m * (m[i] * rho_inverse);
      }
      result[dim + 1] = m * (rho_inverse * rho_theta);

      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    HyperbolicSystemView<dim, Number, MemorySpace>::inverse_theta_mean(
        const Number &theta_i, const Number &theta_j) const
    {
      const auto sum = theta_i + theta_j;
      const auto ratio =
          internal::arithmetic_over_logarithmic_mean(theta_i, theta_j);
      return sum / (ScalarNumber(2.) * theta_i * theta_j * ratio);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::star_state(
        const flux_contribution_type &flux,
        const Number &theta,
        const Number &theta_other,
        const Number &xd_other) const -> std::tuple<state_type, Number>
    {
      const auto &[U, xd, rho_pow, theta_pow, p] = flux;

      /*
       * If the state lies above its neighbor the reconstruction leaves it
       * unchanged. This is the case for (at least) one of the two states
       * of every pair, so we skip the expensive pow() if possible:
       */
      const auto delta_xd = std::min(Number(0.), xd - xd_other);
      if (delta_xd == Number(0.))
        return {U, p};

      /*
       * We lift the state to the height of its neighbor along the
       * hydrostatic profile with exponential potential temperature through
       * theta and theta_other (constant Brunt-Vaisala frequency):
       *   (rho theta)_star^(gamma - 1)
       *     = (rho theta)^(gamma - 1) + c_star (1 / theta)_mean delta_xd,
       * with c_star = g (gamma - 1) / (K gamma), and theta_star =
       * theta_other. We bound theta_star from below by theta so that
       * rho_star <= rho:
       */
      const auto rho_theta = potential_temperature(U);
      const auto rho_theta_pow = rho_pow * theta_pow;

      const Number c_star = (gravity() * gamma_minus_one()) /
                            (eos_constant() * gamma()) *
                            inverse_theta_mean(theta, theta_other);
      const auto factor =
          std::max(Number(0.), rho_theta_pow + c_star * delta_xd);

      /*
       * With s = factor / (rho theta)^(gamma - 1) we have for the ratio of
       * the reconstructed and the original rho theta:
       *   r = (rho theta)_star / (rho theta) = s^(1 / (gamma - 1)),
       * and for the pressure of the reconstructed state:
       *   p_star = K (r rho theta)^gamma = p r^gamma = p r s.
       */
      const Number s = factor / rho_theta_pow;
      const Number r = ryujin::pow(s, gamma_minus_one_inverse());

      /*
       * theta_star = min(max(theta, theta_other), theta / r), i.e., we cap
       * theta_star by the lift of rho theta so that theta_star - theta =
       * O(delta_xd). With r = 1 for delta_xd = 0 this leaves theta
       * unchanged. We compute theta / theta_star without dividing by r:
       */
      const auto theta_ratio =
          std::max(theta / std::max(theta, theta_other), r);

      /* Scale density and momentum by rho_star / rho: */
      state_type U_star = (r * theta_ratio) * U;
      U_star[dim + 1] = r * rho_theta;

      return {U_star, p * r * s};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::equilibrated_states(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j) const -> std::array<state_type, 2>
    {
      const auto &[U_i, xd_i, rho_pow_i, theta_pow_i, p_i] = flux_i;
      const auto &[U_j, xd_j, rho_pow_j, theta_pow_j, p_j] = flux_j;

      const auto theta_i = potential_temperature(U_i) / density(U_i);
      const auto theta_j = potential_temperature(U_j) / density(U_j);
      const auto [U_star_i, p_star_i] =
          star_state(flux_i, theta_i, theta_j, xd_j);
      const auto [U_star_j, p_star_j] =
          star_state(flux_j, theta_j, theta_i, xd_i);

      return {U_star_i, U_star_j};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::flux_contribution(
        const PrecomputedVectorView &pv,
        const InitialPrecomputedVectorView &ipv,
        const unsigned int i,
        const state_type &U_i) const -> flux_contribution_type
    {
      const auto xd_i = ipv.template read_tensor<Number>(i)[0];
      const auto &[p_i, eta_i, rho_pow_i, theta_pow_i] =
          pv.template read_tensor<Number, precomputed_type>(i);

      return {U_i, xd_i, rho_pow_i, theta_pow_i, p_i};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::flux_contribution(
        const PrecomputedVectorView &pv,
        const InitialPrecomputedVectorView &ipv,
        const unsigned int *js,
        const state_type &U_j) const -> flux_contribution_type
    {
      const auto xd_j = ipv.template read_tensor<Number>(js)[0];
      const auto &[p_j, eta_j, rho_pow_j, theta_pow_j] =
          pv.template read_tensor<Number, precomputed_type>(js);

      return {U_j, xd_j, rho_pow_j, theta_pow_j, p_j};
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::flux_divergence(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij) const -> state_type
    {
      const auto &[U_i, xd_i, rho_pow_i, theta_pow_i, p_i] = flux_i;
      const auto &[U_j, xd_j, rho_pow_j, theta_pow_j, p_j] = flux_j;

      const auto theta_i = potential_temperature(U_i) / density(U_i);
      const auto theta_j = potential_temperature(U_j) / density(U_j);
      const auto [U_star_i, p_star_i] =
          star_state(flux_i, theta_i, theta_j, xd_j);
      const auto [U_star_j, p_star_j] =
          star_state(flux_j, theta_j, theta_i, xd_i);

      const auto g_i = g(U_star_i);
      const auto g_j = g(U_star_j);

      auto result = -add(g_j, g_i);

      const auto balance_term = p_star_j - p_star_i + Number(2.) * p_i;

      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i][i] -= balance_term;
      }

      return contract(result, c_ij);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::high_order_flux_divergence(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij) const -> state_type
    {
      const auto &[U_i, xd_i, rho_pow_i, theta_pow_i, p_i] = flux_i;
      const auto &[U_j, xd_j, rho_pow_j, theta_pow_j, p_j] = flux_j;

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      using dealii::SIMDComparison;
      using ryujin::compare_and_apply_mask;

      const auto g_i = g(U_i);
      const auto g_j = g(U_j);

      auto result = -add(g_j, g_i);

      const auto rho_theta_i = potential_temperature(U_i);
      const auto rho_theta_j = potential_temperature(U_j);

      const auto theta_i = rho_theta_i / density(U_i);
      const auto theta_j = rho_theta_j / density(U_j);
      const auto theta_mean = inverse_theta_mean(theta_i, theta_j);

      /*
       * Averaged density
       *   rho_tilde = (p_j - p_i) (1 / theta)_mean
       *               / (c_tilde ((rho theta)_j^(gamma - 1)
       *                           - (rho theta)_i^(gamma - 1))),
       * with c_tilde = K gamma / (gamma - 1), and rho_tilde = (rho
       * theta)_i (1 / theta)_mean if (rho theta)_i = (rho theta)_j. This
       * ensures rho_tilde g (z_j - z_i) = -(p_j - p_i) at rest:
       */
      const auto rho_theta_pow_i = rho_pow_i * theta_pow_i;
      const auto rho_theta_pow_j = rho_pow_j * theta_pow_j;
      const auto delta = rho_theta_pow_j - rho_theta_pow_i;
      const auto tolerance = eps * (rho_theta_pow_i + rho_theta_pow_j);

      const auto c_tilde = eos_constant() * gamma() * gamma_minus_one_inverse();
      const auto safe_delta = compare_and_apply_mask<SIMDComparison::less_than>(
          std::abs(delta), tolerance, Number(1.), delta);
      const auto rho_tilde = compare_and_apply_mask<SIMDComparison::less_than>(
          std::abs(delta),
          tolerance,
          rho_theta_i * theta_mean,
          (p_j - p_i) * theta_mean / (c_tilde * safe_delta));

      const auto balance_term =
          p_i + p_j + rho_tilde * gravity() * (xd_j - xd_i);

      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i][i] -= balance_term;
      }

      return contract(result, c_ij);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::affine_shift(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij,
        const Number &d_ij) const -> state_type
    {
      const auto &[U_i, xd_i, rho_pow_i, theta_pow_i, p_i] = flux_i;
      const auto &[U_j, xd_j, rho_pow_j, theta_pow_j, p_j] = flux_j;

      const auto m_i = momentum(U_i);
      const auto rho_inverse_i = ScalarNumber(1.) / density(U_i);

      const auto theta_i = potential_temperature(U_i) * rho_inverse_i;
      const auto theta_j = potential_temperature(U_j) / density(U_j);
      const auto [U_star_i, p_star_i] =
          star_state(flux_i, theta_i, theta_j, xd_j);

      return -ScalarNumber(2.) * (d_ij + rho_inverse_i * (m_i * c_ij)) *
             (U_star_i - U_i);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::rayleigh_damping(
        const state_type &U, const initial_precomputed_type &initial) const
        -> state_type
    {
      const auto sigma = initial[1];
      const auto rho = density(U);
      const auto m = momentum(U);

      state_type result;
      for (unsigned int k = 0; k < dim; ++k)
        result[1 + k] = -sigma * (m[k] - rho * initial[2 + k]);
      result[dim + 1] =
          -sigma * (potential_temperature(U) - rho * initial[2 + dim]);
      return result;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::nodal_source(
        const PrecomputedVectorView & /*pv*/,
        const InitialPrecomputedVectorView &ipv,
        const unsigned int i,
        const state_type &U_i,
        const ScalarNumber /*tau*/) const -> state_type
    {
      return rayleigh_damping(
          U_i, ipv.template read_tensor<Number, initial_precomputed_type>(i));
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::nodal_source(
        const PrecomputedVectorView & /*pv*/,
        const InitialPrecomputedVectorView &ipv,
        const unsigned int *js,
        const state_type &U_j,
        const ScalarNumber /*tau*/) const -> state_type
    {
      return rayleigh_damping(
          U_j, ipv.template read_tensor<Number, initial_precomputed_type>(js));
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
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::from_initial_state(
        const ST &initial_state) const -> state_type
    {
      const auto primitive_state = expand_state(initial_state);
      return from_primitive_state(primitive_state);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::from_primitive_state(
        const state_type &primitive_state) const -> state_type
    {
      const auto &rho = primitive_state[0];

      auto state = primitive_state;
      /* Fix up momentum: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        state[i] *= rho;
      /* Compute potential temperature */
      state[dim + 1] = rho * primitive_state[dim + 1];

      return state;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    HyperbolicSystemView<dim, Number, MemorySpace>::to_primitive_state(
        const state_type &state) const -> state_type
    {
      const auto &rho = state[0];
      const auto rho_inverse = Number(1.) / rho;
      const auto theta = rho_inverse * potential_temperature(state);

      auto primitive_state = state;
      /* Fix up velocity: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        primitive_state[i] *= rho_inverse;
      /* Set theta: */
      primitive_state[dim + 1] = theta;

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
  } // namespace EulerAtmospheric
} // namespace ryujin
