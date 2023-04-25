//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "equation_of_state.h"

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
  namespace EulerAEOS
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
      static std::string problem_name;

      /**
       * The dimension of the state space.
       */
      template <int dim>
      static constexpr unsigned int problem_dimension = 2 + dim;

      /**
       * The storage type used for a (conserved) state vector \f$\boldsymbol
       * U\f$.
       */
      template <int dim, typename Number>
      using state_type = dealii::Tensor<1, problem_dimension<dim>, Number>;

      /**
       * An array holding all component names of the conserved state as a
       * string.
       */
      template <int dim>
      static const std::array<std::string, problem_dimension<dim>>
          component_names;

      /**
       * The storage type used for a primitive state vector.
       */
      template <int dim, typename Number>
      using primitive_state_type =
          dealii::Tensor<1, problem_dimension<dim>, Number>;

      /**
       * An array holding all component names of the primitive state as a
       * string.
       */
      template <int dim>
      static const std::array<std::string, problem_dimension<dim>>
          primitive_component_names;

      /**
       * The storage type used for the flux \f$\mathbf{f}\f$.
       */
      template <int dim, typename Number>
      using flux_type = dealii::
          Tensor<1, problem_dimension<dim>, dealii::Tensor<1, dim, Number>>;

      /**
       * The storage type used for flux contributions.
       */
      template <int dim, typename Number>
      using flux_contribution_type = flux_type<dim, Number>;

      /**
       * Constructor.
       */
      HyperbolicSystem(const std::string &subsection = "HyperbolicSystem");

      /**
       * Callback for ParameterAcceptor::initialize(). After we read in
       * configuration parameters from the parameter file we have to do some
       * (minor) preparatory work in this class to precompute some common
       * quantities. Do this with a callback.
       */
      void parse_parameters_callback();

      /**
       * @name Precomputed quantities
       */
      //@{

      /**
       * The number of precomputed initial values.
       */
      template <int dim>
      static constexpr unsigned int n_precomputed_initial_values = 0;

      /**
       * Array type used for precomputed initial values.
       */
      template <int dim, typename Number>
      using precomputed_initial_type =
          std::array<Number, n_precomputed_initial_values<dim>>;

      /**
       * An array holding all component names of the precomputed values.
       */
      template <int dim>
      static const std::array<std::string, n_precomputed_initial_values<dim>>
          precomputed_initial_names;

      /**
       * The number of precomputed values.
       */
      template <int dim>
      static constexpr unsigned int n_precomputed_values = 4;

      /**
       * Array type used for precomputed values.
       */
      template <int dim, typename Number>
      using precomputed_type = std::array<Number, n_precomputed_values<dim>>;

      /**
       * An array holding all component names of the precomputed values.
       */
      template <int dim>
      static const std::array<std::string, n_precomputed_values<dim>>
          precomputed_names;

      /**
       * The number of precomputation cycles.
       */
      static constexpr unsigned int n_precomputation_cycles = 2;

      /**
       * Precomputed values for a given state.
       */
      template <unsigned int cycle,
                typename Number,
                typename ScalarNumber = typename get_value_type<Number>::type,
                int problem_dim,
                typename MCV,
                typename SPARSITY>
      void
      precomputation(MCV &precomputed_values,
                     const MultiComponentVector<ScalarNumber, problem_dim> &U,
                     const SPARSITY &sparsity_simd,
                     unsigned int i) const;

      //@}
      /**
       * @name Equation of state
       */
      //@{

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, query
       * the pressure oracle and return \f$p\f$.
       */
      template <int problem_dim, typename Number>
      Number pressure(const dealii::Tensor<1, problem_dim, Number> &U) const;

      //@}
      /**
       * @name Computing derived physical quantities
       */
      //@{

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, return
       * the density <code>U[0]</code>
       */
      template <int problem_dim, typename Number>
      static Number density(const dealii::Tensor<1, problem_dim, Number> &U);

      /**
       * Given a density @ref rho this function returns 0 if rho is in the
       * interval [-relaxation * rho_cutoff, relaxation * rho_cutoff],
       * otherwise rho is returned unmodified. Here, rho_cutoff is the
       * reference density multiplied by eps.
       */
      template <typename Number>
      Number filter_vacuum_density(const Number &rho) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, return
       * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
       */
      template <int problem_dim, typename Number>
      static dealii::Tensor<1, problem_dim - 2, Number>
      momentum(const dealii::Tensor<1, problem_dim, Number> &U);

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, return
       * the total energy <code>U[1+dim]</code>
       */
      template <int problem_dim, typename Number>
      static Number
      total_energy(const dealii::Tensor<1, problem_dim, Number> &U);

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the internal energy \f$\varepsilon = (\rho e)\f$.
       */
      template <int problem_dim, typename Number>
      static Number
      internal_energy(const dealii::Tensor<1, problem_dim, Number> &U);

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the derivative of the internal energy
       * \f$\varepsilon = (\rho e)\f$.
       */
      template <int problem_dim, typename Number>
      static dealii::Tensor<1, problem_dim, Number> internal_energy_derivative(
          const dealii::Tensor<1, problem_dim, Number> &U);

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the (scaled) specific entropy
       * \f[
       *   e^{(\gamma-1)s} = \frac{\rho\,e}{\rho^\gamma}
       *   (1 - b * \rho)^(\gamma -1).
       * \f]
       */
      template <int problem_dim, typename Number>
      Number specific_entropy(const dealii::Tensor<1, problem_dim, Number> &U,
                              const Number gamma_min) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the Harten-type entropy
       * \f[
       *   \eta = (\rho^2 e * (1 - b_interp * \rho))^{1 / (\gamma + 1)}.
       * \f]
       */
      template <int problem_dim, typename Number>
      Number harten_entropy(const dealii::Tensor<1, problem_dim, Number> &U,
                            const Number gamma_min) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, compute
       * and return the derivative \f$\eta'\f$ of the Harten-type entropy
       * \f[
       *   \eta = (\rho^2 e * (1 - b_interp * \rho))^{1 / (\gamma + 1)},
       * \f]
       *
       */
      template <int problem_dim, typename Number>
      dealii::Tensor<1, problem_dim, Number>
      harten_entropy_derivative(const dealii::Tensor<1, problem_dim, Number> &U,
                                const Number &eta,
                                const Number &gamma_min) const;

      /**
       * For a given (2+dim dimensional) state vector <code>U</code> and
       * pressure <code>p</code>, compute a surrogate gamma:
       * \f[
       *   \gamma(\rho, p, e) = 1 + \frac{p * (1 - b * \rho)}{\rho * e}
       * \f]
       *
       * This function is used in various places to interpolate of the
       * pressure.
       */
      template <int problem_dim, typename Number>
      Number surrogate_gamma(const dealii::Tensor<1, problem_dim, Number> &U,
                             const Number &p) const;

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
      template <int problem_dim, typename Number>
      Number surrogate_pressure(const dealii::Tensor<1, problem_dim, Number> &U,
                                const Number &gamma) const;

      /**
       * Returns whether the state @ref U is admissible. If @ref U is a
       * vectorized state then @ref U is admissible if all vectorized values
       * are admissible.
       */
      template <int problem_dim, typename Number>
      bool is_admissible(const dealii::Tensor<1, problem_dim, Number> &U) const;

      //@}
      /**
       * @name Special functions for boundary states
       */
      //@{

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
      template <int problem_dim, typename Number, typename Lambda>
      dealii::Tensor<1, problem_dim, Number> apply_boundary_conditions(
          dealii::types::boundary_id id,
          dealii::Tensor<1, problem_dim, Number> U,
          const dealii::Tensor<1, problem_dim - 2, Number> &normal,
          Lambda get_dirichlet_data) const;

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
      template <typename ST,
                int dim = ST::dimension - 2,
                typename T = typename ST::value_type>
      flux_type<dim, T> f(const ST &U, const T &p) const;

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
      template <typename MultiComponentVector,
                typename MultiComponentVector2,
                typename ST,
                int dim = ST::dimension - 2,
                typename T = typename ST::value_type>
      flux_contribution_type<dim, T> flux_contribution(
          const MultiComponentVector &precomputed_values,
          const MultiComponentVector2 & /*precomputed_initial_values*/,
          const unsigned int i,
          const ST &U_i) const;

      template <typename MultiComponentVector,
                typename MultiComponentVector2,
                typename ST,
                int dim = ST::dimension - 2,
                typename T = typename ST::value_type>
      flux_contribution_type<dim, T> flux_contribution(
          const MultiComponentVector &precomputed_values,
          const MultiComponentVector2 & /*precomputed_initial_values*/,
          const unsigned int *js,
          const ST &U_j) const;

      /**
       * Given flux contributions @p flux_i and @p flux_j compute the flux
       * <code>(-f(U_i) - f(U_j)</code>
       */
      template <typename FT, int dim = FT::dimension - 2>
      FT flux(const FT &flux_i, const FT &flux_j) const;

      /**
       * The low-order and high-order fluxes are the same:
       */
      static constexpr bool have_high_order_flux = false;

      template <typename FT, int dim = FT::dimension - 2>
      FT high_order_flux(const FT &, const FT &) const = delete;

      /** We do not perform state equilibration */
      static constexpr bool have_equilibrated_states = false;

      template <typename FT,
                int dim = FT::dimension - 2,
                typename TT = typename FT::value_type,
                typename T = typename TT::value_type>
      std::array<state_type<dim, T>, 2>
      equilibrated_states(const FT &flux_i, const FT &flux_j) = delete;

      //@}
      /**
       * @name Computing stencil source terms
       */
      //@{

      /** We do not have source terms */
      static constexpr bool have_source_terms = false;

      template <typename MultiComponentVector, typename ST>
      ST low_order_nodal_source(const MultiComponentVector &,
                                const unsigned int,
                                const ST &) const = delete;

      template <typename MultiComponentVector, typename ST>
      ST high_order_nodal_source(const MultiComponentVector &,
                                 const unsigned int i,
                                 const ST &) const = delete;

      template <typename FT,
                int dim = FT::dimension - 2,
                typename TT = typename FT::value_type,
                typename T = typename TT::value_type>
      state_type<dim, T> low_order_stencil_source(
          const FT &,
          const FT &,
          const T,
          const dealii::Tensor<1, dim, T> &) const = delete;

      template <typename FT,
                int dim = FT::dimension - 2,
                typename TT = typename FT::value_type,
                typename T = typename TT::value_type>
      state_type<dim, T> high_order_stencil_source(
          const FT &,
          const FT &,
          const T,
          const dealii::Tensor<1, dim, T> &) const = delete;

      template <typename FT,
                int dim = FT::dimension - 2,
                typename TT = typename FT::value_type,
                typename T = typename TT::value_type>
      state_type<dim, T> affine_shift_stencil_source(
          const FT &,
          const FT &,
          const T,
          const dealii::Tensor<1, dim, T> &) const = delete;

      //@}
      /**
       * @name State transformations (primitive states, expanding
       * dimensionality, Galilei transform, etc.)
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
      template <int dim,
                typename ST,
                typename T = typename ST::value_type,
                int dim2 = ST::dimension - 2,
                typename = typename std::enable_if<(dim >= dim2)>::type>
      state_type<dim, T> expand_state(const ST &state) const;

      /*
       * Given a primitive state [rho, u_1, ..., u_d, p] return a conserved
       * state
       */
      template <int problem_dim, typename Number>
      dealii::Tensor<1, problem_dim, Number> from_primitive_state(
          const dealii::Tensor<1, problem_dim, Number> &primitive_state) const;

      /*
       * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
       * p]
       */
      template <int problem_dim, typename Number>
      dealii::Tensor<1, problem_dim, Number> to_primitive_state(
          const dealii::Tensor<1, problem_dim, Number> &state) const;

      /*
       * Transform the current state according to a  given operator @ref
       * momentum_transform acting on a @p dim dimensional momentum (or
       * velocity) vector.
       */
      template <int problem_dim, typename Number, typename Lambda>
      dealii::Tensor<1, problem_dim, Number> apply_galilei_transform(
          const dealii::Tensor<1, problem_dim, Number> &state,
          const Lambda &lambda) const;

      /*
       * Functions from the user-defined equation of state.
       */
      std::function<double(const double, const double)> pressure_;
      std::function<double(const double, const double)>
          specific_internal_energy_;
      std::function<double(const double, const double)> material_sound_speed_;


      //@}
      /**
       * @name Run time options
       */
      //@{

      ACCESSOR_READ_ONLY(equation_of_state)

      ACCESSOR_READ_ONLY(b_interp)
      ACCESSOR_READ_ONLY(pinf_interp)
      ACCESSOR_READ_ONLY(q_interp)

      ACCESSOR_READ_ONLY(reference_density)

      ACCESSOR_READ_ONLY(vacuum_state_relaxation)

      ACCESSOR_READ_ONLY(compute_expensive_bounds)

    private:
      std::string equation_of_state_;

      double b_interp_;
      double pinf_interp_;
      double q_interp_;

      double reference_density_;

      double vacuum_state_relaxation_;

      bool compute_expensive_bounds_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      std::set<std::unique_ptr<EquationOfState>> equation_of_state_list_;

      //@}
    };


    /* Inline definitions */


    template <unsigned int cycle,
              typename Number,
              typename ScalarNumber,
              int problem_dim,
              typename MCV,
              typename SPARSITY>
    DEAL_II_ALWAYS_INLINE inline void HyperbolicSystem::precomputation(
        MCV &precomputed_values,
        const MultiComponentVector<ScalarNumber, problem_dim> &U,
        const SPARSITY &sparsity_simd,
        unsigned int i) const
    {
      static_assert(cycle <= 1, "internal error");

      unsigned int stride_size = get_stride_size<Number>;
      constexpr int dim = problem_dim - 2;

      const auto U_i = U.template get_tensor<Number>(i);

      if constexpr (cycle == 0) {
        const auto p_i = pressure(U_i);
        const auto gamma_i = surrogate_gamma(U_i, p_i);
        const precomputed_type<dim, Number> prec_i{
            p_i, gamma_i, Number(0.), Number(0.)};
        precomputed_values.template write_tensor<Number>(prec_i, i);
      }

      if constexpr (cycle == 1) {
        using PT = precomputed_type<dim, Number>;

        auto prec_i = precomputed_values.template get_tensor<Number, PT>(i);
        auto &[p_i, gamma_min_i, s_i, eta_i] = prec_i;

        const unsigned int row_length = sparsity_simd.row_length(i);
        const unsigned int *js = sparsity_simd.columns(i) + stride_size;
        for (unsigned int col_idx = 1; col_idx < row_length;
             ++col_idx, js += stride_size) {

          const auto U_j = U.template get_tensor<Number>(js);
          const auto prec_j =
              precomputed_values.template get_tensor<Number, PT>(js);
          auto &[p_j, gamma_min_j, s_j, eta_j] = prec_j;
          const auto gamma_j = surrogate_gamma(U_j, p_j);
          gamma_min_i = std::min(gamma_min_i, gamma_j);
        }

        s_i = specific_entropy(U_i, gamma_min_i);
        eta_i = harten_entropy(U_i, gamma_min_i);

        precomputed_values.template write_tensor<Number>(prec_i, i);
      }
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::density(const dealii::Tensor<1, problem_dim, Number> &U)
    {
      return U[0];
    }


    template <typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::filter_vacuum_density(const Number &rho) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;
      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
      const Number rho_cutoff_big =
          Number(reference_density_ * vacuum_state_relaxation_) * eps;

      return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
          std::abs(rho), rho_cutoff_big, Number(0.), rho);
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim - 2, Number>
    HyperbolicSystem::momentum(const dealii::Tensor<1, problem_dim, Number> &U)
    {
      constexpr int dim = problem_dim - 2;

      dealii::Tensor<1, dim, Number> result;
      for (unsigned int i = 0; i < dim; ++i)
        result[i] = U[1 + i];
      return result;
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::total_energy(
        const dealii::Tensor<1, problem_dim, Number> &U)
    {
      constexpr int dim = problem_dim - 2;
      return U[1 + dim];
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::internal_energy(
        const dealii::Tensor<1, problem_dim, Number> &U)
    {
      /*
       * rho e = (E - 1/2*m^2/rho)
       */

      constexpr int dim = problem_dim - 2;
      using ScalarNumber = typename get_value_type<Number>::type;

      const Number rho_inverse = ScalarNumber(1.) / U[0];
      const auto m = momentum(U);
      const Number E = U[dim + 1];
      return E - ScalarNumber(0.5) * m.norm_square() * rho_inverse;
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
    HyperbolicSystem::internal_energy_derivative(
        const dealii::Tensor<1, problem_dim, Number> &U)
    {
      /*
       * With
       *   rho e = E - 1/2 |m|^2 / rho
       * we get
       *   (rho e)' = (1/2m^2/rho^2, -m/rho , 1 )^T
       */

      constexpr int dim = problem_dim - 2;
      using ScalarNumber = typename get_value_type<Number>::type;

      const Number rho_inverse = ScalarNumber(1.) / U[0];
      const auto u = momentum(U) * rho_inverse;

      dealii::Tensor<1, problem_dim, Number> result;

      result[0] = ScalarNumber(0.5) * u.norm_square();
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = -u[i];
      }
      result[dim + 1] = ScalarNumber(1.);

      return result;
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::pressure(
        const dealii::Tensor<1, problem_dim, Number> &U) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      if constexpr (std::is_same<ScalarNumber, Number>::value) {
        const auto rho = density(U);
        const auto e = internal_energy(U);
        return pressure_(rho, e);

      } else {
        Number result = Number(0.);
        const auto rho = density(U);
        const auto e = internal_energy(U);

        for (unsigned int k = 0; k < Number::size(); ++k) {
          result[k] = pressure_(rho[k], e[k]);
        }

        return result;
      }
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::specific_entropy(
        const dealii::Tensor<1, problem_dim, Number> &U,
        const Number gamma_min) const
    {
      /* FIXME: exp((gamma - 1)s) = (rho e) / rho ^ gamma */

      using ScalarNumber = typename get_value_type<Number>::type;

      const auto &rho = U[0];

      const auto rho_inverse = ScalarNumber(1.) / rho;
      const auto covolume = Number(1.) - Number(b_interp_) * U[0];
      return internal_energy(U) *
             ryujin::pow(rho_inverse - Number(b_interp_), gamma_min) / covolume;
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::harten_entropy(
        const dealii::Tensor<1, problem_dim, Number> &U,
        const Number gamma) const
    {
      constexpr int dim = problem_dim - 2;
      using ScalarNumber = typename get_value_type<Number>::type;

      const Number rho = U[0];
      const auto m = momentum(U);
      const Number E = U[dim + 1];

      const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

      const Number exponent = ScalarNumber(1.) / (gamma + Number(1.));

      const Number covolume = Number(1.) - Number(b_interp_) * rho;
      const Number covolume_term = ryujin::pow(covolume, gamma - Number(1.));

      return ryujin::pow(rho_rho_e * covolume_term, exponent);
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
    HyperbolicSystem::harten_entropy_derivative(
        const dealii::Tensor<1, problem_dim, Number> &U,
        const Number &eta,
        const Number &gamma) const
    {
      /*
       * With
       *   eta = (rho^2 e * (1 - b_interp * rho)) ^ {1 / (gamma + 1)},
       *   rho^2 e = rho * E - 1/2 |m|^2,
       *
       * we get
       *
       *   eta' = factor * (1 - b_interp rho) * (E,-m,rho)^T +
       *          factor * rho^2 e * (gamma - 1) * b * (1,0,0)^T
       *
       *   factor = 1/(gamma+1) * (eta/(1-b_interp rho)^-gamma
       *                        / (1-b_interp rho)^2
       */

      constexpr int dim = problem_dim - 2;
      using ScalarNumber = typename get_value_type<Number>::type;

      const Number rho = U[0];
      const auto m = momentum(U);
      const Number E = U[dim + 1];
      const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

      const Number b = Number(b_interp_);

      const Number covolume = Number(1.) - b * rho;
      const Number covolume_inverse = Number(1.) / covolume;

      const Number factor = ryujin::pow(eta * covolume_inverse, -gamma) *
                            fixed_power<2>(covolume_inverse) /
                            (gamma + Number(1.));

      dealii::Tensor<1, problem_dim, Number> result;

      result[0] =
          factor * (covolume * E - (gamma - Number(1.)) * rho_rho_e * b);
      for (unsigned int i = 0; i < dim; ++i)
        result[1 + i] = -factor * covolume * m[i];
      result[dim + 1] = factor * covolume * rho;

      return result;
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::surrogate_gamma(
        const dealii::Tensor<1, problem_dim, Number> &U, const Number &p) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const Number rho = density(U);
      const Number rho_e = internal_energy(U);
      const Number covolume = Number(1.) - Number(b_interp_) * rho;

      return Number(1.) + p * covolume / rho_e;
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::surrogate_pressure(
        const dealii::Tensor<1, problem_dim, Number> &U,
        const Number &gamma) const
    {
      const Number rho = density(U);
      const Number rho_e = internal_energy(U);
      const Number covolume = Number(1.) - Number(b_interp_) * rho;

      return (gamma - Number(1.)) * rho_e / covolume;
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline bool HyperbolicSystem::is_admissible(
        const dealii::Tensor<1, problem_dim, Number> &U) const
    {
      const auto rho_new = density(U);
      const auto e_new = internal_energy(U);

      constexpr auto gt = dealii::SIMDComparison::greater_than;
      using T = Number;
      const auto test =
          dealii::compare_and_apply_mask<gt>(rho_new, T(0.), T(0.), T(-1.)) + //
          dealii::compare_and_apply_mask<gt>(e_new, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: Negative state [rho, e] detected!\n";
        std::cout << "\t\trho: " << rho_new << "\n";
        std::cout << "\t\tint: " << e_new << "\n";
      }
#endif

      return (test == Number(0.));
    }


    template <int problem_dim, typename Number, typename Lambda>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
    HyperbolicSystem::apply_boundary_conditions(
        dealii::types::boundary_id id,
        dealii::Tensor<1, problem_dim, Number> U,
        const dealii::Tensor<1, problem_dim - 2, Number> &normal,
        Lambda get_dirichlet_data) const
    {
      constexpr auto dim = problem_dim - 2;

      if (id == Boundary::dirichlet) {
        U = get_dirichlet_data();

      } else if (id == Boundary::slip) {
        auto m = momentum(U);
        m -= 1. * (m * normal) * normal;
        for (unsigned int k = 0; k < dim; ++k)
          U[k + 1] = m[k];

      } else if (id == Boundary::no_slip) {
        for (unsigned int k = 0; k < dim; ++k)
          U[k + 1] = Number(0.);

      } else if (id == Boundary::dynamic) {
        AssertThrow(false, dealii::ExcNotImplemented());
        __builtin_trap();
      }

      return U;
    }


    template <typename ST, int dim, typename T>
    DEAL_II_ALWAYS_INLINE inline auto HyperbolicSystem::f(const ST &U,
                                                          const T &p) const
        -> flux_type<dim, T>
    {
      using ScalarNumber = typename get_value_type<T>::type;

      const T rho_inverse = ScalarNumber(1.) / U[0];
      const auto m = momentum(U);
      const T E = U[dim + 1];

      flux_type<dim, T> result;

      result[0] = m;
      for (unsigned int i = 0; i < dim; ++i) {
        result[1 + i] = m * (m[i] * rho_inverse);
        result[1 + i][i] += p;
      }
      result[dim + 1] = m * (rho_inverse * (E + p));

      return result;
    }


    template <typename MCV, typename MICV, typename ST, int dim, typename T>
    DEAL_II_ALWAYS_INLINE inline auto HyperbolicSystem::flux_contribution(
        const MCV &precomputed_values,
        const MICV & /*precomputed_initial_values*/,
        const unsigned int i,
        const ST &U_i) const -> flux_contribution_type<dim, T>
    {
      const auto &[p_i, gamma_min_i, s_i, eta_i] =
          precomputed_values.template get_tensor<T, precomputed_type<dim, T>>(
              i);
      return f(U_i, p_i);
    }


    template <typename MCV, typename MICV, typename ST, int dim, typename T>
    DEAL_II_ALWAYS_INLINE inline auto HyperbolicSystem::flux_contribution(
        const MCV &precomputed_values,
        const MICV & /*precomputed_initial_values*/,
        const unsigned int *js,
        const ST &U_j) const -> flux_contribution_type<dim, T>
    {
      const auto &[p_j, gamma_min_j, s_j, eta_j] =
          precomputed_values.template get_tensor<T, precomputed_type<dim, T>>(
              js);
      return f(U_j, p_j);
    }


    template <typename FT, int dim>
    DEAL_II_ALWAYS_INLINE inline FT
    HyperbolicSystem::flux(const FT &flux_i, const FT &flux_j) const
    {
      return -add(flux_i, flux_j);
    }


    template <int dim, typename ST, typename T, int dim2, typename>
    auto HyperbolicSystem::expand_state(const ST &state) const
        -> state_type<dim, T>
    {
      state_type<dim, T> result;
      result[0] = state[0];
      result[dim + 1] = state[dim2 + 1];
      for (unsigned int i = 1; i < dim2 + 1; ++i)
        result[i] = state[i];

      return result;
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
    HyperbolicSystem::from_primitive_state(
        const dealii::Tensor<1, problem_dim, Number> &primitive_state) const
    {
      constexpr auto dim = problem_dim - 2;

      const auto &rho = primitive_state[0];

      /* extract velocity */
      const auto u = /*SIC!*/ momentum(primitive_state);

      /* get specific internal energy: */
      const auto &e = primitive_state[dim + 1];

      auto state = primitive_state;

      /* Fix up momentum: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        state[i] *= rho;

      /* Compute total energy: */
      state[dim + 1] = rho * e + Number(0.5) * rho * u * u;

      return state;
    }


    template <int problem_dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
    HyperbolicSystem::to_primitive_state(
        const dealii::Tensor<1, problem_dim, Number> &state) const
    {
      constexpr auto dim = problem_dim - 2;

      const auto &rho = state[0];
      const auto rho_inverse = Number(1.) / rho;
      const auto rho_e = internal_energy(state);

      auto primitive_state = state;

      /* Fix up velocity: */
      for (unsigned int i = 1; i < dim + 1; ++i)
        primitive_state[i] *= rho_inverse;

      /* Set specific internal energy */
      primitive_state[dim + 1] = rho_e * rho_inverse;

      return primitive_state;
    }


    template <int problem_dim, typename Number, typename Lambda>
    dealii::Tensor<1, problem_dim, Number>
    HyperbolicSystem::apply_galilei_transform(
        const dealii::Tensor<1, problem_dim, Number> &state,
        const Lambda &lambda) const
    {
      constexpr auto dim = problem_dim - 2;

      auto result = state;
      auto M = lambda(momentum(state));
      for (unsigned int d = 0; d < dim; ++d)
        result[1 + d] = M[d];
      return result;
    }

  } // namespace EulerAEOS
} // namespace ryujin
