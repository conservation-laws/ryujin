//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>
#include <discretization.h>
#include <patterns_conversion.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <functional>

namespace ryujin
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
     * The name of the hyperbolic system.
     */
    static const std::string problem_name;

    /**
     * The dimension of the state space.
     */
    template <int dim>
    static constexpr unsigned int problem_dimension = 1 + dim;

    /**
     * The storage type used for a (conserved) state vector \f$\boldsymbol U\f$.
     */
    template <int dim, typename Number>
    using state_type = dealii::Tensor<1, problem_dimension<dim>, Number>;

    /**
     * An array holding all component names of the conserved state.
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
     * An array holding all component names of the primitive state.
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
     * The storage type used for the flux precomputations.
     */
    template <int dim, typename Number>
    using flux_contribution_type = std::tuple<state_type<dim, Number>, Number>;

    /**
     * Constructor.
     */
    HyperbolicSystem(const std::string &subsection = "HyperbolicSystem");

    /**
     * @name Precomputation of flux quantities
     */
    //@{

    /**
     * The number of precomputed initial values.
     */
    template <int dim>
    static constexpr unsigned int n_precomputed_initial_values = 1;

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
    static constexpr unsigned int n_precomputed_values = 1;

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
     * Precomputed values for a given state.
     */
    template <typename MultiComponentVector, int problem_dim, typename Number>
    void
    nodal_precomputation(MultiComponentVector &precomputed_values,
                         unsigned int i,
                         const dealii::Tensor<1, problem_dim, Number> &U) const;

    //@}
    /**
     * @name Computing derived physical quantities.
     */
    //@{

    /**
     * For a given (1+dim dimensional) state vector <code>U</code>, return
     * the water depth <code>U[0]</code>
     */
    template <int problem_dim, typename Number>
    static Number water_depth(const dealii::Tensor<1, problem_dim, Number> &U);

    /**
     * For a given (1+dim dimensional) state vector <code>U</code>, return
     * the regularized water depth <code>U[0]</code> This function returns
     * max(h, h_cutoff), where h_cutoff is the reference water depth
     * multiplied by eps.
     */
    template <int problem_dim, typename Number>
    Number
    water_depth_sharp(const dealii::Tensor<1, problem_dim, Number> &U) const;

    /**
     * For a given (1+dim dimensional) state vector <code>U</code>, return
     * a regularized inverse of the water depth. This function returns 1 /
     * max(h, h_cutoff), where h_cutoff is the reference water depth
     * multiplied by eps.
     */
    template <int problem_dim, typename Number>
    Number inverse_water_depth_mollified(
        const dealii::Tensor<1, problem_dim, Number> &U) const;

    template <int problem_dim, typename Number>
    Number inverse_water_depth_sharp(
        const dealii::Tensor<1, problem_dim, Number> &U) const;

    /**
     * Given a water depth @ref h this function returns 0 if h is in the
     * interval [-relaxation * h_cutoff, relaxation * h_cutoff], otherwise
     * h is returned unmodified. Here, h_cutoff is the reference water
     * depth multiplied by eps.
     */
    template <typename Number>
    Number filter_dry_water_depth(const Number &h) const;

    /**
     * For a given (1+dim dimensional) state vector <code>U</code>, return
     * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
     */
    template <int problem_dim, typename Number>
    static dealii::Tensor<1, problem_dim - 1, Number>
    momentum(const dealii::Tensor<1, problem_dim, Number> &U);

    /**
     * For a given (1+dim dimensional) state vector <code>U</code>, compute
     * and return the kinetic energy.
     * \f[
     *   KE = 1/2 |m|^2 / h
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    kinetic_energy(const dealii::Tensor<1, problem_dim, Number> &U) const;

    /**
     * For a given (state dimensional) state vector <code>U</code>, compute
     * and return the hydrostatic pressure \f$p\f$:
     * \f[
     *   p = 1/2 g h^2
     * \f]
     */
    template <int problem_dim, typename Number>
    Number pressure(const dealii::Tensor<1, problem_dim, Number> &U) const;

    /**
     * For a given (1+dim dimensional) state vector <code>U</code>, compute
     * the (physical) speed of sound:
     * \f[
     *   c^2 = g * h
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    speed_of_sound(const dealii::Tensor<1, problem_dim, Number> &U) const;

    /**
     * For a given (1+dim dimensional) state vector <code>U</code>, compute
     * and return the entropy \f$\eta = 1/2 g h^2 + 1/2 |m|^2 / h\f$.
     */
    template <int problem_dim, typename Number>
    Number
    mathematical_entropy(const dealii::Tensor<1, problem_dim, Number> &U) const;

    /**
     * For a given (1+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \f$\eta'\f$ of the entropy defined above.
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> mathematical_entropy_derivative(
        const dealii::Tensor<1, problem_dim, Number> &U) const;

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
     * Decomposes a given state @p U into Riemann invariants and then
     * replaces the first or second Riemann characteristic from the one
     * taken from @p U_bar state.
     */
    template <int component, int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> prescribe_riemann_characteristic(
        const dealii::Tensor<1, problem_dim, Number> &U,
        const dealii::Tensor<1, problem_dim, Number> &U_bar,
        const dealii::Tensor<1, problem_dim - 1, Number> &normal) const;

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
        const dealii::Tensor<1, problem_dim - 1, Number> &normal,
        Lambda get_dirichlet_data) const;

    //@}
    /**
     * @name Computing fluxes.
     */
    //@{

    /**
     * For a given (1+dim dimensional) state vector <code>U</code> and
     * left/right topography states <code>Z_left</code> and
     * <code>Z_right</code>, return the star_state <code>U_star</code>
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number>
    star_state(const dealii::Tensor<1, problem_dim, Number> &U,
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
    template <typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    flux_type<dim, T> f(const ST &U) const;

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
     */
    template <typename MultiComponentVector,
              typename MultiComponentVector2,
              typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    flux_contribution_type<dim, T>
    flux_contribution(const MultiComponentVector &precomputed_values,
                      const MultiComponentVector2 &precomputed_initial_values,
                      const unsigned int i,
                      const ST &U_i) const;

    template <typename MultiComponentVector,
              typename MultiComponentVector2,
              typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    flux_contribution_type<dim, T>
    flux_contribution(const MultiComponentVector &precomputed_values,
                      const MultiComponentVector2 &precomputed_initial_values,
                      const unsigned int *js,
                      const ST &U_j) const;

    /**
     * Given precomputed flux contributions @p prec_i and @p prec_j compute
     * the equilibrated, low-order flux \f$(f(U_i^{\ast,j}) +
     * f(U_j^{\ast,i})\f$
     */
    template <typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    flux_type<dim, T> flux(const std::tuple<ST, T> &prec_i,
                           const std::tuple<ST, T> &prec_j) const;

    static constexpr bool have_high_order_flux = true;

    /**
     * Given precomputed flux contributions @p prec_i and @p prec_j compute
     * the high-order flux \f$(f(U_i}) + f(U_j\f$
     */
    template <typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    flux_type<dim, T> high_order_flux(const std::tuple<ST, T> &prec_i,
                                      const std::tuple<ST, T> &prec_j) const;

    static constexpr bool have_equilibrated_states = true;

    /**
     * Given precomputed flux contributions @p prec_i and @p prec_j compute
     * the equilibrated, low-order flux \f$f(U_i^{\ast,j}) +
     * f(U_j^{\ast,i})\f$
     */
    template <typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    std::array<ST, 2>
    equilibrated_states(const std::tuple<ST, T> &prec_i,
                        const std::tuple<ST, T> &prec_j) const;

    //@}
    /**
     * @name Computing stencil source terms
     */
    //@{

    static constexpr bool have_source_terms = true;

    /**
     * FIXME
     */
    template <typename MultiComponentVector,
              typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    ST low_order_nodal_source(const MultiComponentVector &precomputed_values,
                              const unsigned int i,
                              const ST &U_i) const;

    /**
     * FIXME
     */
    template <typename MultiComponentVector,
              typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    ST high_order_nodal_source(const MultiComponentVector &precomputed_values,
                               const unsigned int i,
                               const ST &U_i) const;

    /**
     * Given precomputed flux contributions @p prec_i and @p prec_j compute
     * the equilibrated, low-order source term
     * \f$-g(H^{\ast,j}_i)^2c_ij\f$.
     */
    template <typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    ST low_order_stencil_source(const std::tuple<ST, T> &prec_i,
                                const std::tuple<ST, T> &prec_j,
                                const T &d_ij,
                                const dealii::Tensor<1, dim, T> &c_ij) const;


    /**
     * Given precomputed flux contributions @p prec_i and @p prec_j compute
     * the high-order source term \f$ g H_i Z_j c_ij\f$.
     */
    template <typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    ST high_order_stencil_source(const std::tuple<ST, T> &prec_i,
                                 const std::tuple<ST, T> &prec_j,
                                 const T &d_ij,
                                 const dealii::Tensor<1, dim, T> &c_ij) const;


    /**
     * Given precomputed flux contributions @p prec_i and @p prec_j compute
     * the equilibrated, low-order affine shift
     * \f$ B_{ij} = -2d_ij(U^{\ast,j}_i)-2f((U^{\ast,j}_i))c_ij\f$.
     */
    template <typename ST,
              int dim = ST::dimension - 1,
              typename T = typename ST::value_type>
    ST affine_shift_stencil_source(const std::tuple<ST, T> &prec_i,
                                   const std::tuple<ST, T> &prec_j,
                                   const T &d_ij,
                                   const dealii::Tensor<1, dim, T> &c_ij) const;


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
     * onto the first @ref dim2 unit directions of the @ref dim1
     * dimensional euclidean space.
     *
     * @precondition dim has to be larger or equal than dim2.
     */
    template <int dim,
              typename ST,
              typename T = typename ST::value_type,
              int dim2 = ST::dimension - 1,
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
     * Given a conserved state return a primitive state [rho, u_1, ..., u_d, p]
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> to_primitive_state(
        const dealii::Tensor<1, problem_dim, Number> &state) const;

    /*
     * Transform the current state according to a  given operator @ref
     * momentum_transform acting on  a @p dim dimensional momentum (or
     * velocity) vector.
     */
    template <int problem_dim, typename Number, typename Lambda>
    dealii::Tensor<1, problem_dim, Number>
    apply_galilei_transform(const dealii::Tensor<1, problem_dim, Number> &state,
                            const Lambda &lambda) const;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    double gravity_;
    ACCESSOR_READ_ONLY(gravity)

    double mannings_;
    ACCESSOR_READ_ONLY(mannings)

    double reference_water_depth_;
    ACCESSOR_READ_ONLY(reference_water_depth)

    double dry_state_relaxation_;
    ACCESSOR_READ_ONLY(dry_state_relaxation)

    //@}
  };


  /* Inline definitions */


  template <typename MCV, int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline void HyperbolicSystem::nodal_precomputation(
      MCV &precomputed_values,
      unsigned int i,
      const dealii::Tensor<1, problem_dim, Number> &U_i) const
  {
    constexpr int dim = problem_dim - 1;

    const precomputed_type<dim, Number> prec_i{mathematical_entropy(U_i)};
    precomputed_values.template write_tensor<Number>(prec_i, i);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  HyperbolicSystem::water_depth(const dealii::Tensor<1, problem_dim, Number> &U)
  {
    return U[0];
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  HyperbolicSystem::inverse_water_depth_mollified(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;
    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    const Number h_cutoff =
        Number(reference_water_depth_ * dry_state_relaxation_) * eps;

    const Number h = water_depth(U);
    const Number h_max = std::max(h, h_cutoff);
    const Number denom = h * h + h_max * h_max;
    return ScalarNumber(2.) * h / denom;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  HyperbolicSystem::water_depth_sharp(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;
    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    const Number h_cutoff =
        Number(reference_water_depth_ * dry_state_relaxation_) * eps;

    const Number h = water_depth(U);
    const Number h_max = std::max(h, h_cutoff);
    return h_max;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  HyperbolicSystem::inverse_water_depth_sharp(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;
    return ScalarNumber(1.) / water_depth_sharp(U);
  }


  template <typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  HyperbolicSystem::filter_dry_water_depth(const Number &h) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;
    constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();
    const Number h_cutoff_big =
        Number(reference_water_depth_ * dry_state_relaxation_) * eps;

    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
        std::abs(h), h_cutoff_big, Number(0.), h);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim - 1, Number>
  HyperbolicSystem::momentum(const dealii::Tensor<1, problem_dim, Number> &U)
  {
    constexpr int dim = problem_dim - 1;

    dealii::Tensor<1, dim, Number> result;

    for (unsigned int i = 0; i < dim; ++i)
      result[i] = U[1 + i];
    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::kinetic_energy(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* KE = 1/2 h |v|^2 */

    using ScalarNumber = typename get_value_type<Number>::type;

    const auto h = water_depth(U);
    const auto vel = momentum(U) * inverse_water_depth_sharp(U);

    return ScalarNumber(0.5) * h * vel.norm_square();
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::pressure(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* p = 1/2 g h^2 */

    using ScalarNumber = typename get_value_type<Number>::type;

    const Number h_sqd = U[0] * U[0];

    return ScalarNumber(0.5 * gravity_) * h_sqd;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::speed_of_sound(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    /* c^2 = g * h */
    return std::sqrt(ScalarNumber(gravity_) * U[0]);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::mathematical_entropy(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    const auto p = pressure(U);
    const auto k_e = kinetic_energy(U);
    return p + k_e;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::mathematical_entropy_derivative(
      const dealii::Tensor<1, problem_dim, Number> &U) const
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

    constexpr int dim = problem_dim - 1;
    using ScalarNumber = typename get_value_type<Number>::type;

    dealii::Tensor<1, problem_dim, Number> result;

    const Number &h = U[0];
    const auto vel = momentum(U) * inverse_water_depth_sharp(U);

    // water depth component
    result[0] =
        ScalarNumber(gravity_) * h - ScalarNumber(0.5) * vel.norm_square();

    // momentum components
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = vel[i];
    }

    return result;
  }


  template <int problem_dim, typename T>
  DEAL_II_ALWAYS_INLINE inline bool HyperbolicSystem::is_admissible(
      const dealii::Tensor<1, problem_dim, T> &U) const
  {
    const auto h = filter_dry_water_depth(water_depth(U));

    constexpr auto gte = dealii::SIMDComparison::greater_than_or_equal;
    const auto test =
        dealii::compare_and_apply_mask<gte>(h, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
    if (!(test == T(0.))) {
      std::cout << std::fixed << std::setprecision(16);
      std::cout << "Bounds violation: Negative state [h] detected!\n";
      std::cout << "\t\trho: " << h << "\n" << std::endl;
    }
#endif

    return (test == T(0.));
  }


  template <int component, int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::prescribe_riemann_characteristic(
      const dealii::Tensor<1, problem_dim, Number> &U,
      const dealii::Tensor<1, problem_dim, Number> &U_bar,
      const dealii::Tensor<1, problem_dim - 1, Number> &normal) const
  {

    constexpr int dim = problem_dim - 1;

    /* Note that U_bar are the dirichlet values that are prescribed */
    static_assert(component == 1 || component == 2,
                  "component has to be 1 or 2");

    const auto m = momentum(U);
    const auto a = speed_of_sound(U);
    const auto vn = m * normal * inverse_water_depth_sharp(U);

    const auto m_bar = momentum(U_bar);
    const auto a_bar = speed_of_sound(U_bar);
    const auto vn_bar = m_bar * normal * inverse_water_depth_sharp(U_bar);

    /* First Riemann characteristic: v * n - 2 * a */

    const auto R_1 = component == 1 ? vn_bar - 2. * a_bar : vn - 2. * a;

    /* Second Riemann characteristic: v * n + 2 * a */

    const auto R_2 = component == 2 ? vn_bar + 2. * a_bar : vn + 2. * a;

    const auto vperp = m * inverse_water_depth_sharp(U) - vn * normal;

    const auto vn_new = 0.5 * (R_1 + R_2);

    const auto h_new = ryujin::pow((R_2 - R_1) / 4., 2) / gravity_;

    state_type<dim, Number> U_new;

    U_new[0] = h_new;

    for (unsigned int d = 0; d < dim; ++d) {
      U_new[1 + d] = h_new * (vn_new * normal + vperp)[d];
    }

    return U_new;
  }


  template <int problem_dim, typename Number, typename Lambda>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::apply_boundary_conditions(
      dealii::types::boundary_id id,
      dealii::Tensor<1, problem_dim, Number> U,
      const dealii::Tensor<1, problem_dim - 1, Number> &normal,
      Lambda get_dirichlet_data) const
  {
    constexpr auto dim = problem_dim - 1;

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
        U = get_dirichlet_data();
      }

      /* Subsonic inflow: */
      if (vn >= -a && vn <= 0.) {
        const auto U_dirichlet = get_dirichlet_data();
        U = prescribe_riemann_characteristic<2>(U_dirichlet, U, normal);
      }

      /* Subsonic outflow: */
      if (vn > 0. && vn <= a) {
        const auto U_dirichlet = get_dirichlet_data();
        U = prescribe_riemann_characteristic<1>(U, U_dirichlet, normal);
      }

      /* Supersonic outflow: do nothing, i.e., keep U as is */
    }

    return U;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::star_state(
      const dealii::Tensor<1, problem_dim, Number> &U_left,
      const Number &Z_left,
      const Number &Z_right) const
  {
    const Number Z_max = std::max(Z_left, Z_right);
    const Number h = water_depth(U_left);
    const Number H_star = std::max(Number(0.), h + Z_left - Z_max);

    return U_left * H_star * inverse_water_depth_sharp(U_left);
  }


  template <typename ST, int dim, typename T>
  DEAL_II_ALWAYS_INLINE inline auto HyperbolicSystem::f(const ST &U) const
      -> flux_type<dim, T>
  {
    const T h_inverse = inverse_water_depth_sharp(U);
    const auto m = momentum(U);
    const auto p = pressure(U);

    flux_type<dim, T> result;

    result[0] = m;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = m * (m[i] * h_inverse);
      result[1 + i][i] += p;
    }
    return result;
  }


  template <typename MCV, typename MICV, typename ST, int dim, typename T>
  DEAL_II_ALWAYS_INLINE inline auto
  HyperbolicSystem::flux_contribution(const MCV & /*precomputed_values*/,
                                      const MICV &precomputed_initial_values,
                                      const unsigned int i,
                                      const ST &U_i) const
      -> flux_contribution_type<dim, T>
  {
    const auto &Z_i = precomputed_initial_values.template get_tensor<T>(i)[0];
    return {U_i, Z_i};
  }


  template <typename MCV, typename MICV, typename ST, int dim, typename T>
  DEAL_II_ALWAYS_INLINE inline auto
  HyperbolicSystem::flux_contribution(const MCV & /*precomputed_values*/,
                                      const MICV &precomputed_initial_values,
                                      const unsigned int *js,
                                      const ST &U_j) const
      -> flux_contribution_type<dim, T>
  {
    const auto &Z_j = precomputed_initial_values.template get_tensor<T>(js)[0];
    return {U_j, Z_j};
  }


  template <typename ST, int dim, typename T>
  DEAL_II_ALWAYS_INLINE inline auto
  HyperbolicSystem::flux(const std::tuple<ST, T> &prec_i,
                         const std::tuple<ST, T> &prec_j) const
      -> flux_type<dim, T>
  {
    const auto &[U_star_ij, U_star_ji] = equilibrated_states(prec_i, prec_j);

    const auto f_i = f(U_star_ij);
    const auto f_j = f(U_star_ji);

    return -add(f_i, f_j);
  }


  template <typename ST, int dim, typename T>
  DEAL_II_ALWAYS_INLINE inline auto
  HyperbolicSystem::high_order_flux(const std::tuple<ST, T> &prec_i,
                                    const std::tuple<ST, T> &prec_j) const
      -> flux_type<dim, T>
  {
    const auto &[U_i, Z_i] = prec_i;
    const auto &[U_j, Z_j] = prec_j;

    const auto f_i = f(U_i);
    const auto f_j = f(U_j);

    return -add(f_i, f_j);
  }


  template <typename ST, int dim, typename T>
  DEAL_II_ALWAYS_INLINE inline std::array<ST, 2>
  HyperbolicSystem::equilibrated_states(const std::tuple<ST, T> &prec_i,
                                        const std::tuple<ST, T> &prec_j) const
  {
    const auto &[U_i, Z_i] = prec_i;
    const auto &[U_j, Z_j] = prec_j;
    const auto U_star_ij = star_state(U_i, Z_i, Z_j);
    const auto U_star_ji = star_state(U_j, Z_j, Z_i);
    return {U_star_ij, U_star_ji};
  }


  template <typename MultiComponentVector, typename ST, int dim, typename T>
  ST HyperbolicSystem::low_order_nodal_source(
      const MultiComponentVector & /*precomputed_values*/,
      const unsigned int /*i*/,
      const ST & /*U_i*/) const
  {
    // FIXME
    return ST();
  }


  template <typename MultiComponentVector, typename ST, int dim, typename T>
  ST HyperbolicSystem::high_order_nodal_source(
      const MultiComponentVector & /*precomputed_values*/,
      const unsigned int /*i*/,
      const ST & /*U_i*/) const
  {
    // FIXME
    return ST();
  }


  template <typename ST, int dim, typename T>
  ST HyperbolicSystem::low_order_stencil_source(
      const std::tuple<ST, T> &prec_i,
      const std::tuple<ST, T> &prec_j,
      const T &,
      const dealii::Tensor<1, dim, T> &c_ij) const
  {
    const auto &[U_i, Z_i] = prec_i;
    const auto H_i = water_depth(U_i);
    const auto &[U_j, Z_j] = prec_j;
    const auto U_star_ij = star_state(U_i, Z_i, Z_j);
    const auto H_star_ij = water_depth(U_star_ij);

    const auto factor = gravity_ * (H_star_ij * H_star_ij - H_i * H_i);
    ST result;
    for (unsigned int d = 1; d < dim + 1; ++d)
      result[d] = factor * c_ij[d - 1];
    return result;
  }


  template <typename ST, int dim, typename T>
  ST HyperbolicSystem::high_order_stencil_source(
      const std::tuple<ST, T> &prec_i,
      const std::tuple<ST, T> &prec_j,
      const T &,
      const dealii::Tensor<1, dim, T> &c_ij) const
  {
    using ScalarNumber = typename get_value_type<T>::type;

    const auto &[U_i, Z_i] = prec_i;
    const auto &[U_j, Z_j] = prec_j;
    const auto H_i = water_depth(U_i);
    const auto H_j = water_depth(U_j);

    const auto factor =
        -gravity_ * H_i * (Z_j - Z_i) +
        ScalarNumber(0.5) * gravity_ * (H_j - H_i) * (H_j - H_i);

    ST result;
    for (unsigned int d = 1; d < dim + 1; ++d)
      result[d] = factor * c_ij[d - 1];
    return result;
  }


  template <typename ST, int dim, typename T>
  ST HyperbolicSystem::affine_shift_stencil_source(
      const std::tuple<ST, T> &prec_i,
      const std::tuple<ST, T> &prec_j,
      const T &d_ij,
      const dealii::Tensor<1, dim, T> &c_ij) const
  {
    using ScalarNumber = typename get_value_type<T>::type;

    const auto &[U_star_ij, U_star_ji] = equilibrated_states(prec_i, prec_j);
    const auto f_star_ij = f(U_star_ij);

    return -ScalarNumber(2.) * d_ij * U_star_ij -
           ScalarNumber(2.) * contract(f_star_ij, c_ij);
  }


  template <int dim, typename ST, typename T, int dim2, typename>
  auto HyperbolicSystem::expand_state(const ST &state) const
      -> state_type<dim, T>
  {
    state_type<dim, T> result;
    result[0] = state[0];
    for (unsigned int i = 1; i < dim2 + 1; ++i)
      result[i] = state[i];

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::from_primitive_state(
      const dealii::Tensor<1, problem_dim, Number> &primitive_state) const
  {
    const auto &h = primitive_state[0];

    auto state = primitive_state;
    /* Fix up momentum: */
    for (unsigned int i = 1; i < problem_dim; ++i)
      state[i] *= h;

    return state;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::to_primitive_state(
      const dealii::Tensor<1, problem_dim, Number> &state) const
  {
    const auto h_inverse = inverse_water_depth_sharp(state);

    auto primitive_state = state;
    /* Fix up velocity: */
    for (unsigned int i = 1; i < problem_dim; ++i)
      primitive_state[i] *= h_inverse;

    return primitive_state;
  }


  template <int problem_dim, typename Number, typename Lambda>
  dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::apply_galilei_transform(
      const dealii::Tensor<1, problem_dim, Number> &state,
      const Lambda &lambda) const
  {
    constexpr auto dim = problem_dim - 1;

    auto result = state;
    auto M = lambda(momentum(state));
    for (unsigned int d = 0; d < dim; ++d)
      result[1 + d] = M[d];
    return result;
  }

} /* namespace ryujin */
