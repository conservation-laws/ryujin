//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
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
     * The name of the hyperbolic system as a string.
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
     * An array holding all component names of the conserved state as a string.
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
     * An array holding all component names of the primitive state as a string.
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
     * the momentum vector <code>[U[1], ..., U[1+dim]]</code>.
     */
    template <int problem_dim, typename Number>
    static dealii::Tensor<1, problem_dim - 1, Number>
    momentum(const dealii::Tensor<1, problem_dim, Number> &U);

    /**
     * For a given (1+dim dimensional) state vector <code>U</code>, return
     * the regularized inverse of the water depth.
     * \f[
     *   1/h -> 2 h / (h^2 + max(h, h_tiny))
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    inverse_water_depth(const dealii::Tensor<1, problem_dim, Number> &U) const;

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
     * @name Functions for physical source terms
     */
    //@{

    /**
     * For given (1+dim dimensional) state vectors <code>U_i</code> and
     * <code>U_j</code> and topography states <code>Z_i</code> and
     * <code>Z_j</code> and <code>c_ij</code> vector, return the Shallow Water
     * topography source
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim - 1, Number> swe_topography_source(
        const dealii::Tensor<1, problem_dim, Number> &U_i,
        const dealii::Tensor<1, problem_dim, Number> &U_j,
        const Number &Z_i,
        const Number &Z_j,
        const dealii::Tensor<1, problem_dim - 1, Number> &c_ij) const;

    /**
     * This functions returns a (1+dim dimensional) state vector for
     * physical source terms that depend on the stencil. These functions
     * often involve the gradient of a scalar field.
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> compute_stencil_sources(
        const dealii::Tensor<1, problem_dim, Number> &U_i,
        const dealii::Tensor<1, problem_dim, Number> &U_j,
        const Number &Z_i,
        const Number &Z_j,
        const dealii::Tensor<1, problem_dim - 1, Number> &c_ij) const;

    /**
     * For a given (1+dim dimensional) state vector <code>U</code> and
     * time step <code>dt<code>, return the Glaucker-Mannings friction source
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim - 1, Number>
    friction_source(const dealii::Tensor<1, problem_dim, Number> &U,
                    const Number &tau) const;

    /**
     * This function returns a (1+dim dimensional) state vector of
     * physical source terms that are nodal based (ie do not depend on the
     * stencil).
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number>
    compute_nodal_sources(const dealii::Tensor<1, problem_dim, Number> &U_i,
                          const Number &dt) const;


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
    dealii::Tensor<1, problem_dim, Number>
    apply_boundary_conditions(dealii::types::boundary_id id,
                              dealii::Tensor<1, problem_dim, Number> U,
                              const dealii::Tensor<1, problem_dim - 1> &normal,
                              Lambda get_dirichlet_data) const;

    //@}
    /**
     * @name Computing fluxes.
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
    template <int problem_dim, typename Number>
    flux_type<problem_dim - 1, Number>
    flux(const dealii::Tensor<1, problem_dim, Number> &U) const;

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
     * @precondition dim1 has to be larger or equal than dim2.
     */
    template <int dim1,
              int prob_dim2,
              typename Number,
              typename = typename std::enable_if<(dim1 + 2 >= prob_dim2)>::type>
    state_type<dim1, Number>
    expand_state(const dealii::Tensor<1, prob_dim2, Number> &state) const;

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

    double reference_water_depth_;
    ACCESSOR_READ_ONLY(reference_water_depth)

    double dry_state_tolerance_;
    ACCESSOR_READ_ONLY(dry_state_tolerance)

    double mannings_;
    ACCESSOR_READ_ONLY(mannings)

    //@}
    /**
     * @name Precomputed scalar quantitites
     */
    //@{
    double h_tiny_;
    ACCESSOR_READ_ONLY(h_tiny)

    double gravity_inverse_;

    double g_mannings_sqd_;

    double reference_speed_;
    ACCESSOR_READ_ONLY(reference_speed)

    double h_kinetic_energy_tiny_;
    ACCESSOR_READ_ONLY(h_kinetic_energy_tiny)

    double tiny_entropy_number_;
    ACCESSOR_READ_ONLY(tiny_entropy_number)

    //@}
  };

  /* Inline definitions */

  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  HyperbolicSystem::water_depth(const dealii::Tensor<1, problem_dim, Number> &U)
  {
    return U[0];
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
  DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::inverse_water_depth(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number &h = U[0];
    const Number h_max = std::max(h, Number(h_tiny_));
    const Number denom = h * h + h_max * h_max;

    return ScalarNumber(2.) * h / denom;
  }

  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::kinetic_energy(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* KE = 1/2 h |v|^2 */

    using ScalarNumber = typename get_value_type<Number>::type;

    const auto h = water_depth(U);
    const auto vel = momentum(U) * inverse_water_depth(U);

    return ScalarNumber(0.5) * h * vel.norm_square();
  }

  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::star_state(const dealii::Tensor<1, problem_dim, Number> &U,
                               const Number &Z_left,
                               const Number &Z_right) const
  {

    dealii::Tensor<1, problem_dim, Number> local_star_state;

    const Number Z_max = std::max(Z_left, Z_right);
    const Number H_star = std::max(Number(0.), U[0] + Z_left - Z_max);

    local_star_state = U * H_star * inverse_water_depth(U);

    return local_star_state;
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
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim - 1, Number>
  HyperbolicSystem::swe_topography_source(
      const dealii::Tensor<1, problem_dim, Number> &U_i,
      const dealii::Tensor<1, problem_dim, Number> &U_j,
      const Number &Z_i,
      const Number &Z_j,
      const dealii::Tensor<1, problem_dim - 1, Number> &c_ij) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number &H_i = U_i[0];
    const Number &H_j = U_j[0];

    const Number left_term = -H_i * (Z_j - Z_i);
    const Number right_term = ScalarNumber(0.5) * (H_j - H_i) * (H_j - H_i);

    const Number source = ScalarNumber(gravity_) * (left_term + right_term);

    return source * c_ij;
  }

  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::compute_stencil_sources(
      const dealii::Tensor<1, problem_dim, Number> &U_i,
      const dealii::Tensor<1, problem_dim, Number> &U_j,
      const Number &Z_i,
      const Number &Z_j,
      const dealii::Tensor<1, problem_dim - 1, Number> &c_ij) const
  {
    constexpr int dim = problem_dim - 1;

    dealii::Tensor<1, problem_dim, Number> stencil_sources;

    /* water depth sources */
    stencil_sources[0] = Number(0.);

    /* Momentum sources */
    const auto topography_source =
        swe_topography_source(U_i, U_j, Z_i, Z_j, c_ij);

    for (unsigned int i = 0; i < dim; ++i) {
      stencil_sources[1 + i] = topography_source[i];
    }

    return stencil_sources;
  }

  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim - 1, Number>
  HyperbolicSystem::friction_source(
      const dealii::Tensor<1, problem_dim, Number> &U, const Number &tau) const
  {
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number h_max = std::max(U[0], Number(h_tiny_));
    const Number h_star = ryujin::pow(h_max, ScalarNumber(4. / 3.));

    const auto velocity = momentum(U) * inverse_water_depth(U);
    const Number velocity_norm = velocity.norm();

    const Number small_number =
        ScalarNumber(2. * g_mannings_sqd_) * tau * velocity_norm;

    const auto numerator =
        -ScalarNumber(2. * g_mannings_sqd_) * momentum(U) * velocity_norm;
    const Number denominator = h_star + std::max(h_star, small_number);

    return numerator / denominator;
  }

  /* Inline functions for source terms */
  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::compute_nodal_sources(
      const dealii::Tensor<1, problem_dim, Number> &U_i, const Number &dt) const
  {
    constexpr int dim = problem_dim - 1;

    dealii::Tensor<1, problem_dim, Number> nodal_sources;

    /* water depth sources */
    nodal_sources[0] = Number(0.);

    /* Momentum sources */
    dealii::Tensor<1, dim, Number> friction = friction_source(U_i, dt);

    for (unsigned int i = 0; i < dim; ++i) {
      nodal_sources[1 + i] = friction[i];
    }

    return nodal_sources;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number HyperbolicSystem::speed_of_sound(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* c^2 = g * h */
    using ScalarNumber = typename get_value_type<Number>::type;
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
    const auto vel = momentum(U) * inverse_water_depth(U);

    // water depth component
    result[0] =
        ScalarNumber(gravity_) * h - ScalarNumber(0.5) * vel.norm_square();

    // momentum components
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = vel[i];
    }

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline bool HyperbolicSystem::is_admissible(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    const auto h_new = water_depth(U);

    constexpr auto gt = dealii::SIMDComparison::greater_than;
    using T = Number;
    const auto test =
        dealii::compare_and_apply_mask<gt>(h_new, T(0.), T(0.), T(-1.));

#ifdef DEBUG_OUTPUT
    if (!(test == Number(0.))) {
      std::cout << std::fixed << std::setprecision(16);
      std::cout << "Bounds violation: Negative state [h] detected!\n";
      std::cout << "\t\trho: " << h_new << "\n" << std::endl;
    }
#endif

    return (test == Number(0.));
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
    const auto vn = m * normal * inverse_water_depth(U);

    const auto m_bar = momentum(U_bar);
    const auto a_bar = speed_of_sound(U_bar);
    const auto vn_bar = m_bar * normal * inverse_water_depth(U_bar);

    /* First Riemann characteristic: v * n - 2 * a */

    const auto R_1 = component == 1 ? vn_bar - 2. * a_bar : vn - 2. * a;

    /* Second Riemann characteristic: v * n + 2 * a */

    const auto R_2 = component == 2 ? vn_bar + 2. * a_bar : vn + 2. * a;

    const auto vperp = m * inverse_water_depth(U) - vn * normal;

    const auto vn_new = 0.5 * (R_1 + R_2);

    const auto h_new = gravity_inverse_ * ryujin::pow((R_2 - R_1) / 4., 2);

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
      const dealii::Tensor<1, problem_dim - 1> &normal,
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
      const auto h_inverse = inverse_water_depth(U);
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
  DEAL_II_ALWAYS_INLINE inline HyperbolicSystem::flux_type<problem_dim - 1,
                                                           Number>
  HyperbolicSystem::flux(const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    constexpr int dim = problem_dim - 1;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number h_inverse = inverse_water_depth(U);
    const auto m = momentum(U);
    const auto p = pressure(U);

    flux_type<dim, Number> result;

    result[0] = m;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = m * (m[i] * h_inverse);
      result[1 + i][i] += p;
    }
    return result;
  }


  template <int dim1, int prob_dim2, typename Number, typename>
  HyperbolicSystem::state_type<dim1, Number> HyperbolicSystem::expand_state(
      const dealii::Tensor<1, prob_dim2, Number> &state) const
  {
    constexpr auto dim2 = prob_dim2 - 1;

    state_type<dim1, Number> result;
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
    constexpr auto dim = problem_dim - 1;

    const auto &h = primitive_state[0];
    /* extract velocity: */
    const auto u = /*SIC!*/ momentum(primitive_state);

    auto state = primitive_state;
    /* Fix up momentum: */
    for (unsigned int i = 1; i < dim + 1; ++i)
      state[i] *= h;

    return state;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  HyperbolicSystem::to_primitive_state(
      const dealii::Tensor<1, problem_dim, Number> &state) const
  {
    constexpr auto dim = problem_dim - 1;

    const auto h_inverse = inverse_water_depth(state);

    auto primitive_state = state;
    /* Fix up velocity: */
    for (unsigned int i = 1; i < dim + 1; ++i)
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
