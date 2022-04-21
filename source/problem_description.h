//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "patterns_conversion.h"
#include "simd.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <functional>

namespace ryujin
{
  /**
   * The chosen problem type
   */
  enum class ProblemType {
    /**
     * The compressible Euler equations
     */
    euler,

    /**
     * The compressible Navier-Stokes equations
     */
    navier_stokes,
  };
}

DECLARE_ENUM(ryujin::ProblemType,
             LIST({ryujin::ProblemType::euler, "Euler"},
                  {ryujin::ProblemType::navier_stokes, "Navier Stokes"}));

namespace ryujin
{
  /**
   * Description of a @p dim dimensional hyperbolic conservation law.
   *
   * We have a (2 + dim) dimensional state space \f$[\rho, \textbf m,
   * E]\f$, where \f$\rho\f$ denotes the density, \f$\textbf m\f$ is the
   * momentum, and \f$E\f$ is the total energy.
   *
   * @ingroup EulerModule
   */
  class ProblemDescription final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * The dimension of the state space.
     */
    template <int dim>
    static constexpr unsigned int problem_dimension = 2 + dim;

    /**
     * An array holding all component names as a string.
     */
    template <int dim>
    static const std::array<std::string, dim + 2> component_names;

    /**
     * The storage type used for a state vector \f$\boldsymbol U\f$.
     */
    template <int dim, typename Number>
    using state_type = dealii::Tensor<1, problem_dimension<dim>, Number>;

    /**
     * The storage type used for the flux \f$\mathbf{f}\f$.
     */
    template <int dim, typename Number>
    using flux_type = dealii::
        Tensor<1, problem_dimension<dim>, dealii::Tensor<1, dim, Number>>;

    /**
     * An enum describing the equation of state.
     */
    enum class EquationOfState {
      /**
       * Ideal polytropic gas equation of state described by the specific
       * entropy
       * \f{align}
       *   s(\rho,e) - s_0 =
       *   \log\left(e^{1/(\gamma-1)}\,\rho^{-1}\right).
       * \f}
       */
      ideal_gas,
      /**
       * Van der Waals gas equation of state described by the specific
       * entropy
       * \f{align}
       *   s(\rho,e) - s_0 =
       *   \log\left(e^{1/(\gamma-1)}\,\left(\rho^{-1}-b\right)\right).
       * \f}
       */
      van_der_waals
    };


    /**
     * Constructor.
     */
    ProblemDescription(const std::string &subsection = "ProblemDescription");

    /**
     * Callback for ParameterAcceptor::initialize(). After we read in
     * configuration parameters from the parameter file we have to do some
     * (minor) preparatory work in this class to precompute some common
     * quantities. Do this with a callback.
     */
    void parse_parameters_callback();

    /**
     * @name ProblemDescription compile time options
     */
    //@{

    /**
     * Selected equation of state.
     *
     * @ingroup CompileTimeOptions
     */
    static constexpr EquationOfState equation_of_state_ =
        EquationOfState::ideal_gas;

    //@}
    /**
     * @name Computing derived physical quantities.
     */
    //@{

    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, return
     * the density <code>U[0]</code>
     */
    template <int problem_dim, typename Number>
    static Number density(const dealii::Tensor<1, problem_dim, Number> &U);

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
    static Number total_energy(const dealii::Tensor<1, problem_dim, Number> &U);


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
    static dealii::Tensor<1, problem_dim, Number>
    internal_energy_derivative(const dealii::Tensor<1, problem_dim, Number> &U);


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
    template <int problem_dim, typename Number>
    Number pressure(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * the (physical) speed of sound:
     * \f[
     *   c^2 = \frac{\gamma * p}{\rho\;(1 - b * \rho)}
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    speed_of_sound(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the (scaled) specific entropy
     * \f[
     *   e^{(\gamma-1)s} = \frac{\rho\,e}{\rho^\gamma}
     *   (1 - b * \rho)^(\gamma -1).
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    specific_entropy(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the Harten-type entropy
     * \f[
     *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
     * \f]
     */
    template <int problem_dim, typename Number>
    Number
    harten_entropy(const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \f$\eta'\f$ of the Harten-type entropy
     * \f[
     *   \eta = (\rho^2 e) ^ {1 / (\gamma + 1)}.
     * \f]
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> harten_entropy_derivative(
        const dealii::Tensor<1, problem_dim, Number> &U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the entropy \f$\eta = p^{1/\gamma}\f$.
     */
    template <int problem_dim, typename Number>
    Number
    mathematical_entropy(const dealii::Tensor<1, problem_dim, Number> U) const;


    /**
     * For a given (2+dim dimensional) state vector <code>U</code>, compute
     * and return the derivative \f$\eta'\f$ of the entropy \f$\eta =
     * p^{1/\gamma}\f$.
     */
    template <int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> mathematical_entropy_derivative(
        const dealii::Tensor<1, problem_dim, Number> U) const;

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
    template <int component, int problem_dim, typename Number>
    std::array<dealii::Tensor<1, problem_dim, Number>, 2>
    linearized_eigenvector(
        const dealii::Tensor<1, problem_dim, Number> &U,
        const dealii::Tensor<1, problem_dim - 2, Number> &normal) const;

    /**
     * Decomposes a given state @p U into Riemann invariants and then
     * replaces the first or second Riemann characteristic from the one
     * taken from @p U_bar state. Note that the @p U_bar state is just the
     * prescribed dirichlet values.
     */
    template <int component, int problem_dim, typename Number>
    dealii::Tensor<1, problem_dim, Number> prescribe_riemann_characteristic(
        const dealii::Tensor<1, problem_dim, Number> &U,
        const dealii::Tensor<1, problem_dim, Number> &U_bar,
        const dealii::Tensor<1, problem_dim - 2, Number> &normal) const;

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
     *   \textbf v(E+p)
     * \end{pmatrix},
     * \f]
     */
    template <int problem_dim, typename Number>
    flux_type<problem_dim - 2, Number>
    f(const dealii::Tensor<1, problem_dim, Number> &U) const;

    //@}
    /**
     * @name Transforming to and from primitive states.
     */
    //@{

    /*
     * Given a primitive 1D state [rho, u, p], compute a conserved state
     * with momentum parallel to e_1.
     */
    template <int dim, typename Number>
    state_type<dim, Number>
    from_primitive_state(const dealii::Tensor<1, 3, Number> &state_1d) const;

    //@}

  private:
    /**
     * @name Run time options
     */
    //@{

    ProblemType problem_type_;
    ACCESSOR_READ_ONLY(problem_type)

    double gamma_;
    ACCESSOR_READ_ONLY(gamma)

    double b_;
    ACCESSOR_READ_ONLY(b)

    double mu_;
    ACCESSOR_READ_ONLY(mu)

    double lambda_;
    ACCESSOR_READ_ONLY(lambda)

    double cv_inverse_kappa_;
    ACCESSOR_READ_ONLY(cv_inverse_kappa)

    //@}
    /**
     * @name Precomputed scalar quantitites
     */
    //@{
    double gamma_inverse_;
    double gamma_plus_one_inverse_;

    //@}
  };

  /* Inline definitions */

  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  ProblemDescription::density(const dealii::Tensor<1, problem_dim, Number> &U)
  {
    return U[0];
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim - 2, Number>
  ProblemDescription::momentum(const dealii::Tensor<1, problem_dim, Number> &U)
  {
    constexpr int dim = problem_dim - 2;

    dealii::Tensor<1, dim, Number> result;
    for (unsigned int i = 0; i < dim; ++i)
      result[i] = U[1 + i];
    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::total_energy(
      const dealii::Tensor<1, problem_dim, Number> &U)
  {
    constexpr int dim = problem_dim - 2;
    return U[1 + dim];
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::internal_energy(
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
  ProblemDescription::internal_energy_derivative(
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
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::pressure(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* p = (gamma - 1) / (1 - b * rho) * (rho e) */

    using ScalarNumber = typename get_value_type<Number>::type;
    return ScalarNumber(gamma_ - 1.) * internal_energy(U);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::speed_of_sound(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* c^2 = gamma * p / rho / (1 - b * rho) */

    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const Number p = pressure(U);
    return std::sqrt(gamma_ * p * rho_inverse);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::specific_entropy(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* exp((gamma - 1)s) = (rho e) / rho ^ gamma */

    using ScalarNumber = typename get_value_type<Number>::type;

    const auto rho_inverse = ScalarNumber(1.) / U[0];
    return internal_energy(U) * ryujin::pow(rho_inverse, gamma_);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::harten_entropy(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /* rho^2 e = \rho E - 1/2*m^2 */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho = U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];

    const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();
    return ryujin::pow(rho_rho_e, gamma_plus_one_inverse_);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::harten_entropy_derivative(
      const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    /*
     * With
     *   eta = (rho^2 e) ^ 1/(gamma+1)
     *   rho^2 e = rho * E - 1/2 |m|^2
     *
     * we get
     *
     *   eta' = 1/(gamma+1) * (rho^2 e) ^ -gamma/(gamma+1) * ( E , -m , rho )^T
     */

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho = U[0];
    const auto m = momentum(U);
    const Number E = U[dim + 1];

    const Number rho_rho_e = rho * E - ScalarNumber(0.5) * m.norm_square();

    const auto factor =
        gamma_plus_one_inverse_ *
        ryujin::pow(rho_rho_e, -gamma_ * gamma_plus_one_inverse_);

    dealii::Tensor<1, problem_dim, Number> result;

    result[0] = factor * E;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -factor * m[i];
    }
    result[dim + 1] = factor * rho;

    return result;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number ProblemDescription::mathematical_entropy(
      const dealii::Tensor<1, problem_dim, Number> U) const
  {
    const auto p = pressure(U);
    return ryujin::pow(p, gamma_inverse_);
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::mathematical_entropy_derivative(
      const dealii::Tensor<1, problem_dim, Number> U) const
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

    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number &rho = U[0];
    const Number rho_inverse = ScalarNumber(1.) / rho;
    const auto u = momentum(U) * rho_inverse;
    const auto p = pressure(U);

    const auto factor = (gamma_ - ScalarNumber(1.0)) * gamma_inverse_ *
                        ryujin::pow(p, gamma_inverse_ - ScalarNumber(1.));

    dealii::Tensor<1, problem_dim, Number> result;

    result[0] = factor * ScalarNumber(0.5) * u.norm_square();
    result[dim + 1] = factor;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = -factor * u[i];
    }

    return result;
  }


  template <int component, int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline std::
      array<dealii::Tensor<1, problem_dim, Number>, 2>
      ProblemDescription::linearized_eigenvector(
          const dealii::Tensor<1, problem_dim, Number> &U,
          const dealii::Tensor<1, problem_dim - 2, Number> &normal) const
  {
    static_assert(component == 1 || component == problem_dim,
                  "Only first and last eigenvectors implemented");

    constexpr int dim = problem_dim - 2;

    const auto rho = density(U);
    const auto m = momentum(U);
    const auto v = m / rho;
    const auto a = speed_of_sound(U);
    const auto gamma = this->gamma();

    state_type<dim, Number> b;
    state_type<dim, Number> c;

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

    case problem_dim:
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


  template <int component, int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, problem_dim, Number>
  ProblemDescription::prescribe_riemann_characteristic(
      const dealii::Tensor<1, problem_dim, Number> &U,
      const dealii::Tensor<1, problem_dim, Number> &U_bar,
      const dealii::Tensor<1, problem_dim - 2, Number> &normal) const
  {
    static_assert(component == 1 || component == 2,
                  "component has to be 1 or 2");

    constexpr int dim = problem_dim - 2;

    const auto m = momentum(U);
    const auto rho = density(U);
    const auto a = speed_of_sound(U);
    const auto vn = m * normal / rho;

    const auto m_bar = momentum(U_bar);
    const auto rho_bar = density(U_bar);
    const auto a_bar = speed_of_sound(U_bar);
    const auto vn_bar = m_bar * normal / rho_bar;

    /* First Riemann characteristic: v* n - 2 / (gamma - 1) * a */

    const auto R_1 = component == 1 ? vn_bar - 2. * a_bar / (gamma_ - 1.)
                                    : vn - 2. * a / (gamma_ - 1.);

    /* Second Riemann characteristic: v* n + 2 / (gamma - 1) * a */

    const auto R_2 = component == 2 ? vn_bar + 2. * a_bar / (gamma_ - 1.)
                                    : vn + 2. * a / (gamma_ - 1.);

    const auto p = pressure(U);
    const auto s = p / ryujin::pow(rho, gamma_);

    const auto vperp = m / rho - vn * normal;

    const auto vn_new = 0.5 * (R_1 + R_2);

    auto rho_new =
        1. / (gamma_ * s) * ryujin::pow((gamma_ - 1.) / 4. * (R_2 - R_1), 2);
    rho_new = ryujin::pow(rho_new, 1. / (gamma_ - 1.));

    const auto p_new = s * std::pow(rho_new, gamma_);

    state_type<dim, Number> U_new;
    U_new[0] = rho_new;
    for (unsigned int d = 0; d < dim; ++d) {
      U_new[1 + d] = rho_new * (vn_new * normal + vperp)[d];
    }
    U_new[1 + dim] = p_new / (gamma_ - 1.) +
                     0.5 * rho_new * (vn_new * vn_new + vperp.norm_square());

    return U_new;
  }


  template <int problem_dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline ProblemDescription::flux_type<problem_dim - 2,
                                                             Number>
  ProblemDescription::f(const dealii::Tensor<1, problem_dim, Number> &U) const
  {
    constexpr int dim = problem_dim - 2;
    using ScalarNumber = typename get_value_type<Number>::type;

    const Number rho_inverse = ScalarNumber(1.) / U[0];
    const auto m = momentum(U);
    const auto p = pressure(U);
    const Number E = U[dim + 1];

    flux_type<dim, Number> result;

    result[0] = m;
    for (unsigned int i = 0; i < dim; ++i) {
      result[1 + i] = m * (m[i] * rho_inverse);
      result[1 + i][i] += p;
    }
    result[dim + 1] = m * (rho_inverse * (E + p));

    return result;
  }


  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline ProblemDescription::state_type<dim, Number>
  ProblemDescription::from_primitive_state(
      const dealii::Tensor<1, 3, Number> &state_1d) const
  {
    const auto &rho = state_1d[0];
    const auto &u = state_1d[1];
    const auto &p = state_1d[2];

    state_type<dim, Number> state;

    state[0] = rho;
    state[1] = rho * u;
    state[dim + 1] = p / (Number(gamma_ - 1.)) + Number(0.5) * rho * u * u;

    return state;
  }

} /* namespace ryujin */
