//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <gpu.h>
#include <newton.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>

#include <array>

// #define DEBUG_WAVE_SPEED_ESTIMATOR

namespace ryujin
{
  namespace EulerAEOS
  {
    /**
     * Compile time options for the NASGRiemannSolver.
     */
    struct NASGRiemannSolverOptions {
      /**
       * Take the interpolatory covolume b into account.
       * If set to false, b = 0 is assumed.
       */
      bool covolume = true;

      /**
       * Take the interpolatory reference pressure pinf into account.
       * If set to false, pinf = 0 is assumed.
       */
      bool pinf = true;

      /**
       * Guard divisions against negative numerators and vanishing
       * denominators (vacuum states). If set to false, plain divisions are
       * used instead.
       */
      bool safe_division = true;

      /**
       * Take the ratio of specific heats gamma from the Riemann data of
       * each state. If set to false, a single gamma is assumed that is
       * passed to the constructor, and all gamma dependent constants are
       * precomputed.
       */
      bool variable_gamma = true;
    };


    template <typename Number,
              NASGRiemannSolverOptions options = NASGRiemannSolverOptions{},
              typename MemorySpace = dealii::MemorySpace::Host>
    class NASGRiemannSolverView;


    /**
     * A fast approximative solver for the 1D Riemann problem with a
     * Noble-Abel stiffened gas equation of state. The solver ensures that
     * the estimate \f$\lambda_{\text{max}}\f$ that is returned for the
     * maximal wavespeed is a strict upper bound.
     *
     * The solver is based on @cite ClaytonGuermondPopov-2022.
     *
     * @ingroup EulerEquations
     */
    template <typename ScalarNumber = double,
              NASGRiemannSolverOptions options = NASGRiemannSolverOptions{}>
    class NASGRiemannSolver : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      /**
       * A structure holding all runtime parameters of the Riemann solver.
       */
      struct Parameters {
        /**
         * @name Run time parameters
         */
        //@{

        ScalarNumber covolume_b;
        ScalarNumber pinf;

        bool compute_expensive_bounds;
        double newton_tolerance;
        unsigned int newton_max_iterations;

        //@}
        /**
         * @name Cached inverses
         *
         * If options.variable_gamma is set to false, we maintain a
         * collection of commonly used expressions with gamma that would
         * otherwise need to be recomputed many times putting unnecessary
         * pressure on the div/sqrt ALU unit.
         */
        //@{

        ScalarNumber gamma;
        ScalarNumber lambda_factor;
        ScalarNumber rarefaction_exponent;
        ScalarNumber rarefaction_exponent_inverse;
        ScalarNumber half_gamma_minus_one;
        ScalarNumber c_of_gamma;

        //@}
      };

      //@}
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor.
       */
      NASGRiemannSolver(const std::string &subsection)
          : ParameterAcceptor(subsection)
          , parameters_("nasg_riemann_solver_parameters",
                        TransferPolicy::implicit_transfers_host_resident)
      {
        /* reference remains valid due to implicit_transfers_host_resident */
        auto &parameters = *parameters_.view();

        if constexpr (std::is_same<ScalarNumber, double>::value)
          parameters.newton_tolerance = 1.e-10;
        else
          parameters.newton_tolerance = 1.e-4;
        add_parameter("newton tolerance",
                      parameters.newton_tolerance,
                      "Tolerance for the quadratic newton stopping criterion");

        parameters.newton_max_iterations = 0;
        add_parameter("newton max iterations",
                      parameters.newton_max_iterations,
                      "Maximal number of quadratic newton iterations performed "
                      "during limiting");

        parameters.covolume_b = ScalarNumber(0.);
        parameters.pinf = ScalarNumber(0.);
        parameters.compute_expensive_bounds = false;

        parameters.gamma = ScalarNumber(0.);
        parameters.lambda_factor = ScalarNumber(0.);
        parameters.rarefaction_exponent = ScalarNumber(0.);
        parameters.rarefaction_exponent_inverse = ScalarNumber(0.);
        parameters.half_gamma_minus_one = ScalarNumber(0.);
        parameters.c_of_gamma = ScalarNumber(0.);

        /* invalidates view on default memory space */
        ParameterAcceptor::parse_parameters_call_back.connect(
            [this] { parameters_.view(); });
      }

      /**
       * Set the (single) ratio of specific heats @p gamma and precompute
       * all gamma dependent constants. Only available if
       * options.variable_gamma is set to false.
       */
      void set_gamma(const double gamma)
        requires(!options.variable_gamma);

      /**
       * Set the interpolatory covolume @p covolume_b, the interpolatory
       * reference pressure @p pinf, and whether to compute expensive
       * bounds.
       */
      void set_equation_of_state(const double covolume_b,
                                 const double pinf,
                                 const bool compute_expensive_bounds);

      /**
       * Return a view on the NASGRiemannSolver for a given choice of number
       * type @p Number (which can be a scalar float, or double, as well as
       * a VectorizedArray holding packed scalars). The optional
       * @p MemorySpace template parameter selects whether the view is
       * intended for the host or device memory space.
       */
      template <typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return NASGRiemannSolverView<Number, options, MemorySpace>{*this};
      }

    private:
      //@}
      /**
       * @name Internal fields, methods, and friends
       */
      //@{

      Mirrored<Parameters> parameters_;

      template <typename, NASGRiemannSolverOptions, typename>
      friend class NASGRiemannSolverView;

      //@}
    };


    /**
     * A view of the NASGRiemannSolver that makes the interface available
     * for a given choice of number type @p Number (which can be a scalar
     * float, or double, as well as a VectorizedArray holding packed
     * scalars).
     *
     * The class operates directly on Riemann data
     * \f$[\rho, u, p, \gamma, a]\f$.
     *
     * @ingroup EulerEquations
     */
    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    class NASGRiemannSolverView
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

      using ScalarNumber = typename get_value_type<Number>::type;

      using Parameters =
          typename NASGRiemannSolver<ScalarNumber, options>::Parameters;

      /**
       * Number of components in a primitive state, we store \f$[\rho, v,
       * p, gamma, a]\f$.
       */
      static constexpr unsigned int riemann_data_size = 5;

      /**
       * The array type to store the expanded primitive state for the
       * Riemann solver \f$[\rho, v, p, gamma, a]\f$
       */
      using primitive_type = std::array<Number, riemann_data_size>;

      //@}
      /**
       * @name Constructor and methods for computing wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a NASGRiemannSolver object as argument.
       */
      NASGRiemannSolverView(
          const NASGRiemannSolver<ScalarNumber, options> &riemann_solver)
          : parameters_(riemann_solver.parameters_.template view<MemorySpace>())
      {
      }

      /**
       * Return the tolerance for the quadratic Newton stopping criterion.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber newton_tolerance() const
      {
        return ScalarNumber(parameters_->newton_tolerance);
      }

      /**
       * Return the maximal number of quadratic Newton iterations.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int
      newton_max_iterations() const
      {
        return parameters_->newton_max_iterations;
      }

      /**
       * Return the interpolatory covolume b.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber covolume_b() const
      {
        return parameters_->covolume_b;
      }

      /**
       * Return the interpolatory reference pressure pinf.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE ScalarNumber pinf() const
      {
        return parameters_->pinf;
      }

      /**
       * Return whether to compute expensive bounds.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE bool compute_expensive_bounds() const
      {
        return parameters_->compute_expensive_bounds;
      }

      /**
       * For two given 1D primitive states riemann_data_i and
       * riemann_data_j, compute an upper bound of the maximum wavespeed
       * lambda.
       */
      DEAL_II_HOST_DEVICE Number
      compute(const primitive_type &riemann_data_i,
              const primitive_type &riemann_data_j) const;

      //@}
      /**
       * @name Low level primitives for the Riemann solver
       */
      //@{

      /**
       * Return the covolume \f$1 - b rho\f$
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
      one_minus_b_rho(const Number &rho) const
      {
        if constexpr (options.covolume)
          return Number(1.) - covolume_b() * rho;
        else
          return Number(1.);
      }

      /**
       * Return the shifted pressure \f$p + pinf\f$
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number shift(const Number &p) const
      {
        if constexpr (options.pinf)
          return p + pinf();
        else
          return p;
      }

      /**
       * Return the "unshifted" pressure \f$p - pinf\f$
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number unshift(const Number &p) const
      {
        if constexpr (options.pinf)
          return p - pinf();
        else
          return p;
      }

      /**
       * If options.safe_devision is enabled, return a safe division of
       * numerator / denominator.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
      safe_division(const Number &numerator, const Number &denominator) const
      {
        if constexpr (options.safe_division)
          return ryujin::safe_division(numerator, denominator);
        else
          return numerator / denominator;
      }

      /**
       * The function c(gamma) as defined in (A.3) of
       * @cite ClaytonGuermondPopov-2022, with a simplified cut-off for
       * gamma > 3.
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      template <typename T>
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE static T c(const T &gamma_Z);

      /**
       * Return gamma for the given state.
       *
       * @note If options.variable_gamma is set to false, then this
       * function return the single (scalar) gamma set via set_gamma().
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      gamma_of(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma)
          return riemann_data[3];
        else
          return parameters_->gamma;
      }


      /**
       * Return (gamma + 1) / (2 gamma) for the given state.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      lambda_factor(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma) {
          const auto &gamma = riemann_data[3];
          return ScalarNumber(0.5) * (gamma + ScalarNumber(1.)) / gamma;
        } else
          return parameters_->lambda_factor;
      }

      /**
       * Return (gamma - 1) / (2 gamma) for the given state.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      rarefaction_exponent(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma) {
          const auto &gamma = riemann_data[3];
          return ScalarNumber(0.5) * (gamma - Number(1.)) / gamma;
        } else
          return parameters_->rarefaction_exponent;
      }


      /**
       * Return 2 gamma / (gamma - 1) for the given state.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      rarefaction_exponent_inverse(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma) {
          const auto &gamma = riemann_data[3];
          return ScalarNumber(2.) * gamma / (gamma - Number(1.));
        } else
          return parameters_->rarefaction_exponent_inverse;
      }


      /**
       * Return (gamma - 1) / 2 for the given state.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      half_gamma_minus_one(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma) {
          const auto &gamma = riemann_data[3];
          return ScalarNumber(0.5) * (gamma - Number(1.));
        } else
          return parameters_->half_gamma_minus_one;
      }


      /**
       * Return c(gamma) for the given state.
       */
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
      c_of_gamma(const primitive_type &riemann_data) const
      {
        if constexpr (options.variable_gamma)
          return c(riemann_data[3]);
        else
          return parameters_->c_of_gamma;
      }


      /**
       * The factor alpha = 2 a (1 - b rho) / (gamma - 1) used in the
       * two-rarefaction and shock-shock bounds of
       * @cite ClaytonGuermondPopov-2022.
       *
       * Cost: 0x pow, 1x division, 0x sqrt
       */
      DEAL_II_HOST_DEVICE Number alpha(const Number &rho,
                                       const Number &gamma,
                                       const Number &a) const;


      /*
       * See @cite GuermondPopov2016b, page 912, (3.4), generalized to the
       * Noble-Abel stiffened gas equation of state, see
       * @cite ClaytonGuermondPopov-2022.
       *
       * Cost: 1x pow, 6x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number f(const primitive_type &riemann_data,
                                   const Number p_star) const;


      /*
       * See @cite GuermondPopov2016b, page 912, (3.4), generalized to the
       * Noble-Abel stiffened gas equation of state, see
       * @cite ClaytonGuermondPopov-2022.
       *
       * Cost: 1x pow, 6x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number df(const primitive_type &riemann_data,
                                    const Number &p_star) const;


      /*
       * See @cite GuermondPopov2016b, page 912, (3.3), generalized to the
       * Noble-Abel stiffened gas equation of state, see
       * @cite ClaytonGuermondPopov-2022.
       *
       * Cost: 2x pow, 12x division, 2x sqrt
       */
      DEAL_II_HOST_DEVICE Number phi(const primitive_type &riemann_data_i,
                                     const primitive_type &riemann_data_j,
                                     const Number p_in) const;


      /*
       * See @cite GuermondPopov2016b, page 912, (3.3), generalized to the
       * Noble-Abel stiffened gas equation of state, see
       * @cite ClaytonGuermondPopov-2022.
       *
       * Cost: 2x pow, 12x division, 2x sqrt
       */
      DEAL_II_HOST_DEVICE Number dphi(const primitive_type &riemann_data_i,
                                      const primitive_type &riemann_data_j,
                                      const Number &p) const;


      /**
       * A specialized variant of phi() that computes phi(p_max), see
       * @cite ClaytonGuermondPopov-2022
       *
       * Cost: 0x pow, 4x division, 2x sqrt
       */
      DEAL_II_HOST_DEVICE Number
      phi_of_p_max(const primitive_type &riemann_data_i,
                   const primitive_type &riemann_data_j) const;


      /**
       * Compute the best available, but expensive, upper bound on the
       * expansion-shock case as described in §5.4, Eqn. (5.7) and (5.8) in
       * @cite ClaytonGuermondPopov-2022
       *
       * Cost: 5x pow, 11x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number
      p_star_RS_full(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;


      /**
       * Compute the best available, but expensive, upper bound on the
       * shock-shock case as described in §5.5, Eqn. (5.10) and (5.12) in
       * @cite ClaytonGuermondPopov-2022
       *
       * Cost: 2x pow, 11x division, 5x sqrt (inclusive)
       */
      DEAL_II_HOST_DEVICE Number
      p_star_SS_full(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;


      /*
       * Compute only the failsafe the failsafe bound for \f$\tilde
       * p_2^\ast\f$ (5.11) in @cite ClaytonGuermondPopov-2022
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */
      DEAL_II_HOST_DEVICE Number
      p_star_failsafe(const primitive_type &riemann_data_i,
                      const primitive_type &riemann_data_j) const;


      /*
       * Compute a simultaneous upper bound on (5.7) second formula for
       * \tilde p_2^\ast (5.8) first formula for \tilde p_1^\ast (5.11)
       * formula for \tilde p_2^\ast in @cite ClaytonGuermondPopov-2022
       *
       * Cost: 3x pow, 9x division, 2x sqrt
       *
       * @todo improve documentation
       */
      DEAL_II_HOST_DEVICE Number
      p_star_interpolated(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j) const;


      /*
       * Compute an upper bound on p_star for the case of a single gamma
       * (gamma_i == gamma_j) that combines the expansion-shock bound
       * (5.7)/(5.8) and the shock-shock bound (5.10) of
       * @cite ClaytonGuermondPopov-2022.
       *
       * Cost: 2x pow, 2x division, 0x sqrt
       */
      DEAL_II_HOST_DEVICE Number
      p_star_single_gamma(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j,
                          const Number &phi_p_max) const;


      /*
       * Compute an upper bound on p_star. (In case of two expansion waves
       * the bound is only guaranteed to be less than or equal to p_min.)
       */
      DEAL_II_HOST_DEVICE Number
      p_star_upper_bound(const primitive_type &riemann_data_i,
                         const primitive_type &riemann_data_j,
                         const Number &phi_p_max) const;


      /*
       * Perform one quadratic Newton step on the bracket p_1 <= p_star <=
       * p_2 of the root of phi, see @cite GuermondPopov2016b, p. 915f
       * (4.8) and (4.9).
       *
       * Cost: 8x pow, 51x division, 10x sqrt (inclusive)
       */
      DEAL_II_HOST_DEVICE void newton_step(const primitive_type &riemann_data_i,
                                           const primitive_type &riemann_data_j,
                                           Number &p_1,
                                           Number &p_2) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.7)
       *
       * Cost: 0x pow, 2x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number lambda1_minus(
          const primitive_type &riemann_data, const Number p_star) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.8)
       *
       * Cost: 0x pow, 2x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number lambda3_plus(
          const primitive_type &primitive_state, const Number p_star) const;


      /**
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and two guesses p_1 <= p* <= p_2,
       * compute the gap in lambda between both guesses.
       *
       * See @cite GuermondPopov2016b, page 914, (4.4a), (4.4b), (4.5), and
       * (4.6)
       *
       * Cost: 0x pow, 8x division, 4x sqrt
       */
      DEAL_II_HOST_DEVICE std::array<Number, 2>
      compute_gap(const primitive_type &riemann_data_i,
                  const primitive_type &riemann_data_j,
                  const Number p_1,
                  const Number p_2) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.9)
       *
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and a guess p_2, compute an upper bound
       * for lambda.
       *
       * Cost: 0x pow, 4x division, 2x sqrt (inclusive)
       */
      DEAL_II_HOST_DEVICE Number
      compute_lambda_max(const primitive_type &riemann_data_i,
                         const primitive_type &riemann_data_j,
                         const Number p_star) const;


    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const Parameters *parameters_;

      //@}

      template <typename, NASGRiemannSolverOptions>
      friend class NASGRiemannSolver;
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <typename ScalarNumber, NASGRiemannSolverOptions options>
    inline void
    NASGRiemannSolver<ScalarNumber, options>::set_gamma(const double gamma)
      requires(!options.variable_gamma)
    {
      auto &parameters = *parameters_.view();

      parameters.gamma = ScalarNumber(gamma);
      parameters.lambda_factor = ScalarNumber(0.5 * (gamma + 1.) / gamma);
      parameters.rarefaction_exponent =
          ScalarNumber(0.5 * (gamma - 1.) / gamma);
      parameters.rarefaction_exponent_inverse =
          ScalarNumber(2. * gamma / (gamma - 1.));
      parameters.half_gamma_minus_one = ScalarNumber(0.5 * (gamma - 1.));
      parameters.c_of_gamma = ScalarNumber(
          NASGRiemannSolverView<ScalarNumber, options>::c(ScalarNumber(gamma)));
    }


    template <typename ScalarNumber, NASGRiemannSolverOptions options>
    inline void NASGRiemannSolver<ScalarNumber, options>::set_equation_of_state(
        const double covolume_b,
        const double pinf,
        const bool compute_expensive_bounds)
    {
      auto &parameters = *parameters_.view();

      parameters.covolume_b = ScalarNumber(covolume_b);
      parameters.pinf = ScalarNumber(pinf);
      parameters.compute_expensive_bounds = compute_expensive_bounds;
    }


    /*
     * The NASGRiemannSolver is a guaranteed maximal wavespeed (GMS)
     * estimate for the extended Riemann problem outlined in
     * @cite ClaytonGuermondPopov-2022. For extenstions on handling negative
     * pressures, we follow @cite clayton2023robust (see §4.6).
     *
     * In contrast to the algorithm outlined in above reference the
     * algorithm takes a couple of shortcuts to significantly decrease the
     * computational footprint. These simplifications still guarantee that
     * we have an upper bound on the maximal wavespeed - but the number
     * bound might be larger. In particular:
     *
     *  - We do not check and treat the case phi(p_min) > 0. This
     *    corresponds to two expansion waves, see §5.2 in the reference. In
     *    this case we have
     *
     *      0 < p_star < p_min <= p_max.
     *
     *    And due to the fact that p_star < p_min the wavespeeds reduce to
     *    a left wavespeed v_L - a_L and right wavespeed v_R + a_R. This
     *    implies that it is sufficient to set p_2 to ANY value provided
     *    that p_2 <= p_min hold true in order to compute the correct
     *    wavespeed.
     *
     *    If p_2 > p_min then a more pessimistic bound is computed.
     *
     *  - The (optional) quadratic Newton iteration requires a valid bracket
     *    p_1 <= p_star <= p_2, i.e., phi(p_1) <= 0 <= phi(p_2). Both, the
     *    expensive bound and the (cheaper) interpolated bound, are upper
     *    bounds of p_star; p_1 is set to p_min or p_max depending on the
     *    sign of phi(p_max).
     *
     *  - FIXME: Simplification in p_star_RS
     */

    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "rho_left: " << rho_i << std::endl;
      std::cout << "u_left: " << u_i << std::endl;
      std::cout << "p_left: " << p_i << std::endl;
      std::cout << "gamma_left: " << gamma_i << std::endl;
      std::cout << "a_left: " << a_i << std::endl;
      std::cout << "rho_right: " << rho_j << std::endl;
      std::cout << "u_right: " << u_j << std::endl;
      std::cout << "p_right: " << p_j << std::endl;
      std::cout << "gamma_right: " << gamma_j << std::endl;
      std::cout << "a_right: " << a_j << std::endl;
#endif

      const Number phi_p_max = phi_of_p_max(riemann_data_i, riemann_data_j);
      Number p_2 =
          p_star_upper_bound(riemann_data_i, riemann_data_j, phi_p_max);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "   p^*_tilde  = " << p_2 << "\n";
      std::cout << "   phi(p_*_t) = "
                << phi(riemann_data_i, riemann_data_j, p_2) << std::endl;
#endif

      /*
       * If we do no Newton iteration, cut it short:
       */

      if (newton_max_iterations() == 0) {
        const auto lambda_max =
            compute_lambda_max(riemann_data_i, riemann_data_j, p_2);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "-> lambda_max = " << lambda_max << std::endl;
#endif
        return lambda_max;
      }

      /*
       * Compute p_1 and ensure that p_1 < p_2. If we hit a case with two
       * expansions we might indeed have that p_star_tilde < p_1. Set p_1 =
       * p_2 in this case.
       */

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      Number p_1 =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max, Number(0.), p_max, p_min);

      p_1 = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::less_than_or_equal>(p_1, p_2, p_1, p_2);

      /*
       * Step 2: Perform quadratic Newton iteration.
       *
       * See @cite GuermondPopov2016b, p. 915f (4.8) and (4.9)
       */

      auto [gap, lambda_max] =
          compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << std::fixed << std::setprecision(16);
      std::cout << "p_1: (start) " << p_1 << std::endl;
      std::cout << "p_2: (start) " << p_2 << std::endl;
      std::cout << "gap: (start) " << gap << std::endl;
      std::cout << "l_m: (start) " << lambda_max << std::endl;
#endif

      for (unsigned int i = 0; i < newton_max_iterations(); ++i) {

        /* We accept our current guess if we reach the tolerance... */
        const Number tolerance(newton_tolerance());
        if (std::max(Number(0.), gap - tolerance) == Number(0.)) {
#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
          std::cout << "converged after " << i << " iterations." << std::endl;
#endif
          break;
        }

        newton_step(riemann_data_i, riemann_data_j, p_1, p_2);

        /* Update  lambda_max and gap: */
        auto [gap_new, lambda_max_new] =
            compute_gap(riemann_data_i, riemann_data_j, p_1, p_2);
        gap = gap_new;
        lambda_max = lambda_max_new;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "p_1: (  " << i << "  ) " << p_1 << std::endl;
        std::cout << "p_2: (  " << i << "  ) " << p_2 << std::endl;
        std::cout << "gap:         " << gap << std::endl;
        std::cout << "l_m:         " << lambda_max << std::endl;
#endif
      }

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "-> lambda_max = " << lambda_max << std::endl;
#endif

      return lambda_max;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    template <typename T>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE T
    NASGRiemannSolverView<Number, options, MemorySpace>::c(const T &gamma)
    {
      /*
       * We implement the continuous and monotonic function c(gamma) as
       * defined in (A.3) on page A469 of @cite ClaytonGuermondPopov-2022.
       * But with a simplified quick cut-off for the case gamma > 3:
       *
       *   c(gamma)^2 = 1                                    for gamma <= 5 / 3
       *   c(gamma)^2 = (3 * gamma + 11) / (6 * gamma + 6)   in between
       *   c(gamma)^2 = max(1/2, 5 / 6 - slope (gamma - 3))  for gamma > 3
       *
       * Due to the fact that the function is monotonic we can simply clip
       * the values without checking the conditions:
       */

      constexpr ScalarNumber slope =
          ScalarNumber(-0.34976871477801828189920753948709);

      const T first_radicand = (ScalarNumber(3.) * gamma + T(11.)) /
                               (ScalarNumber(6.) * gamma + T(6.));

      const T second_radicand = T(5. / 6.) + slope * (gamma - T(3.));

      T radicand = std::min(first_radicand, second_radicand);
      radicand = std::min(T(1.), radicand);
      radicand = std::max(T(1. / 2.), radicand);

      return std::sqrt(radicand);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::alpha(
        const Number &rho, const Number &gamma, const Number &a) const
    {
      const Number numerator = ScalarNumber(2.) * a * one_minus_b_rho(rho);

      const Number denominator = gamma - Number(1.);

      return safe_division(numerator, denominator);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::f(
        const primitive_type &riemann_data, const Number p_star) const
    {
      constexpr ScalarNumber min = std::numeric_limits<ScalarNumber>::min();

      const auto &[rho, u, p, gamma_Z, a] = riemann_data;
      const auto gamma = gamma_of(riemann_data);

      const Number one_minus_b_rho = this->one_minus_b_rho(rho);
      const Number gamma_minus_one = gamma - Number(1.);

      const Number Az =
          ScalarNumber(2.) * one_minus_b_rho / (rho * (gamma + Number(1.)));

      const Number Bz = gamma_minus_one / (gamma + Number(1.)) * shift(p);

      const Number radicand = safe_division(Az, shift(p_star) + Bz);

      /* true_value is shock case */
      const Number true_value = (p_star - p) * std::sqrt(radicand);

      const auto exponent = rarefaction_exponent(riemann_data);

      const Number ratio = safe_division(shift(p_star), shift(p));
      const Number factor = ryujin::pow(ratio, exponent) - Number(1.);

      /* false_value is rarefaction case */
      const auto false_value = ScalarNumber(2.) * a * one_minus_b_rho * factor /
                               std::max(gamma_minus_one, Number(min));

      return ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::df(
        const primitive_type &riemann_data, const Number &p_star) const
    {
      const auto &[rho, u, p, gamma_Z, a] = riemann_data;
      const auto gamma = gamma_of(riemann_data);

      const Number one_minus_b_rho = this->one_minus_b_rho(rho);

      const Number radicand_inverse =
          safe_division(ScalarNumber(0.5) * rho, one_minus_b_rho) *
          ((gamma + Number(1.)) * shift(p_star) +
           (gamma - Number(1.)) * shift(p));
      const Number denominator =
          shift(p_star) +
          ((gamma - Number(1.)) / (gamma + Number(1.)) * shift(p));

      /* true_value is shock case */
      const Number true_value =
          (denominator - ScalarNumber(0.5) * (p_star - p)) /
          (denominator * std::sqrt(radicand_inverse));

      const auto exponent = -lambda_factor(riemann_data);

      const Number ratio = safe_division(shift(p_star), shift(p));

      /*
       * false_value is rarefaction case. Note that the factor (gamma - 1)
       * of the derivative of the exponent cancels with the denominator of
       * alpha, so we do not have to divide by (gamma - 1):
       */
      const auto false_value =
          safe_division(a * one_minus_b_rho * ryujin::pow(ratio, exponent),
                        Number(gamma * shift(p)));

      return ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::phi(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_in) const
    {
      const Number &u_i = riemann_data_i[1];
      const Number &u_j = riemann_data_j[1];

      return f(riemann_data_i, p_in) + f(riemann_data_j, p_in) + u_j - u_i;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::dphi(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number &p) const
    {
      return df(riemann_data_i, p) + df(riemann_data_j, p);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::phi_of_p_max(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      /*
       * The approximate Riemann solver is based on a function phi(p) that is
       * montone increasing in p, concave down and whose (weak) third
       * derivative is non-negative and locally bounded. Because we actually
       * do not perform any iteration for computing our wavespeed estimate we
       * can get away by only implementing a specialized variant of the phi
       * function that computes phi(p_max). It inlines the implementation of
       * the "f" function and eliminates all unnecessary branches in "f".
       */

      const auto &[rho_i, u_i, p_i, gamma_Z_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_Z_j, a_j] = riemann_data_j;
      const auto gamma_i = gamma_of(riemann_data_i);
      const auto gamma_j = gamma_of(riemann_data_j);

      const Number p_max = std::max(p_i, p_j);

      const Number radicand_inverse_i =
          safe_division(ScalarNumber(0.5) * rho_i, one_minus_b_rho(rho_i)) *
          ((gamma_i + Number(1.)) * shift(p_max) +
           (gamma_i - Number(1.)) * shift(p_i));

      const Number value_i =
          safe_division(p_max - p_i, std::sqrt(radicand_inverse_i));

      const Number radicand_inverse_j =
          safe_division(ScalarNumber(0.5) * rho_j, one_minus_b_rho(rho_j)) *
          ((gamma_j + Number(1.)) * shift(p_max) +
           (gamma_j - Number(1.)) * shift(p_j));

      const Number value_j =
          safe_division(p_max - p_j, std::sqrt(radicand_inverse_j));

      return value_i + value_j + u_j - u_i;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_RS_full(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;
      const auto alpha_i = alpha(rho_i, gamma_i, a_i);
      const auto alpha_j = alpha(rho_j, gamma_j, a_j);

      /*
       * First get p_min, p_max.
       *
       * Then, we get gamma_min/max, and alpha_min/max. Note that the
       * *_min/max values are associated with p_min/max and are not
       * necessarily the minimum/maximum of *_i vs *_j.
       */

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      const Number gamma_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number alpha_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c(gamma_min) * alpha_min;

      const Number alpha_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number gamma_m = std::min(gamma_i, gamma_j);
      const Number gamma_M = std::max(gamma_i, gamma_j);

      const Number numerator =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              shift(p_max),
              Number(0.),
              Number(0.),
              positive_part(alpha_hat_min + alpha_max - (u_j - u_i)));

      /*
       * The admissible set is p_min >= pinf. But numerically let's avoid
       * division by zero and ensure positivity:
       */
      const Number p_ratio = safe_division(shift(p_min), shift(p_max));

      /*
       * Here, we use a trick: The r-factor only shows up in the formula
       * for the case \gamma_min = \gamma_m, otherwise the r-factor
       * vanishes. We can accomplish this by using the following modified
       * exponent (where we substitute gamma_m by gamma_min):
       */
      const Number r_exponent =
          (gamma_M - gamma_min) / (ScalarNumber(2.) * gamma_min * gamma_M);

      /*
       * Compute (5.7) first formula for \tilde p_1^\ast and (5.8)
       * second formula for \tilde p_2^\ast at the same time:
       */

      const Number first_exponent =
          (gamma_M - Number(1.)) / (ScalarNumber(2.) * gamma_M);

      const Number first_exponent_inverse =
          safe_division(Number(1.), first_exponent);

      const Number first_denom =
          alpha_hat_min * ryujin::pow(p_ratio, r_exponent - first_exponent) +
          alpha_max;

      const Number p_1_tilde = unshift(
          shift(p_max) * ryujin::pow(safe_division(numerator, first_denom),
                                     first_exponent_inverse));

      /*
       * Compute (5.7) second formula for \tilde p_2^\ast and (5.8) first
       * formula for \tilde p_1^\ast at the same time:
       */

      const Number second_exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);

      const Number second_exponent_inverse =
          safe_division(Number(1.), second_exponent);

      Number second_denom =
          alpha_hat_min * ryujin::pow(p_ratio, -second_exponent) +
          alpha_max * ryujin::pow(p_ratio, r_exponent);

      const Number p_2_tilde = unshift(
          shift(p_max) * ryujin::pow(safe_division(numerator, second_denom),
                                     second_exponent_inverse));

      const Number p_star = std::min(p_1_tilde, p_2_tilde);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_RS_full = " << p_star << std::endl;
#endif
      return p_star;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_SS_full(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      const Number gamma_m = std::min(gamma_i, gamma_j);

      const Number alpha_hat_i = c(gamma_i) * alpha(rho_i, gamma_i, a_i);
      const Number alpha_hat_j = c(gamma_j) * alpha(rho_j, gamma_j, a_j);

      /*
       * Compute (5.10) formula for \tilde p_1^\ast:
       *
       * Cost: 2x pow, 4x division, 0x sqrt
       */

      const Number exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exponent_inverse = Number(1.) / exponent;

      const Number numerator =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              shift(p_j),
              Number(0.),
              Number(0.),
              positive_part(alpha_hat_i + alpha_hat_j - (u_j - u_i)));

      const Number denominator =
          alpha_hat_i *
              ryujin::pow(safe_division(shift(p_i), shift(p_j)), -exponent) +
          alpha_hat_j;

      const Number p_1_tilde = unshift(
          shift(p_j) *
          ryujin::pow(safe_division(numerator, denominator), exponent_inverse));

      const auto p_2_tilde = p_star_failsafe(riemann_data_i, riemann_data_j);

      const Number p_star = std::min(p_1_tilde, p_2_tilde);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_SS_full = " << p_star << std::endl;
#endif
      return p_star;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_failsafe(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_Z_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_Z_j, a_j] = riemann_data_j;
      const auto gamma_i = gamma_of(riemann_data_i);
      const auto gamma_j = gamma_of(riemann_data_j);

      /*
       * Compute (5.11) formula for \tilde p_2^\ast:
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */

      const Number p_max = shift(std::max(p_i, p_j));

      const Number radicand_i =
          safe_division(ScalarNumber(2.) * one_minus_b_rho(rho_i) * p_max,
                        rho_i * ((gamma_i + Number(1.)) * p_max +
                                 (gamma_i - Number(1.)) * shift(p_i)));

      const Number x_i = std::sqrt(radicand_i);

      const Number radicand_j =
          safe_division(ScalarNumber(2.) * one_minus_b_rho(rho_j) * p_max,
                        rho_j * ((gamma_j + Number(1.)) * p_max +
                                 (gamma_j - Number(1.)) * shift(p_j)));

      const Number x_j = std::sqrt(radicand_j);

      const Number a = x_i + x_j;
      const Number b =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              a, Number(0.), Number(0.), u_j - u_i);

      const Number c = -shift(p_i) * x_i - shift(p_j) * x_j;

      const Number base = safe_division(
          std::abs(-b +
                   std::sqrt(positive_part(b * b - ScalarNumber(4.) * a * c))),
          std::abs(ScalarNumber(2.) * a));

      const Number p_2_tilde = unshift(base * base);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_failsafe = " << p_2_tilde << std::endl;
#endif
      return p_2_tilde;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_interpolated(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;
      const auto alpha_i = alpha(rho_i, gamma_i, a_i);
      const auto alpha_j = alpha(rho_j, gamma_j, a_j);

      /*
       * First get p_min, p_max.
       *
       * Then, we get gamma_min/max, and alpha_min/max. Note that the
       * *_min/max values are associated with p_min/max and are not
       * necessarily the minimum/maximum of *_i vs *_j.
       */

      const Number p_min = shift(std::min(p_i, p_j));
      const Number p_max = shift(std::max(p_i, p_j));

      const Number gamma_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number alpha_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c(gamma_min) * alpha_min;

      const Number gamma_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, gamma_i, gamma_j);

      const Number alpha_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_max = c(gamma_max) * alpha_max;

      const Number gamma_m = std::min(gamma_i, gamma_j);
      const Number gamma_M = std::max(gamma_i, gamma_j);

      const Number p_ratio = safe_division(p_min, p_max);

      /*
       * Here, we use a trick: The r-factor only shows up in the formula
       * for the case \gamma_min = \gamma_m, otherwise the r-factor
       * vanishes. We can accomplish this by using the following modified
       * exponent (where we substitute gamma_m by gamma_min):
       */
      const Number r_exponent =
          (gamma_M - gamma_min) / (ScalarNumber(2.) * gamma_min * gamma_M);

      /*
       * Compute a simultaneous upper bound on
       *   (5.7) second formula for \tilde p_2^\ast
       *   (5.8) first formula for \tilde p_1^\ast
       *   (5.11) formula for \tilde p_2^\ast
       */

      const Number exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exponent_inverse = Number(1.) / exponent;

      const Number numerator =
          positive_part(alpha_hat_min + /*SIC!*/ alpha_max - (u_j - u_i));

      Number denominator = alpha_hat_min * ryujin::pow(p_ratio, -exponent) +
                           alpha_hat_max * ryujin::pow(p_ratio, r_exponent);

      const auto temp = safe_division(numerator, denominator);

      const Number p_tilde =
          unshift(p_max * ryujin::pow(temp, exponent_inverse));

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_interpolated = " << p_tilde << std::endl;
#endif
      return p_tilde;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_single_gamma(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number &phi_p_max) const
    {
      /*
       * For a single gamma the expansion-shock bound (5.7)/(5.8) and the
       * shock-shock bound (5.10) of @cite ClaytonGuermondPopov-2022 reduce
       * to
       *
       *   p_max * (N / D)^{1/e},  e = (gamma - 1) / (2 gamma),
       *   N = alpha_hat_min + X - (u_j - u_i),
       *   D = alpha_hat_min (p_min / p_max)^{-e} + X,
       *
       * with X = alpha_hat_max for phi(p_max) < 0 (5.10), and X = alpha_max
       * otherwise (5.7)/(5.8).
       */

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      /* We have gamma_i == gamma_j: */
      const auto c_gamma = c_of_gamma(riemann_data_i);

      /*
       * alpha_Z = 2 a_Z (1 - b rho_Z) / (gamma - 1). We drop the common
       * factor 2 / (gamma - 1) and rescale (u_j - u_i) accordingly:
       */
      const Number alpha_i = a_i * one_minus_b_rho(rho_i);
      const Number alpha_j = a_j * one_minus_b_rho(rho_j);

      const Number p_min = shift(std::min(p_i, p_j));
      const Number p_max = shift(std::max(p_i, p_j));

      const Number alpha_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c_gamma * alpha_min;

      /*
       * The shock-shock bound (5.10) uses alpha_hat_max, the
       * expansion-shock bound (5.7)/(5.8) uses alpha_max:
       */
      const Number alpha_select =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max, Number(0.), c_gamma * alpha_max, alpha_max);

      const auto exponent = rarefaction_exponent(riemann_data_i);
      const auto exponent_inverse =
          rarefaction_exponent_inverse(riemann_data_i);

      const Number numerator =
          positive_part(alpha_hat_min + alpha_select -
                        half_gamma_minus_one(riemann_data_i) * (u_j - u_i));

      const Number denominator =
          alpha_hat_min * ryujin::pow(safe_division(p_min, p_max), -exponent) +
          alpha_select;

      const Number p_tilde =
          unshift(p_max * ryujin::pow(safe_division(numerator, denominator),
                                      exponent_inverse));

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_single_gamma = " << p_tilde << std::endl;
#endif
      return p_tilde;
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::lambda1_minus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor = lambda_factor(riemann_data);

      const Number p_inverse = safe_division(Number(1.), shift(p));
      const Number tmp = positive_part(p_star - p) * p_inverse;

      return u - a * std::sqrt(Number(1.) + factor * tmp);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::lambda3_plus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor = lambda_factor(riemann_data);

      const Number p_inverse = safe_division(Number(1.), shift(p));
      const Number tmp = positive_part(p_star - p) * p_inverse;

      return u + a * std::sqrt(Number(1.) + factor * tmp);
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE std::array<Number, 2>
    NASGRiemannSolverView<Number, options, MemorySpace>::compute_gap(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_1,
        const Number p_2) const
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_2 /*SIC!*/);
      const Number nu_12 = lambda1_minus(riemann_data_i, p_1 /*SIC!*/);

      const Number nu_31 = lambda3_plus(riemann_data_j, p_1);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_2);

      const Number lambda_max =
          std::max(positive_part(nu_32), negative_part(nu_11));

      const Number gap =
          std::max(std::abs(nu_32 - nu_31), std::abs(nu_12 - nu_11));

      return {{gap, lambda_max}};
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::compute_lambda_max(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_star) const
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_star);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_star);

      return std::max(positive_part(nu_32), negative_part(nu_11));
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    NASGRiemannSolverView<Number, options, MemorySpace>::p_star_upper_bound(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number &phi_p_max) const
    {
      /*
       * Depending on the compile time options and on
       * compute_expensive_bounds() we use the single gamma bound, the
       * interpolated bound, or the expensive bounds, each combined with
       * the failsafe bound or p_max.
       */

      const Number &p_i = riemann_data_i[2];
      const Number &p_j = riemann_data_j[2];

      const Number p_max = std::max(p_i, p_j);

      if constexpr (!options.variable_gamma) {
        /*
         * For a single gamma the expensive bounds (5.7), (5.8), and (5.10)
         * reduce to a single formula of the same cost as the interpolated
         * bound:
         */
        const Number p_star_tilde =
            p_star_single_gamma(riemann_data_i, riemann_data_j, phi_p_max);
        const Number p_star_backup =
            p_star_failsafe(riemann_data_i, riemann_data_j);

        return ryujin::compare_and_apply_mask<
            dealii::SIMDComparison::less_than>(
            phi_p_max,
            Number(0.),
            std::min(p_star_tilde, p_star_backup),
            std::min(p_max, p_star_tilde));

      } else if (!compute_expensive_bounds()) {
#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        const Number p_star_RS = p_star_RS_full(riemann_data_i, riemann_data_j);
        const Number p_star_SS = p_star_SS_full(riemann_data_i, riemann_data_j);
        const Number p_strict =
            ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                phi_p_max, Number(0.), p_star_SS, std::min(p_max, p_star_RS));
        std::cout << "   p^*_strict = " << p_strict << "\n";
        std::cout << "   phi(p_*_s) = "
                  << phi(riemann_data_i, riemann_data_j, p_strict) << "\n";
        std::cout << "-> lambda_str = "
                  << compute_lambda_max(
                         riemann_data_i, riemann_data_j, p_strict)
                  << std::endl;
#endif

        const Number p_star_tilde =
            p_star_interpolated(riemann_data_i, riemann_data_j);
        const Number p_star_backup =
            p_star_failsafe(riemann_data_i, riemann_data_j);

        return ryujin::compare_and_apply_mask<
            dealii::SIMDComparison::less_than>(
            phi_p_max,
            Number(0.),
            std::min(p_star_tilde, p_star_backup),
            std::min(p_max, p_star_tilde));

      } else {

        const Number p_star_RS = p_star_RS_full(riemann_data_i, riemann_data_j);
        const Number p_star_SS = p_star_SS_full(riemann_data_i, riemann_data_j);

        return ryujin::compare_and_apply_mask<
            dealii::SIMDComparison::less_than>(
            phi_p_max, Number(0.), p_star_SS, std::min(p_max, p_star_RS));
      }
    }


    template <typename Number,
              NASGRiemannSolverOptions options,
              typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    NASGRiemannSolverView<Number, options, MemorySpace>::newton_step(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        Number &p_1,
        Number &p_2) const
    {
      // FIXME: Fuse these computations:
      const Number phi_p_1 = phi(riemann_data_i, riemann_data_j, p_1);
      const Number phi_p_2 = phi(riemann_data_i, riemann_data_j, p_2);
      const Number dphi_p_1 = dphi(riemann_data_i, riemann_data_j, p_1);
      const Number dphi_p_2 = dphi(riemann_data_i, riemann_data_j, p_2);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "phi_p_1:     " << phi_p_1 << std::endl;
      std::cout << "phi_p_2:     " << phi_p_2 << std::endl;
      std::cout << "dphi_p_1:    " << dphi_p_1 << std::endl;
      std::cout << "dphi_p_2:    " << dphi_p_2 << std::endl;
#endif

      ryujin::quadratic_newton_step(
          p_1, p_2, phi_p_1, phi_p_2, dphi_p_1, dphi_p_2);
    }

  } // namespace EulerAEOS
} // namespace ryujin
