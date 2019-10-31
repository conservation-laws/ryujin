#ifndef COMPILE_TIME_OPTIONS_H
#define COMPILE_TIME_OPTIONS_H

/*
 * General compile-time options:
 */

#ifndef DIM
// #define DIM 1
#define DIM 2
// #define DIM 3
#endif

#ifndef NUMBER
// #define NUMBER double
#define NUMBER float
#endif

// #define USE_SIMD

// #define USE_CUSTOM_POW

// #define CHECK_BOUNDS

#if defined(DEBUG) && !defined(CHECK_BOUNDS)
#define CHECK_BOUNDS
#endif

/*
 * class TimeStep:
 */

#ifndef ORDER
// #define ORDER Order::first_order
#define ORDER Order::second_order
#endif

#ifndef TIME_STEP_ORDER
// #define TIME_STEP_ORDER TimeStepOrder::first_order
#define TIME_STEP_ORDER TimeStepOrder::second_order
// #define TIME_STEP_ORDER TimeStepOrder::third_order
#endif

#ifndef LIMITER_ITER
#define LIMITER_ITER 2
#endif

/*
 * quadratic Newton:
 */

#ifndef NEWTON_EPS_DOUBLE
#define NEWTON_EPS_DOUBLE 1.e-10
#endif

#ifndef NEWTON_EPS_FLOAT
#define NEWTON_EPS_FLOAT 1.e-4
#endif

#ifndef NEWTON_MAX_ITER
#define NEWTON_MAX_ITER 1
#endif

/*
 * class RiemannSolver
 */

#ifndef RIEMANN_NEWTON_MAX_ITER
#define RIEMANN_NEWTON_MAX_ITER 1
#endif

#ifndef RIEMANN_GREEDY_DIJ
#define RIEMANN_GREEDY_DIJ false
#endif

#ifndef RIEMANN_GREEDY_THRESHOLD
#define RIEMANN_GREEDY_THRESHOLD 1.00
#endif

#ifndef RIEMANN_GREEDY_RELAX_BOUNDS
#define RIEMANN_GREEDY_RELAX_BOUNDS false
#endif

/*
 * class Indicator
 */

#ifndef INDICATOR
// #define INDICATOR Indicators::zero
// #define INDICATOR Indicators::one
// #define INDICATOR Indicators::smoothness_indicator
#define INDICATOR Indicators::entropy_viscosity_commutator
#endif

#ifndef COMPUTE_SECOND_VARIATIONS
#define COMPUTE_SECOND_VARIATIONS true
#endif

#ifndef SMOOTHNESS_INDICATOR
// #define SMOOTHNESS_INDICATOR SmoothnessIndicators::rho
// #define SMOOTHNESS_INDICATOR SmoothnessIndicators::internal_energy
#define SMOOTHNESS_INDICATOR SmoothnessIndicators::pressure
#endif

#ifndef SMOOTHNESS_INDICATOR_ALPHA_0
#define SMOOTHNESS_INDICATOR_ALPHA_0 0.
#endif

#ifndef SMOOTHNESS_INDICATOR_POWER
#define SMOOTHNESS_INDICATOR_POWER 3
#endif

/*
 * class Limiter
 */

#ifndef LIMITER
// #define LIMITER Limiters::none
// #define LIMITER Limiters::rho
#define LIMITER Limiters::specific_entropy
// #define LIMITER Limiters::entropy_inequality
#endif

#ifndef LIMITER_RELAX_BOUNDS
#define LIMITER_RELAX_BOUNDS true
#endif

#ifndef LIMITER_RELAXATION_ORDER
#define LIMITER_RELAXATION_ORDER 3
#endif

#endif /* COMPILE_TIME_OPTIONS_H */
