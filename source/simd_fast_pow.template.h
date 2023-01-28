//
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2014 - 2022 by Agner Fog
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include "simd.h"

#include <cmath>

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
#define VCL_NAMESPACE vcl
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include "../simd-math/vectorclass.h"
#include "../simd-math/vectormath_exp.h"
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS
#undef VCL_NAMESPACE

#else
namespace ryujin
{
  template <typename T>
  T fast_pow_impl(const T x, const T b, const Bias)
  {
    return std::pow(x, b);
  }
} // namespace ryujin
#endif


#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
namespace ryujin
{
  template <typename VTYPE>
  inline DEAL_II_ALWAYS_INLINE VTYPE fast_pow_impl(VTYPE const x0,
                                                   VTYPE const y,
                                                   Bias)
  {
    /* clang-format off */
    using namespace vcl;

    const float ln2f_hi  =  0.693359375f;        // log(2), split in two for extended precision
    const auto log2e = static_cast<float>(VM_LOG2E); // 1/log(2)

    const float P0logf  =  3.3333331174E-1f;     // coefficients for logarithm expansion
    const float P1logf  = -2.4999993993E-1f;
    const float P2logf  =  2.0000714765E-1f;
    const float P3logf  = -1.6668057665E-1f;
    const float P4logf  =  1.4249322787E-1f;
    const float P5logf  = -1.2420140846E-1f;
    const float P6logf  =  1.1676998740E-1f;
    const float P7logf  = -1.1514610310E-1f;
    const float P8logf  =  7.0376836292E-2f;

    const float p2expf   =  1.f/2.f;             // coefficients for Taylor expansion of exp
    const float p3expf   =  1.f/6.f;
    const float p4expf   =  1.f/24.f;
    const float p5expf   =  1.f/120.f;
    const float p6expf   =  1.f/720.f;
    const float p7expf   =  1.f/5040.f;

    typedef decltype(roundi(x0)) ITYPE;          // integer vector type
    typedef decltype(x0 < x0) BVTYPE;            // boolean vector type

    // data vectors
    VTYPE x, x1, x2;                             // x variable
    VTYPE ef, e1, e2, e3, ee;                    // exponent
    VTYPE yr;                                    // remainder
    VTYPE lg, lg1, lgerr, x2err, v;              // logarithm
    VTYPE z;                                     // pow(x,y)
    VTYPE yodd(0);                               // has sign bit set if y is an odd integer
    // integer vectors
    ITYPE ei, ej;                                // exponent
    // boolean vectors
    BVTYPE blend, xzero;                  // x conditions
    BVTYPE overflow, underflow;           // error conditions

    // remove sign
    x1 = abs(x0);

    // Separate mantissa from exponent
    // This gives the mantissa * 0.5
    x  = fraction_2(x1);

    // reduce range of x = +/- sqrt(2)/2
    blend = x > static_cast<float>(VM_SQRT2 * 0.5);
    x  = if_add(!blend, x, x);                   // conditional add

    // Taylor expansion, high precision
    x   -= 1.0f;
    x2   = x * x;
    lg1  = polynomial_8(x, P0logf, P1logf, P2logf, P3logf, P4logf, P5logf, P6logf, P7logf, P8logf);
    lg1 *= x2 * x;

    // extract exponent
    ef = exponent_f(x1);
    ef = if_add(blend, ef, 1.0f);                // conditional add

    // multiply exponent by y
    // nearest integer e1 goes into exponent of result, remainder yr is added to log
    e1 = round(ef * y);
    yr = mul_sub(ef, y, e1);                   // calculate remainder yr. precision very important here

    // add initial terms to expansion
    lg = nmul_add(0.5f, x2, x) + lg1;            // lg = (x - 0.5f * x2) + lg1;

    // calculate rounding errors in lg
    // rounding error in multiplication 0.5*x*x
    x2err = mul_sub(0.5f*x, x, 0.5f * x2);
    // rounding error in additions and subtractions
    lgerr = mul_add(0.5f, x2, lg - x) - lg1;     // lgerr = ((lg - x) + 0.5f * x2) - lg1;

    // extract something for the exponent
    e2 = round(lg * y * static_cast<float>(VM_LOG2E));
    // subtract this from lg, with extra precision
    v = mul_sub(lg, y, e2 * ln2f_hi);

    // correct for previous rounding errors
    v -= mul_sub(lgerr + x2err, y, yr * static_cast<float>(VM_LN2)); // v -= (lgerr + x2err) * y - yr * float(VM_LN2) ;

    // exp function

    // extract something for the exponent if possible
    x = v;
    e3 = round(x*log2e);
    // high precision multiplication not needed here because abs(e3) <= 1
    x = nmul_add(e3, static_cast<float>(VM_LN2), x);          // x -= e3 * float(VM_LN2);

    // Taylor polynomial
    x2  = x  * x;
    z = polynomial_5(x, p2expf, p3expf, p4expf, p5expf, p6expf, p7expf)*x2 + x + 1.0f;

    // contributions to exponent
    ee = e1 + e2 + e3;
    ei = roundi(ee);
    // biased exponent of result:
    ej = ei + (ITYPE(reinterpret_i(abs(z))) >> 23);

    // add exponent by integer addition
    z = reinterpret_f(ITYPE(reinterpret_i(z)) + (ei << 23)); // the extra 0x10000 is shifted out here


    // check for overflow and underflow

    overflow = BVTYPE(ej >= 0x0FF) | (ee > 300.f);
    underflow = BVTYPE(ej <= 0x000) | (ee < -300.f);
    if (horizontal_or(overflow | underflow)) {
      // handle errors
      z = select(underflow, VTYPE(0.f), z);
      z = select(overflow, infinite_vec<VTYPE>(), z);
    }

    // check for x == 0

    xzero = is_zero_or_subnormal(x0);
    z = wm_pow_case_x0(xzero, y, z);

    return z;

    /* clang-format on */
  }
} // namespace ryujin

#endif
