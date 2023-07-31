//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "simd.template.h"

namespace ryujin
{
#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 3 && defined(__AVX512F__)
  template dealii::VectorizedArray<double, 8>
  pow(const dealii::VectorizedArray<double, 8>, const double);

  template dealii::VectorizedArray<double, 8>
  pow(const dealii::VectorizedArray<double, 8>,
      const dealii::VectorizedArray<double, 8>);

  template dealii::VectorizedArray<float, 16>
  pow(const dealii::VectorizedArray<float, 16>, const float);

  template dealii::VectorizedArray<float, 16>
  pow(const dealii::VectorizedArray<float, 16>,
      const dealii::VectorizedArray<float, 16>);

  template dealii::VectorizedArray<double, 8>
  fast_pow(const dealii::VectorizedArray<double, 8>, const double, const Bias);

  template dealii::VectorizedArray<double, 8>
  fast_pow(const dealii::VectorizedArray<double, 8>,
           const dealii::VectorizedArray<double, 8>,
           const Bias);

  template dealii::VectorizedArray<float, 16>
  fast_pow(const dealii::VectorizedArray<float, 16>, const float, const Bias);

  template dealii::VectorizedArray<float, 16>
  fast_pow(const dealii::VectorizedArray<float, 16>,
           const dealii::VectorizedArray<float, 16>,
           const Bias);
#endif

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 2 && defined(__AVX__)
  template dealii::VectorizedArray<double, 4>
  pow(const dealii::VectorizedArray<double, 4>, const double);

  template dealii::VectorizedArray<double, 4>
  pow(const dealii::VectorizedArray<double, 4>,
      const dealii::VectorizedArray<double, 4>);

  template dealii::VectorizedArray<float, 8>
  pow(const dealii::VectorizedArray<float, 8>, const float);

  template dealii::VectorizedArray<float, 8>
  pow(const dealii::VectorizedArray<float, 8>,
      const dealii::VectorizedArray<float, 8>);

  template dealii::VectorizedArray<double, 4>
  fast_pow(const dealii::VectorizedArray<double, 4>, const double, const Bias);

  template dealii::VectorizedArray<double, 4>
  fast_pow(const dealii::VectorizedArray<double, 4>,
           const dealii::VectorizedArray<double, 4>,
           const Bias);

  template dealii::VectorizedArray<float, 8>
  fast_pow(const dealii::VectorizedArray<float, 8>, const float, const Bias);

  template dealii::VectorizedArray<float, 8>
  fast_pow(const dealii::VectorizedArray<float, 8>,
           const dealii::VectorizedArray<float, 8>,
           const Bias);
#endif

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)
  template dealii::VectorizedArray<double, 2>
  pow(const dealii::VectorizedArray<double, 2>, const double);

  template dealii::VectorizedArray<double, 2>
  pow(const dealii::VectorizedArray<double, 2>,
      const dealii::VectorizedArray<double, 2>);

  template dealii::VectorizedArray<float, 4>
  pow(const dealii::VectorizedArray<float, 4>, const float);

  template dealii::VectorizedArray<float, 4>
  pow(const dealii::VectorizedArray<float, 4>,
      const dealii::VectorizedArray<float, 4>);

  template dealii::VectorizedArray<double, 2>
  fast_pow(const dealii::VectorizedArray<double, 2>, const double, const Bias);

  template dealii::VectorizedArray<double, 2>
  fast_pow(const dealii::VectorizedArray<double, 2>,
           const dealii::VectorizedArray<double, 2>,
           const Bias);

  template dealii::VectorizedArray<float, 4>
  fast_pow(const dealii::VectorizedArray<float, 4>, const float, const Bias);

  template dealii::VectorizedArray<float, 4>
  fast_pow(const dealii::VectorizedArray<float, 4>,
           const dealii::VectorizedArray<float, 4>,
           const Bias);
#endif

  template dealii::VectorizedArray<double, 1>
  pow(const dealii::VectorizedArray<double, 1>, const double);

  template dealii::VectorizedArray<double, 1>
  pow(const dealii::VectorizedArray<double, 1>,
      const dealii::VectorizedArray<double, 1>);

  template dealii::VectorizedArray<float, 1>
  pow(const dealii::VectorizedArray<float, 1>, const float);

  template dealii::VectorizedArray<float, 1>
  pow(const dealii::VectorizedArray<float, 1>,
      const dealii::VectorizedArray<float, 1>);

  template dealii::VectorizedArray<double, 1>
  fast_pow(const dealii::VectorizedArray<double, 1>, const double, const Bias);

  template dealii::VectorizedArray<double, 1>
  fast_pow(const dealii::VectorizedArray<double, 1>,
           const dealii::VectorizedArray<double, 1>,
           const Bias);

  template dealii::VectorizedArray<float, 1>
  fast_pow(const dealii::VectorizedArray<float, 1>, const float, const Bias);

  template dealii::VectorizedArray<float, 1>
  fast_pow(const dealii::VectorizedArray<float, 1>,
           const dealii::VectorizedArray<float, 1>,
           const Bias);

} // namespace ryujin
