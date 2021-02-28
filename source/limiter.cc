//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#include "limiter.template.h"

using namespace dealii;

namespace ryujin
{
  /* instantiations */

#ifndef OBSESSIVE_INLINING
  template NUMBER
  Limiter<DIM, NUMBER>::limit<Limiter<DIM, NUMBER>::Limiters::specific_entropy>(
      const ProblemDescription &,
      const std::array<NUMBER, 3> &,
      const rank1_type &,
      const rank1_type &,
      const NUMBER,
      const NUMBER);

  template VectorizedArray<NUMBER> Limiter<DIM, VectorizedArray<NUMBER>>::limit<
      Limiter<DIM, VectorizedArray<NUMBER>>::Limiters::specific_entropy>(
      const ProblemDescription &,
      const std::array<VectorizedArray<NUMBER>, 3> &,
      const rank1_type &,
      const rank1_type &,
      const VectorizedArray<NUMBER>,
      const VectorizedArray<NUMBER>);

  template NUMBER Limiter<DIM, NUMBER>::limit<
      Limiter<DIM, NUMBER>::Limiters::entropy_inequality>(
      const ProblemDescription &,
      const std::array<NUMBER, 5> &,
      const rank1_type &,
      const rank1_type &,
      const NUMBER,
      const NUMBER);

  template VectorizedArray<NUMBER> Limiter<DIM, VectorizedArray<NUMBER>>::limit<
      Limiter<DIM, VectorizedArray<NUMBER>>::Limiters::entropy_inequality>(
      const ProblemDescription &,
      const std::array<VectorizedArray<NUMBER>, 5> &,
      const rank1_type &,
      const rank1_type &,
      const VectorizedArray<NUMBER>,
      const VectorizedArray<NUMBER>);
#endif

} // namespace ryujin
