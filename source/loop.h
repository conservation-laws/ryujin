//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2025 - 2025 by the ryujin authors
//

#pragma once

#include <instrumentation.h>
#include <openmp.h>
#include <simd.h>

#include <string>

namespace ryujin
{
  /*
   * A thread-parallelized and vectorized loop running on the CPU. The loop
   * traverses the index range [left, internal) SIMD vectorized stepping
   * forward with a stride size equal to the number of packed
   * doubles/singles that the loop body operates on at the same time. For
   * the remainder of the index range, i.e., [internal, right) a serial
   * loop is invoked.
   */
  template <typename ScalarNumber, typename Functor>
  inline void cpu_simd_loop(const std::string &region_name [[maybe_unused]],
                            const Functor &functor,
                            const unsigned int left,
                            const unsigned int internal,
                            const unsigned int right)
  {
    Assert(left <= internal && internal <= right,
           dealii::ExcMessage("Invalid index range: it must hold left <= "
                              "internal, internal <= right"));

    using VA = dealii::VectorizedArray<ScalarNumber>;

    RYUJIN_PARALLEL_REGION_BEGIN
    LIKWID_MARKER_START(region_name.c_str());

    constexpr unsigned int simd_stride_size = get_stride_size<VA>;
    Assert(internal % simd_stride_size == 0,
           dealii::ExcMessage("Invalid index range: internal not divisible by "
                              "simd_stride_size."));

    /* SIMD vectorized loop: */

    RYUJIN_OMP_FOR
    for (unsigned int i = left; i < internal; i += simd_stride_size)
      functor(VA(), i);

    /* Serial loop: */

    RYUJIN_OMP_FOR
    for (unsigned int i = internal; i < right; i += 1)
      functor(ScalarNumber(), i);

    LIKWID_MARKER_STOP(region_name.c_str());
    RYUJIN_PARALLEL_REGION_END
  }
} // namespace ryujin
