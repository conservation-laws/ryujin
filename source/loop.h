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
  template <typename Number, typename Functor>
  inline void loop(const std::string &region_name [[maybe_unused]],
                   const Functor &functor,
                   unsigned int left,
                   unsigned int right)
  {

    RYUJIN_PARALLEL_REGION_BEGIN
    LIKWID_MARKER_START(region_name.c_str());

    constexpr unsigned int stride_size = get_stride_size<Number>;

    RYUJIN_OMP_FOR
    for (unsigned int i = left; i < right; i += stride_size)
      functor(Number(), i);

    LIKWID_MARKER_STOP(region_name.c_str());
    RYUJIN_PARALLEL_REGION_END
  }
} // namespace ryujin
