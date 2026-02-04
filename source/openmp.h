//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/config.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

/**
 * @name OpenMP parallel for macros
 *
 * Intended use:
 * ```
 * // serial work
 *
 * RYUJIN_PARALLEL_REGION_BEGIN
 *
 * // per thread work and thread-local storage declarations
 *
 * RYUJIN_OMP_FOR
 * for (unsigned int i = 0; i < size_internal; i += simd_length) {
 *   // parallel for loop that is statically distributed on all available
 *   // worker threads by slicing the interval [0,size_internal)
 * }
 *
 * RYUJIN_PARALLEL_REGION_END
 * ```
 */
//@{

/**
 * Macro expanding to a `#pragma` directive that can be used in other
 * preprocessor macro definitions.
 *
 * @ingroup Miscellaneous
 */
#define RYUJIN_PRAGMA(x) _Pragma(#x)

/**
 * Begin an openmp parallel region.
 *
 * @ingroup Miscellaneous
 */
#define RYUJIN_PARALLEL_REGION_BEGIN                                           \
  RYUJIN_PRAGMA(omp parallel default(shared))                                  \
  {

/**
 * End an openmp parallel region.
 *
 * @ingroup Miscellaneous
 */
#define RYUJIN_PARALLEL_REGION_END }

/**
 * Enter a parallel for loop.
 *
 * @ingroup Miscellaneous
 */
#define RYUJIN_OMP_FOR RYUJIN_PRAGMA(omp for)

/**
 * Enter a parallel for loop with "nowait" declaration, i.e., the end of
 * the for loop does not include an implicit thread synchronization
 * barrier.
 *
 * @ingroup Miscellaneous
 */
#define RYUJIN_OMP_FOR_NOWAIT RYUJIN_PRAGMA(omp for nowait)

/**
 * Declare an explicit Thread synchronization barrier.
 *
 * @ingroup Miscellaneous
 */
#define RYUJIN_OMP_BARRIER RYUJIN_PRAGMA(omp barrier)

/**
 * Annotate a critical section that has to be accessed sequentially.
 *
 * @ingroup Miscellaneous
 */
#define RYUJIN_OMP_CRITICAL RYUJIN_PRAGMA(omp critical)

/**
 * Annotate a section that has to be executed on one thread only.
 *
 * @ingroup Miscellaneous
 */
#define RYUJIN_OMP_SINGLE RYUJIN_PRAGMA(omp single)

//@}
