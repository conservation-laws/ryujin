//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef OPENMP_MACROS_H
#define OPENMP_MACROS_H

/**
 * @name OpenMP parallel for loop options and macros:
 */
//@{

#define GRENDEL_PRAGMA(x) _Pragma(#x)

#define GRENDEL_PARALLEL_REGION_BEGIN                                          \
  GRENDEL_PRAGMA(omp parallel default(shared))                                 \
  {

#define GRENDEL_PARALLEL_REGION_END }

#define GRENDEL_OMP_FOR GRENDEL_PRAGMA(omp for)
#define GRENDEL_OMP_FOR_NOWAIT GRENDEL_PRAGMA(omp for nowait)
#define GRENDEL_OMP_BARRIER GRENDEL_PRAGMA(omp barrier)

#define GRENDEL_LIKELY(x) (__builtin_expect(!!(x), 1))
#define GRENDEL_UNLIKELY(x) (__builtin_expect(!!(x), 0))

//@}

#endif /* OPENMP_MACROS_H */
