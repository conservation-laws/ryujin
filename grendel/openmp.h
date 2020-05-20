//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef OPENMP_H
#define OPENMP_H

#include <compile_time_options.h>

#include <deal.II/base/config.h>

#include <atomic>
#include <omp.h>

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


/**
 * FIXME: Documentation
 */
template <typename Payload>
class SynchronizationDispatch
{
public:
  SynchronizationDispatch(const Payload &payload)
      : payload_(payload)
      , executed_payload_(false)
      , n_threads_ready_(0)
  {
  }

  ~SynchronizationDispatch()
  {
    if (!executed_payload_)
      payload_();
  }

  DEAL_II_ALWAYS_INLINE inline void check(bool &thread_ready,
                                          const bool condition)
  {
#ifdef USE_COMMUNICATION_HIDING
    if (GRENDEL_UNLIKELY(thread_ready == false && condition)) {
#else
    (void)thread_ready;
    (void)condition;
    if constexpr (false) {
#endif
      thread_ready = true;
      if (++n_threads_ready_ == omp_get_num_threads()) {
        executed_payload_ = true;
        payload_();
      }
    }
  }

private:
  const Payload payload_;
  bool executed_payload_;
  std::atomic_int n_threads_ready_;
};

//@}

#endif /* OPENMP_H */
