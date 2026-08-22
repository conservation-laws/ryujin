//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>
#include <simd.h>

#include <deal.II/base/config.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/parallel.h>

#include <string>
#include <type_traits>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace ryujin
{
  /*
   * A thread-parallelized and vectorized loop running on the CPU. The loop
   * traverses the index range [left, internal) SIMD vectorized stepping
   * forward with a stride size equal to the number of packed
   * doubles/singles that the loop body operates on at the same time. For
   * the remainder of the index range, i.e., [internal, right) a serial
   * loop is invoked.
   *
   * @note the index internal is rounded down to the next integer multiple
   * of the SIMD stride size.
   *
   * @note Here, @p body is a functor that must accept a "sentinel" type as
   * first argument and the current index i as last argument. Additional
   * `args` may be specified in the cpu_simd_loop() invocation that will be
   * forwarded to the loop body:
   * `body(Number(), std::forward<Args>(args)..., i);`
   */
  template <typename ScalarNumber, typename Functor, typename... Args>
  inline void cpu_simd_loop(const std::string &region_name [[maybe_unused]],
                            const Functor &body,
                            const unsigned int left,
                            const unsigned int internal,
                            const unsigned int right,
                            Args &&...args)
  {
    Assert(left <= internal && internal <= right,
           dealii::ExcMessage("Invalid index range: it must hold left <= "
                              "internal, internal <= right"));

    using VA = dealii::VectorizedArray<ScalarNumber>;

    constexpr unsigned int stride_size = get_stride_size<VA>;
    const unsigned int regular =
        left + (internal - left) / stride_size * stride_size;

#if defined(WITH_OPENMP)
    /* Variant using OpenMP: */

    RYUJIN_PRAGMA(omp parallel default(shared))
    {
      /* SIMD vectorized loop: */
      RYUJIN_PRAGMA(omp for nowait)
      for (unsigned int i = left; i < regular; i += stride_size)
        body(VA(), std::forward<Args>(args)..., i);

      /* Serial loop: */
      RYUJIN_PRAGMA(omp for)
      for (unsigned int i = regular; i < right; i += 1)
        body(ScalarNumber(), std::forward<Args>(args)..., i);
    }

#elif defined(WITH_DEAL_II_THREADS)
    /* Variant using dealii's parallel for: */
    {
      /*
       * We have to ensure that the deal.II routine only schedules a
       * workload that is divisible by stride_size.
       */
      Assert((regular - left) % stride_size == 0, dealii::ExcInternalError());
      dealii::parallel::apply_to_subranges(
          0,
          (regular - left) / stride_size,
          [&](const unsigned int begin, const unsigned int end) {
            /* SIMD vectorized loop: */
            for (unsigned int i = begin; i < end; ++i)
              body(VA(), std::forward<Args>(args)..., left + stride_size * i);
          },
          1000);

      dealii::parallel::apply_to_subranges(
          regular,
          right,
          [&](const unsigned int begin, const unsigned int end) {
            /* Serial loop: */
            for (unsigned int i = begin; i < end; ++i)
              body(ScalarNumber(), std::forward<Args>(args)..., i);
          },
          1000);
    }

#else
    /* Execute loops in serial: */
    {
      /* SIMD vectorized loop: */
      for (unsigned int i = left; i < regular; i += stride_size)
        body(VA(), std::forward<Args>(args)..., i);

      /* Serial loop: */
      for (unsigned int i = regular; i < right; i += 1)
        body(ScalarNumber(), std::forward<Args>(args)..., i);
    }
#endif
  }


  /*
   * A loop running on the device (i.e., in the default memory space). The
   * loop traverses the index range [left, right) with a suitable Kokkos
   * parallel_for using a range policy.
   *
   * @note The index range [left, internal) that is used for SIMD
   * vectorization in cpu_simd_loop() is currently ignored: On the device
   * every "lane" operates on a scalar value and the loop body is thus
   * always called with a scalar sentinel type.
   *
   * @note Here, @p body is a functor that must accept a "sentinel" type as
   * first argument and the current index i as last argument. Additional
   * `args` may be specified in the gpu_loop() invocation that will be
   * forwarded to the loop body:
   * `body(ScalarNumber(), std::forward<Args>(args)..., i);`
   *
   * @note The loop body (and everything it references) has to be callable
   * on the device. In particular, @p body and all @p args are copied into
   * the kernel, meaning that the functor must capture by value and must be
   * trivially copyable.
   *
   * @note The function fences the execution space before returning. It
   * thus has the same (synchronous) semantics as cpu_simd_loop().
   */
  template <typename ScalarNumber, typename Functor, typename... Args>
  inline void gpu_loop(const std::string &region_name,
                       const Functor &body,
                       const unsigned int left,
                       const unsigned int internal [[maybe_unused]],
                       const unsigned int right,
                       Args &&...args)
  {
    Assert(left <= internal && internal <= right,
           dealii::ExcMessage("Invalid index range: it must hold left <= "
                              "internal, internal <= right"));

    using MemorySpace = dealii::MemorySpace::Default;
    using ExecutionSpace = typename MemorySpace::kokkos_space::execution_space;
    using Policy =
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<unsigned int>>;

    const auto exec = ExecutionSpace{};

    Kokkos::parallel_for(
        region_name,
        Policy(exec, left, right),
        KOKKOS_LAMBDA(const unsigned int i) {
          body(ScalarNumber(), args..., i);
        });

    exec.fence();
  }


  /*
   * A loop running either on the CPU, or on the device depending on the
   * selected memory space: For dealii::MemorySpace::Host the loop is
   * dispatched to cpu_simd_loop(), and for dealii::MemorySpace::Default to
   * gpu_loop().
   *
   * @note Here, @p body is a functor that must accept a "sentinel" type as
   * first argument and the current index i as last argument. Additional
   * `args` may be specified in the loop() invocation that will be forwarded
   * to the loop body.
   */
  template <typename MemorySpace,
            typename ScalarNumber,
            typename Functor,
            typename... Args>
  inline void loop(const std::string &region_name,
                   const Functor &body,
                   const unsigned int left,
                   const unsigned int internal,
                   const unsigned int right,
                   Args &&...args)
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      cpu_simd_loop<ScalarNumber>(region_name,
                                  body,
                                  left,
                                  internal,
                                  right,
                                  std::forward<Args>(args)...);
    } else {
      gpu_loop<ScalarNumber>(region_name,
                             body,
                             left,
                             internal,
                             right,
                             std::forward<Args>(args)...);
    }
  }
} // namespace ryujin
