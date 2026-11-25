//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/computing_timer.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/instrumentation.h>
#include <ryujin/base/simd.h>

#include <deal.II/base/config.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/parallel.h>

#include <concepts>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace ryujin
{
  /**
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
   *
   * @ingroup Utilities
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

    if (!region_name.empty()) {
      LIKWID_MARKER_START(region_name.c_str());
    }

    using VA = dealii::VectorizedArray<ScalarNumber>;

    constexpr unsigned int stride_size = VA::size();
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

    if (!region_name.empty()) {
      LIKWID_MARKER_STOP(region_name.c_str());
    }
  }


  /**
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
   *
   * @ingroup Utilities
   */
  template <typename ScalarNumber, typename Functor, typename... Args>
  inline void gpu_loop(const std::string &region_name,
                       const Functor &body,
                       const unsigned int left,
                       const unsigned int internal [[maybe_unused]],
                       const unsigned int right,
                       Args &&...args)
  {
    DeviceTimer::Scope scope;

    Assert(left <= internal && internal <= right,
           dealii::ExcMessage("Invalid index range: it must hold left <= "
                              "internal, internal <= right"));

    using MemorySpace = dealii::MemorySpace::Default;
    using ExecutionSpace = typename MemorySpace::kokkos_space::execution_space;
    using Policy =
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<unsigned int>>;

    const auto exec = ExecutionSpace{};

    if (!region_name.empty()) {
      NVTX_MARKER_START(region_name.c_str());
    }

    Kokkos::parallel_for(
        region_name,
        Policy(exec, left, right),
        KOKKOS_LAMBDA(const unsigned int i) {
          body(ScalarNumber(), args..., i);
        });

    exec.fence();

    if (!region_name.empty()) {
      NVTX_MARKER_STOP(region_name.c_str());
    }
  }


  /**
   * A loop running either on the CPU, or on the device depending on the
   * selected memory space: For dealii::MemorySpace::Host the loop is
   * dispatched to cpu_simd_loop(), and for dealii::MemorySpace::Default to
   * gpu_loop().
   *
   * @note Here, @p body is a functor that must accept a "sentinel" type as
   * first argument and the current index i as last argument. Additional
   * `args` may be specified in the loop() invocation that will be forwarded
   * to the loop body.
   *
   * @ingroup Utilities
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


  /**
   * The reduction operations of a reduction_loop(): Reduction::sum,
   * Reduction::min, and Reduction::max.
   *
   * @ingroup Utilities
   */
  namespace Reduction
  {
    template <typename Number>
    using sum = Kokkos::Sum<Number>;

    template <typename Number>
    using min = Kokkos::Min<Number>;

    template <typename Number>
    using max = Kokkos::Max<Number>;
  } // namespace Reduction


  namespace internal
  {
    /**
     * A Kokkos reducer that reduces an array of values elementwise with a
     * scalar Kokkos reducer (such as Kokkos::Sum, Kokkos::Min, or
     * Kokkos::Max). The loop body returns a callable object `j -> Number`
     * returning the j-th partial result.
     */
    template <typename ElementReducer>
    struct ArrayReducer {
      using scalar_type = typename ElementReducer::value_type;
      using value_type = scalar_type[];

      const unsigned int value_count;

      explicit ArrayReducer(std::vector<scalar_type> &values)
          : value_count(static_cast<unsigned int>(values.size()))
          , data_(values.data())
          , element_reducer_(typename ElementReducer::result_view_type())
      {
      }

      KOKKOS_INLINE_FUNCTION
      void init(scalar_type *values) const
      {
        for (unsigned int k = 0; k < value_count; ++k)
          element_reducer_.init(values[k]);
      }

      KOKKOS_INLINE_FUNCTION
      void join(scalar_type *destination, const scalar_type *source) const
      {
        for (unsigned int k = 0; k < value_count; ++k)
          element_reducer_.join(destination[k], source[k]);
      }

      template <typename Contribution>
        requires std::invocable<const Contribution &, unsigned int>
      KOKKOS_INLINE_FUNCTION void join(scalar_type *destination,
                                       const Contribution &contribution) const
      {
        for (unsigned int k = 0; k < value_count; ++k)
          element_reducer_.join(destination[k], contribution(k));
      }

      scalar_type *reference() const
      {
        return data_;
      }

    private:
      scalar_type *const data_;

      /*
       * The element reducer is only used for its join() and init()
       * operations; it is constructed with an empty result view.
       */
      const ElementReducer element_reducer_;
    };


    /**
     * Return the Kokkos reducer for the reduction operation @p Op that
     * folds contributions into @p result: either a scalar Number, or a
     * std::vector<Number> that is reduced elementwise.
     */
    template <template <typename> class Op, typename Result>
    auto make_reducer(Result &result)
    {
      if constexpr (std::is_arithmetic_v<Result>) {
        return Op<Result>(result);
      } else {
        using Number = typename Result::value_type;
        return ArrayReducer<Op<Number>>(result);
      }
    }


    /**
     * A local storage container for a partial result of a reduction into
     * @p Result, which is either a scalar Number, or a std::vector<Number>.
     */
    template <typename Result>
    struct LocalResult {
      Result storage;

      /*
       * Size the storage like @p result and initialize it with the neutral
       * element of the @p reducer.
       */
      template <typename Reducer>
      LocalResult(const Reducer &reducer, const Result &result)
      {
        if constexpr (!std::is_arithmetic_v<Result>)
          storage.resize(result.size());
        reducer.init(get());
      }

      /*
       * Return the stored data in the form that the join() and init()
       * operations of the reducer expect: a reference to a scalar, or a
       * pointer to the first element of an array.
       */
      decltype(auto) get()
      {
        if constexpr (std::is_arithmetic_v<Result>)
          return (storage);
        else
          return storage.data();
      }

      /* Construct a view for our storage element: */
      auto view()
      {
        using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
        if constexpr (std::is_arithmetic_v<Result>)
          return Kokkos::View<Result, Kokkos::HostSpace, Unmanaged>(&storage);
        else {
          using Number = typename Result::value_type;
          return Kokkos::View<Number *, Kokkos::HostSpace, Unmanaged>(
              storage.data(), storage.size());
        }
      }
    };


    /**
     * A Kokkos functor for gpu_reduction_loop() that combines the reducer
     * with the loop body. We simply augment the reducer itself with an
     * operator() calling the loop body.
     */
    template <typename Reducer, typename Body>
    struct ReductionFunctor : Reducer {
      const Body body;

      ReductionFunctor(const Reducer &reducer, const Body &body)
          : Reducer(reducer)
          , body(body)
      {
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(const unsigned int i, auto &&local_result) const
      {
        Reducer::join(local_result, body(i));
      }
    };
  } // namespace internal


  /*
   * A thread-parallelized reduction loop running on the CPU. The loop
   * traverses the index range [left, right) and reduces the contributions
   * returned by the loop body into @p result, see reduction_loop().
   *
   * @note In contrast to cpu_simd_loop() the loop is not SIMD vectorized:
   * the loop body is always called with a scalar sentinel type.
   *
   * @note Here, @p body is a functor that must accept a "sentinel" type as
   * first argument and the current index i as last argument, and that
   * returns its contribution to the reduction. Additional `args` may be
   * specified in the cpu_reduction_loop() invocation that will be
   * forwarded to the loop body:
   * `body(Number(), std::forward<Args>(args)..., i)`
   */
  template <template <typename> class Op,
            typename Result,
            typename Functor,
            typename... Args>
  inline void cpu_reduction_loop(const std::string &region_name
                                 [[maybe_unused]],
                                 const Functor &body,
                                 Result &result,
                                 const unsigned int left,
                                 const unsigned int right,
                                 Args &&...args)
  {
    Assert(
        left <= right,
        dealii::ExcMessage("Invalid index range: it must hold left <= right"));

    if (!region_name.empty()) {
      LIKWID_MARKER_START(region_name.c_str());
    }

    const auto reducer = internal::make_reducer<Op>(result);
    using Reducer = std::remove_const_t<decltype(reducer)>;
    using scalar_type = std::remove_extent_t<typename Reducer::value_type>;

#if defined(WITH_OPENMP)
    /* Variant using OpenMP: */

    RYUJIN_PRAGMA(omp parallel default(shared))
    {
      internal::LocalResult<Result> local_result(reducer, result);

      RYUJIN_PRAGMA(omp for nowait)
      for (unsigned int i = left; i < right; ++i)
        reducer.join(local_result.get(),
                     body(scalar_type(), std::forward<Args>(args)..., i));

      RYUJIN_PRAGMA(omp critical)
      reducer.join(reducer.reference(), local_result.get());
    }

#elif defined(WITH_DEAL_II_THREADS)
    /* Variant using dealii's parallel for: */
    {
      std::mutex mutex;

      dealii::parallel::apply_to_subranges(
          left,
          right,
          [&](const unsigned int begin, const unsigned int end) {
            /* per thread */
            internal::LocalResult<Result> local_result(reducer, result);

            for (unsigned int i = begin; i < end; ++i)
              reducer.join(local_result.get(),
                           body(scalar_type(), std::forward<Args>(args)..., i));

            std::lock_guard<std::mutex> lock(mutex);
            reducer.join(reducer.reference(), local_result.get());
          },
          1000);
    }

#else
    /* Execute loop in serial: */
    {
      for (unsigned int i = left; i < right; ++i)
        reducer.join(reducer.reference(),
                     body(scalar_type(), std::forward<Args>(args)..., i));
    }
#endif

    if (!region_name.empty()) {
      LIKWID_MARKER_STOP(region_name.c_str());
    }
  }


  /*
   * A reduction loop running on the device (i.e., in the default memory
   * space). The loop traverses the index range [left, right) with a
   * Kokkos::parallel_reduce using a range policy and reduces the
   * contributions returned by the loop body into @p result, see
   * reduction_loop().
   *
   * @note Here, @p body is a functor that must accept a "sentinel" type as
   * first argument and the current index i as last argument, and that
   * returns its contribution to the reduction. Additional `args` may be
   * specified in the gpu_reduction_loop() invocation that will be
   * forwarded to the loop body:
   * `body(Number(), args..., i)`
   *
   * @note The loop body (and everything it references) has to be callable
   * on the device, see the discussion in gpu_loop().
   *
   * @note The function fences the execution space before returning. It
   * thus has the same (synchronous) semantics as cpu_reduction_loop().
   */
  template <template <typename> class Op,
            typename Result,
            typename Functor,
            typename... Args>
  inline void gpu_reduction_loop(const std::string &region_name,
                                 const Functor &body,
                                 Result &result,
                                 const unsigned int left,
                                 const unsigned int right,
                                 Args &&...args)
  {
    DeviceTimer::Scope scope;

    Assert(
        left <= right,
        dealii::ExcMessage("Invalid index range: it must hold left <= right"));

    const auto reducer = internal::make_reducer<Op>(result);
    using Reducer = std::remove_const_t<decltype(reducer)>;
    using scalar_type = std::remove_extent_t<typename Reducer::value_type>;

    using MemorySpace = dealii::MemorySpace::Default;
    using ExecutionSpace = typename MemorySpace::kokkos_space::execution_space;
    using Policy =
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<unsigned int>>;

    const auto exec = ExecutionSpace{};

    const auto kernel = KOKKOS_LAMBDA(const unsigned int i)
    {
      return body(scalar_type(), args..., i);
    };

    const auto functor =
        internal::ReductionFunctor<Reducer, decltype(kernel)>(reducer, kernel);

    internal::LocalResult<Result> local_result(reducer, result);

    if (!region_name.empty()) {
      NVTX_MARKER_START(region_name.c_str());
    }

    Kokkos::parallel_reduce(
        region_name, Policy(exec, left, right), functor, local_result.view());

    exec.fence();

    if (!region_name.empty()) {
      NVTX_MARKER_STOP(region_name.c_str());
    }

    reducer.join(reducer.reference(), local_result.get());
  }


  /*
   * A reduction loop running either on the CPU, or on the device depending
   * on the selected memory space: For dealii::MemorySpace::Host the loop is
   * dispatched to cpu_reduction_loop(), and for dealii::MemorySpace::Default to
   * gpu_reduction_loop().
   *
   * The loop body computes and returns a contribution for every index.
   * The reduction operation @p Op (Reduction::sum, Reduction::min, or
   * Reduction::max) folds all contributions into @p result,
   * which is either a scalar Number, or a std::vector<Number> for an
   * elementwise reduction of an array of values. The initial contents of
   * @p result take part in the reduction:
   * ```
   * const auto body = [=](auto, unsigned int i) -> Number {
   *   // ...
   *   return local_contribution;
   * };
   *
   * reduction_loop<MemorySpace, Reduction::min>(
   *     "loop name", body, value, 0, n_owned);
   * ```
   * For reducing an array of values elementwise the loop body returns a
   * callable `j -> Number` that is evaluated for all j < `sums.size()`:
   * ```
   * const auto body = [=](auto, unsigned int i) {
   *   // ...
   *   return [=](unsigned int j) { return local_contribution[j]; };
   * };
   *
   * std::vector<Number> sums(n_values, Number(0.));
   * reduction_loop<MemorySpace, Reduction::sum>(
   *     "loop name", body, sums, 0, n_owned);
   * ```
   *
   * @note Here, @p body is a functor that must accept a "sentinel" type as
   * first argument and the current index i as last argument, and that
   * returns its contribution to the reduction. Additional `args` may be
   * specified in the reduction_loop() invocation that will be forwarded to
   * the loop body.
   */
  template <typename MemorySpace,
            template <typename> class Op,
            typename Result,
            typename Functor,
            typename... Args>
  inline void reduction_loop(const std::string &region_name,
                             const Functor &body,
                             Result &result,
                             const unsigned int left,
                             const unsigned int right,
                             Args &&...args)
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      cpu_reduction_loop<Op>(
          region_name, body, result, left, right, std::forward<Args>(args)...);
    } else {
      gpu_reduction_loop<Op>(
          region_name, body, result, left, right, std::forward<Args>(args)...);
    }
  }
} // namespace ryujin
