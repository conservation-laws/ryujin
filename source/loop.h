//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <computing_timer.h>
#include <convenience_macros.h>
#include <instrumentation.h>
#include <simd.h>

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
   * A reducer for summing up an array of values that can be used with
   * reduction_loop(). The reducer takes a result object as argument in
   * which it folds in contributions of the loop body:
   * ```
   * std::vector<Number> sums(n_values, Number(0.));
   * reduction_loop<MemorySpace>("name", body, ArraySum<Number>(sums), 0, n);
   * ```
   *
   * The loop body itself has to return a callable object `j -> Number`
   * returning the j-th partial result that will be folded back into the
   * result.
   */
  template <typename Number>
  struct ArraySum {
    using value_type = Number[];

    const unsigned int value_count;

    ArraySum(Number *data, const unsigned int n)
        : value_count(n)
        , data_(data)
    {
    }

    explicit ArraySum(std::vector<Number> &values)
        : ArraySum(values.data(), static_cast<unsigned int>(values.size()))
    {
    }

    KOKKOS_INLINE_FUNCTION
    void init(Number *values) const
    {
      for (unsigned int k = 0; k < value_count; ++k)
        values[k] = Number(0.);
    }

    KOKKOS_INLINE_FUNCTION
    void join(Number *destination, const Number *source) const
    {
      for (unsigned int k = 0; k < value_count; ++k)
        destination[k] += source[k];
    }

    template <typename Contribution>
      requires std::invocable<const Contribution &, unsigned int>
    KOKKOS_INLINE_FUNCTION void join(Number *destination,
                                     const Contribution &contribution) const
    {
      for (unsigned int k = 0; k < value_count; ++k)
        destination[k] += contribution(k);
    }

    Number *reference() const
    {
      return data_;
    }

  private:
    Number *const data_;
  };


  namespace internal
  {
    /**
     * A local storage container for a reducer with either a scalar Number
     * or an array Number[] value_type. The Kokkos convention for the
     * latter is to also provide a `value_count` member annotating the size
     * of the array. We use this for setting up a type trait is_array.
     */
    template <typename Reducer>
    struct LocalResult {
      static constexpr bool is_array =
          std::is_array_v<typename Reducer::value_type>;
      using scalar_type = std::remove_extent_t<typename Reducer::value_type>;

      std::conditional_t<is_array, std::vector<scalar_type>, scalar_type>
          storage;

      /* Initialize our storage element: */
      LocalResult(const Reducer &reducer)
      {
        if constexpr (is_array)
          storage.resize(reducer.value_count);
        reducer.init(get());
      }

      /* Return the stored data: */
      std::conditional_t<is_array, scalar_type *, scalar_type &> get()
      {
        if constexpr (is_array)
          return storage.data();
        else
          return storage;
      }

      /* Construct a view for our storage element: */
      auto view()
      {
        using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
        if constexpr (is_array)
          return Kokkos::View<scalar_type *, Kokkos::HostSpace, Unmanaged>(
              storage.data(), storage.size());
        else
          return Kokkos::View<scalar_type, Kokkos::HostSpace, Unmanaged>(
              &storage);
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
   * returned by the loop body with the supplied @p reducer into the result
   * storage that the reducer references, see reduction_loop().
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
  template <typename Reducer, typename Functor, typename... Args>
  inline void cpu_reduction_loop(const std::string &region_name
                                 [[maybe_unused]],
                                 const Functor &body,
                                 const Reducer &reducer,
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

    using scalar_type = std::remove_extent_t<typename Reducer::value_type>;

#if defined(WITH_OPENMP)
    /* Variant using OpenMP: */

    RYUJIN_PRAGMA(omp parallel default(shared))
    {
      internal::LocalResult<Reducer> local_result(reducer);

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
            internal::LocalResult<Reducer> local_result(reducer);

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
   * contributions returned by the loop body with the supplied @p reducer
   * into the result storage that the reducer references, see reduction_loop().
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
  template <typename Reducer, typename Functor, typename... Args>
  inline void gpu_reduction_loop(const std::string &region_name,
                                 const Functor &body,
                                 const Reducer &reducer,
                                 const unsigned int left,
                                 const unsigned int right,
                                 Args &&...args)
  {
    DeviceTimer::Scope scope;

    Assert(
        left <= right,
        dealii::ExcMessage("Invalid index range: it must hold left <= right"));

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

    internal::LocalResult<Reducer> result(reducer);

    if (!region_name.empty()) {
      NVTX_MARKER_START(region_name.c_str());
    }

    Kokkos::parallel_reduce(
        region_name, Policy(exec, left, right), functor, result.view());

    exec.fence();

    if (!region_name.empty()) {
      NVTX_MARKER_STOP(region_name.c_str());
    }

    reducer.join(reducer.reference(), result.get());
  }


  /*
   * A reduction loop running either on the CPU, or on the device depending
   * on the selected memory space: For dealii::MemorySpace::Host the loop is
   * dispatched to cpu_reduction_loop(), and for dealii::MemorySpace::Default to
   * gpu_reduction_loop().
   *
   * The loop body computes and returns a contribution for every index. The
   * reduction operation itself is selected with a @p reducer object (such
   * as Kokkos::Min, Kokkos::Max, or Kokkos::Sum) that folds all
   * contributions into the result storage it references. The value_type of
   * the reducer also determines the number type of the loop. The initial
   * contents of the result storage take part in the reduction:
   * ```
   * const auto body = [=](auto, unsigned int i) -> Number {
   *   // ...
   *   return local_contribution;
   * };
   *
   * reduction_loop<MemorySpace>(
   *     "loop name", body, Kokkos::Min<Number>(value), 0, n_owned);
   * ```
   * For summing up an array of values use the ArraySum reducer, in which
   * case the loop body returns a callable `j -> Number` that the reducer
   * evaluates for all j < `value_count`:
   * ```
   * const auto body = [=](auto, unsigned int i) {
   *   // ...
   *   return [=](unsigned int j) { return local_contribution[j]; };
   * };
   *
   * std::vector<Number> sums(n_values, Number(0.));
   * reduction_loop<MemorySpace>(
   *     "loop name", body, ArraySum<Number>(sums), 0, n_owned);
   * ```
   *
   * @note Here, @p body is a functor that must accept a "sentinel" type as
   * first argument and the current index i as last argument, and that
   * returns its contribution to the reduction. Additional `args` may be
   * specified in the reduction_loop() invocation that will be forwarded to
   * the loop body.
   */
  template <typename MemorySpace,
            typename Reducer,
            typename Functor,
            typename... Args>
  inline void reduction_loop(const std::string &region_name,
                             const Functor &body,
                             const Reducer &reducer,
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
      cpu_reduction_loop(
          region_name, body, reducer, left, right, std::forward<Args>(args)...);
    } else {
      gpu_reduction_loop(
          region_name, body, reducer, left, right, std::forward<Args>(args)...);
    }
  }
} // namespace ryujin
