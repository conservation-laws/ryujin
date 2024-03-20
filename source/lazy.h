//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by Matthias Maier
// Copyright (C) 2024 - 2024 by the ryujin authors
//

#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_consumption.h>

#if DEAL_II_VERSION_GTE(9, 5, 0)
#include <deal.II/base/mutex.h>
#else
#include <deal.II/base/thread_management.h>
#endif

#include <atomic>
#include <mutex>
#include <optional>


namespace ryujin
{
  /**
   * This is a slightly minimized variant of the Lazy<T> initialization class
   * shipped with the current development version of deal.II.
   */
  template <typename T>
  class Lazy
  {
  public:
    Lazy();
    Lazy(const Lazy &other);
    Lazy(Lazy &&other) noexcept;

    Lazy &operator=(const Lazy &other);
    Lazy &operator=(Lazy &&other) noexcept;

    void reset() noexcept;

    template <typename Callable>
    void ensure_initialized(const Callable &creator) const;

    bool has_value() const;

    const T &value() const;
    T &value();

  private:
    mutable std::optional<T> object;
    mutable std::atomic<bool> object_is_initialized;
    mutable dealii::Threads::Mutex initialization_mutex;
  };


  // ------------------------------- inline functions --------------------------


  template <typename T>
  inline Lazy<T>::Lazy()
      : object_is_initialized(false)
  {
  }


  template <typename T>
  inline Lazy<T>::Lazy(const Lazy &other)
      : object(other.object)
  {
    object_is_initialized.store(other.object_is_initialized.load());
  }


  template <typename T>
  inline Lazy<T>::Lazy(Lazy &&other) noexcept
      : object(std::move(other.object))
  {
    object_is_initialized.store(other.object_is_initialized.load());

    other.object_is_initialized.store(false);
    other.object.reset();
  }


  template <typename T>
  inline Lazy<T> &Lazy<T>::operator=(const Lazy &other)
  {
    object = other.object;
    object_is_initialized.store(other.object_is_initialized.load());

    return *this;
  }


  template <typename T>
  inline Lazy<T> &Lazy<T>::operator=(Lazy &&other) noexcept
  {
    object = std::move(other.object);
    object_is_initialized.store(other.object_is_initialized.load());

    other.object_is_initialized.store(false);
    other.object.reset();

    return *this;
  }


  template <typename T>
  inline void Lazy<T>::reset() noexcept
  {
    object_is_initialized.store(false);
    object.reset();
  }


  template <typename T>
  template <typename Callable>
  inline DEAL_II_ALWAYS_INLINE void
  Lazy<T>::ensure_initialized(const Callable &creator) const
  {
    // Check the object_is_initialized atomic with "acquire" semantics.
    if (!object_is_initialized.load(std::memory_order_acquire))
#ifdef DEAL_II_HAVE_CXX20
        [[unlikely]]
#endif
    {
      std::lock_guard<std::mutex> lock(initialization_mutex);

      //
      // Check again. If this thread won the race to the lock then we
      // initialize the object. Otherwise another thread has already
      // initialized the object and flipped the object_is_initialized
      // bit. (Here, the initialization_mutex ensures consistent ordering
      // with a memory fence, so we will observe the updated bool without
      // acquire semantics.)
      //
      if (!object_is_initialized.load(std::memory_order_relaxed)) {
        Assert(object.has_value() == false, dealii::ExcInternalError());
        object.emplace(std::move(creator()));

        // Flip the object_is_initialized boolean with "release" semantics.
        object_is_initialized.store(true, std::memory_order_release);
      }
    }
  }


  template <typename T>
  inline DEAL_II_ALWAYS_INLINE bool Lazy<T>::has_value() const
  {
    return (object_is_initialized && object.has_value());
  }


  template <typename T>
  inline DEAL_II_ALWAYS_INLINE const T &Lazy<T>::value() const
  {
    Assert(object_is_initialized && object.has_value(),
           dealii::ExcMessage(
               "value() has been called but the contained object has not been "
               "initialized. Did you forget to call 'ensure_initialized()' "
               "first?"));

    return object.value();
  }


  template <typename T>
  inline DEAL_II_ALWAYS_INLINE T &Lazy<T>::value()
  {
    Assert(object_is_initialized && object.has_value(),
           dealii::ExcMessage(
               "value() has been called but the contained object has not been "
               "initialized. Did you forget to call 'ensure_initialized()' "
               "first?"));

    return object.value();
  }

} // namespace ryujin
