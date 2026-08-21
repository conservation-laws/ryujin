//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/config.h>
#include <deal.II/base/memory_space.h>

#include <string>
#include <type_traits>

namespace ryujin
{
  /**
   * A small helper class that stores a (trivially copyable) object of type
   * @p T in host memory and maintains a mirror of it in the default memory
   * space.
   *
   * This allows to encapsulate a "POD style" payload (such as runtime
   * parameters of a class) in a single structure that is moved into device
   * memory once - instead of copying individual parameters field by field
   * into a "view" object that is then captured by value in a computation
   * loop.
   *
   * Intended usage:
   * ```
   * class HyperbolicSystem {
   * public:
   *   struct Parameters { double gamma; double gamma_inverse; };
   *
   *   HyperbolicSystem()
   *     : parameters_("label")
   *   {
   *     auto &parameters = parameters_.value();
   *     parameters.gamma = 42.;
   *     add_parameter("gamma", parameters.gamma, "...");
   *     // ... and call update_parameters() whenever "gamma" changes.
   *   }
   *
   *   void update_parameters()
   *   {
   *     auto &parameters = parameters_.value();
   *     parameters.gamma_inverse = 1. / parameters.gamma;
   *     parameters_.update();
   *   }
   *
   * private:
   *   Mirrored<Parameters> parameters_;
   * };
   * ```
   *
   * @note An object of this class allocates memory with Kokkos. It thus
   * has to be created after Kokkos has been initialized, which is done by
   * the dealii::Utilities::MPI::MPI_InitFinalize constructor.
   *
   * @ingroup Miscellaneous
   */
  template <typename T>
  class Mirrored
  {
  public:
    static_assert(std::is_trivially_copyable_v<T>,
                  "The stored type has to be trivially copyable so that we "
                  "can move it into device memory");

    /**
     * A read only view of the stored object residing in the selected
     * memory space.
     *
     * @note The view is unmanaged on purpose: copying a managed
     * Kokkos::View updates a reference count with an atomic operation,
     * whereas copying an unmanaged view is a plain pointer copy. This
     * matters because "view" classes holding on to such a Mirrored::View
     * are routinely created within loop bodies. The view is only valid as
     * long as the underlying Mirrored object is alive.
     */
    template <typename MemorySpace>
    using View = Kokkos::View<const T,
                              typename MemorySpace::kokkos_space,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    /**
     * Constructor. Allocates the host and default memory space storage.
     * The @p label is used as the Kokkos allocation label.
     */
    Mirrored(const std::string &label);

    /**
     * Return a (read and write) reference to the object stored in host
     * memory. Call update() after modifying the object in order to
     * synchronize the mirror in the default memory space.
     */
    T &value();

    /**
     * Return a read only reference to the object stored in host memory.
     */
    const T &value() const;

    /**
     * Copy the object stored in host memory over to the default memory
     * space.
     */
    void update();

    /**
     * Return a (read only) view on the stored object for the selected
     * memory space.
     */
    template <typename MemorySpace>
    View<MemorySpace> get_view() const;

  private:
    /**
     * @name Internal data
     */
    //@{

    using KokkosHost = dealii::MemorySpace::Host::kokkos_space;
    using KokkosDefault = dealii::MemorySpace::Default::kokkos_space;

    Kokkos::View<T, KokkosHost> host_;
    Kokkos::View<T, KokkosDefault> default_;

    //@}
  };


#ifndef DOXYGEN
  /*
   * -------------------------------------------------------------------------
   * Inline function definitions
   * -------------------------------------------------------------------------
   */


  template <typename T>
  Mirrored<T>::Mirrored(const std::string &label)
  {
    host_ = Kokkos::View<T, KokkosHost>(label);

    /*
     * Note: If the host and default memory space happen to be the same
     * then create_mirror_view() simply returns the host view and no
     * additional memory is allocated.
     */
    default_ = Kokkos::create_mirror_view(
        typename KokkosDefault::execution_space(), host_);
  }


  template <typename T>
  inline T &Mirrored<T>::value()
  {
    return host_();
  }


  template <typename T>
  inline const T &Mirrored<T>::value() const
  {
    return host_();
  }


  template <typename T>
  inline void Mirrored<T>::update()
  {
    /*
     * Note: deep_copy() is a no-op if both views reference the same
     * allocation.
     */
    Kokkos::deep_copy(/*dst*/ default_, /*src*/ host_);
  }


  template <typename T>
  template <typename MemorySpace>
  inline auto Mirrored<T>::get_view() const -> View<MemorySpace>
  {
    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      return host_;
    } else {
      return default_;
    }
  }
#endif

} // namespace ryujin
