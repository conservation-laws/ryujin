//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>

#include <string>
#include <type_traits>

namespace ryujin
{
  /**
   * A convenience wrapper around MirroredStorage that maintains a single
   * (trivially copyable) object of type @p T mirrored between the host and
   * default (device) memory spaces.
   *
   * This allows to encapsulate a "POD style" payload (such as the runtime
   * parameters of a class) in a single structure that is moved into device
   * memory once - instead of copying individual parameters field by field
   * into a "view" object that is then captured by value in a computation
   * loop.
   *
   * Intended usage, here with the TransferPolicy::implicit_transfers
   * policy so that the payload is transferred on demand:
   * ```
   * class HyperbolicSystem {
   * public:
   *   struct Parameters { double gamma; double gamma_inverse; };
   *
   *   HyperbolicSystem()
   *     : parameters_("hyperbolic system parameters",
   *                   TransferPolicy::implicit_transfers)
   *   {
   *     auto *parameters = parameters_.view();
   *     parameters->gamma = 1.4;
   *     add_parameter("gamma", parameters->gamma, "...");
   *     // ... and call update_parameters() whenever "gamma" changes.
   *   }
   *
   *   void update_parameters()
   *   {
   *     // A writable view drops the (now stale) mirror in the default
   *     // memory space:
   *     auto *parameters = parameters_.view();
   *     parameters->gamma_inverse = 1. / parameters->gamma;
   *   }
   *
   *   template <typename MemorySpace>
   *   class View {
   *   public:
   *     View(const HyperbolicSystem &hyperbolic_system)
   *         // ... which is copied over again here:
   *         : parameters_(
   *               hyperbolic_system.parameters_.template view<MemorySpace>())
   *     {
   *     }
   *
   *     DEAL_II_HOST_DEVICE double gamma() const { return parameters_->gamma; }
   *
   *   private:
   *     const Parameters *parameters_;
   *   };
   *
   *   template <typename MemorySpace>
   *   View<MemorySpace> view() const { return View<MemorySpace>(*this); }
   *
   * private:
   *   Mirrored<Parameters> parameters_;
   * };
   * ```
   *
   * @note The transfer policy defaults to
   * TransferPolicy::explicit_transfers, where view() never triggers a
   * memory transfer on its own and all transfers have to be requested
   * manually. Keep in mind that copy_to_memory_space() is a no-op if the
   * selected memory space is already resident: updating the payload under
   * that policy requires a move_to_memory_space<Host>() before the write,
   * and a copy_to_memory_space<Default>() afterwards.
   *
   * @note The pointer returned by view() may only be dereferenced on the
   * selected memory space.
   *
   * @note In contrast to a copied Kokkos::View the pointer returned by
   * view() does not pin the underlying storage: it dangles after a
   * move_to_memory_space() away from the corresponding memory space - such
   * as the implicit one performed by a writable view() under the
   * implicit_transfers policy. Recreate the View objects of the example
   * above after every parameter update.
   *
   * @note An object of this class allocates memory with Kokkos. It thus has
   * to be created after Kokkos has been initialized, which is done by the
   * dealii::Utilities::MPI::MPI_InitFinalize constructor.
   *
   * @ingroup GPU
   */
  template <typename T>
  class Mirrored : public MirroredStorage<Mirrored<T>>
  {
  public:
    static_assert(std::is_trivially_copyable_v<T>,
                  "The stored type has to be trivially copyable so that we "
                  "can move it into device memory");

    /**
     * Constructor. Allocates the storage in the host memory space. The
     * @p label is used as the Kokkos allocation label and
     * @p transfer_policy selects the transfer policy, see the
     * documentation of TransferPolicy.
     */
    Mirrored(const std::string &label = "mirrored object",
             const TransferPolicy transfer_policy =
                 TransferPolicy::explicit_transfers);

    /**
     * Return a writable pointer to the stored object residing in the
     * selected memory space.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    T *view();

    /**
     * Return a read only pointer to the stored object residing in the
     * selected memory space.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    const T *view() const;

  private:
    /**
     * @name Internal fields, methods, and friends
     */
    //@{

    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    template <typename MemorySpace>
    using KokkosView = Kokkos::View<T, typename MemorySpace::kokkos_space>;

    /**
     * Return the storage of the selected memory space.
     */
    template <typename MemorySpace>
    KokkosView<MemorySpace> &storage() const;

    /*
     * Storage primitives required by the MirroredStorage base class:
     */

    template <typename MemorySpace>
    void allocate_storage() const;

    template <typename To, typename From>
    void deep_copy_storage() const;

    template <typename MemorySpace>
    void deallocate_storage();

    void refresh_direct_interface();

    std::string label_;

    mutable KokkosView<HostSpace> host_;
    mutable KokkosView<DefaultSpace> default_;

    friend class MirroredStorage<Mirrored<T>>;

    //@}
  };


#ifndef DOXYGEN
  /*
   * -------------------------------------------------------------------------
   * Inline function definitions
   * -------------------------------------------------------------------------
   */


  template <typename T>
  Mirrored<T>::Mirrored(const std::string &label,
                        const TransferPolicy transfer_policy)
      : label_(label)
  {
    this->set_transfer_policy(transfer_policy);

    allocate_storage<HostSpace>();
    this->reset_residency(/*host*/ true, /*default*/ false);
  }


  template <typename T>
  template <typename MemorySpace>
  inline T *Mirrored<T>::view()
  {
    this->template prepare_write_access<MemorySpace>();

    /*
     * Note: The returned pointer must only be dereferenced on the selected
     * memory space.
     *
     * Note: If the host and default memory spaces coincide then only the
     * host storage is ever allocated.
     */
    if constexpr (have_separate_memory_spaces &&
                  !std::is_same_v<MemorySpace, HostSpace>) {
      return default_.data();
    } else {
      return host_.data();
    }
  }


  template <typename T>
  template <typename MemorySpace>
  inline const T *Mirrored<T>::view() const
  {
    this->template prepare_read_access<MemorySpace>();

    /* See the comment in the writable view() variant above. */
    if constexpr (have_separate_memory_spaces &&
                  !std::is_same_v<MemorySpace, HostSpace>) {
      return default_.data();
    } else {
      return host_.data();
    }
  }


  template <typename T>
  template <typename MemorySpace>
  inline auto Mirrored<T>::storage() const -> KokkosView<MemorySpace> &
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (std::is_same_v<MemorySpace, HostSpace>)
      return host_;
    else
      return default_;
  }


  template <typename T>
  template <typename MemorySpace>
  inline void Mirrored<T>::allocate_storage() const
  {
    storage<MemorySpace>() = KokkosView<MemorySpace>(label_);
  }


  template <typename T>
  template <typename To, typename From>
  inline void Mirrored<T>::deep_copy_storage() const
  {
    Kokkos::deep_copy(/*dst*/ storage<To>(), /*src*/ storage<From>());
  }


  template <typename T>
  template <typename MemorySpace>
  inline void Mirrored<T>::deallocate_storage()
  {
    storage<MemorySpace>() = KokkosView<MemorySpace>();
  }


  template <typename T>
  inline void Mirrored<T>::refresh_direct_interface()
  {
    // do nothing
  }

#endif
} // namespace ryujin
