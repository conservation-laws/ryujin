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
   * A constexpr boolean that is true if the host and default (device)
   * memory spaces are distinct. If ryujin is configured without device
   * support then dealii::MemorySpace::Default coincides with
   * dealii::MemorySpace::Host and no memory transfers are necessary.
   *
   * @ingroup GPU
   */
  inline constexpr bool have_separate_memory_spaces =
      !std::is_same_v<dealii::MemorySpace::Host::kokkos_space,
                      dealii::MemorySpace::Default::kokkos_space>;


  /**
   * A template alias that maps a given memory space to the respective
   * other one: dealii::MemorySpace::Host maps to
   * dealii::MemorySpace::Default and vice versa.
   *
   * @ingroup GPU
   */
  template <typename MemorySpace>
  using other_space_t =
      std::conditional_t<std::is_same_v<MemorySpace, dealii::MemorySpace::Host>,
                         dealii::MemorySpace::Default,
                         dealii::MemorySpace::Host>;


  /**
   * A policy describing how host/device memory transfers are performed
   * when accessing a MirroredStorage object via view().
   *
   * @ingroup GPU
   */
  enum class TransferPolicy {
    /**
     * view<MemorySpace>() asserts that the selected memory space is
     * resident; all transfers must be requested manually via
     * copy_to_memory_space() and move_to_memory_space().
     */
    explicit_transfers,
    /**
     * Requesting a read-only view triggers an implicit
     * copy_to_memory_space() if the selected memory space is not resident
     * (both memory spaces remain resident afterwards).
     *
     * Requesting a writable view triggers an implicit
     * move_to_memory_space(): the data is copied over if necessary and the
     * other memory space is deallocated and marked non-resident. Writable
     * access invalidates the stale mirror.
     */
    implicit_transfers,
  };


  /**
   * A CRTP base class that manages the residency state of data mirrored
   * between the host and default (device) memory spaces. It provides a
   * unified interface for querying residency (is_resident()) and for
   * transferring data between memory spaces (copy_to_memory_space() and
   * move_to_memory_space()) that is used by the SparsityPattern,
   * SparseMatrix, and MultiComponentVector classes.
   *
   * Data can be resident on either memory space, or on both (after a
   * copy_to_memory_space() operation).
   *
   * The @p Derived class must provide the following (private) primitives
   * and declare `friend class MirroredStorage<Derived>;`:
   * ```
   *   template <typename MemorySpace> void allocate_storage() const;
   *   template <typename To, typename From> void deep_copy_storage() const;
   *   template <typename MemorySpace> void deallocate_storage();
   * ```
   * allocate_storage() and deep_copy_storage() must be const and may only
   * modify `mutable` members: they are reachable from a const view()
   * under the TransferPolicy::implicit_transfers policy.
   *
   * @note Under TransferPolicy::implicit_transfers a const object mutates
   * internal (mutable) state when a view is requested for a non-resident
   * memory space. Concurrent access is thus not thread safe.
   *
   * @note Views handed out by view() pin the underlying (reference
   * counted) storage of the corresponding memory space; they dangle after
   * a move_to_memory_space() away from that memory space. Managing view
   * lifetime is the caller's responsibility.
   *
   * @ingroup GPU
   */
  template <typename Derived>
  class MirroredStorage
  {
  public:
    /**
     * @name Memory space access and synchronization
     */
    //@{

    /**
     * Returns true if data is available on the selected memory space,
     * i.e., if view<MemorySpace>() may be called. Both memory spaces
     * can be resident simultaneously. If the host and default memory
     * spaces coincide the function returns true for both memory spaces
     * (once the object has been initialized).
     */
    template <typename MemorySpace>
    bool is_resident() const;

    /**
     * Make a deep copy of the data to the selected memory space. Both
     * memory spaces remain valid/resident.
     *
     * @note The copy is skipped if the selected memory space is already
     * resident. No consistency check is performed, whether both memory
     * spaces hold the same data.
     */
    template <typename MemorySpace>
    void copy_to_memory_space() const;

    /**
     * Move the data to the selected memory space, then deallocate the
     * storage of the other memory space and mark it non-resident.
     *
     * @note The copy is skipped if the selected memory space is already
     * resident. No consistency check is performed, whether both memory
     * spaces hold the same data.
     */
    template <typename MemorySpace>
    void move_to_memory_space();

    /**
     * Return the currently selected transfer policy.
     */
    TransferPolicy transfer_policy() const;

    /**
     * Select a transfer policy, see the documentation of TransferPolicy.
     */
    void set_transfer_policy(const TransferPolicy transfer_policy);

  protected:
    //@}
    /**
     * @name Internal methods used by derived classes
     */
    //@{

    MirroredStorage() = default;

    /**
     * Prepare read access on the selected memory space.
     *
     * This function is called by the derived class at the beginning of
     * every (const) view() method returning a read-only view: Under the
     * implicit_transfers policy the function performs an implicit
     * copy_to_memory_space(), otherwise it asserts that the selected
     * memory space is resident.
     */
    template <typename MemorySpace>
    void prepare_read_access() const;

    /**
     * Prepare write access on the selected memory space.
     *
     * This function is called by the derived class at the beginning of
     * every view() method returning a writable view: Under the
     * implicit_transfers policy the function performs an implicit
     * move_to_memory_space() (invalidating a stale mirror on the other
     * memory space), otherwise it asserts that the selected memory space
     * is resident.
     */
    template <typename MemorySpace>
    void prepare_write_access();

    /**
     * Reset the residency flags. This function is called by the derived
     * class at the end of reinit().
     */
    void reset_residency(const bool host_resident, const bool default_resident);

    //@}

  private:
    /**
     * @name Internal fields and methods
     */
    //@{

    using HostSpace = dealii::MemorySpace::Host;
    using DefaultSpace = dealii::MemorySpace::Default;

    template <typename MemorySpace>
    bool &residency_flag() const;

    Derived &derived();
    const Derived &derived() const;

    mutable bool host_resident_ = false;
    mutable bool default_resident_ = false;

    TransferPolicy transfer_policy_ = TransferPolicy::explicit_transfers;

    //@}
  };


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


  template <typename Derived>
  template <typename MemorySpace>
  inline bool MirroredStorage<Derived>::is_resident() const
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (!have_separate_memory_spaces)
      return host_resident_ || default_resident_;

    return residency_flag<MemorySpace>();
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::copy_to_memory_space() const
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if (is_resident<MemorySpace>())
      return;

    if constexpr (have_separate_memory_spaces) {
      using OtherSpace = other_space_t<MemorySpace>;
      Assert(is_resident<OtherSpace>(),
             dealii::ExcMessage(
                 "Unable to copy to the chosen memory space: the object has "
                 "not been properly initialized."));

      derived().template allocate_storage<MemorySpace>();
      derived().template deep_copy_storage<MemorySpace, OtherSpace>();
      residency_flag<MemorySpace>() = true;

    } else {
      Assert(false,
             dealii::ExcMessage(
                 "Unable to copy to the chosen memory space: the object has "
                 "not been properly initialized."));
    }
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::move_to_memory_space()
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (!have_separate_memory_spaces) {
      Assert(is_resident<MemorySpace>(),
             dealii::ExcMessage(
                 "Unable to move to the chosen memory space: the object has "
                 "not been properly initialized."));
      return;

    } else {
      using OtherSpace = other_space_t<MemorySpace>;

      if (!is_resident<MemorySpace>()) {
        Assert(is_resident<OtherSpace>(),
               dealii::ExcMessage(
                   "Unable to move to the chosen memory space: the object has "
                   "not been properly initialized."));

        derived().template allocate_storage<MemorySpace>();
        derived().template deep_copy_storage<MemorySpace, OtherSpace>();
        residency_flag<MemorySpace>() = true;
      }

      /*
       * Note: If both memory spaces happen to be resident we skip the deep
       * copy above and simply deallocate the other memory space. This is
       * the "writable access invalidates the stale mirror" primitive used
       * by prepare_write_access().
       */
      if (residency_flag<OtherSpace>()) {
        derived().template deallocate_storage<OtherSpace>();
        residency_flag<OtherSpace>() = false;
      }
    }
  }


  template <typename Derived>
  inline TransferPolicy MirroredStorage<Derived>::transfer_policy() const
  {
    return transfer_policy_;
  }


  template <typename Derived>
  inline void MirroredStorage<Derived>::set_transfer_policy(
      const TransferPolicy transfer_policy)
  {
    transfer_policy_ = transfer_policy;
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::prepare_read_access() const
  {
    if (transfer_policy_ == TransferPolicy::implicit_transfers) {
      copy_to_memory_space<MemorySpace>();

    } else {
      Assert(is_resident<MemorySpace>(),
             dealii::ExcMessage(
                 "The chosen memory space is not resident. Either call "
                 "copy_to_memory_space() / move_to_memory_space() prior to "
                 "requesting a view, or select the "
                 "TransferPolicy::implicit_transfers policy."));
    }
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::prepare_write_access()
  {
    if (transfer_policy_ == TransferPolicy::implicit_transfers) {
      move_to_memory_space<MemorySpace>();

    } else {
      Assert(is_resident<MemorySpace>(),
             dealii::ExcMessage(
                 "The chosen memory space is not resident. Either call "
                 "copy_to_memory_space() / move_to_memory_space() prior to "
                 "requesting a view, or select the "
                 "TransferPolicy::implicit_transfers policy."));
    }
  }


  template <typename Derived>
  inline void
  MirroredStorage<Derived>::reset_residency(const bool host_resident,
                                            const bool default_resident)
  {
    host_resident_ = host_resident;
    default_resident_ = default_resident;
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline bool &MirroredStorage<Derived>::residency_flag() const
  {
    if constexpr (std::is_same_v<MemorySpace, HostSpace>)
      return host_resident_;
    else
      return default_resident_;
  }


  template <typename Derived>
  inline Derived &MirroredStorage<Derived>::derived()
  {
    return static_cast<Derived &>(*this);
  }


  template <typename Derived>
  inline const Derived &MirroredStorage<Derived>::derived() const
  {
    return static_cast<const Derived &>(*this);
  }


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


#endif
} // namespace ryujin
