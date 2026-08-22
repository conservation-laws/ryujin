//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>

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
   *   void refresh_direct_interface();
   * ```
   * allocate_storage() and deep_copy_storage() must be const and may only
   * modify `mutable` members: they are reachable from a const view()
   * under the TransferPolicy::implicit_transfers policy. In addition,
   * deallocate_storage<Host>() must also release the inherited
   * direct-access (host) view subobject that all derived classes carry.
   * Conversely, refresh_direct_interface() re-attaches the inherited
   * direct-access view; it is called by move_to_memory_space<Host>()
   * whenever host residency changes.
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
   * @note After a *const* implicit copy to the host memory space the
   * inherited direct-access interface of the derived class remains
   * detached (the base-class view subobject cannot be modified under
   * const). Access the data through the view returned by view<Host>()
   * instead; a non-const move_to_memory_space<Host>() or reinit()
   * re-attaches the direct interface.
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

      const bool needs_copy = !is_resident<MemorySpace>();
      if (needs_copy) {
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
      const bool needs_deallocate = residency_flag<OtherSpace>();
      if (needs_deallocate) {
        derived().template deallocate_storage<OtherSpace>();
        residency_flag<OtherSpace>() = false;
      }

      /*
       * Re-attach the direct-access (host) interface of the derived class
       * whenever the residency state changed. Note that a preceding const
       * copy_to_memory_space<HostSpace>() leaves the direct interface
       * detached, which we repair here as well (in this case needs_copy is
       * false but needs_deallocate is true).
       */
      if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
        if (needs_copy || needs_deallocate)
          derived().refresh_direct_interface();
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

#endif
} // namespace ryujin
