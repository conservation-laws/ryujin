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
    /**
     * The same policy as implicit_transfers, but with the host memory
     * space "pinned": operations that would deallocate the host storage,
     * i.e., move_to_memory_space<dealii::MemorySpace::Default>() and
     * requesting a writable view on the default memory space, are
     * disallowed. The data thus always remains resident on the host memory
     * space and pointers (and views) into the host storage remain valid.
     *
     * If the host and default memory spaces coincide the restriction is
     * lifted: there is only a single allocation that no transfer can
     * deallocate.
     */
    implicit_transfers_host_resident,
    /**
     * The converse of implicit_transfers_host_resident: the default
     * (device) memory space is pinned and
     * move_to_memory_space<dealii::MemorySpace::Host>() as well as
     * requesting a writable view on the host memory space are disallowed.
     *
     * If the host and default memory spaces coincide the restriction is
     * lifted.
     */
    implicit_transfers_default_resident,
  };


  /**
   * Return true if the given transfer policy performs implicit memory
   * transfers when a view is requested, i.e., for all implicit_transfers*
   * policies.
   *
   * @ingroup GPU
   */
  inline constexpr bool performs_implicit_transfers(const TransferPolicy policy)
  {
    return policy == TransferPolicy::implicit_transfers ||
           policy == TransferPolicy::implicit_transfers_host_resident ||
           policy == TransferPolicy::implicit_transfers_default_resident;
  }


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
   * under an implicit transfer policy.
   *
   * @note Under an implicit transfer policy a const object mutates
   * internal (mutable) state when a view is requested for a non-resident
   * memory space. Concurrent access is thus not thread safe.
   *
   * @note move_to_memory_space() invalidates all references and pointers
   * to data on the memory space we move away from. This invalidates any
   * existing View on that memory space. Managing view lifetime is the
   * caller's responsibility.
   *
   * @note if host and device memory spaces coincide, then the following
   * special rules apply: the residency flags for host and default space
   * must coincide: is_resident<>() always returns true for a properly
   * initialized object. is_pinned<>() always returns false.
   * move_to_memory_space<>() and copy_to_memory_space<>() assert that both
   * memory spaces are resident and simply return.
   *
   * @note A memory space pinned by the current transfer policy has to be
   * resident, see is_pinned(). The @p Derived class thus has to initialize
   * an object in the following order: first put the data in place, then
   * call reset_residency(), and only then select the transfer policy with
   * set_transfer_policy().
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
     * Returns true if the selected memory space is "pinned" by the current
     * transfer policy, i.e., if the data has to remain resident on it and
     * all operations that would deallocate its storage - a
     * move_to_memory_space() to the other memory space and a writable view
     * on the other memory space - are disallowed.
     *
     * The function always returns false if the host and default memory
     * spaces coincide.
     */
    template <typename MemorySpace>
    bool is_pinned() const;

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
     *
     * @note The operation is disallowed if the other memory space is
     * pinned by the current transfer policy, see is_pinned().
     */
    template <typename MemorySpace>
    void move_to_memory_space();

    /**
     * Return the currently selected transfer policy.
     */
    TransferPolicy transfer_policy() const;

    /**
     * Select a transfer policy, see the documentation of TransferPolicy.
     *
     * @note The function asserts that a memory space pinned by the selected
     * policy is resident. Transfer the data to the memory space that is
     * about to be pinned prior to calling this function.
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
     * every (const) view() method returning a read-only view: Under an
     * implicit transfer policy the function performs an implicit
     * copy_to_memory_space(), otherwise it asserts that the selected
     * memory space is resident.
     */
    template <typename MemorySpace>
    void prepare_read_access() const;

    /**
     * Prepare write access on the selected memory space.
     *
     * This function is called by the derived class at the beginning of
     * every view() method returning a writable view: Under an implicit
     * transfer policy the function performs an implicit
     * move_to_memory_space() (invalidating a stale mirror on the other
     * memory space), otherwise it asserts that the selected memory space
     * is resident. The operation is disallowed if the other memory space
     * is pinned by the current transfer policy, see is_pinned().
     */
    template <typename MemorySpace>
    void prepare_write_access();

    /**
     * Reset the residency flags. This function is called by the derived
     * class at the end of reinit(), prior to selecting the transfer policy
     * with set_transfer_policy().
     *
     * @note If the host and default memory spaces coincide both flags have
     * to coincide as well. The function further asserts that a memory space
     * pinned by the current transfer policy remains resident.
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
   * A convenience wrapper around MirroredStorage that maintains a
   * (trivially copyable) payload mirrored between the host and default
   * (device) memory spaces.
   *
   * This allows to encapsulate a "POD style" payload (such as the runtime
   * parameters of a class) in a single structure that is moved into device
   * memory once - instead of copying individual parameters field by field
   * into a "view" object that is then captured by value in a computation
   * loop.
   *
   * The class operates in one of two modes that are selected by the
   * template argument:
   *  - Mirrored<T> maintains a single object of type @p T.
   *  - Mirrored<T *> maintains a one dimensional array of objects of type
   *    @p T. The length is set in the constructor, or with reinit() and
   *    queried with size().
   *
   * Intended usage of the scalar mode, here with the
   * TransferPolicy::implicit_transfers_host_resident policy so that the
   * payload is transferred on demand and the host storage - into which the
   * runtime parameters are bound - is never deallocated:
   * ```
   * class HyperbolicSystem {
   * public:
   *   struct Parameters { double gamma; double gamma_inverse; };
   *
   *   HyperbolicSystem()
   *     : parameters_("hyperbolic system parameters",
   *                   TransferPolicy::implicit_transfers_host_resident)
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
   * The array mode uses the same raw pointer interface, but the storage is
   * sized explicitly with reinit():
   * ```
   * class Stencil {
   * public:
   *   struct Entry { unsigned int index; double weight; };
   *
   *   Stencil() : entries_("stencil entries") {}
   *
   *   void prepare(const std::size_t n_entries)
   *   {
   *     entries_.reinit(n_entries, TransferPolicy::implicit_transfers);
   *
   *     auto *entries = entries_.view();
   *     for (std::size_t i = 0; i < entries_.size(); ++i)
   *       entries[i] = {static_cast<unsigned int>(i), 1.};
   *   }
   *
   * private:
   *   Mirrored<Entry *> entries_;
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
   * selected memory space. In the array mode the pointer references the
   * first of size() contiguously stored objects; the length is not part of
   * the returned pointer and has to be carried along by the caller.
   *
   * @note In contrast to a copied Kokkos::View the pointer returned by
   * view() does not pin the underlying storage: it dangles after a
   * move_to_memory_space() away from the corresponding memory space - such
   * as the implicit one performed by a writable view() under an implicit
   * transfer policy. The
   * TransferPolicy::implicit_transfers_host_resident policy used in the
   * example above guarantees that the pointer into the host storage
   * remains valid, but the device mirror is still dropped by every
   * writable host view(): Recreate the View objects of the example above
   * after every parameter update.
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
    /**
     * True if the class maintains a one dimensional array of objects, i.e.,
     * if the template argument is of the form `T *`, and false if the class
     * maintains a single object of type `T`.
     */
    static constexpr bool is_array = std::is_pointer_v<T>;

    /**
     * The type of the stored object: `T` itself in the scalar mode, and the
     * pointee type in the array mode.
     */
    using value_type = std::remove_pointer_t<T>;

    static_assert(std::is_trivially_copyable_v<value_type>,
                  "The stored type has to be trivially copyable so that we "
                  "can move it into device memory");

    static_assert(!std::is_pointer_v<value_type>,
                  "Only a single level of indirection is supported: a "
                  "Mirrored<T *> object maintains a one dimensional array of "
                  "objects of type T");

    static_assert(!std::is_array_v<T>,
                  "An array of objects is spelled Mirrored<T *>, and not "
                  "Mirrored<T[]>");

    /**
     * Constructor for the scalar mode. Allocates the storage of a single
     * object on the selected memory space. The @p label is used as the
     * Kokkos allocation label, @p transfer_policy selects the transfer
     * policy, see the documentation of TransferPolicy, and the (defaulted)
     * last argument selects the memory space that is initialized.
     *
     * @note The selected memory space has to be consistent with the
     * transfer policy: a policy that pins the other memory space is not
     * admissible, see MirroredStorage.
     */
    template <typename InitializedMemorySpace = dealii::MemorySpace::Host>
    Mirrored(const std::string &label = "mirrored object",
             const TransferPolicy transfer_policy =
                 TransferPolicy::explicit_transfers,
             InitializedMemorySpace = {})
      requires(!is_array);

    /**
     * Constructor for the array mode. Allocates the storage of a single
     * object on the selected memory space. The @p label is used as the
     * Kokkos allocation label, @p transfer_policy selects the transfer
     * policy, see the documentation of TransferPolicy, and the (defaulted)
     * last argument selects the memory space that is initialized.
     *
     * @note The selected memory space has to be consistent with the
     * transfer policy: a policy that pins the other memory space is not
     * admissible, see MirroredStorage.
     */
    template <typename InitializedMemorySpace = dealii::MemorySpace::Host>
    Mirrored(const std::string &label = "mirrored array",
             const std::size_t size = 0,
             const TransferPolicy transfer_policy =
                 TransferPolicy::explicit_transfers,
             InitializedMemorySpace = {})
      requires(is_array);

    /**
     * Reinitialize the object to maintain @p size objects. The storage is
     * (re)allocated (and zero initialized) on the selected default memory
     * space.
     */
    template <typename InitializedMemorySpace = dealii::MemorySpace::Host>
    void reinit(const std::size_t size,
                const TransferPolicy transfer_policy =
                    TransferPolicy::explicit_transfers,
                InitializedMemorySpace = {})
      requires(is_array);

    /**
     * Return the number of stored objects.
     */
    std::size_t size() const
      requires(is_array);

    /**
     * Return a writable pointer to the stored object residing in the
     * selected memory space. In the array mode the pointer references the
     * first of size() contiguously stored objects.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    value_type *view();

    /**
     * Return a read only pointer to the stored object residing in the
     * selected memory space. In the array mode the pointer references the
     * first of size() contiguously stored objects.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    const value_type *view() const;

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

    /**
     * Allocate the storage on the selected memory space, drop the storage
     * of the other memory space, and reset the residency flags
     * accordingly.
     */
    template <typename InitializedMemorySpace>
    void initialize_storage();

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

    /* The number of stored objects, only used in the array mode: */
    std::size_t size_ = 0;

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

    if constexpr (!have_separate_memory_spaces) {
      Assert(host_resident_ == default_resident_,
             dealii::ExcMessage(
                 "Internal error: the host and default memory spaces coincide "
                 "but the two residency flags do not. There is only a single "
                 "allocation, so the object has to be resident on both memory "
                 "spaces, or on neither of them."));
      return host_resident_ || default_resident_;
    }

    return residency_flag<MemorySpace>();
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline bool MirroredStorage<Derived>::is_pinned() const
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (!have_separate_memory_spaces) {
      return false;
    }

    if constexpr (std::is_same_v<MemorySpace, HostSpace>) {
      return transfer_policy_ ==
             TransferPolicy::implicit_transfers_host_resident;

    } else {
      return transfer_policy_ ==
             TransferPolicy::implicit_transfers_default_resident;
    }
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::copy_to_memory_space() const
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (!have_separate_memory_spaces) {
      /*
       * If both memory spaces coincide there is only a single allocation
       * that is resident on both of them: there is nothing to copy.
       */
      Assert(host_resident_ && default_resident_,
             dealii::ExcMessage(
                 "Unable to copy to the chosen memory space: the host and "
                 "default memory spaces coincide but the object is not "
                 "resident on them. The object has not been properly "
                 "initialized."));
      return;
    }

    Assert(!is_pinned<MemorySpace>() || is_resident<MemorySpace>(),
           dealii::ExcMessage(
               "Unable to copy to the chosen memory space: the selected "
               "transfer policy pins the chosen memory space but the object "
               "is not resident on it. The object has not been properly "
               "initialized."));

    if (is_resident<MemorySpace>())
      return;

    using OtherSpace = other_space_t<MemorySpace>;
    Assert(is_resident<OtherSpace>(),
           dealii::ExcMessage(
               "Unable to copy to the chosen memory space: the object has "
               "not been properly initialized."));

    derived().template allocate_storage<MemorySpace>();
    derived().template deep_copy_storage<MemorySpace, OtherSpace>();
    residency_flag<MemorySpace>() = true;
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::move_to_memory_space()
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    if constexpr (!have_separate_memory_spaces) {
      /*
       * If both memory spaces coincide there is only a single allocation
       * that no memory transfer can ever deallocate: there is nothing to
       * move.
       */
      Assert(host_resident_ && default_resident_,
             dealii::ExcMessage(
                 "Unable to move to the chosen memory space: the host and "
                 "default memory spaces coincide but the object is not "
                 "resident on them. The object has not been properly "
                 "initialized."));
      return;
    }

    Assert(!is_pinned<MemorySpace>() || is_resident<MemorySpace>(),
           dealii::ExcMessage(
               "Unable to move to the chosen memory space: the selected "
               "transfer policy pins the chosen memory space but the object "
               "is not resident on it. The object has not been properly "
               "initialized."));

    using OtherSpace = other_space_t<MemorySpace>;

    Assert(!is_pinned<OtherSpace>(),
           dealii::ExcMessage(
               "Unable to move to the chosen memory space: the selected "
               "transfer policy requires the data to remain resident on "
               "the other memory space."));

    if (!is_resident<MemorySpace>()) {
      Assert(is_resident<OtherSpace>(),
             dealii::ExcMessage(
                 "Unable to move to the chosen memory space: the object has "
                 "not been properly initialized."));

      derived().template allocate_storage<MemorySpace>();
      derived().template deep_copy_storage<MemorySpace, OtherSpace>();
      residency_flag<MemorySpace>() = true;
    }

    if (residency_flag<OtherSpace>()) {
      derived().template deallocate_storage<OtherSpace>();
      residency_flag<OtherSpace>() = false;
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

    Assert(!is_pinned<HostSpace>() || is_resident<HostSpace>(),
           dealii::ExcMessage(
               "Unable to select the given transfer policy: the policy pins "
               "the host memory space but the object is not resident on it. "
               "Transfer the data to the pinned memory space before selecting "
               "the transfer policy."));

    Assert(!is_pinned<DefaultSpace>() || is_resident<DefaultSpace>(),
           dealii::ExcMessage(
               "Unable to select the given transfer policy: the policy pins "
               "the default memory space but the object is not resident on "
               "it. Transfer the data to the pinned memory space before "
               "selecting the transfer policy."));
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::prepare_read_access() const
  {
    if (performs_implicit_transfers(transfer_policy_)) {
      copy_to_memory_space<MemorySpace>();

    } else {
      Assert(is_resident<MemorySpace>(),
             dealii::ExcMessage(
                 "The chosen memory space is not resident. Either call "
                 "copy_to_memory_space() / move_to_memory_space() prior to "
                 "requesting a view, or select one of the "
                 "TransferPolicy::implicit_transfers* policies."));
    }
  }


  template <typename Derived>
  template <typename MemorySpace>
  inline void MirroredStorage<Derived>::prepare_write_access()
  {
    if (performs_implicit_transfers(transfer_policy_)) {
      Assert(!is_pinned<other_space_t<MemorySpace>>(),
             dealii::ExcMessage(
                 "Unable to request a writable view on the chosen memory "
                 "space: the selected transfer policy requires the data to "
                 "remain resident on the other memory space."));

      move_to_memory_space<MemorySpace>();

    } else {
      Assert(is_resident<MemorySpace>(),
             dealii::ExcMessage(
                 "The chosen memory space is not resident. Either call "
                 "copy_to_memory_space() / move_to_memory_space() prior to "
                 "requesting a view, or select one of the "
                 "TransferPolicy::implicit_transfers* policies."));
    }
  }


  template <typename Derived>
  inline void
  MirroredStorage<Derived>::reset_residency(const bool host_resident,
                                            const bool default_resident)
  {
    if constexpr (!have_separate_memory_spaces) {
      Assert(host_resident == default_resident,
             dealii::ExcMessage(
                 "Unable to reset the residency flags: the host and default "
                 "memory spaces coincide, so there is only a single "
                 "allocation and both flags have to coincide as well."));
    }

    host_resident_ = host_resident;
    default_resident_ = default_resident;

    Assert(!is_pinned<HostSpace>() || is_resident<HostSpace>(),
           dealii::ExcMessage(
               "Unable to reset the residency flags: the selected transfer "
               "policy pins the host memory space and requires the data to "
               "remain resident on it."));

    Assert(!is_pinned<DefaultSpace>() || is_resident<DefaultSpace>(),
           dealii::ExcMessage(
               "Unable to reset the residency flags: the selected transfer "
               "policy pins the default memory space and requires the data "
               "to remain resident on it."));
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
  template <typename InitializedMemorySpace>
  Mirrored<T>::Mirrored(const std::string &label,
                        const TransferPolicy transfer_policy,
                        InitializedMemorySpace)
    requires(!is_array)
      : label_(label)
  {
    initialize_storage<InitializedMemorySpace>();

    /* The transfer policy is selected last, see MirroredStorage: */
    this->set_transfer_policy(transfer_policy);
  }


  template <typename T>
  template <typename InitializedMemorySpace>
  Mirrored<T>::Mirrored(const std::string &label,
                        const std::size_t size,
                        const TransferPolicy transfer_policy,
                        InitializedMemorySpace)
    requires(is_array)
      : label_(label)
  {
    reinit(size, transfer_policy, InitializedMemorySpace{});
  }


  template <typename T>
  template <typename InitializedMemorySpace>
  void Mirrored<T>::reinit(const std::size_t size,
                           const TransferPolicy transfer_policy,
                           InitializedMemorySpace)
    requires(is_array)
  {
    /*
     * Drop the transfer policy for the duration of the reinit so that a
     * pinned memory space of a previous policy does not interfere:
     */
    this->set_transfer_policy(TransferPolicy::explicit_transfers);

    size_ = size;

    initialize_storage<InitializedMemorySpace>();

    /* The transfer policy is selected last, see MirroredStorage: */
    this->set_transfer_policy(transfer_policy);
  }


  template <typename T>
  inline std::size_t Mirrored<T>::size() const
    requires(is_array)
  {
    return size_;
  }


  template <typename T>
  template <typename InitializedMemorySpace>
  void Mirrored<T>::initialize_storage()
  {
    static_assert(std::is_same_v<InitializedMemorySpace, HostSpace> ||
                      std::is_same_v<InitializedMemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    /*
     * Note: If the host and default memory spaces coincide then only the
     * host storage is ever allocated.
     */
    if constexpr (have_separate_memory_spaces &&
                  !std::is_same_v<InitializedMemorySpace, HostSpace>) {
      /* Drop possibly stale storage from a previous reinit(): */
      deallocate_storage<HostSpace>();
      allocate_storage<DefaultSpace>();

      this->reset_residency(/*host*/ false, /*default*/ true);

    } else {
      if constexpr (have_separate_memory_spaces)
        deallocate_storage<DefaultSpace>();
      allocate_storage<HostSpace>();

      /*
       * If both memory spaces coincide the single host allocation is
       * trivially resident on both of them:
       */
      this->reset_residency(/*host*/ true,
                            /*default*/ !have_separate_memory_spaces);
    }
  }


  template <typename T>
  template <typename MemorySpace>
  inline auto Mirrored<T>::view() -> value_type *
  {
    this->template prepare_write_access<MemorySpace>();

    return storage<MemorySpace>().data();
  }


  template <typename T>
  template <typename MemorySpace>
  inline auto Mirrored<T>::view() const -> const value_type *
  {
    this->template prepare_read_access<MemorySpace>();

    return storage<MemorySpace>().data();
  }


  template <typename T>
  template <typename MemorySpace>
  inline auto Mirrored<T>::storage() const -> KokkosView<MemorySpace> &
  {
    static_assert(std::is_same_v<MemorySpace, HostSpace> ||
                      std::is_same_v<MemorySpace, DefaultSpace>,
                  "Unexpected memory space");

    /*
     * Note: If the host and default memory spaces coincide then only the
     * host storage is ever allocated.
     */
    if constexpr (std::is_same_v<MemorySpace, HostSpace>)
      return host_;
    else
      return default_;
  }


  template <typename T>
  template <typename MemorySpace>
  inline void Mirrored<T>::allocate_storage() const
  {
    if constexpr (is_array)
      storage<MemorySpace>() = KokkosView<MemorySpace>(label_, size_);
    else
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
