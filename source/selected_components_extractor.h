//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "gpu.h"
#include "loop.h"
#include "observer_pointer.h"
#include "offline_data.h"
#include "state_vector.h"

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace ryujin
{
  template <typename Description,
            int dim,
            typename Number,
            typename MemorySpace>
  class SelectedComponentsExtractorView;

  /**
   * A helper class that extracts a selection of named components from a
   * state vector into a vector of scalar vectors suitable for output and
   * postprocessing.
   *
   * A component can be selected by any of its conserved, primitive,
   * precomputed, initial-precomputed, or parabolic name, or by one of the
   * additional names supplied by the caller.
   *
   * Intended usage:
   * ```
   * SelectedComponentsExtractor<Description, dim, Number> extractor(
   *     offline_data, hyperbolic_system, parabolic_system,
   *     initial_precomputed, {"alpha"}, {alpha});
   * extractor.prepare(selected_component_names);
   * // ...
   * extractor.prepare_extraction(state_vector);
   * const auto components = extractor.view().extract();
   * ```
   *
   * The extract() call populates one full scalar vector per selected
   * component. Alternatively, all selected components can be read for a
   * single degree of freedom - in a computation loop running on the host,
   * or on the device:
   * ```
   * extractor.template prepare_extraction<MemorySpace>(state_vector);
   * const auto extractor_view = extractor.template view<MemorySpace>();
   *
   * const auto body = [=](auto sentinel, unsigned int i) {
   *   using T = decltype(sentinel);
   *
   *   extractor_view.template extract_element<T>(
   *       i, [&](unsigned int k, const T &value) {
   *         // ...
   *       });
   * };
   * ```
   *
   * @note A view is only valid as long as neither prepare(), nor
   * prepare_extraction() is called again, and as long as the state vector
   * the extraction was prepared with is not modified.
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number>
  class SelectedComponentsExtractor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using StateVector = typename View::StateVector;
    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;

    using ScalarVector = Vectors::ScalarVector<Number>;

    /*
     * A selected component is identified by an offset into the linear
     * range formed by concatenating all conserved, primitive, precomputed,
     * initial-precomputed, parabolic, and additional components (in this
     * order). The following constants mark the beginning of the respective
     * sections. The parabolic section has a run time size, the offset of
     * the additional section is thus stored in additional_offset_.
     */

    static constexpr unsigned int conserved_offset = 0;
    static constexpr unsigned int primitive_offset = View::problem_dimension;
    static constexpr unsigned int precomputed_offset =
        2 * View::problem_dimension;
    static constexpr unsigned int initial_offset =
        precomputed_offset + View::n_precomputed_values;
    static constexpr unsigned int parabolic_offset =
        initial_offset + View::n_initial_precomputed_values;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     *
     * In addition to the conserved, primitive, precomputed,
     * initial-precomputed, and parabolic components a caller can supply a
     * list of @p additional_names with corresponding @p additional_vectors
     * that can be selected as well.
     */
    SelectedComponentsExtractor(
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const ParabolicSystem &parabolic_system,
        const InitialPrecomputedVector &initial_precomputed,
        const std::vector<std::string> &additional_names = {},
        const std::vector<std::reference_wrapper<const ScalarVector>>
            &additional_vectors = {});

    /**
     * Validate the list of @p selected component names and set up all
     * internal index bookkeeping that is independent of a state vector. A
     * call to prepare() is necessary before an extraction can be prepared.
     *
     * @note A call to this function invalidates all views previously
     * returned by view().
     */
    void prepare(const std::vector<std::string> &selected);

    /**
     * Set up all internal data structures necessary for reading the
     * selected components out of @p state_vector on the selected memory
     * space. This function creates a view for all selected components of
     * the state vector (and all selected additional vectors) on the
     * selected memory space. Depending on the transfer policy of the
     * individual vectors this either asserts that the memory space is
     * resident, or triggers an implicit memory transfer.
     *
     * A call to prepare_extraction() is necessary before a view for the
     * selected memory space can be created with view().
     *
     * @note A call to this function discards all information stored for a
     * previous extraction and invalidates all views previously returned by
     * view().
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    void prepare_extraction(const StateVector &state_vector) const;

    //@}
    /**
     * @name Queries
     */
    //@{

    /**
     * Return the number of selected components, i.e., the number of entries
     * of the vector returned by extract() and the number of values passed
     * to the writer by extract_element().
     */
    unsigned int n_selected() const;

    //@}
    /**
     * @name Memory space access
     */
    //@{

    /**
     * Return a view on the extractor for the selected memory space that
     * reads all selected components out of the state vector the extraction
     * has been prepared with.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>
    view() const;

  private:
    //@}
    /**
     * @name Internal fields and friends
     */
    //@{

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<const ParabolicSystem> parabolic_system_;

    const InitialPrecomputedVector &initial_precomputed_;

    const std::vector<std::string> additional_names_;
    const std::vector<std::reference_wrapper<const ScalarVector>>
        additional_vectors_;

    /**
     * The offset of the additional section, see parabolic_offset.
     */
    const unsigned int additional_offset_;

    /**
     * The offset of every selected component, in the order in which the
     * components have been selected.
     */
    std::vector<unsigned int> selection_;

    /**
     * Record which sections of the linear range of offsets have selected
     * components. This allows prepare_extraction() to only create views
     * for vectors that we are actually reading from.
     */
    bool read_conserved_ = false;
    bool read_primitive_ = false;
    bool read_precomputed_ = false;
    bool read_initial_ = false;

    template <typename MemorySpace>
    using ExtractorView =
        SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>;

    /**
     * A variant storing the view (on the memory space) that the extraction
     * has last been prepared for.
     */
    mutable std::variant<std::monostate,
                         ExtractorView<dealii::MemorySpace::Host>,
                         ExtractorView<dealii::MemorySpace::Default>>
        view_;

    //@}
  };


  /**
   * A "view" of a SelectedComponentsExtractor that extracts the selected
   * components on the host or device memory space.
   *
   * A view is created with SelectedComponentsExtractor::view() and is a
   * cheap, copyable object that can be captured by value in a computation
   * loop. It is only valid as long as neither
   * SelectedComponentsExtractor::prepare(), nor
   * SelectedComponentsExtractor::prepare_extraction() is called again.
   *
   * @ingroup TimeLoop
   */
  template <typename Description,
            int dim,
            typename Number,
            typename MemorySpace>
  class SelectedComponentsExtractorView
  {
  public:
    static_assert(std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
                      std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
                  "Unexpected memory space");

    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using Extractor = SelectedComponentsExtractor<Description, dim, Number>;

    /**
     * Shorthand typedef for the scalar
     * dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace> that
     * a single extracted component is stored in.
     */
    using ScalarVector =
        dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace>;

    //@}
    /**
     * @name Extracting components
     */
    //@{

    /**
     * Extract all selected components out of the state vector the
     * extraction was prepared with and return them as a vector of scalar
     * vectors that is populated on the memory space of the view. The
     * returned vector contains one entry per selected component in the
     * order in which the components have been selected in prepare().
     */
    std::vector<ScalarVector> extract() const;

    /**
     * Return the number of selected components, i.e., the number of values
     * passed to the writer by extract_element().
     */
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int n_selected() const
    {
      return static_cast<unsigned int>(entries_.extent(0));
    }

    /**
     * Extract all selected components for the single degree of freedom
     * @p i and hand them to @p write: The function calls
     * `write(k, value)` exactly once for every k = 0, ..., n_selected()-1
     * in increasing order, where `value` (of type @p T) is the value of
     * the k-th selected component, i.e., it corresponds to the k-th entry
     * of the vector returned by extract().
     *
     * @note The function neither allocates memory, nor does it perform
     * any memory space transfers: it can be called on the memory space of
     * the view. Correspondingly, @p write has to be callable on the memory
     * space of the view.
     *
     * If the template parameter @a T is a VectorizedArray then the
     * function hands out SIMD vectorized values populated with the entries
     * stored at indices i, i+1, ..., i+simd_length-1. Correspondingly,
     * @p i has to be divisible by the SIMD length.
     */
    template <typename T = Number, typename Writer>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    extract_element(unsigned int i, const Writer &write) const;

  private:
    //@}
    /**
     * @name Constructor, internal fields, and friends
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using EquationView = typename HyperbolicSystem::template View<dim, Number>;
    using StateVector = typename EquationView::StateVector;

    template <typename Vector>
    using VectorView =
        decltype(std::declval<const Vector &>().template view<MemorySpace>());

    using HyperbolicVectorView =
        VectorView<std::tuple_element_t<0, StateVector>>;
    using PrecomputedVectorView =
        VectorView<std::tuple_element_t<1, StateVector>>;
    using InitialPrecomputedVectorView =
        VectorView<typename EquationView::InitialPrecomputedVector>;
    using ScalarVectorView = VectorView<Vectors::ScalarVector<Number>>;

    /*
     * A selected component: its offset, and for a parabolic component or
     * an additional vector a view of the corresponding scalar vector.
     */
    struct Entry {
      unsigned int offset;
      ScalarVectorView scalar_view;
    };

    /**
     * Constructor. All remaining fields are set up by
     * SelectedComponentsExtractor::prepare_extraction().
     */
    SelectedComponentsExtractorView(
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system);

    const OfflineData<dim, Number> *offline_data_;

    /* Record which vectors have views set up: */
    bool read_conserved_ = false;
    bool read_primitive_ = false;
    bool read_precomputed_ = false;
    bool read_initial_ = false;

    SelectView<dim, Number, MemorySpace, HyperbolicSystem> system_views_;

    HyperbolicVectorView U_view_;
    PrecomputedVectorView precomputed_view_;
    InitialPrecomputedVectorView initial_view_;

    /*
     * One entry per selected component, in the order in which the
     * components have been selected.
     */
    Kokkos::View<const Entry *, typename MemorySpace::kokkos_space> entries_;

    friend class SelectedComponentsExtractor<Description, dim, Number>;

    //@}
  };


#ifndef DOXYGEN
  /*
   * -------------------------------------------------------------------------
   * Inline function definitions
   * -------------------------------------------------------------------------
   */


  template <typename Description, int dim, typename Number>
  SelectedComponentsExtractor<Description, dim, Number>::
      SelectedComponentsExtractor(
          const OfflineData<dim, Number> &offline_data,
          const HyperbolicSystem &hyperbolic_system,
          const ParabolicSystem &parabolic_system,
          const InitialPrecomputedVector &initial_precomputed,
          const std::vector<std::string> &additional_names,
          const std::vector<std::reference_wrapper<const ScalarVector>>
              &additional_vectors)
      : offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , initial_precomputed_(initial_precomputed)
      , additional_names_(additional_names)
      , additional_vectors_(additional_vectors)
      , additional_offset_(parabolic_offset +
                           parabolic_system.parabolic_component_names().size())
  {
    Assert(additional_names_.size() == additional_vectors_.size(),
           dealii::ExcMessage("The number of additional component names does "
                              "not match the number of additional vectors."));
  }


  template <typename Description, int dim, typename Number>
  void SelectedComponentsExtractor<Description, dim, Number>::prepare(
      const std::vector<std::string> &selected)
  {
    std::vector<unsigned int> selection;
    selection.reserve(selected.size());

    for (const auto &entry : selected) {
      const auto search = [&](const auto &names, const unsigned int offset) {
        const auto pos = std::find(std::begin(names), std::end(names), entry);
        if (pos == std::end(names))
          return false;
        const unsigned int index = std::distance(std::begin(names), pos);
        selection.push_back(offset + index);
        return true;
      };

      const bool found =
          search(View::component_names, conserved_offset) ||
          search(View::primitive_component_names, primitive_offset) ||
          search(View::precomputed_names, precomputed_offset) ||
          search(View::initial_precomputed_names, initial_offset) ||
          search(parabolic_system_->parabolic_component_names(),
                 parabolic_offset) ||
          search(additional_names_, additional_offset_);

      AssertThrow(found,
                  dealii::ExcMessage(
                      "Invalid component name: \"" + entry +
                      "\" is not a valid conserved, primitive, precomputed, "
                      "initial, parabolic, or additional component name."));
    }

    /*
     * Record which sections of the linear range of offsets have selected
     * components:
     */

    const auto selects = [&](const unsigned int begin, const unsigned int end) {
      return std::any_of(
          selection.begin(), selection.end(), [&](const auto offset) {
            return begin <= offset && offset < end;
          });
    };

    read_conserved_ = selects(conserved_offset, primitive_offset);
    read_primitive_ = selects(primitive_offset, precomputed_offset);
    read_precomputed_ = selects(precomputed_offset, initial_offset);
    read_initial_ = selects(initial_offset, parabolic_offset);

    selection_ = std::move(selection);

    /* Invalidate all views: */
    view_ = std::monostate{};
  }


  template <typename Description, int dim, typename Number>
  template <typename MemorySpace>
  void
  SelectedComponentsExtractor<Description, dim, Number>::prepare_extraction(
      const StateVector &state_vector) const
  {
    ExtractorView<MemorySpace> view(*offline_data_, *hyperbolic_system_);

    view.read_conserved_ = read_conserved_;
    view.read_primitive_ = read_primitive_;
    view.read_precomputed_ = read_precomputed_;
    view.read_initial_ = read_initial_;

    /*
     * Only set up views for vectors that we are actually reading from,
     * and only ensure residency of those parts of the state vector:
     */

    using Vectors::StateVectorPart;

    if (read_conserved_ || read_primitive_) {
      Vectors::copy_to_memory_space<MemorySpace>(state_vector,
                                                 StateVectorPart::hyperbolic);
      view.U_view_ = std::get<0>(state_vector).template view<MemorySpace>();
    }

    if (read_precomputed_) {
      Vectors::copy_to_memory_space<MemorySpace>(state_vector,
                                                 StateVectorPart::precomputed);
      view.precomputed_view_ =
          std::get<1>(state_vector).template view<MemorySpace>();
    }

    if (read_initial_) {
      Vectors::copy_to_memory_space<MemorySpace>(initial_precomputed_);
      view.initial_view_ = initial_precomputed_.template view<MemorySpace>();
    }

    /*
     * Set up an entry for every selected component on the host, with a
     * view for every selected parabolic component and additional vector,
     * and copy them into an array residing on the memory space of the view:
     */

    Kokkos::View<typename ExtractorView<MemorySpace>::Entry *,
                 Kokkos::HostSpace>
        entries("selected_components_entries", n_selected());

    const auto &parabolic = std::get<2>(state_vector);

    for (unsigned int k = 0; k < n_selected(); ++k) {
      const auto offset = selection_[k];
      entries[k].offset = offset;

      if (offset < parabolic_offset)
        continue;

      const auto &vector =
          offset < additional_offset_
              ? parabolic[offset - parabolic_offset]
              : additional_vectors_[offset - additional_offset_].get();

      Vectors::copy_to_memory_space<MemorySpace>(vector);
      entries[k].scalar_view = vector.template view<MemorySpace>();
    }

    view.entries_ = Kokkos::create_mirror_view_and_copy(
        typename MemorySpace::kokkos_space{}, entries);

    /* Discard all information stored for a previous extraction: */
    view_.template emplace<ExtractorView<MemorySpace>>(std::move(view));
  }


  template <typename Description, int dim, typename Number>
  unsigned int
  SelectedComponentsExtractor<Description, dim, Number>::n_selected() const
  {
    return static_cast<unsigned int>(selection_.size());
  }


  template <typename Description, int dim, typename Number>
  template <typename MemorySpace>
  SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>
  SelectedComponentsExtractor<Description, dim, Number>::view() const
  {
    Assert(std::holds_alternative<ExtractorView<MemorySpace>>(view_),
           dealii::ExcMessage(
               "Invalid state: prepare_extraction() has to be called for the "
               "selected memory space before a view can be created."));

    return std::get<ExtractorView<MemorySpace>>(view_);
  }


  template <typename Description,
            int dim,
            typename Number,
            typename MemorySpace>
  SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>::
      SelectedComponentsExtractorView(
          const OfflineData<dim, Number> &offline_data,
          const HyperbolicSystem &hyperbolic_system)
      : offline_data_(&offline_data)
      , system_views_(hyperbolic_system)
  {
  }


  template <typename Description,
            int dim,
            typename Number,
            typename MemorySpace>
  auto SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>::
      extract() const -> std::vector<ScalarVector>
  {
    const auto &offline_data = *offline_data_;

    /*
     * Set up all destination vectors and an array of pointers to their
     * data residing on the memory space of the view:
     */

    std::vector<ScalarVector> extracted_components(n_selected());

    struct Destination {
      Number *data;
    };

    Kokkos::View<Destination *, Kokkos::HostSpace> host_destinations(
        "selected_components_destinations", n_selected());

    for (unsigned int k = 0; k < n_selected(); ++k) {
      extracted_components[k].reinit(offline_data.scalar_partitioner());
      host_destinations[k].data = extracted_components[k].begin();
    }

    const auto destinations = Kokkos::create_mirror_view_and_copy(
        typename MemorySpace::kokkos_space{}, host_destinations);

    /*
     * Extract all selected components in a single sweep:
     */

    const auto view = *this;

    const auto body = [=](auto sentinel, unsigned int i) {
      using T = decltype(sentinel);

      view.template extract_element<T>(
          i, [&](const unsigned int k, const T &value) {
            if constexpr (std::is_same_v<T, dealii::VectorizedArray<Number>>)
              value.store(destinations[k].data + i);
            else
              destinations[k].data[i] = value;
          });
    };

    loop<MemorySpace, Number>("extract_selected_components",
                              body,
                              0,
                              offline_data.n_locally_internal(),
                              offline_data.n_locally_owned());

    for (auto &it : extracted_components)
      it.update_ghost_values();

    return extracted_components;
  }


  template <typename Description,
            int dim,
            typename Number,
            typename MemorySpace>
  template <typename T, typename Writer>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>::
      extract_element(const unsigned int i, const Writer &write) const
  {
    /*
     * Read all involved tensors once. Reading a single component out of a
     * MultiComponentVector is not any cheaper than reading the full
     * tensor...
     */

    T staging[Extractor::parabolic_offset];

    if (read_conserved_ || read_primitive_) {
      const auto U_i = U_view_.template read_tensor<T>(i);
      for (unsigned int d = 0; d < EquationView::problem_dimension; ++d)
        staging[Extractor::conserved_offset + d] = U_i[d];

      if (read_primitive_) {
        const auto primitive_i =
            system_views_.template view<T>().to_primitive_state(U_i);
        for (unsigned int d = 0; d < EquationView::problem_dimension; ++d)
          staging[Extractor::primitive_offset + d] = primitive_i[d];
      }
    }

    if (read_precomputed_) {
      const auto precomputed_i = precomputed_view_.template read_tensor<T>(i);
      for (unsigned int d = 0; d < EquationView::n_precomputed_values; ++d)
        staging[Extractor::precomputed_offset + d] = precomputed_i[d];
    }

    if (read_initial_) {
      const auto initial_i = initial_view_.template read_tensor<T>(i);
      for (unsigned int d = 0; d < EquationView::n_initial_precomputed_values;
           ++d)
        staging[Extractor::initial_offset + d] = initial_i[d];
    }

    /*
     * Parabolic components and additional vectors are read out of the
     * corresponding scalar vector directly:
     */

    for (unsigned int k = 0; k < n_selected(); ++k) {
      const auto &entry = entries_[k];

      if (entry.offset < Extractor::parabolic_offset)
        write(k, staging[entry.offset]);
      else
        write(k, entry.scalar_view.template read_entry<T>(i));
    }
  }

#endif
} // namespace ryujin
