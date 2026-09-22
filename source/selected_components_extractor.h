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
   *   T values[n_selected];
   *   extractor_view.extract_element(values, i);
   *   // ...
   * };
   * ```
   *
   * All bookkeeping and all memory space transfers happen in prepare() and
   * prepare_extraction(); a view is a cheap, copyable collection of
   * pointers, indices, and booleans.
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

    using HyperbolicVector = std::tuple_element_t<0, StateVector>;
    using PrecomputedVector = std::tuple_element_t<1, StateVector>;
    using ParabolicVector = std::tuple_element_t<2, StateVector>;

    template <typename MemorySpace>
    using HyperbolicVectorView =
        decltype(std::declval<const HyperbolicVector &>()
                     .template view<MemorySpace>());
    template <typename MemorySpace>
    using PrecomputedVectorView =
        decltype(std::declval<const PrecomputedVector &>()
                     .template view<MemorySpace>());
    template <typename MemorySpace>
    using InitialPrecomputedVectorView =
        decltype(std::declval<const InitialPrecomputedVector &>()
                     .template view<MemorySpace>());
    template <typename MemorySpace>
    using ScalarVectorView = decltype(std::declval<const ScalarVector &>()
                                          .template view<MemorySpace>());

    /*
     *
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
     * @note A call to this function invalidates all views for the selected
     * memory space previously returned by view().
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    void prepare_extraction(const StateVector &state_vector) const;

    //@}
    /**
     * @name Queries
     */
    //@{

    /**
     * Return the number of selected components, i.e., the number of
     * entries of the vector returned by extract() and the number of
     * values written by extract_element().
     */
    std::size_t n_selected() const;

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
     * The combined number of parabolic components and additional vectors,
     * i.e., the size of the scalar view array maintained in the payload.
     */
    const unsigned int n_scalar_;

    /*
     * Index bookkeeping that is independent of the state vector and the
     * memory space. All of the following is set up by prepare():
     */

    /**
     * The offset of every selected component, in the order in which the
     * components have been selected. The array is mirrored between the
     * host and device memory spaces so that it can be captured in a
     * computation loop.
     */
    Mirrored<unsigned int *> selection_{"selected_components_selection"};

    /**
     * The number of selected components, i.e., the size of selection_.
     */
    unsigned int n_selected_ = 0;

    /**
     * Record which sections of the linear range of offsets have selected
     * components. This allows prepare_extraction() to only create views
     * for vectors that we are actually reading from.
     */
    bool read_conserved_ = false;
    bool read_primitive_ = false;
    bool read_precomputed_ = false;
    bool read_initial_ = false;
    bool read_scalar_ = false;

    /**
     * All state vector and memory space dependent data set up by
     * prepare_extraction(), maintained once per memory space.
     */
    template <typename MemorySpace>
    struct Payload {
      const unsigned int *selection_ = nullptr;

      HyperbolicVectorView<MemorySpace> U_view_;
      PrecomputedVectorView<MemorySpace> precomputed_view_;
      InitialPrecomputedVectorView<MemorySpace> initial_view_;

      /*
       * One view per parabolic component and additional vector, indexed by
       * `offset - parabolic_offset`: the parabolic components come first,
       * the additional vectors last. Only the views of selected components
       * are populated. The array is mirrored between the host and device
       * memory spaces so that it can be captured in a computation loop.
       */
      Mirrored<ScalarVectorView<MemorySpace> *> scalar_views_storage_{
          "selected_components_scalar_views"};
      const ScalarVectorView<MemorySpace> *scalar_views_ = nullptr;

      bool prepared_ = false;
    };

    mutable Payload<dealii::MemorySpace::Host> host_payload_;
    mutable Payload<dealii::MemorySpace::Default> default_payload_;

    /**
     * Return the payload of the selected memory space.
     */
    template <typename MemorySpace>
    Payload<MemorySpace> &payload() const;

    template <typename, int, typename, typename>
    friend class SelectedComponentsExtractorView;

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

    using HyperbolicSystem = typename Extractor::HyperbolicSystem;
    using StateVector = typename Extractor::StateVector;

    using EquationView = typename Extractor::View;

    static constexpr auto problem_dimension = EquationView::problem_dimension;
    static constexpr auto n_precomputed_values =
        EquationView::n_precomputed_values;
    static constexpr auto n_initial_precomputed_values =
        EquationView::n_initial_precomputed_values;

    using HyperbolicVectorView =
        typename Extractor::template HyperbolicVectorView<MemorySpace>;
    using PrecomputedVectorView =
        typename Extractor::template PrecomputedVectorView<MemorySpace>;
    using InitialPrecomputedVectorView =
        typename Extractor::template InitialPrecomputedVectorView<MemorySpace>;
    using ScalarVectorView =
        typename Extractor::template ScalarVectorView<MemorySpace>;

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
     * written by extract_element().
     */
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int n_selected() const
    {
      return n_selected_;
    }

    /**
     * Extract all selected components for the single degree of freedom
     * @p i and store them in @p result. The values are stored in the order
     * in which the components have been selected in prepare(), i.e.,
     * `result[k]` corresponds to the k-th entry of the vector returned by
     * extract().
     *
     * @note The caller has to supply storage for n_selected() values of
     * type @p T. The function neither allocates memory, nor does it
     * perform any memory space transfers: it can be called on the memory
     * space of the view.
     *
     * If the template parameter @a T is a VectorizedArray then the
     * function returns SIMD vectorized values populated with the entries
     * stored at indices i, i+1, ..., i+simd_length-1. Correspondingly,
     * @p i has to be divisible by the SIMD length.
     */
    template <typename T = Number>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
    extract_element(T *result, unsigned int i) const;

  private:
    //@}
    /**
     * @name Constructor, internal fields, and friends
     */
    //@{

    using Payload = typename Extractor::template Payload<MemorySpace>;

    /**
     * Constructor.
     */
    SelectedComponentsExtractorView(const Extractor &extractor,
                                    const Payload &payload);

    const Extractor *extractor_;

    const unsigned int *selection_;
    unsigned int n_selected_;

    /* Record which vectors have views set up: */
    bool read_conserved_;
    bool read_primitive_;
    bool read_precomputed_;
    bool read_initial_;

    HyperbolicVectorView U_view_;
    PrecomputedVectorView precomputed_view_;
    InitialPrecomputedVectorView initial_view_;

    SelectView<dim, Number, MemorySpace, HyperbolicSystem> system_views_;

    /*
     * One view per parabolic component and additional vector, see the
     * documentation of SelectedComponentsExtractor::Payload.
     */
    const ScalarVectorView *scalar_views_;

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
      , n_scalar_(additional_offset_ - parabolic_offset +
                  additional_vectors.size())
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

    n_selected_ = static_cast<unsigned int>(selection.size());

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
    read_scalar_ = selects(parabolic_offset, parabolic_offset + n_scalar_);

    selection_.reinit(selection.size(), TransferPolicy::implicit_transfers);
    std::copy(selection.begin(), selection.end(), selection_.view());

    /*
     * (Re)size the scalar view arrays and invalidate all views that we have
     * handed out so far:
     */

    const std::size_t size = read_scalar_ ? n_scalar_ : 0;
    host_payload_.scalar_views_storage_.reinit(
        size, TransferPolicy::implicit_transfers);
    default_payload_.scalar_views_storage_.reinit(
        size, TransferPolicy::implicit_transfers);

    host_payload_.prepared_ = false;
    default_payload_.prepared_ = false;
  }


  template <typename Description, int dim, typename Number>
  template <typename MemorySpace>
  void
  SelectedComponentsExtractor<Description, dim, Number>::prepare_extraction(
      const StateVector &state_vector) const
  {
    using HostSpace = dealii::MemorySpace::Host;

    auto &payload = this->template payload<MemorySpace>();

    payload.selection_ = selection_.template view<MemorySpace>();

    /*
     * Only set up views for vectors that we are actually reading from:
     * creating a view triggers a residency assertion (or an implicit memory
     * transfer).
     */

    if (read_conserved_ || read_primitive_)
      payload.U_view_ = std::get<0>(state_vector).template view<MemorySpace>();

    if (read_precomputed_)
      payload.precomputed_view_ =
          std::get<1>(state_vector).template view<MemorySpace>();

    if (read_initial_)
      payload.initial_view_ = initial_precomputed_.template view<MemorySpace>();

    /*
     * Create a view for every selected parabolic component and additional
     * vector and store them in an array residing on the memory space of the
     * view:
     */

    if (read_scalar_) {
      auto *scalar_views = payload.scalar_views_storage_.view();

      /* We iterate over the selection on the host: */
      const auto *selection = selection_.template view<HostSpace>();
      const auto &parabolic = std::get<2>(state_vector);

      for (unsigned int k = 0; k < n_selected_; ++k) {
        const auto offset = selection[k];
        if (offset < parabolic_offset)
          continue;

        const auto component = offset - parabolic_offset;
        if (offset < additional_offset_)
          scalar_views[component] =
              parabolic[component].template view<MemorySpace>();
        else
          scalar_views[component] =
              additional_vectors_[offset - additional_offset_]
                  .get()
                  .template view<MemorySpace>();
      }

      payload.scalar_views_ = std::as_const(payload.scalar_views_storage_)
                                  .template view<MemorySpace>();
    }

    payload.prepared_ = true;
  }


  template <typename Description, int dim, typename Number>
  std::size_t
  SelectedComponentsExtractor<Description, dim, Number>::n_selected() const
  {
    return n_selected_;
  }


  template <typename Description, int dim, typename Number>
  template <typename MemorySpace>
  SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>
  SelectedComponentsExtractor<Description, dim, Number>::view() const
  {
    Assert(this->template payload<MemorySpace>().prepared_,
           dealii::ExcMessage(
               "Invalid state: prepare_extraction() has to be called for the "
               "selected memory space before a view can be created."));

    return SelectedComponentsExtractorView<Description,
                                           dim,
                                           Number,
                                           MemorySpace>(
        *this, this->template payload<MemorySpace>());
  }


  template <typename Description, int dim, typename Number>
  template <typename MemorySpace>
  auto SelectedComponentsExtractor<Description, dim, Number>::payload() const
      -> Payload<MemorySpace> &
  {
    static_assert(std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
                      std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
                  "Unexpected memory space");

    if constexpr (std::is_same_v<MemorySpace, dealii::MemorySpace::Host>)
      return host_payload_;
    else
      return default_payload_;
  }


  template <typename Description,
            int dim,
            typename Number,
            typename MemorySpace>
  SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>::
      SelectedComponentsExtractorView(const Extractor &extractor,
                                      const Payload &payload)
      : extractor_(&extractor)
      , selection_(payload.selection_)
      , n_selected_(extractor.n_selected_)
      , read_conserved_(extractor.read_conserved_)
      , read_primitive_(extractor.read_primitive_)
      , read_precomputed_(extractor.read_precomputed_)
      , read_initial_(extractor.read_initial_)
      , U_view_(payload.U_view_)
      , precomputed_view_(payload.precomputed_view_)
      , initial_view_(payload.initial_view_)
      , system_views_(*extractor.hyperbolic_system_)
      , scalar_views_(payload.scalar_views_)
  {
  }


  template <typename Description,
            int dim,
            typename Number,
            typename MemorySpace>
  auto SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>::
      extract() const -> std::vector<ScalarVector>
  {
    using HostSpace = dealii::MemorySpace::Host;

    const auto &offline_data = *extractor_->offline_data_;
    const auto &scalar_partitioner = offline_data.scalar_partitioner();

    /* We iterate over the selection on the host: */
    const auto *selection = extractor_->selection_.template view<HostSpace>();

    std::vector<ScalarVector> extracted_components(n_selected_);
    for (auto &it : extracted_components)
      it.reinit(scalar_partitioner);

    for (unsigned int k = 0; k < n_selected_; ++k) {
      const auto offset = selection[k];
      auto &destination = extracted_components[k];

      if (offset < Extractor::primitive_offset) {
        U_view_.extract_component(destination,
                                  offset - Extractor::conserved_offset);

      } else if (offset < Extractor::precomputed_offset) {
        /*
         * Primitive components are computed from the conserved state:
         */

        const auto U_view = U_view_;
        const auto system_views = system_views_;
        const auto component = offset - Extractor::primitive_offset;
        auto *data = destination.begin();

        const auto body = [=](auto sentinel, unsigned int i) {
          using T = decltype(sentinel);

          const auto U_i = U_view.template read_tensor<T>(i);
          const auto primitive_i =
              system_views.template view<T>().to_primitive_state(U_i);

          if constexpr (std::is_same_v<T, dealii::VectorizedArray<Number>>)
            primitive_i[component].store(data + i);
          else
            data[i] = primitive_i[component];
        };

        loop<MemorySpace, Number>("extract_primitive_component",
                                  body,
                                  0,
                                  offline_data.n_locally_internal(),
                                  offline_data.n_locally_owned());

      } else if (offset < Extractor::initial_offset) {
        precomputed_view_.extract_component(
            destination, offset - Extractor::precomputed_offset);

      } else if (offset < Extractor::parabolic_offset) {
        initial_view_.extract_component(destination,
                                        offset - Extractor::initial_offset);

      } else {
        scalar_views_[offset - Extractor::parabolic_offset].extract_component(
            destination, 0);
      }
    }

    return extracted_components;
  }


  template <typename Description,
            int dim,
            typename Number,
            typename MemorySpace>
  template <typename T>
  DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
  SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>::
      extract_element(T *result, const unsigned int i) const
  {
    /*
     * Read all involved tensors once. Reading a single component out of a
     * MultiComponentVector is not any cheaper than reading the full
     * tensor...
     */

    T staging[Extractor::parabolic_offset];

    if (read_conserved_ || read_primitive_) {
      const auto U_i = U_view_.template read_tensor<T>(i);
      for (unsigned int d = 0; d < problem_dimension; ++d)
        staging[Extractor::conserved_offset + d] = U_i[d];

      if (read_primitive_) {
        const auto primitive_i =
            system_views_.template view<T>().to_primitive_state(U_i);
        for (unsigned int d = 0; d < problem_dimension; ++d)
          staging[Extractor::primitive_offset + d] = primitive_i[d];
      }
    }

    if (read_precomputed_) {
      const auto precomputed_i = precomputed_view_.template read_tensor<T>(i);
      for (unsigned int d = 0; d < n_precomputed_values; ++d)
        staging[Extractor::precomputed_offset + d] = precomputed_i[d];
    }

    if (read_initial_) {
      const auto initial_i = initial_view_.template read_tensor<T>(i);
      for (unsigned int d = 0; d < n_initial_precomputed_values; ++d)
        staging[Extractor::initial_offset + d] = initial_i[d];
    }

    /*
     * Parabolic components and additional vectors are read out of the
     * corresponding scalar vector directly:
     */

    for (unsigned int k = 0; k < n_selected_; ++k) {
      const auto offset = selection_[k];

      if (offset < Extractor::parabolic_offset)
        result[k] = staging[offset];
      else
        result[k] = scalar_views_[offset - Extractor::parabolic_offset]
                        .template read_entry<T>(i);
    }
  }

#endif
} // namespace ryujin
