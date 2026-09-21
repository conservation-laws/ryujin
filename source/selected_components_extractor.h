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
   * const auto components = extractor.view(state_vector).extract();
   * ```
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number>
  class SelectedComponentsExtractor
  {
  public:
    /**
     * @name Typedefs
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using StateVector = typename View::StateVector;
    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;

    using ScalarVector = Vectors::ScalarVector<Number>;

    //@}
    /**
     * @ name Constexpr constants
     *
     * A selected component is identified by an offset into the linear
     * range formed by concatenating all conserved, primitive, precomputed,
     * initial-precomputed, parabolic, and additional components (in this
     * order). The following constants mark the beginning of the respective
     * sections. The parabolic section has a run time size, the offset of
     * the additional section is thus stored in additional_offset_.
     */
    //@{
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
     * internal index bookkeeping. A call to prepare() is necessary before
     * a view can be used to extract components.
     */
    void prepare(const std::vector<std::string> &selected);

    //@}
    /**
     * @name Queries
     */
    //@{

    /**
     * Return the number of selected components, i.e., the number of
     * entries of the vector returned by extract().
     */
    std::size_t n_selected() const;

    //@}
    /**
     * @name Memory space access
     */
    //@{

    /**
     * Return a view on the extractor for the selected memory space that
     * reads all selected components out of @p state_vector.
     *
     * @note Creating the view decays the state vector into views on the
     * selected memory space. Depending on the transfer policy of the
     * individual vectors this either asserts that the memory space is
     * resident, or triggers an implicit memory transfer.
     */
    template <typename MemorySpace = dealii::MemorySpace::Host>
    SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>
    view(const StateVector &state_vector) const;

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

    template <typename, int, typename, typename>
    friend class SelectedComponentsExtractorView;

    //@}
  };


  /**
   * A "view" of a SelectedComponentsExtractor that extracts the selected
   * components on the host or device memory space.
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

    using HyperbolicVector = std::tuple_element_t<0, StateVector>;
    using PrecomputedVector = std::tuple_element_t<1, StateVector>;
    using ParabolicVector = std::tuple_element_t<2, StateVector>;
    using InitialPrecomputedVector =
        typename Extractor::InitialPrecomputedVector;

    using HyperbolicVectorView =
        decltype(std::declval<const HyperbolicVector &>()
                     .template view<MemorySpace>());
    using PrecomputedVectorView =
        decltype(std::declval<const PrecomputedVector &>()
                     .template view<MemorySpace>());
    using InitialPrecomputedVectorView =
        decltype(std::declval<const InitialPrecomputedVector &>()
                     .template view<MemorySpace>());

    /**
     * Shorthand typedef for the scalar
     * dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace> that
     * a single extracted component is stored in.
     */
    using ScalarVector =
        dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace>;

    //@}
    /**
     * @name Constructor
     */
    //@{

    SelectedComponentsExtractorView(const Extractor &extractor,
                                    const StateVector &state_vector);

    //@}
    /**
     * @name Extracting components
     */
    //@{

    /**
     * Extract all selected components out of the state vector the view was
     * created with and return them as a vector of scalar vectors that is
     * populated on the memory space of the view. The returned vector
     * contains one entry per selected component in the order in which the
     * components have been selected in prepare().
     */
    std::vector<ScalarVector> extract() const;

  private:
    //@}
    /**
     * @name Internal fields
     */
    //@{

    const Extractor *extractor_;

    HyperbolicVectorView U_view_;
    PrecomputedVectorView precomputed_view_;
    InitialPrecomputedVectorView initial_view_;

    SelectView<dim, Number, MemorySpace, HyperbolicSystem> system_views_;

    const ParabolicVector *parabolic_;

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
    selection_.clear();
    selection_.reserve(selected.size());

    for (const auto &entry : selected) {
      const auto search = [&](const auto &names, const unsigned int offset) {
        const auto pos = std::find(std::begin(names), std::end(names), entry);
        if (pos == std::end(names))
          return false;
        const unsigned int index = std::distance(std::begin(names), pos);
        selection_.push_back(offset + index);
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
  }


  template <typename Description, int dim, typename Number>
  std::size_t
  SelectedComponentsExtractor<Description, dim, Number>::n_selected() const
  {
    return selection_.size();
  }


  template <typename Description, int dim, typename Number>
  template <typename MemorySpace>
  SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>
  SelectedComponentsExtractor<Description, dim, Number>::view(
      const StateVector &state_vector) const
  {
    return SelectedComponentsExtractorView<Description,
                                           dim,
                                           Number,
                                           MemorySpace>(*this, state_vector);
  }


  template <typename Description,
            int dim,
            typename Number,
            typename MemorySpace>
  SelectedComponentsExtractorView<Description, dim, Number, MemorySpace>::
      SelectedComponentsExtractorView(const Extractor &extractor,
                                      const StateVector &state_vector)
      : extractor_(&extractor)
      , system_views_(*extractor.hyperbolic_system_)
      , parabolic_(&std::get<2>(state_vector))
  {
    using HostSpace = dealii::MemorySpace::Host;

    const auto &selection = extractor.selection_;

    /*
     * Only set up views for vectors that we are actually reading from:
     * creating a view triggers a residency assertion (or an implicit memory
     * transfer).
     */

    const auto selects = [&](const unsigned int begin, const unsigned int end) {
      return std::any_of(
          selection.begin(), selection.end(), [&](const auto offset) {
            return begin <= offset && offset < end;
          });
    };

    if (selects(Extractor::conserved_offset, Extractor::precomputed_offset))
      U_view_ = std::get<0>(state_vector).template view<MemorySpace>();

    if (selects(Extractor::precomputed_offset, Extractor::initial_offset))
      precomputed_view_ =
          std::get<1>(state_vector).template view<MemorySpace>();

    if (selects(Extractor::initial_offset, Extractor::parabolic_offset))
      initial_view_ =
          extractor.initial_precomputed_.template view<MemorySpace>();
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

    const auto &selection = extractor_->selection_;
    const auto additional_offset = extractor_->additional_offset_;

    std::vector<ScalarVector> extracted_components(selection.size());
    for (auto &it : extracted_components)
      it.reinit(scalar_partitioner);

    for (std::size_t k = 0; k < selection.size(); ++k) {
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

      } else if (offset < additional_offset) {
        if constexpr (std::is_same_v<MemorySpace, HostSpace>)
          destination = (*parabolic_)[offset - Extractor::parabolic_offset]
                            .deal_ii_vector();
        else
          AssertThrow(false, dealii::ExcNotImplemented());

      } else {
        extractor_->additional_vectors_[offset - additional_offset]
            .get()
            .template view<MemorySpace>()
            .extract_component(destination, 0);
      }
    }

    return extracted_components;
  }

#endif
} // namespace ryujin
