//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "state_vector.h"

namespace ryujin
{
  /**
   * A helper class that extracts a selection of named components from a
   * state vector into a vector of scalar vectors suitable for output and
   * postprocessing.
   *
   * A component can be selected by any of its conserved, primitive,
   * parabolic, precomputed, or initial-precomputed name, or by one of the
   * additional names supplied by the caller.
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number>
  struct SelectedComponentsExtractor {
    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using StateVector = typename View::StateVector;
    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;

    using ScalarVector = Vectors::ScalarVector<Number>;
    using ScalarHostVector = Vectors::ScalarHostVector<Number>;

    SelectedComponentsExtractor() = delete;

    static void check(const std::vector<std::string> &parabolic_component_names,
                      const std::vector<std::string> &additional_names,
                      const std::vector<std::string> &selected)
    {
      const auto search = [&](const auto entry, const auto &names) {
        const auto pos = std::find(std::begin(names), std::end(names), entry);
        return pos != std::end(names);
      };

      for (const auto &entry : selected) {
        const auto found = search(entry, View::component_names) ||
                           search(entry, View::primitive_component_names) ||
                           search(entry, parabolic_component_names) ||
                           search(entry, View::precomputed_names) ||
                           search(entry, View::initial_precomputed_names) ||
                           search(entry, additional_names);
        AssertThrow(found,
                    dealii::ExcMessage(
                        "Invalid component name: \"" + entry +
                        "\" is not a valid conserved, primitive, parabolic, "
                        "precomputed, or initial component name."));
      }
    }

    static std::vector<ScalarHostVector>
    extract(const OfflineData<dim, Number> &offline_data,
            const HyperbolicSystem &hyperbolic_system,
            const ParabolicSystem &parabolic_system,
            const StateVector &state_vector,
            const InitialPrecomputedVector &initial_precomputed,
            const std::vector<std::string> &additional_names,
            const std::vector<std::reference_wrapper<const ScalarVector>>
                &additional_vectors,
            const std::vector<std::string> &selected)
    {
      /*
       * Match the selected_components strings against conserved,
       * primitive, and initial component names and record an index pair
       * matching return vector position and component index:
       */

      std::vector<std::tuple<std::size_t, std::size_t>> conserved_indices;
      std::vector<std::tuple<std::size_t, std::size_t>> primitive_indices;
      std::vector<std::tuple<std::size_t, std::size_t>> precomputed_indices;
      std::vector<std::tuple<std::size_t, std::size_t>> parabolic_indices;
      std::vector<std::tuple<std::size_t, std::size_t>> initial_indices;
      std::vector<std::tuple<std::size_t, std::size_t>> additional_indices;

      for (std::size_t i = 0; const auto &entry : selected) {
        const auto search = [&](const auto &names, auto &indices) {
          const auto pos = std::find(std::begin(names), std::end(names), entry);
          if (pos != std::end(names)) {
            const auto index = std::distance(std::begin(names), pos);
            indices.push_back({i++, index});
            return true;
          }
          return false;
        };

        if (search(View::component_names, conserved_indices))
          ;
        else if (search(View::primitive_component_names, primitive_indices))
          ;
        else if (search(View::precomputed_names, precomputed_indices))
          ;
        else if (search(parabolic_system.parabolic_component_names(),
                        parabolic_indices))
          ;
        else if (search(View::initial_precomputed_names, initial_indices))
          ;
        else if (search(additional_names, additional_indices))
          ;
        else
          AssertThrow(false, dealii::ExcInternalError());
      }

      std::vector<ScalarHostVector> extracted_components(selected.size());
      const auto &scalar_partitioner = offline_data.scalar_partitioner();
      for (auto &it : extracted_components)
        it.reinit(scalar_partitioner);

      for (const auto &[i, k] : conserved_indices) {
        const auto &U = std::get<0>(state_vector);
        U.extract_component(extracted_components[i], k);
      }

      if (!primitive_indices.empty()) {
        const auto &U = std::get<0>(state_vector);
        const unsigned int n_owned = scalar_partitioner->locally_owned_size();
        const auto view = hyperbolic_system.template view<dim, Number>();
        for (unsigned int i = 0; i < n_owned; ++i) {
          const auto U_i = U.read_tensor(i);
          const auto PU_i = view.to_primitive_state(U_i);
          for (const auto &[j, k] : primitive_indices)
            extracted_components[j].local_element(i) = PU_i[k];
        }
      }

      for (const auto &[i, k] : precomputed_indices) {
        const auto &prec = std::get<1>(state_vector);
        prec.extract_component(extracted_components[i], k);
      }

      for (const auto &[i, k] : parabolic_indices) {
        const auto &parabolic = std::get<2>(state_vector);
        extracted_components[i] = parabolic.block(k);
      }

      for (const auto &[i, k] : initial_indices) {
        initial_precomputed.extract_component(extracted_components[i], k);
      }

      for (const auto &[i, k] : additional_indices) {
        additional_vectors[k].get().extract_component( //
            extracted_components[i],
            0);
      }

      return extracted_components;
    }
  };
} // namespace ryujin
