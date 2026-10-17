//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <ryujin/interface/parabolic_system.h>
#include <ryujin/interface/postprocessor.h>

#include <gpu.h>
#include <loop.h>
#include <observer_pointer.h>
#include <offline_data.h>
#include <simd.h>
#include <state_vector.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>

#include <Kokkos_Pair.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ryujin
{
  /**
   * The typed, HyperbolicSystem dependent, counterpart of
   * Interface::Postprocessor; see there for details on the selectable
   * components and postprocessed quantities.
   *
   * Intended usage:
   * ```
   * Postprocessor<Description, dim, Number> postprocessor(
   *     offline_data, hyperbolic_system, parabolic_system,
   *     initial_precomputed, {"alpha"}, {alpha});
   * const auto selection = postprocessor.prepare(selected_component_names);
   * // ...
   * const auto components = postprocessor.extract(selection, state_vector);
   * ```
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number>
  class Postprocessor final : public Interface::Postprocessor<dim, Number>
  {
  public:
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using Selection = typename Interface::Postprocessor<dim, Number>::Selection;

    /**
     * Constructor.
     */
    Postprocessor(const OfflineData<dim, Number> &offline_data,
                  const HyperbolicSystem &hyperbolic_system,
                  const Interface::ParabolicSystem &parabolic_system,
                  const MultiComponentVector<Number> &initial_precomputed);

    Selection prepare(const std::vector<std::string> &selected) const override;

    std::vector<ScalarHostVector<Number>>
    extract(const Selection &selection,
            const StateVector<Number> &state_vector) const override;

    void extract(const Selection &selection,
                 const StateVector<Number> &state_vector,
                 const unsigned int *indices,
                 const unsigned int n_points,
                 Number *values) const override;

  private:
    using View = typename HyperbolicSystem::template View<dim, Number>;

    template <int n_comp, typename MemorySpace>
    using VectorView =
        MultiComponentVectorView<Number, n_comp, MemorySpace, false>;

    /**
     * The section a component is read from.
     */
    enum class Section : std::uint8_t {
      state,
      primitive,
      precomputed,
      initial_precomputed,
      dynamic_state,
    };

    /**
     * The selected postprocessing operation:
     */
    enum class Operation : std::uint8_t { value, schlieren, vorticity };

    /**
     * Returns a lambda kernel read(sentinel, i, write) that calls
     * write(k, value) for every component k of @p selection at the degree of
     * freedom i on MemorySpace (alternatively, i, ..., i + simd_length - 1
     * if the type of sentinel is a VectorizedArray).
     *
     * @note The parts of the state vector that we need to read from are
     * made resident via copy_to_memory_space<>().
     */
    template <typename MemorySpace>
    auto reader(const Selection &selection,
                const StateVector<Number> &state_vector) const;

    dealii::ObserverPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<const Interface::ParabolicSystem> parabolic_system_;
    const MultiComponentVector<Number> &initial_precomputed_;
  };


#ifndef DOXYGEN
  /*
   * -------------------------------------------------------------------------
   * Inline function definitions
   * -------------------------------------------------------------------------
   */


  template <typename Description, int dim, typename Number>
  Postprocessor<Description, dim, Number>::Postprocessor(
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const Interface::ParabolicSystem &parabolic_system,
      const MultiComponentVector<Number> &initial_precomputed)
      : offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , initial_precomputed_(initial_precomputed)
  {
  }


  template <typename Description, int dim, typename Number>
  auto Postprocessor<Description, dim, Number>::prepare(
      const std::vector<std::string> &selected) const -> Selection
  {
    const auto &hyperbolic_system = *hyperbolic_system_;

    Assert(hyperbolic_system.state_names(dim).size() ==
                   View::problem_dimension &&
               hyperbolic_system.primitive_names(dim).size() ==
                   View::problem_dimension &&
               hyperbolic_system.precomputed_names(dim).size() ==
                   View::n_precomputed_values &&
               hyperbolic_system.initial_precomputed_names(dim).size() ==
                   View::n_initial_precomputed_values,
           dealii::ExcMessage("The component names of the hyperbolic system "
                              "do not match the counts of its view."));

    const std::vector<std::pair<Section, const std::vector<std::string> *>>
        sections{
            {Section::state, &hyperbolic_system.state_names(dim)},
            {Section::primitive, &hyperbolic_system.primitive_names(dim)},
            {Section::precomputed, &hyperbolic_system.precomputed_names(dim)},
            {Section::initial_precomputed,
             &hyperbolic_system.initial_precomputed_names(dim)},
            {Section::dynamic_state, &parabolic_system_->state_names(dim)},
    };

    /*
     * Look up a component name and return its section, its index, and
     * the number of components of the section:
     */
    const auto find = [&](const std::string &name)
        -> std::optional<std::tuple<Section, unsigned int, unsigned int>> {
      for (const auto &[section, names] : sections) {
        const auto pos = std::find(names->begin(), names->end(), name);
        if (pos != names->end())
          return std::make_tuple(
              section,
              static_cast<unsigned int>(pos - names->begin()),
              static_cast<unsigned int>(names->size()));
      }
      return std::nullopt;
    };

    const std::vector<std::pair<Operation, std::string>> prefixes{
        {Operation::schlieren, "schlieren_"},
        {Operation::vorticity, "vorticity_"},
    };

    Selection selection;

    for (const auto &entry : selected) {
      auto operation = Operation::value;
      auto found = find(entry);

      for (const auto &[prefix_operation, prefix] : prefixes) {
        if (found || entry.rfind(prefix, 0) != 0)
          continue;
        operation = prefix_operation;
        found = find(entry.substr(prefix.size()));
      }

      AssertThrow(found,
                  dealii::ExcMessage(
                      "Invalid component name: \"" + entry +
                      "\" is not a valid conserved, primitive, precomputed, "
                      "initial, parabolic, or postprocessed "
                      "component name."));

      const auto [section, index, n_names] = *found;
      const unsigned int n_operands =
          operation == Operation::vorticity ? dim : 1;

      AssertThrow(operation != Operation::vorticity || dim > 1,
                  dealii::ExcMessage("Invalid component name: \"" + entry +
                                     "\": the vorticity is not defined in "
                                     "one spatial dimension."));
      AssertThrow(index + n_operands <= n_names,
                  dealii::ExcMessage("Invalid component name: \"" + entry +
                                     "\": the vorticity requires dim "
                                     "consecutive components."));

      selection.quantities.push_back(
          {std::uint8_t(operation),
           static_cast<unsigned int>(selection.components.size())});
      for (unsigned int d = 0; d < n_operands; ++d)
        selection.components.push_back({std::uint8_t(section), index + d});
    }

    return selection;
  }


  template <typename Description, int dim, typename Number>
  template <typename MemorySpace>
  auto Postprocessor<Description, dim, Number>::reader(
      const Selection &selection, const StateVector<Number> &state_vector) const
  {
    const auto &selected = selection.components;
    const auto selects = [&](const Section section) {
      return std::any_of(selected.begin(), selected.end(), [&](const auto &it) {
        return Section(it.first) == section;
      });
    };

    const bool read_primitive = selects(Section::primitive);
    const bool read_state = read_primitive || selects(Section::state);
    const bool read_precomputed = selects(Section::precomputed);
    const bool read_initial = selects(Section::initial_precomputed);

    /*
     * Only set up views for vectors that we are actually reading from,
     * and only ensure residency of those parts of the state vector:
     */

    VectorView<View::problem_dimension, MemorySpace> U_view;
    VectorView<View::n_precomputed_values, MemorySpace> precomputed_view;
    VectorView<View::n_initial_precomputed_values, MemorySpace> initial_view;

    if (read_state) {
      copy_to_memory_space<MemorySpace>(state_vector, StateVectorPart::state);
      U_view = state_vector.state
                   .template view<View::problem_dimension, MemorySpace>();
    }

    if (read_precomputed) {
      copy_to_memory_space<MemorySpace>(state_vector,
                                        StateVectorPart::precomputed);
      precomputed_view =
          state_vector.precomputed
              .template view<View::n_precomputed_values, MemorySpace>();
    }

    if (read_initial) {
      copy_to_memory_space<MemorySpace>(initial_precomputed_);
      initial_view =
          initial_precomputed_
              .template view<View::n_initial_precomputed_values, MemorySpace>();
    }

    /*
     * Set up the components, and a view for every parabolic component, on
     * the host, and copy them into arrays residing on the memory space:
     */

    const auto n_components = selected.size();
    Kokkos::View<Kokkos::pair<Section, unsigned int> *, Kokkos::HostSpace>
        host_components("postprocessor_components", n_components);
    Kokkos::View<VectorView<1, MemorySpace> *, Kokkos::HostSpace> host_scalars(
        "postprocessor_scalars", n_components);

    for (unsigned int k = 0; k < n_components; ++k) {
      const auto section = Section(selected[k].first);
      const auto index = selected[k].second;
      host_components[k] = {section, index};

      if (section == Section::dynamic_state) {
        const auto &vector = state_vector.dynamic_state[index];
        copy_to_memory_space<MemorySpace>(vector);
        host_scalars[k] = vector.template view<1, MemorySpace>();
      }
    }

    const auto space = typename MemorySpace::kokkos_space{};
    const auto components =
        Kokkos::create_mirror_view_and_copy(space, host_components);
    const auto scalars =
        Kokkos::create_mirror_view_and_copy(space, host_scalars);

    const SelectView<dim, Number, MemorySpace, HyperbolicSystem> system_views(
        *hyperbolic_system_);

    return [=](auto sentinel, const unsigned int i, const auto &write) {
      using T = decltype(sentinel);

      /*
       * Read all involved tensors once. Reading a single component out of
       * a MultiComponentVector is not any cheaper than reading the full
       * tensor...
       */

      decltype(U_view.template read_tensor<T>(i)) U_i;
      if (read_state)
        U_i = U_view.template read_tensor<T>(i);

      auto primitive_i = U_i;
      if (read_primitive)
        primitive_i = system_views.template view<T>().to_primitive_state(U_i);

      decltype(precomputed_view.template read_tensor<T>(i)) precomputed_i;
      if (read_precomputed)
        precomputed_i = precomputed_view.template read_tensor<T>(i);

      decltype(initial_view.template read_tensor<T>(i)) initial_i;
      if (read_initial)
        initial_i = initial_view.template read_tensor<T>(i);

      for (unsigned int k = 0; k < components.extent(0); ++k) {
        const auto [section, index] = components[k];
        switch (section) {
        case Section::state:
          write(k, U_i[index]);
          break;
        case Section::primitive:
          write(k, primitive_i[index]);
          break;
        case Section::precomputed:
          write(k, precomputed_i[index]);
          break;
        case Section::initial_precomputed:
          write(k, initial_i[index]);
          break;
        default:
          write(k, scalars[k].template read_entry<T>(i));
        }
      }
    };
  }


  template <typename Description, int dim, typename Number>
  std::vector<ScalarHostVector<Number>>
  Postprocessor<Description, dim, Number>::extract(
      const Selection &selection, const StateVector<Number> &state_vector) const
  {
    using MemorySpace = dealii::MemorySpace::Host;

    const auto &offline_data = *offline_data_;
    const unsigned int n_internal = offline_data.n_locally_internal();
    const unsigned int n_owned = offline_data.n_locally_owned();

    /*
     * Extract all components in a single pass:
     */

    std::vector<ScalarHostVector<Number>> components(
        selection.components.size());
    std::vector<Number *> destinations;
    for (auto &it : components) {
      it.reinit(offline_data.scalar_partitioner());
      destinations.push_back(it.begin());
    }

    const auto read = reader<MemorySpace>(selection, state_vector);
    auto *const *data = destinations.data();

    loop<MemorySpace, Number>(
        "postprocessor_extract",
        [=](auto sentinel, const unsigned int i) {
          using T = decltype(sentinel);
          read(sentinel, i, [&](const unsigned int k, const T &value) {
            if constexpr (std::is_same_v<T, dealii::VectorizedArray<Number>>)
              value.store(data[k] + i);
            else
              data[k][i] = value;
          });
        },
        0,
        n_internal,
        n_owned);

    for (auto &it : components)
      it.update_ghost_values();

    /*
     * Then postprocess extracted components:
     */

    const auto sparsity_simd_view = offline_data.sparsity_pattern_simd().view();
    const auto lumped_mass_matrix_view =
        offline_data.lumped_mass_matrix().template view<1>();
    const auto cij_matrix_view = offline_data.cij_matrix().view();

    std::vector<ScalarHostVector<Number>> extracted(selection.size());
    for (unsigned int k = 0; k < selection.size(); ++k) {
      const auto operation = Operation(selection.quantities[k].first);
      const auto first = selection.quantities[k].second;
      auto &result = extracted[k];

      if (operation == Operation::value) {
        result = std::move(components[first]);
        continue;
      }

      result.reinit(offline_data.scalar_partitioner());
      const auto *operands = &components[first];

      const auto body = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd_view.row_length(i);
        if (row_length == 1)
          return;

        const unsigned int stride_size = sparsity_simd_view.stride_of_row(i);

        dealii::Tensor<1, dim, T> gradient;
        dealii::Tensor<1, dim == 2 ? 1 : dim, T> curl;

        const unsigned int *js = sparsity_simd_view.columns(i);
        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += stride_size) {

          const auto c_ij = cij_matrix_view.template read_tensor<T>(i, col_idx);

          if (operation == Operation::schlieren) {
            gradient -= c_ij * read_entry<T>(operands[0], js);

          } else {
            dealii::Tensor<1, dim, T> u_j;
            for (unsigned int d = 0; d < dim; ++d)
              u_j[d] = read_entry<T>(operands[d], js);

            if constexpr (dim == 2) {
              curl[0] -= cross_product_2d(c_ij) * u_j;
            } else if constexpr (dim == 3) {
              curl -= cross_product_3d(c_ij, u_j);
            }
          }
        }

        const auto m_i = lumped_mass_matrix_view.template read_entry<T>(i);

        T value_i;
        if (operation == Operation::schlieren)
          value_i = gradient.norm() / m_i;
        else
          value_i = (dim == 2 ? curl[0] / m_i : curl.norm() / m_i);

        write_entry<T>(result, value_i, i);
      };

      cpu_simd_loop<Number>(
          "postprocessor_postprocess", body, 0, n_internal, n_owned);

      offline_data.affine_constraints().distribute(result);
      result.update_ghost_values();
    }

    return extracted;
  }


  template <typename Description, int dim, typename Number>
  void Postprocessor<Description, dim, Number>::extract(
      const Selection &selection,
      const StateVector<Number> &state_vector,
      const unsigned int *indices,
      const unsigned int n_points,
      Number *values) const
  {
    AssertThrow(
        std::all_of(selection.quantities.begin(),
                    selection.quantities.end(),
                    [](const auto &it) {
                      return Operation(it.first) == Operation::value;
                    }),
        dealii::ExcMessage("Postprocessed quantities cannot be extracted at "
                           "individual points."));

    /* Every selected quantity is a value, thus components == quantities: */
    const auto read = reader<selected_memory_space_t>(selection, state_vector);
    const auto n_components =
        static_cast<unsigned int>(selection.components.size());

    /* The points are scattered: run a scalar loop only. */
    loop<selected_memory_space_t, Number>(
        "postprocessor_extract_points",
        [=](auto /*sentinel*/, const unsigned int p) {
          auto *values_p = values + p * n_components;
          read(Number(),
               indices[p],
               [=](const unsigned int k, const Number value) {
                 values_p[k] = value;
               });
        },
        0,
        0,
        n_points);
  }

#endif
} // namespace ryujin
