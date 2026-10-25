//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/loop.h>
#include <ryujin/base/simd.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/hyperbolic_system.h>
#include <ryujin/linear_algebra/multicomponent_vector.h>
#include <ryujin/linear_algebra/state_vector.h>
#include <ryujin/typed/hyperbolic_module.h>
#include <ryujin/typed/initial_state.h>
#include <ryujin/typed/initial_values.h>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <array>
#include <string>
#include <tuple>
#include <vector>

//
// A skeleton hyperbolic description: a minimal, instantiable reference
// implementation of everything the generic classes (HyperbolicModule,
// InitialValues, SolutionTransfer, Postprocessor, Simulation) require from
// a hyperbolic system, its compute kernels, and its initial states.
//
// The state is a single scalar u with n_precomputed precomputed values.
// All fluxes vanish and all kernels are trivial; the point of the skeleton
// is that it compiles, not that it computes anything meaningful.
//

namespace ryujin
{
  namespace Skeleton
  {
    template <int n_precomputed,
              int dim,
              typename Number,
              typename MemorySpace = dealii::MemorySpace::Host>
    class HyperbolicSystemView;


    /**
     * The skeleton hyperbolic system with one run time parameter.
     */
    template <int n_precomputed>
    class HyperbolicSystem : public Interface::HyperbolicSystem
    {
    public:
      HyperbolicSystem(const std::string &subsection = "/HyperbolicSystem")
          : Interface::HyperbolicSystem("Skeleton",
                                        {{{"u"}, {"u"}, {"u"}}},
                                        {{{"u"}, {"u"}, {"u"}}},
                                        {{names(), names(), names()}},
                                        {},
                                        subsection)
      {
        add_parameter("skeleton parameter", parameter_, "A parameter");
      }

      template <int dim,
                typename Number = double,
                typename MemorySpace = dealii::MemorySpace::Host>
      using View =
          HyperbolicSystemView<n_precomputed, dim, Number, MemorySpace>;

      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return View<dim, Number, MemorySpace>{*this};
      }

      /**
       * Fill all precomputed values with zero.
       */
      template <typename MemorySpace = dealii::MemorySpace::Host,
                int dim,
                typename ScalarNumber>
      void fill_precomputed_values(
          const OfflineData<dim, ScalarNumber> &offline_data,
          StateVector<ScalarNumber> &state_vector,
          const bool skip_constrained_dofs = true) const
      {
        const unsigned int n_internal = offline_data.n_locally_internal();
        const unsigned int n_owned = offline_data.n_locally_owned();

        const auto sparsity_simd_view =
            offline_data.sparsity_pattern_simd().template view<MemorySpace>();
        const auto precomputed_view =
            state_vector.precomputed
                .template view<n_precomputed, MemorySpace>();

        const auto body = [=](auto sentinel, unsigned int i) {
          using T = decltype(sentinel);

          if (skip_constrained_dofs && sparsity_simd_view.row_length(i) == 1)
            return;

          std::array<T, n_precomputed> precomputed;
          precomputed.fill(T(0.));
          precomputed_view.template write_tensor<T>(precomputed, i);
        };

        loop<MemorySpace, ScalarNumber>(
            "skeleton_kernel_01b", body, 0, n_internal, n_owned);
      }

      ACCESSOR_READ_ONLY(parameter)

    private:
      static std::vector<std::string> names()
      {
        std::vector<std::string> names;
        for (int k = 0; k < n_precomputed; ++k)
          names.push_back("q_" + std::to_string(k));
        return names;
      }

      double parameter_ = 0.;
    };


    /**
     * The view of the skeleton hyperbolic system.
     */
    template <int n_precomputed, int dim, typename Number, typename MemorySpace>
    class HyperbolicSystemView
    {
    public:
      using ScalarNumber = typename get_value_type<Number>::type;

      static constexpr unsigned int problem_dimension = 1;

      using state_type = dealii::Tensor<1, problem_dimension, Number>;

      using flux_contribution_type =
          dealii::Tensor<1, problem_dimension, dealii::Tensor<1, dim, Number>>;

      static constexpr unsigned int n_precomputed_values = n_precomputed;

      using precomputed_type = std::array<Number, n_precomputed_values>;

      static constexpr unsigned int n_initial_precomputed_values = 0;

      using initial_precomputed_type =
          std::array<Number, n_initial_precomputed_values>;

      template <int n_comp>
      using VectorView =
          MultiComponentVectorView<ScalarNumber, n_comp, MemorySpace, false>;

      using PrecomputedVectorView = VectorView<n_precomputed_values>;

      using InitialPrecomputedVectorView =
          VectorView<n_initial_precomputed_values>;

      static constexpr bool have_high_order_flux = false;

      static constexpr bool have_source_terms = false;

      HyperbolicSystemView(const HyperbolicSystem<n_precomputed> &) {}

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE bool
      is_admissible(const state_type & /*U*/) const
      {
        return true;
      }

      template <typename Lambda>
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE state_type
      apply_boundary_conditions(const dealii::types::boundary_id /*id*/,
                                const state_type &U,
                                const dealii::Tensor<1, dim, Number> & /*n*/,
                                const Lambda & /*get_dirichlet_data*/) const
      {
        return U;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE flux_contribution_type
      flux_contribution(const PrecomputedVectorView & /*pv*/,
                        const InitialPrecomputedVectorView & /*piv*/,
                        const unsigned int /*i*/,
                        const state_type & /*U_i*/) const
      {
        return flux_contribution_type{};
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE flux_contribution_type
      flux_contribution(const PrecomputedVectorView & /*pv*/,
                        const InitialPrecomputedVectorView & /*piv*/,
                        const unsigned int * /*js*/,
                        const state_type & /*U_j*/) const
      {
        return flux_contribution_type{};
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE state_type
      flux_divergence(const flux_contribution_type & /*flux_i*/,
                      const flux_contribution_type & /*flux_j*/,
                      const dealii::Tensor<1, dim, Number> & /*c_ij*/) const
      {
        return state_type{};
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE state_type
      to_primitive_state(const state_type &U) const
      {
        return U;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE state_type
      from_primitive_state(const state_type &U) const
      {
        return U;
      }

      template <typename Lambda>
      DEAL_II_HOST_DEVICE_ALWAYS_INLINE state_type apply_galilei_transform(
          const state_type &U, const Lambda & /*lambda*/) const
      {
        return U;
      }
    };


    /**
     * The types of the view of the hyperbolic system System that all
     * kernel views share.
     */
    template <typename System, int dim, typename Number, typename MemorySpace>
    struct KernelViewTypes {
      using View = typename System::template View<dim, Number, MemorySpace>;
      using state_type = typename View::state_type;
      using flux_contribution_type = typename View::flux_contribution_type;
      using PrecomputedVectorView = typename View::PrecomputedVectorView;
    };


    /**
     * A trivial indicator view.
     */
    template <typename System, int dim, typename Number, typename MemorySpace>
    struct IndicatorView : KernelViewTypes<System, dim, Number, MemorySpace> {
      using Base = KernelViewTypes<System, dim, Number, MemorySpace>;
      using typename Base::flux_contribution_type;
      using typename Base::PrecomputedVectorView;
      using typename Base::state_type;

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
      reset(const PrecomputedVectorView & /*pv*/,
            const unsigned int /*i*/,
            const state_type & /*U_i*/,
            const flux_contribution_type & /*flux_i*/)
      {
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
      accumulate(const PrecomputedVectorView & /*pv*/,
                 const unsigned int * /*js*/,
                 const state_type & /*U_j*/,
                 const flux_contribution_type & /*flux_j*/,
                 const dealii::Tensor<1, dim, Number> & /*c_ij*/)
      {
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
      alpha(const Number /*hd_i*/) const
      {
        return Number(0.);
      }
    };


    /**
     * A trivial limiter view that does not limit.
     */
    template <typename System, int dim, typename Number, typename MemorySpace>
    struct LimiterView : KernelViewTypes<System, dim, Number, MemorySpace> {
      using Base = KernelViewTypes<System, dim, Number, MemorySpace>;
      using typename Base::flux_contribution_type;
      using typename Base::PrecomputedVectorView;
      using typename Base::state_type;

      static constexpr unsigned int n_bounds = 1;

      using Bounds = std::array<Number, n_bounds>;

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE unsigned int iterations() const
      {
        return 0;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
      reset(const PrecomputedVectorView & /*pv*/,
            const unsigned int /*i*/,
            const state_type & /*U_i*/,
            const flux_contribution_type & /*flux_i*/)
      {
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE void
      accumulate(const PrecomputedVectorView & /*pv*/,
                 const unsigned int * /*js*/,
                 const state_type & /*U_j*/,
                 const flux_contribution_type & /*flux_j*/,
                 const dealii::Tensor<1, dim, Number> & /*scaled_c_ij*/,
                 const state_type & /*affine_shift*/)
      {
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Bounds
      bounds(const Number /*hd_i*/) const
      {
        return Bounds{};
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE std::tuple<Number, bool>
      limit(const Bounds & /*bounds*/,
            const state_type & /*U*/,
            const state_type & /*P*/,
            const Number /*t_min*/ = Number(0.),
            const Number t_max = Number(1.)) const
      {
        return {t_max, true};
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Bounds
      projection_bounds_from_state(const PrecomputedVectorView & /*pv*/,
                                   const unsigned int /*i*/,
                                   const state_type & /*U_i*/) const
      {
        return Bounds{};
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Bounds combine_bounds(
          const Bounds &bounds_left, const Bounds & /*bounds_right*/) const
      {
        return bounds_left;
      }

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Bounds
      fully_relax_bounds(const Bounds &bounds, const Number & /*hd*/) const
      {
        return bounds;
      }
    };


    /**
     * A trivial wave speed estimator view.
     */
    template <typename System, int dim, typename Number, typename MemorySpace>
    struct WaveSpeedEstimatorView
        : KernelViewTypes<System, dim, Number, MemorySpace> {
      using Base = KernelViewTypes<System, dim, Number, MemorySpace>;
      using typename Base::flux_contribution_type;
      using typename Base::PrecomputedVectorView;
      using typename Base::state_type;

      DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
      compute(const PrecomputedVectorView & /*pv*/,
              const unsigned int /*i*/,
              const unsigned int * /*js*/,
              const state_type & /*U_i*/,
              const state_type & /*U_j*/,
              const flux_contribution_type & /*flux_i*/,
              const flux_contribution_type & /*flux_j*/,
              const dealii::Tensor<1, dim, Number> & /*n_ij*/) const
      {
        return Number(1.);
      }
    };


    /**
     * A compute kernel for the hyperbolic system System: constructed from
     * the hyperbolic system and a subsection, it hands out KernelView
     * objects.
     */
    template <typename System,
              template <typename, int, typename, typename>
              class KernelView>
    class Kernel
    {
    public:
      Kernel(const System & /*hyperbolic_system*/,
             const std::string & /*subsection*/)
      {
      }

      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return KernelView<System, dim, Number, MemorySpace>{};
      }
    };

    template <typename System, typename ScalarNumber = double>
    using Indicator = Kernel<System, IndicatorView>;

    template <typename System, typename ScalarNumber = double>
    using Limiter = Kernel<System, LimiterView>;

    template <typename System, typename ScalarNumber = double>
    using WaveSpeedEstimator = Kernel<System, WaveSpeedEstimatorView>;


    /**
     * The (only) initial state "uniform", returning the zero state.
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    class Uniform : public InitialState<HyperbolicSystem, dim, Number>
    {
    public:
      using state_type =
          typename InitialState<HyperbolicSystem, dim, Number>::state_type;

      Uniform(const HyperbolicSystem & /*hyperbolic_system*/,
              const std::string &subsection)
          : InitialState<HyperbolicSystem, dim, Number>("uniform", subsection)
      {
      }

      state_type compute(const dealii::Point<dim> & /*point*/,
                         Number /*t*/) final
      {
        return state_type{};
      }
    };


    /**
     * The default initial states for the hyperbolic system HyperbolicSystem.
     * Use it to specialize InitialStateLibrary::defaults().
     */
    template <typename HyperbolicSystem, int dim, typename Number>
    auto initial_states()
    {
      using L = InitialStateLibrary<HyperbolicSystem, dim, Number>;
      return std::vector<typename L::Creator>{
          L::template creator<Uniform<HyperbolicSystem, dim, Number>>(),
      };
    }


    /**
     * The skeleton hyperbolic description for the hyperbolic system
     * System.
     */
    template <typename System>
    struct HyperbolicDescription {
      using HyperbolicSystem = System;

      template <int dim, typename Number>
      using InitialValues =
          ryujin::InitialValues<HyperbolicSystem, dim, Number>;

      template <typename ScalarNumber = double>
      using Indicator = Skeleton::Indicator<HyperbolicSystem, ScalarNumber>;

      template <typename ScalarNumber = double>
      using Limiter = Skeleton::Limiter<HyperbolicSystem, ScalarNumber>;

      template <typename ScalarNumber = double>
      using WaveSpeedEstimator =
          Skeleton::WaveSpeedEstimator<HyperbolicSystem, ScalarNumber>;

      template <int dim, typename Number>
      using HyperbolicModule =
          ryujin::HyperbolicModule<HyperbolicDescription, dim, Number>;
    };
  } // namespace Skeleton
} // namespace ryujin
