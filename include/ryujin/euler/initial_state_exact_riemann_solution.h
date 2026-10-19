//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 - 2026 by the ryujin authors
// Copyright (C) 2025 by Triad National Security, LLC
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/euler_aeos/nasg_riemann_solver.h>
#include <ryujin/typed/initial_state_library.h>

#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    /**
     * The exact Riemann solution.
     *
     * This initial state class computes the analytic (self-similar)
     * solution of the 1D Riemann problem for the compressible Euler
     * equations with a Noble-Abel-Stiffened Gas equation of state with the
     * help of EulerAEOS::NASGRiemannSolver, see
     * @cite ClaytonGuermondPopov-2022 and @cite Toro2009. The covolume b
     * and reference pressure pinf are taken from the equation of state of
     * the hyperbolic system (if available), otherwise a polytropic gas
     * equation of state is assumed.
     *
     * @note This class returns the analytic solution as a function of time
     * @p t and position @p x.
     *
     * @note If the hyperbolic system does not provide gamma (for example,
     * EulerAEOS), the "gamma" parameter of this class must coincide with
     * the gamma of the equation of state. Similarly, the equation of state
     * should be a Noble-Abel-Stiffened Gas EOS. Neither condition is
     * checked: with a mismatched gamma or a different equation of state,
     * the returned solution will not be an exact solution of the
     * hyperbolic system.
     *
     * @ingroup EulerEquations
     */
    template <typename Description, int dim, typename Number>
    class ExactRiemannSolution : public InitialState<Description, dim, Number>
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View = typename HyperbolicSystem::template View<dim, Number>;
      using state_type = typename View::state_type;

      using ScalarNumber = typename View::ScalarNumber;

      /**
       * Compile time options of the Riemann solver: a single, constant
       * gamma, and covolume and reference pressure only if provided by the
       * hyperbolic system. safe_division is enabled so that the solution
       * remains well defined if a vacuum is formed.
       */
      static constexpr EulerAEOS::NASGRiemannSolverOptions
          riemann_solver_options{.covolume = View::have_covolume_constant,
                                 .pinf = View::have_covolume_constant,
                                 .safe_division = true,
                                 .variable_gamma = false};

      using RiemannSolver =
          EulerAEOS::NASGRiemannSolver<ScalarNumber, riemann_solver_options>;
      using RiemannSolverView =
          EulerAEOS::NASGRiemannSolverView<Number, riemann_solver_options>;
      using primitive_type = typename RiemannSolverView::primitive_type;


      ExactRiemannSolution(const HyperbolicSystem &hyperbolic_system,
                           const std::string subsection)
          : InitialState<Description, dim, Number>("exact riemann solution",
                                                   subsection)
          , hyperbolic_system_(hyperbolic_system)
          , riemann_solver_(subsection +
                            "/exact riemann solution/riemann solver")
      {
        gamma_ = 1.4;
        if constexpr (!View::have_gamma) {
          this->add_parameter("gamma", gamma_, "The ratio of specific heats");
        }

        primitive_left_[0] = 1.4;
        primitive_left_[1] = 0.0;
        primitive_left_[2] = 1.0;
        this->add_parameter("primitive state left",
                            primitive_left_,
                            "1d primitive state [rho, u, p] on the left");

        primitive_right_[0] = 1.4;
        primitive_right_[1] = 0.0;
        primitive_right_[2] = 1.0;
        this->add_parameter("primitive state right",
                            primitive_right_,
                            "1d primitive state [rho, u, p] on the right");

        // Convert the primitive states to a full Riemann solution:
        const auto prepare_riemann_data = [&]() {
          const auto view = hyperbolic_system_.template view<dim, Number>();
          if constexpr (View::have_gamma) {
            gamma_ = view.gamma();
          }

          riemann_solver_.set_gamma(gamma_);

          if constexpr (View::have_covolume_constant) {
            riemann_solver_.set_equation_of_state(
                view.eos_covolume_constant(),
                view.eos_interpolation_pinfty(),
                /*compute_expensive_bounds*/ false);
          }

          const auto riemann_solver_view =
              riemann_solver_.template view<Number>();

          riemann_data_left_ =
              expand_state(riemann_solver_view, primitive_left_);
          riemann_data_right_ =
              expand_state(riemann_solver_view, primitive_right_);

          riemann_solution_ = riemann_solver_view.solve(riemann_data_left_,
                                                        riemann_data_right_);
        };

        this->parse_parameters_call_back.connect(prepare_riemann_data);
        prepare_riemann_data();
      }


      state_type compute(const dealii::Point<dim> &point, Number t) final
      {
        const auto view = hyperbolic_system_.template view<dim, Number>();

        const double &x = point[0];

        dealii::Tensor<1, 3, Number> primitive_state;

        if (t < 1.e-14 && x < 0.) {
          /* Left state: */
          primitive_state = primitive_left_;

        } else if (t < 1.e-14 && x > 0.) {
          /* Right state: */
          primitive_state = primitive_right_;

        } else {
          const Number xi = x / t;

          const auto riemann_solver_view =
              riemann_solver_.template view<Number>();
          const auto sampled =
              riemann_solver_view.sample(riemann_solution_, xi);

          for (unsigned int i = 0; i < 3; ++i)
            primitive_state[i] = sampled[i];
        }

        using state_type_1d =
            typename HyperbolicSystem::template View<1, Number>::state_type;
        static_assert(state_type_1d::dimension <=
                      dealii::Tensor<1, 3, Number>::dimension);

        state_type_1d result;
        for (unsigned int i = 0; i < state_type_1d::dimension; ++i)
          result[i] = primitive_state[i];
        return view.from_initial_state(result);
      }

    private:
      //@}
      /**
       * @name Run time options
       */
      //@{

      Number gamma_;

      dealii::Tensor<1, 3, Number> primitive_left_;
      dealii::Tensor<1, 3, Number> primitive_right_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      const HyperbolicSystem &hyperbolic_system_;

      RiemannSolver riemann_solver_;

      primitive_type riemann_data_left_;
      primitive_type riemann_data_right_;
      typename RiemannSolverView::RiemannSolution riemann_solution_;

      //@}
      /**
       * @name Internal methods
       */
      //@{

      /**
       * Expand a 1d primitive state \f$[\rho, u, p]\f$ into the Riemann
       * data \f$[\rho, u, p, \gamma, a]\f$ expected by
       * EulerAEOS::NASGRiemannSolver.
       */
      primitive_type
      expand_state(const RiemannSolverView &riemann_solver_view,
                   const dealii::Tensor<1, 3, Number> &primitive_state) const
      {
        primitive_type result;
        result[0] = primitive_state[0];
        result[1] = primitive_state[1];
        result[2] = primitive_state[2];
        result[3] = gamma_;
        result[4] =
            riemann_solver_view.speed_of_sound(result[0], result[2], result[3]);
        return result;
      }

      //@}
    };
  } // namespace EulerInitialStates
} // namespace ryujin
