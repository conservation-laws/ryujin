//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "flux_library.h"

#include <convenience_macros.h>
#include <deal.II/base/config.h>
#include <discretization.h>
#include <multicomponent_vector.h>
#include <openmp.h>
#include <patterns_conversion.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>

namespace ryujin
{
  namespace ScalarConservation
  {
    /**
     * A scalar conservation equation for a state \f$u\f$ with a user
     * specified flux \f$f(u)\f$.
     *
     * @ingroup ScalarConservationEquations
     */
    class HyperbolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the hyperbolic system as a string.
       */
      static inline std::string problem_name = "Scalar conservation equation";

      /**
       * Constructor.
       */
      HyperbolicSystem(const std::string &subsection = "/HyperbolicSystem");

    private:
      /**
       * @name Runtime parameters, internal fields and methods
       */
      //@{
      std::string flux_;

      bool riemann_solver_greedy_wavespeed_;
      bool riemann_solver_averaged_entropy_;
      unsigned int riemann_solver_random_entropies_;

      FluxLibrary::flux_list_type flux_list_;
      using Flux = FluxLibrary::Flux;
      std::shared_ptr<Flux> selected_flux_;

      //@}


    public:
      /**
       * A view on the HyperbolicSystem for a given dimension @p dim and
       * choice of number type @p Number (which can be a scalar float, or
       * double, as well as a VectorizedArray holding packed scalars.
       */
      template <int dim, typename Number>
      class View
      {
      public:
        /**
         * Constructor taking a reference to the underlying
         * HyperbolicSystem
         */
        View(const HyperbolicSystem &hyperbolic_system)
            : hyperbolic_system_(hyperbolic_system)
        {
        }

        /**
         * Create a modified view from the current one:
         */
        template <int dim2, typename Number2>
        auto view() const
        {
          return View<dim2, Number2>{hyperbolic_system_};
        }

        /**
         * The underlying scalar number type.
         */
        using ScalarNumber = typename get_value_type<Number>::type;

        /**
         * @name Access to runtime parameters
         */
        //@{

        DEAL_II_ALWAYS_INLINE inline const std::string &flux() const
        {
          return hyperbolic_system_.flux_;
        }

        DEAL_II_ALWAYS_INLINE inline bool
        riemann_solver_greedy_wavespeed() const
        {
          return hyperbolic_system_.riemann_solver_greedy_wavespeed_;
        }

        DEAL_II_ALWAYS_INLINE inline bool
        riemann_solver_averaged_entropy() const
        {
          return hyperbolic_system_.riemann_solver_averaged_entropy_;
        }

        DEAL_II_ALWAYS_INLINE inline unsigned int
        riemann_solver_random_entropies() const
        {
          return hyperbolic_system_.riemann_solver_random_entropies_;
        }

        DEAL_II_ALWAYS_INLINE inline ScalarNumber
        riemann_solver_approximation_delta() const
        {
          const auto &flux = hyperbolic_system_.selected_flux_;
          return ScalarNumber(flux->derivative_approximation_delta());
        }

        //@}
        /**
         * @name Low-level access to the flux function parser:
         */
        //@{

        /**
         * For a given state \f$u\f$ compute the flux \f$f(u)\f$.
         */
        DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
        flux_function(const Number &u) const;

        /**
         * For a given state \f$u\f$ compute the flux gradient \f$f'(u)\f$.
         */
        DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
        flux_gradient_function(const Number &u) const;

        //@}
        /**
         * @name Internal data
         */
        //@{

      private:
        const HyperbolicSystem &hyperbolic_system_;

      public:
        //@}
        /**
         * @name Types and compile time constants
         */
        //@{

        /**
         * The dimension of the state space.
         */
        static constexpr unsigned int problem_dimension = 1;

        /**
         * The storage type used for a (conserved) state vector \f$\boldsymbol
         * U\f$.
         */
        using state_type = dealii::Tensor<1, problem_dimension, Number>;

        /**
         * MulticomponentVector for storing a vector of conserved states:
         */
        using vector_type =
            MultiComponentVector<ScalarNumber, problem_dimension>;

        /**
         * An array holding all component names of the conserved state as a
         * string.
         */
        static inline const auto component_names =
            std::array<std::string, problem_dimension>{"u"};

        /**
         * The storage type used for a primitive state vector.
         */
        using primitive_state_type =
            dealii::Tensor<1, problem_dimension, Number>;

        /**
         * An array holding all component names of the primitive state as a
         * string.
         */
        static inline const auto primitive_component_names =
            std::array<std::string, problem_dimension>{"u"};

        /**
         * The storage type used for the flux \f$\mathbf{f}\f$.
         */
        using flux_type = dealii::
            Tensor<1, problem_dimension, dealii::Tensor<1, dim, Number>>;

        /**
         * The storage type used for flux contributions.
         */
        using flux_contribution_type = flux_type;

        //@}
        /**
         * @name Precomputed quantities
         */
        //@{

        /**
         * The number of precomputed initial values.
         */
        static constexpr unsigned int n_precomputed_initial_values = 0;

        /**
         * Array type used for precomputed initial values.
         */
        using precomputed_initial_state_type =
            std::array<Number, n_precomputed_initial_values>;

        /**
         * MulticomponentVector for storing a vector of precomputed initial
         * states:
         */
        using precomputed_initial_vector_type =
            MultiComponentVector<ScalarNumber, n_precomputed_initial_values>;

        /**
         * An array holding all component names of the precomputed values.
         */
        static inline const auto precomputed_initial_names =
            std::array<std::string, n_precomputed_initial_values>{};

        /**
         * The number of precomputed values.
         */
        static constexpr unsigned int n_precomputed_values = 2. * dim;

        /**
         * Array type used for precomputed values.
         */
        using precomputed_state_type = std::array<Number, n_precomputed_values>;

        /**
         * MulticomponentVector for storing a vector of precomputed states:
         */
        using precomputed_vector_type =
            MultiComponentVector<ScalarNumber, n_precomputed_values>;

        /**
         * An array holding all component names of the precomputed values.
         */
        static inline const auto precomputed_names =
            []() -> std::array<std::string, n_precomputed_values> {
          if constexpr (dim == 1)
            return {"f", "df"};
          else if constexpr (dim == 2)
            return {"f_1", "f_2", "df_1", "df_2"};
          else if constexpr (dim == 3)
            return {"f_1", "f_2", "f_3", "df_1", "df_2", "df_3"};
          __builtin_trap();
        }();

        /**
         * The number of precomputation cycles.
         */
        static constexpr unsigned int n_precomputation_cycles = 1;

        /**
         * Step 0: precompute values for hyperbolic update. This routine is
         * called within our usual loop() idiom in HyperbolicModule
         */
        template <typename DISPATCH, typename SPARSITY>
        void
        precomputation_loop(unsigned int /*cycle*/,
                            const DISPATCH &dispatch_check,
                            precomputed_vector_type & /*precomputed_values*/,
                            const SPARSITY & /*sparsity_simd*/,
                            const vector_type & /*U*/,
                            unsigned int /*left*/,
                            unsigned int /*right*/) const;

        //@}
        /**
         * @name Computing derived physical quantities
         */
        //@{

        /**
         * Return the scalar value stored in the state "vector". Given the
         * fact that a scalar conservation equation is indeed scalar this
         * simply unwraps the Tensor from the state_type and returns the
         * one and only entry.
         */
        static Number state(const state_type &U);

        /**
         * For a given state <code>U</code>, compute the square entropy
         * \f[
         *   \eta = 1/2 u^2.
         * \f]
         */
        Number square_entropy(const Number &u) const;

        /**
         * For a given state <code>U</code>, compute the derivative of the
         * square entropy
         * \f[
         *   \eta' = u.
         * \f]
         */
        Number square_entropy_derivative(const Number &u) const;

        /**
         * For a given state <code>U</code>, compute the Krŭzkov entropy
         * \f[
         *   \eta = |u-k|.
         * \f]
         */
        Number kruzkov_entropy(const Number &k, const Number &u) const;

        /**
         * For a given state <code>U</code>, compute the derivative of the
         * Krŭzkov entropy:
         * \f[
         *   \eta' = \text{sgn}(u-k).
         * \f]
         */
        Number kruzkov_entropy_derivative(const Number &k,
                                          const Number &u) const;

        /**
         * Returns whether the state @p U is admissible. If @p U is a
         * vectorized state then @p U is admissible if all vectorized
         * values are admissible.
         */
        bool is_admissible(const state_type & /*U*/) const
        {
          return true;
        }

        //@}
        /**
         * @name Special functions for boundary states
         */
        //@{

        /**
         * Apply boundary conditions.
         */
        template <typename Lambda>
        state_type apply_boundary_conditions(
            const dealii::types::boundary_id /*id*/,
            const state_type &U,
            const dealii::Tensor<1, dim, Number> & /*normal*/,
            const Lambda & /*get_dirichlet_data*/) const;

        //@}
        /**
         * @name Flux computations
         */
        //@{

        /**
         * Helper function that given a @p precomputed_state constructs a
         * dealii::Tensor with the flux @f$f(u)@f$.
         */
        dealii::Tensor<1, dim, Number> construct_flux_tensor(
            const precomputed_state_type &precomputed_state) const;

        /**
         * Helper function that given a @p precomputed_state constructs a
         * dealii::Tensor with the gradient of the flux @f$f(u)@f$ with
         * respect to the state @f$u@f$.
         */
        dealii::Tensor<1, dim, Number> construct_flux_gradient_tensor(
            const precomputed_state_type &precomputed_state) const;

        /**
         * Given a state @p U_i and an index @p i compute flux contributions.
         *
         * Intended usage:
         * ```
         * Indicator<dim, Number> indicator;
         * for (unsigned int i = n_internal; i < n_owned; ++i) {
         *   // ...
         *   const auto flux_i = flux_contribution(precomputed..., i, U_i);
         *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
         *     // ...
         *     const auto flux_j = flux_contribution(precomputed..., js, U_j);
         *     const auto flux_ij = flux(flux_i, flux_j);
         *   }
         * }
         * ```
         *
         * For the Euler equations we simply compute <code>f(U_i)</code>.
         */
        flux_contribution_type
        flux_contribution(const precomputed_vector_type &pv,
                          const precomputed_initial_vector_type & /*piv*/,
                          const unsigned int i,
                          const state_type & /*U_i*/) const;

        flux_contribution_type
        flux_contribution(const precomputed_vector_type &pv,
                          const precomputed_initial_vector_type & /*piv*/,
                          const unsigned int *js,
                          const state_type & /*U_j*/) const;

        /**
         * Given flux contributions @p flux_i and @p flux_j compute the flux
         * <code>(-f(U_i) - f(U_j)</code>
         */
        flux_type flux(const flux_contribution_type &flux_i,
                       const flux_contribution_type &flux_j) const;

        /** The low-order and high-order fluxes are the same */
        static constexpr bool have_high_order_flux = false;

        flux_type
        high_order_flux(const flux_contribution_type &,
                        const flux_contribution_type &) const = delete;

        //@}
        /**
         * @name Computing stencil source terms
         */
        //@{

        /** We do not have source terms */
        static constexpr bool have_source_terms = false;

        state_type low_order_source(const precomputed_vector_type &pv,
                                    const unsigned int i,
                                    const state_type &U_i,
                                    const ScalarNumber t,
                                    const ScalarNumber tau) const = delete;

        state_type high_order_source(const precomputed_vector_type &pv,
                                     const unsigned int i,
                                     const state_type &U_i,
                                     const ScalarNumber t,
                                     const ScalarNumber tau) const = delete;

        //@}
        /**
         * @name State transformations
         */
        //@{

        /**
         * Given a state vector associated with a different spatial
         * dimensions than the current one, return an "expanded" version of
         * the state vector associated with @a dim spatial dimensions where
         * the momentum vector of the conserved state @p state is expaned
         * with zeros to a total length of @a dim entries.
         *
         * @note @a dim has to be larger or equal than the dimension of the
         * @a ST vector.
         */
        template <typename ST>
        state_type expand_state(const ST &state) const
        {
          return state;
        }

        /**
         * Given a primitive state [rho, u_1, ..., u_d, p] return a conserved
         * state
         */
        state_type
        from_primitive_state(const primitive_state_type &primitive_state) const
        {
          return primitive_state;
        }

        /**
         * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
         * p]
         */
        primitive_state_type to_primitive_state(const state_type &state) const
        {
          return state;
        }

        /**
         * Transform the current state according to a  given operator
         * @p lambda acting on a @a dim dimensional momentum (or velocity)
         * vector.
         */
        template <typename Lambda>
        state_type apply_galilei_transform(const state_type &state,
                                           const Lambda & /*lambda*/) const
        {
          return state;
        }

      }; /* HyperbolicSystem::View */

      template <int dim, typename Number>
      friend class View;

      /**
       * Return a view on the Hyperbolic System for a given dimension @p
       * dim and choice of number type @p Number (which can be a scalar
       * float, or double, as well as a VectorizedArray holding packed
       * scalars.
       */
      template <int dim, typename Number>
      auto view() const
      {
        return View<dim, Number>{*this};
      }
    }; /* HyperbolicSystem */


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    inline HyperbolicSystem::HyperbolicSystem(const std::string &subsection)
        : ParameterAcceptor(subsection)
    {
      flux_ = "burgers";
      add_parameter("flux",
                    flux_,
                    "The scalar flux. Valid names are given by any of the "
                    "subsections defined below");

      riemann_solver_greedy_wavespeed_ = false;
      add_parameter(
          "riemann solver greedy wavespeed",
          riemann_solver_greedy_wavespeed_,
          "Use a greedy wavespeed estimate instead of a guaranteed upper bound "
          "on the maximal wavespeed (for convex fluxes).");

      riemann_solver_averaged_entropy_ = false;
      add_parameter(
          "riemann solver averaged entropy",
          riemann_solver_averaged_entropy_,
          "In addition to the wavespeed estimate based on the Roe average and "
          "flux gradients of the left and right state also enforce an entropy "
          "inequality on the averaged Krŭzkov entropy.");

      riemann_solver_random_entropies_ = 0;
      add_parameter(
          "riemann solver random entropies",
          riemann_solver_random_entropies_,
          "In addition to the wavespeed estimate based on the Roe average and "
          "flux gradients of the left and right state also enforce an entropy "
          "inequality on the prescribed number of random Krŭzkov entropies.");

      /*
       * And finally populate the flux list with all flux configurations
       * defined in the FluxLibrary namespace:
       */
      FluxLibrary::populate_flux_list(flux_list_, subsection);

      const auto populate_functions = [this]() {
        bool initialized = false;
        for (auto &it : flux_list_)

          /* Populate flux functions: */
          if (it->name() == flux_) {
            selected_flux_ = it;
            it->parse_parameters_call_back();
            problem_name = "Scalar conservation equation (" + it->name() +
                           ": " + it->flux_formula() + ")";
            initialized = true;
            break;
          }

        AssertThrow(initialized,
                    dealii::ExcMessage(
                        "Could not find a flux description with name \"" +
                        flux_ + "\""));
      };

      ParameterAcceptor::parse_parameters_call_back.connect(populate_functions);
      populate_functions();
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
    HyperbolicSystem::View<dim, Number>::flux_function(const Number &u) const
    {
      const auto &flux = hyperbolic_system_.selected_flux_;
      dealii::Tensor<1, dim, Number> result;

      /* This access by calling into value() repeatedly is terrible: */
      for (unsigned int k = 0; k < dim; ++k) {
        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          result[k] = flux->value(u, k);
        } else {
          for (unsigned int s = 0; s < Number::size(); ++s) {
            result[k][s] = flux->value(u[s], k);
          }
        }
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
    HyperbolicSystem::View<dim, Number>::flux_gradient_function(
        const Number &u) const
    {
      const auto &flux = hyperbolic_system_.selected_flux_;
      dealii::Tensor<1, dim, Number> result;

      /* This access by calling into value() repeatedly is terrible: */
      for (unsigned int k = 0; k < dim; ++k) {
        if constexpr (std::is_same_v<ScalarNumber, Number>) {
          result[k] = flux->gradient(u, k);
        } else {
          for (unsigned int s = 0; s < Number::size(); ++s) {
            result[k][s] = flux->gradient(u[s], k);
          }
        }
      }

      return result;
    }


    template <int dim, typename Number>
    template <typename DISPATCH, typename SPARSITY>
    DEAL_II_ALWAYS_INLINE inline void
    HyperbolicSystem::View<dim, Number>::precomputation_loop(
        unsigned int cycle [[maybe_unused]],
        const DISPATCH &dispatch_check,
        precomputed_vector_type &precomputed_values,
        const SPARSITY &sparsity_simd,
        const vector_type &U,
        unsigned int left,
        unsigned int right) const
    {
      Assert(cycle == 0, dealii::ExcInternalError());

      /* We are inside a thread parallel context */

      unsigned int stride_size = get_stride_size<Number>;

      RYUJIN_OMP_FOR
      for (unsigned int i = left; i < right; i += stride_size) {

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        dispatch_check(i);

        const auto U_i = U.template get_tensor<Number>(i);
        const auto u_i = state(U_i);

        const auto f_i = flux_function(u_i);
        const auto df_i = flux_gradient_function(u_i);

        precomputed_state_type prec_i;

        for (unsigned int k = 0; k < n_precomputed_values / 2; ++k) {
          prec_i[k] = f_i[k];
          prec_i[dim + k] = df_i[k];
        }

        precomputed_values.template write_tensor<Number>(prec_i, i);
      }
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::state(const state_type &U)
    {
      return U[0];
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::square_entropy(const Number &u) const
    {
      return ScalarNumber(0.5) * u * u;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::square_entropy_derivative(
        const Number &u) const
    {
      return u;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::kruzkov_entropy(const Number &k,
                                                         const Number &u) const
    {
      return std::abs(k - u);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystem::View<dim, Number>::kruzkov_entropy_derivative(
        const Number &k, const Number &u) const
    {
      constexpr auto gte = dealii::SIMDComparison::greater_than_or_equal;
      // return sgn(u-k):
      return dealii::compare_and_apply_mask<gte>(u, k, Number(1.), Number(-1.));
    }


    template <int dim, typename Number>
    template <typename Lambda>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::apply_boundary_conditions(
        dealii::types::boundary_id id,
        const state_type &U,
        const dealii::Tensor<1, dim, Number> & /*normal*/,
        const Lambda &get_dirichlet_data) const -> state_type
    {
      state_type result = U;

      if (id == Boundary::dirichlet) {
        result = get_dirichlet_data();

      } else if (id == Boundary::slip) {
        AssertThrow(
            false,
            dealii::ExcMessage("Invalid boundary ID »Boundary::slip«, slip "
                               "boundary conditions are unavailable for scalar "
                               "conservation equations."));
        __builtin_trap();

      } else if (id == Boundary::no_slip) {
        AssertThrow(
            false,
            dealii::ExcMessage("Invalid boundary ID »Boundary::no_slip«, "
                               "no-slip boundary conditions are unavailable "
                               "for scalar conservation equations."));
        __builtin_trap();

      } else if (id == Boundary::dynamic) {
        AssertThrow(
            false,
            dealii::ExcMessage("Invalid boundary ID »Boundary::dynamic«, "
                               "dynamic boundary conditions are unavailable "
                               "for scalar conservation equations."));
        __builtin_trap();
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
    HyperbolicSystem::View<dim, Number>::construct_flux_tensor(
        const precomputed_state_type &precomputed_state) const
    {
      dealii::Tensor<1, dim, Number> result;

      if constexpr (dim == 1) {
        const auto &[f, df] = precomputed_state;
        result[0] = f;

      } else if constexpr (dim == 2) {
        const auto &[f_1, f_2, df_1, df_2] = precomputed_state;
        result[0] = f_1;
        result[1] = f_2;

      } else if constexpr (dim == 3) {
        const auto &[f_1, f_2, f_3, df_1, df_2, df_3] = precomputed_state;
        result[0] = f_1;
        result[1] = f_2;
        result[2] = f_3;
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
    HyperbolicSystem::View<dim, Number>::construct_flux_gradient_tensor(
        const precomputed_state_type &precomputed_state) const
    {
      dealii::Tensor<1, dim, Number> result;

      if constexpr (dim == 1) {
        const auto &[f, df] = precomputed_state;
        result[0] = df;

      } else if constexpr (dim == 2) {
        const auto &[f_1, f_2, df_1, df_2] = precomputed_state;
        result[0] = df_1;
        result[1] = df_2;

      } else if constexpr (dim == 3) {
        const auto &[f_1, f_2, f_3, df_1, df_2, df_3] = precomputed_state;
        result[0] = df_1;
        result[1] = df_2;
        result[2] = df_3;
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::flux_contribution(
        const precomputed_vector_type &pv,
        const precomputed_initial_vector_type & /*piv*/,
        const unsigned int i,
        const state_type & /*U_i*/) const -> flux_contribution_type
    {
      /* The flux contribution is a rank 2 tensor, thus a little dance: */
      flux_contribution_type result;
      result[0] = construct_flux_tensor(
          pv.template get_tensor<Number, precomputed_state_type>(i));
      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystem::View<dim, Number>::flux_contribution(
        const precomputed_vector_type &pv,
        const precomputed_initial_vector_type & /*piv*/,
        const unsigned int *js,
        const state_type & /*U_j*/) const -> flux_contribution_type
    {
      /* The flux contribution is a rank 2 tensor, thus a little dance: */
      flux_contribution_type result;
      result[0] = construct_flux_tensor(
          pv.template get_tensor<Number, precomputed_state_type>(js));
      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto HyperbolicSystem::View<dim, Number>::flux(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j) const -> flux_type
    {
      return -add(flux_i, flux_j);
    }

  } // namespace ScalarConservation
} // namespace ryujin
