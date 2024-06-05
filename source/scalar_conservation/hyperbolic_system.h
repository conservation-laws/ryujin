//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
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
#include <state_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>

namespace ryujin
{
  namespace ScalarConservation
  {
    template <int dim, typename Number>
    class HyperbolicSystemView;

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

      /**
       * Return a view on the Hyperbolic System for a given dimension @p
       * dim and choice of number type @p Number (which can be a scalar
       * float, or double, as well as a VectorizedArray holding packed
       * scalars.
       */
      template <int dim, typename Number>
      auto view() const
      {
        return HyperbolicSystemView<dim, Number>{*this};
      }

    private:
      /**
       * @name Runtime parameters, internal fields, methods, and friends
       */
      //@{
      std::string flux_;

      FluxLibrary::flux_list_type flux_list_;
      using Flux = FluxLibrary::Flux;
      std::shared_ptr<Flux> selected_flux_;

      template <int dim, typename Number>
      friend class HyperbolicSystemView;
      //@}
    }; /* HyperbolicSystem */


    /**
     * A view on the HyperbolicSystem for a given dimension @p dim and
     * choice of number type @p Number (which can be a scalar float, or
     * double, as well as a VectorizedArray holding packed scalars.
     */
    template <int dim, typename Number>
    class HyperbolicSystemView
    {
    public:
      /**
       * Constructor taking a reference to the underlying
       * HyperbolicSystem
       */
      HyperbolicSystemView(const HyperbolicSystem &hyperbolic_system)
          : hyperbolic_system_(hyperbolic_system)
      {
      }

      /**
       * Create a modified view from the current one:
       */
      template <int dim2, typename Number2>
      auto view() const
      {
        return HyperbolicSystemView<dim2, Number2>{hyperbolic_system_};
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

      DEAL_II_ALWAYS_INLINE inline ScalarNumber
      derivative_approximation_delta() const
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
       * @name Types and constexpr constants
       */
      //@{

      /**
       * The dimension of the state space.
       */
      static constexpr unsigned int problem_dimension = 1;

      /**
       * Storage type for a (conserved) state vector \f$\boldsymbol U\f$.
       */
      using state_type = dealii::Tensor<1, problem_dimension, Number>;

      /**
       * Storage type for the flux \f$\mathbf{f}\f$.
       */
      using flux_type =
          dealii::Tensor<1, problem_dimension, dealii::Tensor<1, dim, Number>>;

      /**
       * The storage type used for flux contributions.
       */
      using flux_contribution_type = flux_type;

      /**
       * An array holding all component names of the conserved state as a
       * string.
       */
      static inline const auto component_names =
          std::array<std::string, problem_dimension>{"u"};

      /**
       * An array holding all component names of the primitive state as a
       * string.
       */
      static inline const auto primitive_component_names =
          std::array<std::string, problem_dimension>{"u"};

      /**
       * The number of precomputed values.
       */
      static constexpr unsigned int n_precomputed_values = 2 * dim;

      /**
       * Array type used for precomputed values.
       */
      using precomputed_type = std::array<Number, n_precomputed_values>;

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
       * The number of precomputed initial values.
       */
      static constexpr unsigned int n_initial_precomputed_values = 0;

      /**
       * Array type used for precomputed initial values.
       */
      using initial_precomputed_type =
          std::array<Number, n_initial_precomputed_values>;

      /**
       * An array holding all component names of the precomputed values.
       */
      static inline const auto initial_precomputed_names =
          std::array<std::string, n_initial_precomputed_values>{};

      /**
       * A compound state vector.
       */
      using StateVector = Vectors::
          StateVector<ScalarNumber, problem_dimension, n_precomputed_values>;

      /**
       * MulticomponentVector for storing the hyperbolic state vector:
       */
      using HyperbolicVector =
          Vectors::MultiComponentVector<ScalarNumber, problem_dimension>;

      /**
       * MulticomponentVector for storing a vector of precomputed states:
       */
      using PrecomputedVector =
          Vectors::MultiComponentVector<ScalarNumber, n_precomputed_values>;

      /**
       * MulticomponentVector for storing a vector of precomputed initial
       * states:
       */
      using InitialPrecomputedVector =
          Vectors::MultiComponentVector<ScalarNumber,
                                        n_initial_precomputed_values>;

      //@}
      /**
       * @name Computing precomputed quantities
       */
      //@{


      /**
       * The number of precomputation cycles.
       */
      static constexpr unsigned int n_precomputation_cycles = 1;

      /**
       * Step 0: precompute values for hyperbolic update. This routine is
       * called within our usual loop() idiom in HyperbolicModule
       */
      template <typename DISPATCH, typename SPARSITY>
      void precomputation_loop(unsigned int cycle,
                               const DISPATCH &dispatch_check,
                               const SPARSITY &sparsity_simd,
                               StateVector &state_vector,
                               unsigned int left,
                               unsigned int right) const;

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
      Number kruzkov_entropy_derivative(const Number &k, const Number &u) const;

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
      dealii::Tensor<1, dim, Number>
      construct_flux_tensor(const precomputed_type &precomputed_state) const;

      /**
       * Helper function that given a @p precomputed_state constructs a
       * dealii::Tensor with the gradient of the flux @f$f(u)@f$ with
       * respect to the state @f$u@f$.
       */
      dealii::Tensor<1, dim, Number> construct_flux_gradient_tensor(
          const precomputed_type &precomputed_state) const;

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
       *     const auto flux_ij = flux_divergence(flux_i, flux_j, c_ij);
       *   }
       * }
       * ```
       *
       * For the Euler equations we simply compute <code>f(U_i)</code>.
       */
      flux_contribution_type
      flux_contribution(const PrecomputedVector &pv,
                        const InitialPrecomputedVector & /*piv*/,
                        const unsigned int i,
                        const state_type & /*U_i*/) const;

      flux_contribution_type
      flux_contribution(const PrecomputedVector &pv,
                        const InitialPrecomputedVector & /*piv*/,
                        const unsigned int *js,
                        const state_type & /*U_j*/) const;

      /**
       * Given flux contributions @p flux_i and @p flux_j compute the flux
       * <code>(-f(U_i) - f(U_j)</code>
       */
      state_type
      flux_divergence(const flux_contribution_type &flux_i,
                      const flux_contribution_type &flux_j,
                      const dealii::Tensor<1, dim, Number> &c_ij) const;

      /** The low-order and high-order fluxes are the same */
      static constexpr bool have_high_order_flux = false;

      state_type high_order_flux_divergence(
          const flux_contribution_type &,
          const flux_contribution_type &,
          const dealii::Tensor<1, dim, Number> &c_ij) const = delete;

      //@}
      /**
       * @name Computing stencil source terms
       */
      //@{

      /** We do not have source terms */
      static constexpr bool have_source_terms = false;

      state_type nodal_source(const PrecomputedVector &pv,
                              const unsigned int i,
                              const state_type &U_i,
                              const ScalarNumber tau) const = delete;

      state_type nodal_source(const PrecomputedVector &pv,
                              const unsigned int *js,
                              const state_type &U_j,
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
      state_type from_primitive_state(const state_type &primitive_state) const
      {
        return primitive_state;
      }

      /**
       * Given a conserved state return a primitive state [rho, u_1, ..., u_d,
       * p]
       */
      state_type to_primitive_state(const state_type &state) const
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

    }; /* HyperbolicSystemView */


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
    HyperbolicSystemView<dim, Number>::flux_function(const Number &u) const
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
    HyperbolicSystemView<dim, Number>::flux_gradient_function(
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
    HyperbolicSystemView<dim, Number>::precomputation_loop(
        unsigned int cycle [[maybe_unused]],
        const DISPATCH &dispatch_check,
        const SPARSITY &sparsity_simd,
        StateVector &state_vector,
        unsigned int left,
        unsigned int right) const
    {
      Assert(cycle == 0, dealii::ExcInternalError());

      const auto &U = std::get<0>(state_vector);
      auto &precomputed = std::get<1>(state_vector);

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

        precomputed_type prec_i;

        for (unsigned int k = 0; k < n_precomputed_values / 2; ++k) {
          prec_i[k] = f_i[k];
          prec_i[dim + k] = df_i[k];
        }

        precomputed.template write_tensor<Number>(prec_i, i);
      }
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::state(const state_type &U)
    {
      return U[0];
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::square_entropy(const Number &u) const
    {
      return ScalarNumber(0.5) * u * u;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::square_entropy_derivative(
        const Number &u) const
    {
      return u;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::kruzkov_entropy(const Number &k,
                                                       const Number &u) const
    {
      return std::abs(k - u);
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    HyperbolicSystemView<dim, Number>::kruzkov_entropy_derivative(
        const Number &k, const Number &u) const
    {
      constexpr auto gte = dealii::SIMDComparison::greater_than_or_equal;
      // return sgn(u-k):
      return dealii::compare_and_apply_mask<gte>(u, k, Number(1.), Number(-1.));
    }


    template <int dim, typename Number>
    template <typename Lambda>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::apply_boundary_conditions(
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
    HyperbolicSystemView<dim, Number>::construct_flux_tensor(
        const precomputed_type &precomputed) const
    {
      dealii::Tensor<1, dim, Number> result;

      if constexpr (dim == 1) {
        const auto &[f, df] = precomputed;
        result[0] = f;

      } else if constexpr (dim == 2) {
        const auto &[f_1, f_2, df_1, df_2] = precomputed;
        result[0] = f_1;
        result[1] = f_2;

      } else if constexpr (dim == 3) {
        const auto &[f_1, f_2, f_3, df_1, df_2, df_3] = precomputed;
        result[0] = f_1;
        result[1] = f_2;
        result[2] = f_3;
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number>
    HyperbolicSystemView<dim, Number>::construct_flux_gradient_tensor(
        const precomputed_type &precomputed) const
    {
      dealii::Tensor<1, dim, Number> result;

      if constexpr (dim == 1) {
        const auto &[f, df] = precomputed;
        result[0] = df;

      } else if constexpr (dim == 2) {
        const auto &[f_1, f_2, df_1, df_2] = precomputed;
        result[0] = df_1;
        result[1] = df_2;

      } else if constexpr (dim == 3) {
        const auto &[f_1, f_2, f_3, df_1, df_2, df_3] = precomputed;
        result[0] = df_1;
        result[1] = df_2;
        result[2] = df_3;
      }

      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_contribution(
        const PrecomputedVector &pv,
        const InitialPrecomputedVector & /*piv*/,
        const unsigned int i,
        const state_type & /*U_i*/) const -> flux_contribution_type
    {
      /* The flux contribution is a rank 2 tensor, thus a little dance: */
      flux_contribution_type result;
      result[0] = construct_flux_tensor(
          pv.template get_tensor<Number, precomputed_type>(i));
      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_contribution(
        const PrecomputedVector &pv,
        const InitialPrecomputedVector & /*piv*/,
        const unsigned int *js,
        const state_type & /*U_j*/) const -> flux_contribution_type
    {
      /* The flux contribution is a rank 2 tensor, thus a little dance: */
      flux_contribution_type result;
      result[0] = construct_flux_tensor(
          pv.template get_tensor<Number, precomputed_type>(js));
      return result;
    }


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline auto
    HyperbolicSystemView<dim, Number>::flux_divergence(
        const flux_contribution_type &flux_i,
        const flux_contribution_type &flux_j,
        const dealii::Tensor<1, dim, Number> &c_ij) const -> state_type
    {
      return -contract(add(flux_i, flux_j), c_ij);
    }

  } // namespace ScalarConservation
} // namespace ryujin
