//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/interface/module.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <limits>
#include <string>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The description-free interface of a ParabolicModule: implicit time
     * stepping for a parabolic system.
     *
     * The interface holds the common state of Interface::Module.
     *
     * @ingroup Interfaces
     */
    template <int dim, typename Number = double>
    class ParabolicModule : public Module<dim, Number>
    {
    public:
      /**
       * @name Typedefs
       */
      //@{

      using typename Module<dim, Number>::StageVectors;

      using typename Module<dim, Number>::StageWeights;

      //@}
      /**
       * @name Functions for performing implicit time steps
       */
      //@{

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p t and a time-step size @p tau perform an implicit
       * backward Euler step (and store the result in @p new_state_vector).
       *
       * The stage state vectors @p stage_state_vectors together with the
       * weights @p stage_weights_explicit and @p stage_weights_implicit
       * are used to construct a modified high order right-hand side /
       * flux, see @cite ErnGuermond2023, Eq. (3.7). Let \f$\boldsymbol
       * G(\boldsymbol U)\f$ denote the parabolic operator and let
       * \f$\boldsymbol G^{\text{lin}}(\boldsymbol U^n;\boldsymbol U)\f$
       * denote its quasi-linearization around the state \f$\boldsymbol
       * U^n\f$ at the beginning of the time step. With the old state
       * \f$\boldsymbol W\f$, the \f$k\f$ stage states \f$\boldsymbol
       * U^{s}\f$, the \f$k\f$ explicit weights \f$\omega^{\text e}_s\f$,
       * and the \f$k+1\f$ implicit weights \f$\omega^{\text i}_s\f$ the
       * function solves
       * \f{align}
       *   \boldsymbol U^{\text{new}}
       *   - \tau\,\omega^{\text i}_{k+1}\,
       *   \boldsymbol G^{\text{lin}}(\boldsymbol U^n;
       *   \boldsymbol U^{\text{new}})
       *   \;=\;
       *   \boldsymbol W + \tau\sum_{s=1}^{k}\Big\{
       *   \omega^{\text e}_s\,\boldsymbol G(\boldsymbol U^{s})
       *   + \big(\omega^{\text i}_s-\omega^{\text e}_s\big)\,
       *   \boldsymbol G^{\text{lin}}(\boldsymbol U^n;\boldsymbol U^{s})
       *   \Big\}.
       * \f}
       * For a linear parabolic operator the explicit weights drop out and
       * only the implicit weights \f$\omega^{\text i}_s\f$ enter.
       *
       * The number of stage state vectors and the number of explicit
       * weights have to agree. The list of implicit weights contains one
       * additional entry: its last element \f$\omega^{\text i}_{k+1}\f$
       * is the weight of the new state. Stages for which both weights
       * vanish can be omitted. A plain backward Euler step is performed
       * with no stage state vectors, no explicit weights, and the single
       * implicit weight 1.
       *
       * In terms of the Butcher tableaux of the explicit and implicit
       * schemes the weights for the stage \f$l\f$ are \f$\omega^{\text e}_s
       * = (a^{\text e}_{l,s}-a^{\text e}_{l-1,s}) / (c_l - c_{l-1})\f$ and
       * \f$\omega^{\text i}_s = (a^{\text i}_{l,s} -a^{\text i}_{l-1,s}) /
       * (c_l - c_{l-1})\f$; the last implicit weight is the scaled
       * diagonal entry \f$a^{\text i}_{l,l} / (c_l - c_{l-1})\f$.
       *
       * @note A parabolic module that makes use of the quasi-linearization
       * expects the state \f$\boldsymbol U^n\f$ as first stage state
       * vector.
       *
       * The function returns the time step size @p tau. A parabolic module
       * cannot choose the time step size: @p tau must be nonzero and @p
       * tau_max is ignored.
       *
       * @todo This function should always operate on the selected memory space.
       */
      virtual Number backward_euler_step(
          const StateVector<Number> &old_state_vector,
          StageVectors stage_state_vectors,
          StageWeights stage_weights_explicit,
          StageWeights stage_weights_implicit,
          StateVector<Number> &new_state_vector,
          Number t,
          Number tau,
          Number tau_max = std::numeric_limits<Number>::max()) const = 0;

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p t and a time-step size @p tau perform an implicit
       * Crank-Nicolson step (and store the result in @p new_state_vector).
       *
       * This variant is used in the TimeIntegrator class for the Strang
       * split variants.
       *
       * The function returns the time step size @p tau. A parabolic module
       * cannot choose the time step size: @p tau must be nonzero and @p
       * tau_max is ignored.
       *
       * @todo This function should always operate on the selected memory space.
       */
      virtual Number crank_nicolson_step(
          const StateVector<Number> &old_state_vector,
          StateVector<Number> &new_state_vector,
          Number t,
          Number tau,
          Number tau_max = std::numeric_limits<Number>::max()) const = 0;

      //@}

    protected:
      using Module<dim, Number>::Module;
    };
  } // namespace Interface
} // namespace ryujin
