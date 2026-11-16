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
       * denote a linearization around the state \f$\boldsymbol U^n\f$ at
       * the beginning of the time step. With the old state \f$\boldsymbol
       * W\f$, the \f$k\f$ stage states \f$\boldsymbol U^{s}\f$, the \f$k\f$
       * explicit weights \f$\omega^{\text e}_s\f$, and the \f$k+1\f$
       * implicit weights \f$\omega^{\text i}_s\f$ the function solves
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
       * The linearization is chosen by the parabolic module. It has to be
       * the same in every term of a time step; \f$\boldsymbol
       * G^{\text{lin}} = \boldsymbol G\f$, i.e., solving the nonlinear
       * problem, is admissible. For a linear parabolic operator the
       * explicit weights drop out and only the implicit weights
       * \f$\omega^{\text i}_s\f$ enter.
       *
       * The number of stage state vectors and the number of explicit
       * weights have to agree. The list of implicit weights contains one
       * additional entry: its last element \f$\omega^{\text i}_{k+1}\f$
       * is the weight of the new state and is nonzero; the case of a
       * vanishing weight is handled by stage_combination_step(). Stages
       * for which both weights vanish can be omitted. A plain backward
       * Euler step is performed with no stage state vectors, no explicit
       * weights, and the single implicit weight 1.
       *
       * The TimeIntegrator computes the stage \f$l\f$ as an increment of
       * its start stage \f$l'\f$ (the old state \f$\boldsymbol W\f$ of
       * the step) with step size \f$(c_l - c_{l'})\tau\f$. In terms of the
       * Butcher tableaux the weights are \f$\omega^{\text e}_s =
       * (a^{\text e}_{l,s}-a^{\text e}_{l',s}) / (c_l - c_{l'})\f$ and
       * \f$\omega^{\text i}_s = (a^{\text i}_{l,s} -a^{\text i}_{l',s}) /
       * (c_l - c_{l'})\f$; the last implicit weight is the scaled diagonal
       * entry \f$a^{\text i}_{l,l} / (c_l - c_{l'})\f$. For a diagonally
       * implicit scheme the explicit tableau is the strictly lower part of
       * the implicit tableau: the known stages enter with the operator
       * \f$\boldsymbol G\f$ and only the new stage is linearized. If the
       * hyperbolic module splits off an implicit flux
       * (Interface::HyperbolicModule::have_implicit_flux()) the explicit
       * weights instead describe the implicit flux in the prediction: one
       * for the start stage and zero otherwise after a hyperbolic step
       * (also of an explicit scheme, whose stages then pass through
       * stage_combination_step()), and zero for a diagonally implicit
       * scheme.
       *
       * @note A parabolic module that makes use of the linearization uses
       * the first stage state vector as linearization point
       * \f$\boldsymbol U^n\f$.
       *
       * The time @p t and the time-step size @p tau refer to the step
       * from the start stage to the new stage. A parabolic module that
       * maintains a constraint with time-dependent data (such as the
       * Gauss law with a time-dependent background density) uses the
       * change of the data from @p t to @p t + @p tau. Along the start
       * stages of a scheme these changes add up to the change over the
       * full time step, so the new state satisfies the constraint at the
       * end of the time step.
       *
       * @note The function must return the time step size @p tau
       * unchanged. @p tau_max is ignored.
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
       * Perform the update of a stage whose implicit weight of the new
       * state vanishes: with the same arguments as backward_euler_step()
       * (without the weight of the new state) compute
       * \f{align}
       *   \boldsymbol U^{\text{new}}
       *   \;=\;
       *   \boldsymbol W + \tau\sum_{s=1}^{k}\Big\{
       *   \omega^{\text e}_s\,\boldsymbol G(\boldsymbol U^{s})
       *   + \big(\omega^{\text i}_s-\omega^{\text e}_s\big)\,
       *   \boldsymbol G^{\text{lin}}(\boldsymbol U^n;\boldsymbol U^{s})
       *   \Big\}
       * \f}
       * and store the result in @p new_state_vector. This is the update
       * of every stage of a scheme whose row of the implicit tableau has
       * a vanishing diagonal entry. The function makes no assumption on
       * the position of the stage in the time step, or on the order of the
       * stage state vectors beyond the first one being the state at the
       * beginning of the time step.
       *
       * @note How the operator is evaluated for the stages is up to the
       * parabolic module.
       *
       * @note If the new state is not admissible the module must signal a
       * restart (if permitted by the IDViolationStrategy), or should limit
       * the update toward the (admissible) old state \f$\boldsymbol W\f$.
       *
       * @note The function must return the time step size @p tau
       * unchanged.
       */
      virtual Number
      stage_combination_step(const StateVector<Number> &old_state_vector,
                             StageVectors stage_state_vectors,
                             StageWeights stage_weights_explicit,
                             StageWeights stage_weights_implicit,
                             StateVector<Number> &new_state_vector,
                             Number t,
                             Number tau) const = 0;

      //@}

    protected:
      using Module<dim, Number>::Module;
    };
  } // namespace Interface
} // namespace ryujin
