//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

namespace ryujin
{
  namespace InitialStateLibrary
  {
    /**
     * Ritter's dam break solution. This is one-dimensional dam break without
     * friction. See: Sec.~7.3 in @cite GuermondEtAl2018SW.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename state_type>
    class RitterDamBreak : public InitialState<dim, Number, state_type, 1>
    {
    public:
      RitterDamBreak(const HyperbolicSystem &hyperbolic_system,
                     const std::string subsection)
          : InitialState<dim, Number, state_type, 1>("ritter_dam_break",
                                                     subsection)
          , hyperbolic_system(hyperbolic_system)
      {
        t_initial_ = 0.;
        this->add_parameter("time initial",
                            t_initial_,
                            "Time at which initial state is prescribed");

        left_depth = 0.005;
        this->add_parameter("left water depth",
                            left_depth,
                            "Depth of water to the left of pseudo-dam (x<0)");
        right_depth = 0.;
        this->add_parameter("right water depth",
                            right_depth,
                            "Depth of water to the right of pseudo-dam (x>0)");
      }

      virtual state_type compute(const dealii::Point<dim> &point,
                                 Number t) final override
      {
        const auto g = hyperbolic_system.gravity();
        const Number x = point[0];

        /* Return initial state if t_initial_ = 0 */

        if (t_initial_ <= 1.e-10) {
          if (x < 0)
            return hyperbolic_system.template expand_state<dim>(
                HyperbolicSystem::state_type<1, Number>{
                    {left_depth, Number(0.)}});
          else
            return hyperbolic_system.template expand_state<dim>(
                HyperbolicSystem::state_type<1, Number>{
                    {right_depth, Number(0.)}});
        }

        AssertThrow(t + t_initial_ > 0.,
                    dealii::ExcMessage("Expansion must be computed at a time "
                                       "greater than 0."));

        /* ... else we compute the expansion profiles at t + t_initial */

        const Number aL = std::sqrt(g * left_depth);
        const Number xA = -(t + t_initial_) * aL;
        const Number xB = Number(2.) * (t + t_initial_) * aL;

        const Number tmp = aL - x / (2. * (t + t_initial_));

        const Number h_expansion = 4. / (9. * g) * tmp * tmp;
        const Number v_expansion = 2. / 3. * (x / (t + t_initial_) + aL);

        if (x <= xA)
          return hyperbolic_system.template expand_state<dim>(
              HyperbolicSystem::state_type<1, Number>{
                  {left_depth, Number(0.)}});
        else if (x <= xB)
          return hyperbolic_system.template expand_state<dim>(
              HyperbolicSystem::state_type<1, Number>{
                  {h_expansion, h_expansion * v_expansion}});
        else
          return hyperbolic_system.template expand_state<dim>(
              HyperbolicSystem::state_type<1, Number>{
                  {right_depth, Number(0.)}});
      }

      /* Default bathymetry of 0 */

    private:
      const HyperbolicSystem &hyperbolic_system;

      Number t_initial_;

      Number left_depth;
      Number right_depth;
    };

  } // namespace InitialStateLibrary
} // namespace ryujin
