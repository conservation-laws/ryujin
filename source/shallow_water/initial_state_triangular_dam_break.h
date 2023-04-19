
#pragma once

#include "hyperbolic_system.h"
#include <initial_state.h>

namespace ryujin
{
  namespace ShallowWater
  {
    /**
     * Dam break over a triangular step.
     *
     * @ingroup InitialValues
     */
    template <int dim, typename Number, typename state_type>
    class TriangularDamBreak : public InitialState<dim, Number, state_type, 1>
    {
    public:
      TriangularDamBreak(const HyperbolicSystem &hyperbolic_system,
                         const std::string s)
          : InitialState<dim, Number, state_type, 1>("triangular dam break", s)
          , hyperbolic_system(hyperbolic_system)
      {
        left_depth = 0.75;
        this->add_parameter("left resevoir depth",
                            left_depth,
                            "Depth of water at left resevoir");

        left_position = 0.00;
        this->add_parameter("left resevoir position",
                            left_position,
                            "Position of left resevoir");

        right_depth = 0.15;
        this->add_parameter("right resevoir depth",
                            right_depth,
                            "Depth of water at right resevoir");

        right_position = 14.5;
        this->add_parameter("right resevoir position",
                            right_position,
                            "Position of right resevoir");


        step_position = 13.0;
        this->add_parameter("step position",
                            step_position,
                            "Center position of the triangular step");

        step_width = 6.0;
        this->add_parameter(
            "step width", step_width, "Total width of the triangular step");

        step_height = 0.4;
        this->add_parameter(
            "step height", step_height, "Height of the triangular step");
      }

      state_type compute(const dealii::Point<dim> &point, Number /*t*/) final
      {
        const Number x = point[0];
        const Number bath = compute_bathymetry(point);

        /* Set water depths at the two resevoirs */
        Number h = 0.;
        if (x < left_position)
          h = std::max(Number(0.), left_depth - bath);
        else if (x > right_position)
          h = std::max(Number(0.), right_depth - bath);

        return hyperbolic_system.template expand_state<dim>(
            HyperbolicSystem::state_type<1, Number>{{h, Number(0.)}});
      }

      auto initial_precomputations(const dealii::Point<dim> &point) ->
          typename InitialState<dim, Number, state_type, 1>::precomputed_type
          final
      {
        /* Compute bathymetry: */
        return {compute_bathymetry(point)};
      }

    private:
      const HyperbolicSystem &hyperbolic_system;

      DEAL_II_ALWAYS_INLINE inline Number
      compute_bathymetry(const dealii::Point<dim> &point) const
      {
        const Number x = point[0];

        const Number slope = Number(2.) * step_height / step_width;
        const Number triangular_step =
            step_height - slope * std::abs(x - step_position);
        return std::max(Number(0.), triangular_step);
      }

      Number left_depth;
      Number left_position;

      Number right_depth;
      Number right_position;

      Number step_position;
      Number step_width;
      Number step_height;
    };

  } // namespace ShallowWater
} // namespace ryujin
