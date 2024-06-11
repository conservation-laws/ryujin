#pragma once

#include <initial_state_library.h>

namespace ryujin
{
  namespace EulerInitialStates
  {
    template <typename Description, int dim, typename Number>
    class Nozzle : public InitialState<Description, dim, Number>
    {
    public:
      using HyperbolicSystem = typename Description::HyperbolicSystem;
      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;
      using state_type = typename View::state_type;
        Nozzle(const HyperbolicSystem &hyperbolic_system,
              const std::string subsection)
          : InitialState<Description, dim, Number>("nozzle", subsection)
          , hyperbolic_system_(hyperbolic_system)
      {
        top_[0] = 1.0;
        top_[1] = 1.0;
        top_[2] = 1.0;
        this->add_parameter("flow state top",
        top_, "Initial Vacuum state");
        bottom_[0] = 1.0;
        bottom_[1] = 1.0;
        bottom_[2] = 1.0;
        this->add_parameter("flow state bottom",
        bottom_, "Initial inflow state");
        const auto convert_states = [&]() {
            const auto view = hyperbolic_system_.template view<dim, Number>();
            state_init_ = view.from_initial_state(top_);
            state_inflow_ = view.from_initial_state(bottom_);
        };
        this->parse_parameters_call_back.connect(convert_states);
        convert_states();
      }
        state_type compute(const dealii::Point<dim> &point,
                         Number /*t*/) final
      {
          state_type result;
          if (point[0] > 1e-6){
            result = state_init_;
          }
          else {

            result = state_inflow_;

          }
          return result;  
        }
    private:
      const HyperbolicSystem &hyperbolic_system_;

      dealii::Tensor<1, 3, Number> top_;
      dealii::Tensor<1, 3, Number> bottom_;

      state_type state_inflow_;
      state_type state_init_;
        };

    }
}