#include "timeloop.h"

#include <fstream>

int main()
{
  ryujin::TimeLoop<DIM> time_loop;

  /*
   * If necessary, create empty parameter file and exit:
   */

  const auto filename = "ryujin.prm";
  if (!std::ifstream(filename)) {
    std::ofstream file(filename);
    dealii::ParameterAcceptor::prm.print_parameters(
        file, dealii::ParameterHandler::OutputStyle::Text);
    return 0;
  }

  time_loop.run();
  return 0;
}
