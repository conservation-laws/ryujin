#include "problem_description.template.h"

namespace grendel
{
  template<>
  const std::array<std::string, 3> ProblemDescription<1>::component_names{
      "rho", "m", "E"};

  template<>
  const std::array<std::string, 4> ProblemDescription<2>::component_names{
      "rho", "m_1", "m_2", "E"};

  template<>
  const std::array<std::string, 5> ProblemDescription<3>::component_names{
      "rho", "m_1", "m_2", "m_3", "E"};

  template class grendel::ProblemDescription<1>;
  template class grendel::ProblemDescription<2>;
  template class grendel::ProblemDescription<3>;

} /* namespace grendel */
