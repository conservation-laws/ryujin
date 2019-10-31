#ifndef PROBLEM_DESCRIPTION_TEMPLATE_H
#define PROBLEM_DESCRIPTION_TEMPLATE_H

#include "problem_description.h"

namespace grendel
{
  template <>
  const std::array<std::string, 3> ProblemDescription<1>::component_names{
      "rho", "m", "E"};

  template <>
  const std::array<std::string, 4> ProblemDescription<2>::component_names{
      "rho", "m_1", "m_2", "E"};

  template <>
  const std::array<std::string, 5> ProblemDescription<3>::component_names{
      "rho", "m_1", "m_2", "m_3", "E"};

  template <>
  const std::array<std::string, 3>
      ProblemDescription<1, float>::component_names{"rho", "m", "E"};

  template <>
  const std::array<std::string, 4>
      ProblemDescription<2, float>::component_names{"rho", "m_1", "m_2", "E"};

  template <>
  const std::array<std::string, 5>
      ProblemDescription<3, float>::component_names{
          "rho", "m_1", "m_2", "m_3", "E"};

} /* namespace grendel */

#endif /* PROBLEM_DESCRIPTION_TEMPLATE_H */
