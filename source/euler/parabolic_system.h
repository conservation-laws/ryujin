//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>
#include <discretization.h>
#include <patterns_conversion.h>
#include <simd.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <functional>

namespace ryujin
{
  class ParabolicSystem final : public dealii::ParameterAcceptor
  {
  public:
    ParabolicSystem(const std::string &subsection = "ParabolicSystem");

    void parse_parameters_callback();

  private:
    /**
     * @name Run time options
     */
    //@{

    double mu_;
    ACCESSOR_READ_ONLY(mu)

    double lambda_;
    ACCESSOR_READ_ONLY(lambda)

    double cv_inverse_kappa_;
    ACCESSOR_READ_ONLY(cv_inverse_kappa)

    //@}
  };

} /* namespace ryujin */
