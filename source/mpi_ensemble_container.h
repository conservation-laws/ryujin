//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "mpi_ensemble.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

namespace ryujin
{
  /**
   * A specialized wrapper container used in the TimeLoop that helps
   * creating instances of HyperbolicSystem / ParabolicSystem /
   * InitialValues that are MPIEnsemble "aware:"
   *  - For regular equation descriptions we populate a vector with a
   *    per-ensemble instance of the class template T. The class template T
   *    must have a constructor that takes a std::string "subsection" as
   *    last argument. We modify this subsection string by appending
   *    "/ensemble n" to the n-th ensemble instance, which then allows to
   *    configure each ensemble instance individually in the parameter
   *    file.
   *
   *  @fixme For HyperbolicSystem/ ParabolicSystem / InitialValues
   *  realizations that are already MPIEnsemble "aware" we should fall back
   *  to only creating one instance - possibly by checking for and calling
   *  a modified constructor that takes an mpi_ensemble as an argument.
   */
  template <typename T>
  class MPIEnsembleContainer
  {
  public:
    /**
     * Create an independent instance of T for every ensemble of the MPI
     * ensemble collection. The subsection string for the ParameterAcceptor
     * is appended by "/ensemble n".
     */
    template <typename... Args>
    MPIEnsembleContainer(const MPIEnsemble &mpi_ensemble,
                         const std::string &subsection,
                         Args &&...args)
    {
      const auto &ensemble = mpi_ensemble.ensemble();
      const auto &n_ensembles = mpi_ensemble.n_ensembles();
      unsigned int digits = dealii::Utilities::needed_digits(n_ensembles - 1);

      payload_.resize(n_ensembles);
      for (int n = 0; n < n_ensembles; ++n) {
        /* Only append "/ensemble n" if we have more than one ensemble: */
        auto modified = subsection;
        if (n_ensembles > 1)
          modified +=
              "/ensemble " + dealii::Utilities::int_to_string(n, digits);
        payload_[n] =
            std::make_unique<T>(std::forward<Args>(args)..., modified);
      }

      ensemble_payload_ = payload_[ensemble].get();
    }

    /**
     * Return a const reference to the appropriate class instance for the
     * current MPI ensemble.
     */
    const T &get() const
    {
      return *ensemble_payload_;
    }

    /**
     * Conversion operator that returns the appropriate class instance for
     * the current MPI ensemble.
     */
    operator const T &() const
    {
      return *ensemble_payload_;
    }

  private:
    std::vector<std::unique_ptr<T>> payload_;
    T *ensemble_payload_;
  };


  /**
   * A trait class that determines whether a given equation Description
   * struct contains a method or field "n_mpi_ensembles".
   */
  namespace
  {
    template <typename C>
    static auto test(...) -> std::false_type;

    template <typename C>
    static auto test(int) -> decltype(C::n_mpi_ensembles(), std::true_type());
  } // namespace

  template <typename T>
  constexpr bool has_n_mpi_ensembles_v = decltype(test<T>(0))::value;
} /* namespace ryujin */
