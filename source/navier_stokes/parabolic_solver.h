//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <convenience_macros.h>
#include <initial_values.h>
#include <offline_data.h>
#include <simd.h>
#include <sparse_matrix_simd.h>

#include "parabolic_solver_gmg_operators.h"

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

namespace ryujin
{
  namespace NavierStokes
  {
    template <int, typename>
    class DiagonalMatrix;

    /**
     * Minimum entropy guaranteeing second-order time stepping for the
     * parabolic limiting equation @cite ryujin-2021-2, Eq. 3.3:
     * \f{align}
     *   \newcommand{\bbm}{{\boldsymbol m}}
     *   \newcommand{\bef}{{\boldsymbol f}}
     *   \newcommand{\bk}{{\boldsymbol k}}
     *   \newcommand{\bu}{{\boldsymbol u}}
     *   \newcommand{\bv}{{\boldsymbol v}}
     *   \newcommand{\bn}{{\boldsymbol n}}
     *   \newcommand{\pols}{{\mathbb s}}
     *   \newcommand{\Hflux}{\bk}
     *   &\partial_t \rho  =  0,
     *   \\
     *   &\partial_t \bbm - \nabla\cdot(\pols(\bv)) = \bef,
     *   \\
     *   &\partial_t E   + \nabla\cdot(\Hflux(\bu)- \pols(\bv) \bv) =
     * \bef\cdot\bv,
     *   \\
     *   &\bv_{|\partial D}=\boldsymbol 0, \qquad \Hflux(\bu)\cdot\bn_{|\partial
     * D}=0 . \f}
     *
     * Internally, the module first performs an explicit second order
     * Crank-Nicolson step updating the velocity @cite ryujin-2021-2, Eq.
     * 5.5:
     * \f{align}
     *   \begin{cases}
     *     \newcommand\bsfV{{\textbf V}}
     *     \newcommand{\polB}{{\mathbb B}}
     *     \newcommand{\calI}{{\mathcal I}}
     *     \newcommand\bsfF{{\textbf F}}
     *     \newcommand\bsfM{{\textbf M}}
     *     \newcommand{\upint}{^\circ}
     *     \newcommand{\upbnd}{^\partial}
     *     \newcommand{\dt}{{\tau}}
     *     \newcommand{\calV}{{\mathcal V}}
     *     \varrho^{n}_i m_i \bsfV^{n+\frac{1}{2}} +
     *     \tfrac12 \dt\sum_{j\in\calI(i)} \polB_{ij} \bsfV^{n+\frac{1}{2}} =
     *     m_i \bsfM_i^{n} + \tfrac12 \dt m_i \bsfF_i^{n+\frac12},
     *     & \forall i\in \calV\upint
     *     \\[0.3em]
     *     \bsfV_i^{n+\frac{1}{2}} = \boldsymbol 0, &  \forall i\in \calV\upbnd,
     *   \end{cases}
     * \f}
     * We then postprocess and compute an internal energy update with a
     * second Crank-Nicolson step, @cite ryujin-2021-2, Eq. 5.13:
     * \f{align}
     *     \newcommand\sfe{{\mathsf e}}
     *     \newcommand{\upHnph}{^{\text{H},n+\frac{1}{2}}}
     *     \newcommand{\calI}{{\mathcal I}}
     *     \newcommand\sfK{{\mathsf K}}
     *     \newcommand{\calV}{{\mathcal V}}
     *     m_i \varrho_i^{n}(\sfe_i{\upHnph} - \sfe_i^{n})+\tfrac12\dt
     *     \sum_{j\in\calI(i)} \beta_{ij}\sfe_i{\upHnph}
     *     = \tfrac12 \dt m_i\sfK_i^{n+\frac{1}{2}}, \qquad \forall i\in \calV.
     * \f}
     *
     * @todo Provide more details about boundary conditions.
     *
     * @ingroup ParabolicModule
     */
    template <typename Description, int dim, typename Number = double>
    class ParabolicSolver final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @copydoc HyperbolicSystem
       */
      using HyperbolicSystem = typename Description::HyperbolicSystem;

      /**
       * @copydoc ParabolicSystem
       */
      using ParabolicSystem = typename Description::ParabolicSystem;

      /**
       * @copydoc HyperbolicSystem::View
       */
      using HyperbolicSystemView =
          typename HyperbolicSystem::template View<dim, Number>;

      /**
       * @copydoc OfflineData::scalar_type
       */
      using scalar_type = typename OfflineData<dim, Number>::scalar_type;

      /**
       * @copydoc HyperbolicSystem::View::vector_type
       */
      using vector_type = typename HyperbolicSystemView::vector_type;

      /**
       * A distributed block vector used for temporary storage of the
       * velocity field.
       */
      using block_vector_type =
          dealii::LinearAlgebra::distributed::BlockVector<Number>;

      /**
       * Constructor.
       */
      ParabolicSolver(
          const MPI_Comm &mpi_communicator,
          std::map<std::string, dealii::Timer> &computing_timer,
          const HyperbolicSystem &hyperbolic_system,
          const ParabolicSystem &parabolic_system,
          const OfflineData<dim, Number> &offline_data,
          const InitialValues<Description, dim, Number> &initial_values,
          const std::string &subsection = "ParabolicSolver");

      /**
       * Prepare time stepping. A call to @ref prepare() allocates temporary
       * storage and is necessary before any of the following time-stepping
       * functions can be called.
       */
      void prepare();

      /**
       * @name Functions for performing implicit time steps
       */
      //@{

      /**
       * Given a reference to a previous state vector @p old_U at time @p
       * old_t and a time-step size @p tau perform an implicit Crank-Nicolson
       * step (and store the result in @p new_U).
       */
      void crank_nicolson_step(const vector_type &old_U,
                               const Number old_t,
                               vector_type &new_U,
                               Number tau,
                               unsigned int cycle) const;

      /**
       * Print a status line with solver statistics. This function is used
       * for constructing the status message displayed periodically in the
       * TimeLoop.
       */
      void print_solver_statistics(std::ostream &output) const;

      //@}
      /**
       * @name Accessors
       */
      //@{

      ACCESSOR_READ_ONLY(n_restarts)
      ACCESSOR_READ_ONLY(n_warnings)

      //@}

    private:
      /**
       * @name Run time options
       */
      //@{

      bool use_gmg_velocity_;
      ACCESSOR_READ_ONLY(use_gmg_velocity)

      bool use_gmg_internal_energy_;
      ACCESSOR_READ_ONLY(use_gmg_internal_energy)

      Number tolerance_;
      bool tolerance_linfty_norm_;

      unsigned int gmg_max_iter_vel_;
      unsigned int gmg_max_iter_en_;
      double gmg_smoother_range_vel_;
      double gmg_smoother_range_en_;
      double gmg_smoother_max_eig_vel_;
      double gmg_smoother_max_eig_en_;
      unsigned int gmg_smoother_degree_;
      unsigned int gmg_smoother_n_cg_iter_;
      unsigned int gmg_min_level_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      const MPI_Comm &mpi_communicator_;
      std::map<std::string, dealii::Timer> &computing_timer_;

      dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
      dealii::SmartPointer<const ParabolicSystem> parabolic_system_;
      dealii::SmartPointer<const ryujin::OfflineData<dim, Number>>
          offline_data_;
      dealii::SmartPointer<
          const ryujin::InitialValues<Description, dim, Number>>
          initial_values_;

      mutable unsigned int n_restarts_;
      mutable unsigned int n_warnings_;
      mutable double n_iterations_velocity_;
      mutable double n_iterations_internal_energy_;

      mutable dealii::MatrixFree<dim, Number> matrix_free_;

      mutable block_vector_type velocity_;
      mutable block_vector_type velocity_rhs_;
      mutable scalar_type internal_energy_;
      mutable scalar_type internal_energy_rhs_;
      mutable scalar_type density_;

      mutable Number tau_;
      mutable Number theta_;

      mutable dealii::MGLevelObject<dealii::MatrixFree<dim, float>>
          level_matrix_free_;
      mutable dealii::MGConstrainedDoFs mg_constrained_dofs_;
      mutable dealii::MGLevelObject<
          dealii::LinearAlgebra::distributed::Vector<float>>
          level_density_;
      mutable MGTransferVelocity<dim, float> mg_transfer_velocity_;
      mutable dealii::MGLevelObject<VelocityMatrix<dim, float, Number>>
          level_velocity_matrices_;
      mutable MGTransferEnergy<dim, float> mg_transfer_energy_;
      mutable dealii::MGLevelObject<EnergyMatrix<dim, float, Number>>
          level_energy_matrices_;

      mutable dealii::mg::SmootherRelaxation<
          dealii::PreconditionChebyshev<
              VelocityMatrix<dim, float, Number>,
              dealii::LinearAlgebra::distributed::BlockVector<float>,
              DiagonalMatrix<dim, float>>,
          dealii::LinearAlgebra::distributed::BlockVector<float>>
          mg_smoother_velocity_;

      mutable dealii::mg::SmootherRelaxation<
          dealii::PreconditionChebyshev<
              EnergyMatrix<dim, float, Number>,
              dealii::LinearAlgebra::distributed::Vector<float>>,
          dealii::LinearAlgebra::distributed::Vector<float>>
          mg_smoother_energy_;

      //@}
    };

  } // namespace NavierStokes
} /* namespace ryujin */
