//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/gpu.h>
#include <ryujin/linear_algebra/sparse_matrix.h>
#include <ryujin/linear_algebra/sparsity_pattern.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace ryujin
{
  namespace Interface
  {
    /**
     * The description-free interface of OfflineData: all precomputed data
     * that the algebraic part of ryujin (hyperbolic systems,
     * HyperbolicModule, TimeIntegrator, Postprocessor) operates on.
     *
     * The interface only depends on the space dimension @p dim. It holds
     * index ranges, partitioners, affine constraints, the boundary map,
     * sparsity patterns, and all matrices, but nothing that refers to a
     * mesh. The typed realization ryujin::OfflineData<structdim, dim,
     * Number> owns the DoFHandlers and populates all data members.
     *
     * @ingroup Interfaces
     */
    template <int dim, typename Number = double>
    class OfflineData : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      /**
       * A tuple describing (local) dof index, boundary normal, normal
       * mass, boundary mass, boundary id, and position of the boundary
       * degree of freedom.
       */
      using BoundaryDescription =
          std::tuple<unsigned int /*i*/,
                     dealii::Tensor<1, dim, Number> /*normal*/,
                     Number /*normal mass*/,
                     Number /*boundary mass*/,
                     dealii::types::boundary_id /*id*/,
                     dealii::Point<dim>> /*position*/;

      /**
       * A structure describing coupling boundary degrees of freedom on
       * directly enforced boundaries for which we have to symmetrize the
       * d_ij matrix.
       */
      struct CouplingDescription {
        unsigned int i;
        unsigned int col_idx;
        unsigned int j;

        bool operator==(const CouplingDescription &) const = default;
      };

      /**
       * A vector of BoundaryDescription entries.
       */
      using BoundaryMap = std::vector<BoundaryDescription>;

      /**
       * A vector of CouplingDescription entries.
       */
      using CouplingBoundaryPairs = std::vector<CouplingDescription>;

      //@}
      /**
       * @name Setup
       */
      //@{

      /**
       * Prepare offline data: set up index ranges, partitioners, sparsity
       * patterns, and assemble all matrices. The problem_dimension and
       * n_precomputed_values parameters are used to set up appropriately
       * sized vector partitioners for the state and precomputed
       * MultiComponentVector.
       */
      virtual void prepare(const unsigned int problem_dimension,
                           const unsigned int n_precomputed_values,
                           const bool with_multigrid_data = true) = 0;

      /**
       * Recompute everything that depends on the geometry of the mesh
       * without changing the numbering of degrees of freedom, sparsity
       * patterns, or partitioners. Requires a prior call to prepare().
       */
      virtual void update_geometry() = 0;

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * Return true if the selected finite element space is
       * discontinuous.
       */
      ACCESSOR_READ_ONLY(have_discontinuous_ansatz)

      /**
       * An index map that assigns every locally owned degree of freedom of
       * the discontinuous ("dG") variant of the selected finite element
       * space (numbered consecutively within the locally owned range) the
       * global index of the collocated degree of freedom of the continuous
       * ("cG") variant. A cG vector is interpolated into the dG space by
       * setting `dg(i) = cg(dg_to_cg_index_map()[i])`.
       *
       * @note For a discontinuous ansatz the numbering of the locally owned
       * range coincides with the local numbering of scalar_partitioner().
       */
      ACCESSOR_READ_ONLY(dg_to_cg_index_map)

      /**
       * The inverse of dg_to_cg_index_map(): An index map that assigns
       * every locally owned degree of freedom of the cG space (numbered
       * consecutively within the locally owned range) the global index of
       * one collocated degree of freedom of the dG space. The choice among
       * the collocated degrees of freedom is arbitrary; the map thus
       * restricts a continuous dG function to the cG space by setting
       * `cg(i) = dg(cg_to_dg_index_map()[i])`.
       *
       * @note For a continuous ansatz the numbering of the locally owned
       * range coincides with the local numbering of scalar_partitioner().
       */
      ACCESSOR_READ_ONLY(cg_to_dg_index_map)

      /**
       * An AffineConstraints object storing constraints for the continuous
       * ("cG") variant of the selected finite element space in (deal.II
       * typical) global numbering.
       *
       * @note The affine constraints object is populated with (a)
       * hanging node constraints, and (b) periodicity constraints, that
       * directly affect the chosen ansatz space.
       *
       * @note If the selected finite element space is continuous then this
       * method simply returns the same object as affine_constraints().
       */
      ACCESSOR_READ_ONLY(affine_constraints_cg)

      /**
       * An AffineConstraints object storing constraints for the
       * discontinuous ("dG") variant of the selected finite element space
       * in (deal.II typical) global numbering.
       *
       * @note The affine constraints object is populated with (a)
       * hanging node constraints, and (b) periodicity constraints, that
       * directly affect the chosen ansatz space.
       *
       * @note If the selected finite element space is discontinuous then
       * this method simply returns the same object as affine_constraints().
       */
      ACCESSOR_READ_ONLY(affine_constraints_dg)

      /**
       * An AffineConstraints object storing constraints in (deal.II
       * typical) global numbering.
       *
       * @note The affine constraints object is populated with (a)
       * hanging node constraints, and (b) periodicity constraints, that
       * directly affect the chosen ansatz space.
       */
      const dealii::AffineConstraints<Number> &affine_constraints() const
      {
        if (have_discontinuous_ansatz_)
          return *affine_constraints_dg_;
        else
          return *affine_constraints_cg_;
      }

      /**
       * An MPI partitioner for the (scalar) Vector storing a scalar-valued
       * quantity.
       */
      ACCESSOR_READ_ONLY_NO_DEREFERENCE(scalar_partitioner)

      /**
       * An MPI partitioner for the MultiComponentVector storing a
       * vector-valued quantity of size HyperbolicSystem::problem_dimension.
       */
      ACCESSOR_READ_ONLY_NO_DEREFERENCE(hyperbolic_vector_partitioner)

      /**
       * An MPI partitioner for the MultiComponentVector storing a
       * vector-valued quantity of size HyperbolicSystem::problem_dimension.
       */
      ACCESSOR_READ_ONLY_NO_DEREFERENCE(precomputed_vector_partitioner)

      /**
       * The subinterval \f$[0,\texttt{n_export_indices()})\f$ contains all
       * (SIMD-vectorized) indices of the interval
       * \f$[0,\texttt{n_locally_internal()})\f$ that are exported to
       * neighboring MPI ranks.
       *
       * @note The interval \f$[\texttt{n_locally_internal()},
       * \texttt{n_locally_relevant()})\f$ (consisting of
       * non-SIMD-vectorized indices) contains additional degrees of
       * freedom that might have to be exported to neighboring MPI ranks.
       */
      ACCESSOR_READ_ONLY(n_export_indices)

      /**
       * Number of locally owned internal degrees of freedom: In (MPI rank)
       * local numbering all indices in the half open interval [0,
       * n_locally_internal_) are owned by this processor, have standard
       * connectivity, and are not situated at a boundary.
       */
      ACCESSOR_READ_ONLY(n_locally_internal)

      /**
       * Number of locally owned degrees of freedom: In (MPI rank) local
       * numbering all indices in the half open interval [0,
       * n_locally_owned_) are owned by this processor.
       */
      ACCESSOR_READ_ONLY(n_locally_owned)

      /**
       * Number of locally relevant degrees of freedom: This number is the
       * total number of degrees of freedom we store locally on this MPI
       * rank. I.e., we can access the half open interval [0,
       * n_locally_relevant_) on this machine.
       */
      ACCESSOR_READ_ONLY(n_locally_relevant)

      /**
       * The boundary map. Local numbering.
       *
       * For every degree of freedom that has nonzero support at the
       * boundary we record the global degree of freedom index along with a
       * weighted boundary normal, the associated boundary id, and
       * position.
       *
       * This map is later used in OfflineData to handle boundary
       * degrees of freedom after every time step (for example to implement
       * reflective boundary conditions).
       */
      ACCESSOR_READ_ONLY(boundary_map)

      /**
       * A compacted, strictly increasing list of all locally owned degrees
       * of freedom that appear in the boundary_map(). The function returns
       * a reference to a Mirrored object that can be accessed on both
       * memory spaces with view() and size().
       */
      ACCESSOR_READ_ONLY(boundary_indices)

      /**
       * For every entry of the boundary_map() the position of its degree
       * of freedom in boundary_indices(). Local numbering.
       *
       * @note The boundary map can contain more than one entry for the
       * same degree of freedom (for example a corner that is
       * simultaneously located on a slip and a dirichlet boundary). All
       * such entries share the same position.
       */
      ACCESSOR_READ_ONLY(boundary_slots)

      /**
       * An array of CouplingDescription structures describing coupling
       * degrees of freedom i and j where both degrees of freedom are
       * collocated at the boundary (and hence the d_ij matrix has to be
       * symmetrized). The function returns a reference to a Mirrored
       * object holding (i, col_idx, j) triples that can be accessed on
       * both memory spaces with view() and size().
       */
      ACCESSOR_READ_ONLY(coupling_boundary_pairs)

      /**
       * A sparsity pattern for (standard deal.II) matrices storing indices
       * in (deal.II typical) global numbering.
       */
      ACCESSOR_READ_ONLY(sparsity_pattern)

      /**
       * A sparsity pattern for matrices in vectorized format. Local
       * numbering.
       */
      ACCESSOR_READ_ONLY(sparsity_pattern_simd)

      /**
       * The mass matrix. (SIMD storage, local numbering)
       */
      ACCESSOR_READ_ONLY(mass_matrix)

      /**
       * The inverse mass matrix. (SIMD storage, local numbering)
       *
       * This matrix is only available for a discontinuous finite Element
       * ansatz.
       */
      ACCESSOR_READ_ONLY(mass_matrix_inverse)

      /**
       * The lumped mass matrix. (stored as vector, local numbering)
       */
      ACCESSOR_READ_ONLY(lumped_mass_matrix)

      /**
       * The inverse of the lumped mass matrix. (stored as vector, local
       * numbering)
       */
      ACCESSOR_READ_ONLY(lumped_mass_matrix_inverse)

      /**
       * The stiffness matrix \f$(beta_{ij})\f$:
       *   \f$\beta_{ij} = \nabla\varphi_{j}\cdot\nabla\varphi_{i}\f$
       * (SIMD storage, local numbering)
       */
      ACCESSOR_READ_ONLY(betaij_matrix)

      /**
       * The \f$(c_{ij})\f$ matrix. (SIMD storage, local numbering)
       */
      ACCESSOR_READ_ONLY(cij_matrix)

      /**
       * The incidence matrix \f$(beta_{ij})\f$: 1 for coupling face
       * degrees of freedom that share the same support point coordinate, 0
       * otherwise.
       *
       * (SIMD storage, local numbering)
       *
       * This matrix is only available for a discontinuous finite Element
       * ansatz.
       */
      ACCESSOR_READ_ONLY(incidence_matrix)

      /**
       * Size of computational domain.
       */
      ACCESSOR_READ_ONLY(measure_of_omega)

      /**
       * The support points of all locally owned degrees of freedom in
       * (MPI rank) local numbering, i.e., the vector has n_locally_owned()
       * entries.
       *
       * All finite element spaces used in ryujin are nodal: the state
       * U_i is the value of the discrete solution at support_points()[i].
       * Interpolating a function into the finite element space thus
       * amounts to evaluating it at the support points.
       *
       * @note Only locally owned degrees of freedom are covered: the
       * ghost range of scalar_partitioner() also contains coupling
       * degrees of freedom on artificial cells whose position is not
       * known on this rank.
       */
      ACCESSOR_READ_ONLY(support_points)

      //@}

    protected:
      OfflineData(const std::string &subsection)
          : ParameterAcceptor(subsection)
      {
        affine_constraints_cg_ =
            std::make_unique<dealii::AffineConstraints<Number>>();
        affine_constraints_dg_ =
            std::make_unique<dealii::AffineConstraints<Number>>();
        boundary_indices_ = std::make_unique<Mirrored<unsigned int *>>(
            "offline_data_boundary_indices");
        coupling_boundary_pairs_ =
            std::make_unique<Mirrored<CouplingDescription *>>(
                "offline_data_coupling_boundary_pairs");
        sparsity_pattern_ = std::make_unique<dealii::DynamicSparsityPattern>();
        sparsity_pattern_simd_ = std::make_unique<SparsityPattern<warp_size>>();
        mass_matrix_ = std::make_unique<SparseMatrix<Number>>();
        mass_matrix_inverse_ = std::make_unique<SparseMatrix<Number>>();
        lumped_mass_matrix_ = std::make_unique<ScalarVector<Number>>();
        lumped_mass_matrix_inverse_ = std::make_unique<ScalarVector<Number>>();
        betaij_matrix_ = std::make_unique<SparseMatrix<Number>>();
        cij_matrix_ = std::make_unique<SparseMatrix<Number, dim>>();
        incidence_matrix_ = std::make_unique<SparseMatrix<Number>>();
      }

      bool have_discontinuous_ansatz_ = false;

      std::vector<dealii::types::global_dof_index> dg_to_cg_index_map_;
      std::vector<dealii::types::global_dof_index> cg_to_dg_index_map_;

      std::unique_ptr<dealii::AffineConstraints<Number>> affine_constraints_cg_;
      std::unique_ptr<dealii::AffineConstraints<Number>> affine_constraints_dg_;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          scalar_partitioner_;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          hyperbolic_vector_partitioner_;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          precomputed_vector_partitioner_;

      unsigned int n_export_indices_ = 0;
      unsigned int n_locally_internal_ = 0;
      unsigned int n_locally_owned_ = 0;
      unsigned int n_locally_relevant_ = 0;

      BoundaryMap boundary_map_;

      std::unique_ptr<Mirrored<unsigned int *>> boundary_indices_;
      std::vector<unsigned int> boundary_slots_;

      std::unique_ptr<Mirrored<CouplingDescription *>> coupling_boundary_pairs_;

      std::unique_ptr<dealii::DynamicSparsityPattern> sparsity_pattern_;

      std::unique_ptr<SparsityPattern<warp_size>> sparsity_pattern_simd_;

      std::unique_ptr<SparseMatrix<Number>> mass_matrix_;
      std::unique_ptr<SparseMatrix<Number>> mass_matrix_inverse_;

      std::unique_ptr<ScalarVector<Number>> lumped_mass_matrix_;
      std::unique_ptr<ScalarVector<Number>> lumped_mass_matrix_inverse_;

      std::unique_ptr<SparseMatrix<Number>> betaij_matrix_;
      std::unique_ptr<SparseMatrix<Number, dim>> cij_matrix_;
      std::unique_ptr<SparseMatrix<Number>> incidence_matrix_;

      Number measure_of_omega_ = 0.;

      std::vector<dealii::Point<dim>> support_points_;
    };
  } // namespace Interface
} // namespace ryujin
