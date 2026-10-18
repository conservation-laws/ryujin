//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <ryujin/interface/discretization.h>

#include <convenience_macros.h>
#include <gpu.h>
#include <observer_pointer.h>
#include <sparse_matrix.h>
#include <sparsity_pattern.h>
#include <state_vector.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
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
     * The interface of an OfflineData object: all data that can be
     * precomputed offline, most notably the DoFHandler, the sparsity
     * patterns, the partitioners, and all matrices (mass matrix, lumped
     * mass matrix, $c_{ij}$ matrices, and $n_{ij}$ matrices).
     *
     * The interface holds all state in protected fields with a
     * non-virtual accessor each. Objects of class type are held in
     * std::unique_ptr. A derived class fills the fields in prepare().
     * After prepare() is called, all getter functions return valid
     * references.
     *
     * @ingroup Mesh
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
       * A tuple describing (local) dof index, boundary normal, normal mass,
       * boundary mass, boundary id, and position of the boundary degree of
       * freedom.
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
       * Prepare offline data. A call to prepare() sets up storage, dof
       * handlers, sparsity patterns, and assembles all matrices.
       *
       * The problem_dimension and n_precomputed_values parameters is used
       * to set up appropriately sized vector partitioners for the state
       * and precomputed MultiComponentVector.
       */
      virtual void prepare(const unsigned int problem_dimension,
                           const unsigned int n_precomputed_values) = 0;

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * Return a read-only const reference to the DoFHandler for the
       * continuous ("cG") variant of the selected finite element space.
       *
       * @note If the selected finite element space is continuous then this
       * method simply returns the same object as dof_handler().
       */
      ACCESSOR_READ_ONLY(dof_handler_cg)

      /**
       * Return a read-only const reference to the DoFHandler for the
       * discontinuous ("dG") variant of the selected finite element space.
       *
       * @note If the selected finite element space is discontinuous then
       * this method simply returns the same object as dof_handler().
       */
      ACCESSOR_READ_ONLY(dof_handler_dg)

      /**
       * Return a read-only const reference to the dof handler.
       */
      const dealii::DoFHandler<dim> &dof_handler() const
      {
        if (discretization_->have_discontinuous_ansatz()) {
          Assert(dof_handler_dg_, dealii::ExcInternalError());
          return *dof_handler_dg_;
        } else {
          Assert(dof_handler_cg_, dealii::ExcInternalError());
          return *dof_handler_cg_;
        }
      }

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
       * discontinuous ("dG") variant of the selected finite element space in
       * (deal.II typical) global numbering.
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
       * An AffineConstraints object storing constraints in (deal.II typical)
       * global numbering.
       *
       * @note The affine constraints object is populated with (a)
       * hanging node constraints, and (b) periodicity constraints, that
       * directly affect the chosen ansatz space.
       */
      const dealii::AffineConstraints<Number> &affine_constraints() const
      {
        if (discretization_->have_discontinuous_ansatz()) {
          return *affine_constraints_dg_;
        } else {
          return *affine_constraints_cg_;
        }
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
       * \texttt{n_locally_relevant()})\f$ (consisting of non-SIMD-vectorized
       * indices) contains additional degrees of freedom that might have to
       * be exported to neighboring MPI ranks.
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
       * total number of degrees of freedom we store locally on this MPI rank.
       * I.e.,  we can access the half open interval [0, n_locally_relevant_)
       * on this machine.
       */
      ACCESSOR_READ_ONLY(n_locally_relevant)

      /**
       * The boundary map. Local numbering.
       *
       * For every degree of freedom that has nonzero support at the boundary
       * we record the global degree of freedom index along with a weighted
       * boundary normal, the associated boundary id, and position.
       *
       * This map is later used in OfflineData to handle boundary
       * degrees of freedom after every time step (for example to implement
       * reflective boundary conditions).
       */
      ACCESSOR_READ_ONLY(boundary_map)

      /**
       * A compacted, strictly increasing list of all locally owned degrees of
       * freedom that appear in the boundary_map(). The function returns a
       * reference to a Mirrored object that can be accessed on both memory
       * spaces with view() and size().
       */
      ACCESSOR_READ_ONLY(boundary_indices)

      /**
       * For every entry of the boundary_map() the position of its degree of
       * freedom in boundary_indices(). Local numbering.
       *
       * @note The boundary map can contain more than one entry for the same
       * degree of freedom (for example a corner that is simultaneously
       * located on a slip and a dirichlet boundary). All such entries share
       * the same position.
       */
      ACCESSOR_READ_ONLY(boundary_slots)

      /**
       * An array of CouplingDescription structures describing coupling
       * degrees of freedom i and j where both degrees of freedom are
       * collocated at the boundary (and hence the d_ij matrix has to be
       * symmetrized). The function returns a reference to a Mirrored object
       * holding (i, col_idx, j) triples that can be accessed on both memory
       * spaces with view() and size().
       */
      ACCESSOR_READ_ONLY(coupling_boundary_pairs)

      /**
       * The boundary map on all levels of the grid in case multilevel
       * support was enabled.
       */
      ACCESSOR_READ_ONLY(level_boundary_map)

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
       * The lumped mass matrix on all levels of the grid in case multilevel
       * support was enabled.
       */
      ACCESSOR_READ_ONLY(level_lumped_mass_matrix)

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
       * The incidence matrix \f$(beta_{ij})\f$: 1 for coupling face degrees
       * of freedom that share the same support point coordinate, 0 otherwise.
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
       * Returns a reference of the underlying Discretization object.
       */
      ACCESSOR_READ_ONLY(discretization)

      //@}

    protected:
      /**
       * Constructor.
       */
      OfflineData(const Discretization<dim> &discretization,
                  const std::string &subsection)
          : ParameterAcceptor(subsection)
          , discretization_(&discretization)
          , n_export_indices_(0)
          , n_locally_internal_(0)
          , n_locally_owned_(0)
          , n_locally_relevant_(0)
          , measure_of_omega_(0.)
      {
      }

      /**
       * @name Internal data
       */
      //@{

      dealii::ObserverPointer<const Discretization<dim>> discretization_;

      std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_cg_;
      std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_dg_;

      std::unique_ptr<dealii::AffineConstraints<Number>> affine_constraints_cg_;
      std::unique_ptr<dealii::AffineConstraints<Number>> affine_constraints_dg_;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          scalar_partitioner_;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          hyperbolic_vector_partitioner_;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
          precomputed_vector_partitioner_;

      unsigned int n_export_indices_;
      unsigned int n_locally_internal_;
      unsigned int n_locally_owned_;
      unsigned int n_locally_relevant_;

      BoundaryMap boundary_map_;
      std::vector<BoundaryMap> level_boundary_map_;

      std::unique_ptr<Mirrored<unsigned int *>> boundary_indices_;
      std::vector<unsigned int> boundary_slots_;

      std::unique_ptr<Mirrored<CouplingDescription *>> coupling_boundary_pairs_;

      std::unique_ptr<dealii::DynamicSparsityPattern> sparsity_pattern_;

      std::unique_ptr<SparsityPattern<warp_size>> sparsity_pattern_simd_;

      std::unique_ptr<SparseMatrix<Number>> mass_matrix_;
      std::unique_ptr<SparseMatrix<Number>> mass_matrix_inverse_;

      std::unique_ptr<ScalarVector<Number>> lumped_mass_matrix_;
      std::unique_ptr<ScalarVector<Number>> lumped_mass_matrix_inverse_;

      std::vector<ScalarHostVector<float>> level_lumped_mass_matrix_;

      std::unique_ptr<SparseMatrix<Number>> betaij_matrix_;
      std::unique_ptr<SparseMatrix<Number, dim>> cij_matrix_;
      std::unique_ptr<SparseMatrix<Number>> incidence_matrix_;

      Number measure_of_omega_;

      //@}
    };
  } // namespace Interface
} /* namespace ryujin */
