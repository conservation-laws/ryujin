#ifndef OFFLINE_DATA_TEMPLATE_H
#define OFFLINE_DATA_TEMPLATE_H

#include "offline_data.h"
#include "scratch_data.h"

#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <boost/range/irange.hpp>
#include <boost/range/iterator_range.hpp>

#include "helper.h"

// Workaround
namespace dealii
{
  template <>
  void dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<1, 1>>(
      dealii::DoFHandler<1, 1> const &,
      unsigned int,
      int,
      dealii::AffineConstraints<double> &,
      dealii::ComponentMask const &)
  {
    // do nothing
  }
}

namespace grendel
{
  using namespace dealii;


  template <int dim>
  OfflineData<dim>::OfflineData(
      const MPI_Comm &mpi_communicator,
      dealii::TimerOutput &computing_timer,
      const grendel::Discretization<dim> &discretization,
      const std::string &subsection /*= "OfflineData"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , discretization_(&discretization)
  {
  }


  template <int dim>
  void OfflineData<dim>::setup()
  {
    deallog << "OfflineData<dim>::setup()" << std::endl;

    IndexSet locally_owned;
    IndexSet locally_relevant;

    {
      deallog << "        distribute dofs" << std::endl;
      TimerOutput::Scope t(computing_timer_, "offline_data - distribute dofs");

      /* Initialize dof_handler and gather all locally owned indices: */

      dof_handler_.initialize(discretization_->triangulation(),
                              discretization_->finite_element());
      DoFRenumbering::Cuthill_McKee(dof_handler_);



      locally_owned = dof_handler_.locally_owned_dofs();
      n_locally_owned_ = locally_owned.n_elements();

      /* Print out the DoF distribution: */

      const auto n_dofs = locally_owned.size();
      deallog << "        " << n_dofs
              << " global DoFs, local DoF distribution:" << std::endl;

      const auto this_mpi_process =
          Utilities::MPI::this_mpi_process(mpi_communicator_);
      const auto n_mpi_processes =
          Utilities::MPI::n_mpi_processes(mpi_communicator_);
      unsigned int n_locally_owned_dofs = dof_handler_.n_locally_owned_dofs();

      if (this_mpi_process > 0) {
        MPI_Send(
            &n_locally_owned_dofs, 1, MPI_UNSIGNED, 0, 0, mpi_communicator_);

      } else {

        deallog << "        ( " << n_locally_owned_dofs << std::flush;
        for (unsigned int p = 1; p < n_mpi_processes; ++p) {
          MPI_Recv(&n_locally_owned_dofs,
                   1,
                   MPI_UNSIGNED,
                   p,
                   0,
                   mpi_communicator_,
                   MPI_STATUS_IGNORE);
          deallog << " + " << n_locally_owned_dofs << std::flush;
        }
        deallog << " )" << std::endl;
      }
    }

    const auto dofs_per_cell =
        discretization_->finite_element().dofs_per_cell;

    {
      deallog << "        create partitioner and affine constraints"
              << std::endl;
      TimerOutput::Scope t(
          computing_timer_,
          "offline_data - create partitioner and affine constraints");

      /*
       * We first populate a "locally extended" index set. This set
       * contains all locally relevant dofs (in our usual definition) and
       * in addition all dofs associated with ghosted cells. Adding these
       * extended degrees of freedom to local matrices simplifies the
       * assembly significantly. FIXME: If it ever becomes necessary we
       * could try to add a locally relevant index set to the  partitioner
       * to only communicate a subset of these degrees of freedom.
       */

      locally_relevant.clear();

      DoFTools::extract_locally_relevant_dofs(dof_handler_, locally_relevant);
      n_locally_relevant_ = locally_relevant.n_elements();

      partitioner_.reset(new dealii::Utilities::MPI::Partitioner(
          locally_owned, locally_relevant, mpi_communicator_));

      /*
       * create a temporary affine constraints object and populate it with
       * global indices:
       */

      AffineConstraints<double> global_constraints(locally_relevant);


      const auto n_periodic_faces =
          discretization_->triangulation().get_periodic_face_map().size();
      if (n_periodic_faces != 0) {
        /*
         * Enforce periodic boundary conditions. We assume that the mesh is
         * in "normal configuration". By convenction we also omit enforcing
         * periodicity in x direction. This avoids accidentally glueing the
         * corner degrees of freedom together whish leads to instability.
         */
        for (int i = 1; i < dim; ++i) /* omit x direction! */
          DoFTools::make_periodicity_constraints(dof_handler_,
                                                 /*b_id */ Boundary::periodic,
                                                 /*direction*/ i,
                                                 global_constraints);
      }

      DoFTools::make_hanging_node_constraints(dof_handler_, global_constraints);

      global_constraints.close();

      /*
       * And translate into local indices:
       */

      affine_constraints_.clear();
      for(auto line : global_constraints.get_lines()) {

        /* translate into local index ranges: */
        line.index = partitioner_->global_to_local(line.index);
        std::transform(line.entries.begin(),
                       line.entries.end(),
                       line.entries.begin(),
                       [&](auto entry) {
                         return std::make_pair(
                             partitioner_->global_to_local(entry.first),
                             entry.second);
                       });

        affine_constraints_.add_line(line.index);
        affine_constraints_.add_entries(line.index, line.entries);
        affine_constraints_.set_inhomogeneity(line.index, line.inhomogeneity);
      }
      affine_constraints_.close();
    }

    /*
     * We need a local view of a couple of matrices. Because they are never
     * used in a matrix-vector product, and in order to avoid unnecessary
     * overhead, we simply assemble these parts into a local
     * dealii::SparseMatrix<dim>.
     *
     * These sparse matrices have to store values for all _locally
     * extended_ degrees of freedom that couple. Unfortunately, the deal.II
     * library doesn't have a helper function to record such a sparsity
     * pattern, so we quickly do the grunt work by hand. While we are at
     * it, we also apply a translation between the global (distributed)
     * degrees of freedom numbering and the local index range [0,
     * locally_relevant_.n_elements()];
     */

    {
      deallog << "        create_sparsity_pattern" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "offline_data - create sparsity pattern");

      DynamicSparsityPattern dsp(n_locally_relevant_, n_locally_relevant_);

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

      for (auto cell : dof_handler_.active_cell_iterators()) {
        /* iterate over locally owned cells and the ghost layer */
        if (cell->is_artificial())
          continue;

        /* translate into local index ranges: */
        cell->get_dof_indices(dof_indices);
        std::transform(dof_indices.begin(),
                       dof_indices.end(),
                       dof_indices.begin(),
                       [&](auto index) {
                         return partitioner_->global_to_local(index);
                       });

        affine_constraints_.add_entries_local_to_global(
            dof_indices, dsp, false);
      }

      sparsity_pattern_.copy_from(dsp);
    }

    /*
     * Next we can (re)initialize all local matrices:
     */

    {
      deallog << "        set up matrices" << std::endl;
      TimerOutput::Scope t(computing_timer_, "offline_data - set up matrices");

      mass_matrix_.reinit(sparsity_pattern_);
      lumped_mass_matrix_.reinit(sparsity_pattern_);
      bij_matrix_.reinit(sparsity_pattern_);
      betaij_matrix_.reinit(sparsity_pattern_);
      norm_matrix_.reinit(sparsity_pattern_);
      for (auto &matrix : cij_matrix_)
        matrix.reinit(sparsity_pattern_);
      for (auto &matrix : nij_matrix_)
        matrix.reinit(sparsity_pattern_);
    }
  }


  template <int dim>
  void OfflineData<dim>::assemble()
  {
    deallog << "OfflineData<dim>::assemble()" << std::endl;

    mass_matrix_ = 0.;
    lumped_mass_matrix_ = 0.;
    norm_matrix_ = 0.;
    for (auto &matrix : cij_matrix_)
      matrix = 0.;
    betaij_matrix_ = 0.;
    for (auto &matrix : nij_matrix_)
      matrix = 0.;
    measure_of_omega_ = 0.;

    const unsigned int dofs_per_cell =
        discretization_->finite_element().dofs_per_cell;

    const unsigned int n_q_points = discretization_->quadrature().size();

    /*
     * First pass: Assemble all matrices:
     */

    /* The local, per-cell assembly routine: */

    const auto local_assemble_system = [&](const auto &cell,
                                           auto &scratch,
                                           auto &copy) {
      /* iterate over locally owned cells and the ghost layer */

      auto &is_artificial = copy.is_artificial_;
      auto &local_dof_indices = copy.local_dof_indices_;

      auto &local_boundary_normal_map = copy.local_boundary_normal_map_;
      auto &cell_mass_matrix = copy.cell_mass_matrix_;
      auto &cell_lumped_mass_matrix = copy.cell_lumped_mass_matrix_;
      auto &cell_betaij_matrix = copy.cell_betaij_matrix_;
      auto &cell_cij_matrix = copy.cell_cij_matrix_;
      auto &cell_measure = copy.cell_measure_;

      auto &fe_values = scratch.fe_values_;
      auto &fe_face_values = scratch.fe_face_values_;

      is_artificial = cell->is_artificial();
      if (is_artificial)
        return;

      cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_lumped_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_betaij_matrix.reinit(dofs_per_cell, dofs_per_cell);
      for (auto &matrix : cell_cij_matrix)
        matrix.reinit(dofs_per_cell, dofs_per_cell);

      fe_values.reinit(cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      /* translate into local index ranges: */
      std::transform(local_dof_indices.begin(),
                     local_dof_indices.end(),
                     local_dof_indices.begin(),
                     [&](auto index) {
                       return partitioner_->global_to_local(index);
                     });

      /* clear out copy data: */
      local_boundary_normal_map.clear();
      cell_mass_matrix = 0.;
      cell_lumped_mass_matrix = 0.;
      cell_betaij_matrix = 0.;
      for (auto &matrix : cell_cij_matrix)
        matrix = 0.;
      cell_measure = 0.;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
        const auto JxW = fe_values.JxW(q_point);

        if (cell->is_locally_owned())
          cell_measure += JxW;

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          const auto value_JxW = fe_values.shape_value(j, q_point) * JxW;
          const auto grad_JxW = fe_values.shape_grad(j, q_point) * JxW;

          cell_lumped_mass_matrix(j, j) += value_JxW;

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {

            const auto value = fe_values.shape_value(i, q_point);
            const auto grad = fe_values.shape_grad(i, q_point);

            cell_mass_matrix(i, j) += value * value_JxW;

            cell_betaij_matrix(i, j) += grad * grad_JxW;

            for (unsigned int d = 0; d < dim; ++d)
              cell_cij_matrix[d](i, j) += (value * grad_JxW)[d];

          } /* for i */
        }   /* for j */
      }     /* for q */

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        const auto face = cell->face(f);
        const auto id = face->boundary_id();

        if (!face->at_boundary())
          continue;

        fe_face_values.reinit(cell, f);
        const unsigned int n_face_q_points = scratch.face_quadrature_.size();

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          if (!discretization_->finite_element().has_support_on_face(j, f))
            continue;

          dealii::Tensor<1, dim> normal;
          if (id == Boundary::slip) {
            /*
             * Only accumulate a normal if the boundary indicator is for
             * slip boundary conditions. Otherwise we create a wrong normal
             * in corners of the computational domain.
             */
            for (unsigned int q = 0; q < n_face_q_points; ++q)
              normal += fe_face_values.normal_vector(q) *
                        fe_face_values.shape_value(j, q);
          }

          const auto index = local_dof_indices[j];

          // FIXME: This is a bloody hack:
          Point<dim> position;
          const auto global_index = partitioner_->local_to_global(index);
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            if (cell->vertex_dof_index(v, 0) == global_index)
              position = cell->vertex(v);

          /*
           * Ensure that we record the highest boundary indicator for a
           * given degree of freedom (higher indicators take precedence):
           */
          const auto old_id = std::get<1>(local_boundary_normal_map[index]);
          local_boundary_normal_map[index] =
              std::make_tuple(normal, std::max(old_id, id), position);
        } /* j */
      }   /* f */
    };

    const auto copy_local_to_global = [&](const auto &copy) {
      const auto &is_artificial = copy.is_artificial_;
      const auto &local_dof_indices = copy.local_dof_indices_;
      const auto &local_boundary_normal_map = copy.local_boundary_normal_map_;
      const auto &cell_mass_matrix = copy.cell_mass_matrix_;
      const auto &cell_lumped_mass_matrix = copy.cell_lumped_mass_matrix_;
      const auto &cell_cij_matrix = copy.cell_cij_matrix_;
      const auto &cell_betaij_matrix = copy.cell_betaij_matrix_;
      const auto &cell_measure = copy.cell_measure_;

      if (is_artificial)
        return;

      for (const auto &it : local_boundary_normal_map) {
        auto &[normal, id, position] = boundary_normal_map_[it.first];
        auto &[new_normal, new_id, new_position] = it.second;

        normal += new_normal;
        /*
         * Ensure that we record the highest boundary indicator for a given
         * degree of freedom (higher indicators take precedence):
         */
        id = std::max(id, new_id);
        position = new_position;
      }

      affine_constraints_.distribute_local_to_global(
          cell_mass_matrix, local_dof_indices, mass_matrix_);

      affine_constraints_.distribute_local_to_global(
          cell_lumped_mass_matrix, local_dof_indices, lumped_mass_matrix_);

      for (int k = 0; k < dim; ++k) {
        affine_constraints_.distribute_local_to_global(
            cell_cij_matrix[k], local_dof_indices, cij_matrix_[k]);
        affine_constraints_.distribute_local_to_global(
            cell_cij_matrix[k], local_dof_indices, nij_matrix_[k]);
      }

      affine_constraints_.distribute_local_to_global(
          cell_betaij_matrix, local_dof_indices, betaij_matrix_);

      measure_of_omega_ += cell_measure;
    };



    {
      deallog << "        assemble mass matrices, beta_ij, and c_ijs"
              << std::endl;
      TimerOutput::Scope t(
          computing_timer_,
          "offline_data - assemble mass matrices, beta_ij, and c_ij");

      WorkStream::run(dof_handler_.begin_active(),
                      dof_handler_.end(),
                      local_assemble_system,
                      copy_local_to_global,
                      AssemblyScratchData<dim>(*discretization_),
                      AssemblyCopyData<dim>());
    }

    measure_of_omega_ =
        Utilities::MPI::sum(measure_of_omega_, mpi_communicator_);

    /*
     * Second part: We have to import the "ghost" layer of the lumped mass
     * matrix in order to compute the b_ij matrices correctly.
     */

    {
      dealii::LinearAlgebra::distributed::Vector<double> temp_(partitioner_);

      for (unsigned int i = 0; i < partitioner_->local_size(); ++i)
        temp_.local_element(i) = lumped_mass_matrix_.diag_element(i);

      temp_.update_ghost_values();

      const auto offset = partitioner_->local_size();
      for (unsigned int i = 0; i < partitioner_->n_ghost_indices(); ++i)
        lumped_mass_matrix_.diag_element(offset + i) =
            temp_.local_element(offset + i);
    }

    /*
     * Third part: Compute norms and n_ijs
     */

    const auto on_subranges = [&](auto i1, const auto i2) {
      for (; i1 < i2; ++i1) {
        const auto row_index = *i1;

        std::for_each(sparsity_pattern_.begin(row_index),
                      sparsity_pattern_.end(row_index),
                      [&](const auto &jt) {
                        const auto value = gather_get_entry(cij_matrix_, &jt);
                        const double norm = value.norm();
                        set_entry(norm_matrix_, &jt, norm);
                      });

        std::for_each(sparsity_pattern_.begin(row_index),
                      sparsity_pattern_.end(row_index),
                      [&](const auto &jt) {
                        const auto col_index = jt.column();
                        const auto m_ij = get_entry(mass_matrix_, &jt);
                        const auto m_j =
                            lumped_mass_matrix_.diag_element(col_index);
                        const auto b_ij =
                            (row_index == col_index ? 1. : 0.) - m_ij / m_j;
                        set_entry(bij_matrix_, &jt, b_ij);
                      });

        for (auto &matrix : nij_matrix_) {
          auto nij_entry = matrix.begin(row_index);
          std::for_each(norm_matrix_.begin(row_index),
                        norm_matrix_.end(row_index),
                        [&](const auto &it) {
                          const auto norm = it.value();
                          nij_entry->value() /= norm;
                          ++nij_entry;
                        });
        }

      } /* row_index */
    };

    {
      deallog << "        compute b_ijs, |c_ij|s, and n_ijs" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "offline_data - compute b_ij, |c_ij|, and n_ij");

      const auto indices = boost::irange<unsigned int>(0, n_locally_relevant_);
      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);

      /*
       * And also normalize our boundary normals:
       */
      for (auto &it : boundary_normal_map_) {
        auto &[normal, id, _] = it.second;
        normal /= (normal.norm() + std::numeric_limits<double>::epsilon());
      }
    }

    /*
     * Second pass: Fix up boundary cijs:
     */

    /* The local, per-cell assembly routine: */

    const auto local_assemble_system_cij = [&](const auto &cell,
                                               auto &scratch,
                                               auto &copy) {
      /* iterate over locally owned cells and the ghost layer */

      auto &is_artificial = copy.is_artificial_;
      auto &local_dof_indices = copy.local_dof_indices_;

      auto &cell_cij_matrix = copy.cell_cij_matrix_;

      auto &fe_face_values = scratch.fe_face_values_;

      is_artificial = cell->is_artificial();
      if (is_artificial)
        return;

      for (auto &matrix : cell_cij_matrix)
        matrix.reinit(dofs_per_cell, dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      std::transform(local_dof_indices.begin(),
                     local_dof_indices.end(),
                     local_dof_indices.begin(),
                     [&](auto index) {
                       return partitioner_->global_to_local(index);
                     });

      /* clear out copy data: */
      for (auto &matrix : cell_cij_matrix)
        matrix = 0.;

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        const auto face = cell->face(f);
        const auto id = face->boundary_id();

        if (!face->at_boundary())
          continue;

        if (id != Boundary::slip)
          continue;

        fe_face_values.reinit(cell, f);
        const unsigned int n_face_q_points = scratch.face_quadrature_.size();

        for (unsigned int q = 0; q < n_face_q_points; ++q) {

          const auto JxW = fe_face_values.JxW(q);
          const auto normal_q = fe_face_values.normal_vector(q);

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            if (!discretization_->finite_element().has_support_on_face(j, f))
              continue;

            const auto &[normal_j, _1, _2] =
                boundary_normal_map_[local_dof_indices[j]];

            const auto value_JxW = fe_face_values.shape_value(j, q) * JxW;

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const auto value = fe_face_values.shape_value(i, q);

              for (unsigned int d = 0; d < dim; ++d)
                cell_cij_matrix[d](i, j) +=
                    (normal_j[d] - normal_q[d]) * (value * value_JxW);
            } /* i */
          }   /* j */
        }     /* q */
      }       /* f */
    };

    const auto copy_local_to_global_cij = [&](const auto &copy) {
      const auto &is_artificial = copy.is_artificial_;
      const auto &local_dof_indices = copy.local_dof_indices_;
      const auto &cell_cij_matrix = copy.cell_cij_matrix_;

      if (is_artificial)
        return;

      for (int k = 0; k < dim; ++k) {
        affine_constraints_.distribute_local_to_global(
            cell_cij_matrix[k], local_dof_indices, cij_matrix_[k]);
      }
    };

    {
      deallog << "        fix slip boundary c_ijs" << std::endl;
      TimerOutput::Scope t(computing_timer_,
                           "offline_data - fix slip boundary c_ij");

      WorkStream::run(dof_handler_.begin_active(),
                      dof_handler_.end(),
                      local_assemble_system_cij,
                      copy_local_to_global_cij,
                      AssemblyScratchData<dim>(*discretization_),
                      AssemblyCopyData<dim>());
    }
  }

} /* namespace grendel */

#endif /* OFFLINE_DATA_TEMPLATE_H */
