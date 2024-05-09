//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "discretization.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

namespace ryujin
{

  /**
   * Internal scratch data for thread parallelized assembly. See the
   * deal.II Workstream documentation for details.
   *
   * @ingroup Mesh
   */
  template <int dim>
  class AssemblyScratchData
  {
  public:
    AssemblyScratchData(const AssemblyScratchData<dim> &assembly_scratch_data)
        : AssemblyScratchData(assembly_scratch_data.discretization_)
    {
    }


    AssemblyScratchData(const Discretization<dim> &discretization)
        : discretization_(discretization)
        , fe_values_(discretization_.mapping(),
                     discretization_.finite_element(),
                     discretization_.quadrature(),
                     dealii::update_values | dealii::update_gradients |
                         dealii::update_quadrature_points |
                         dealii::update_JxW_values)
        , fe_face_values_(
              discretization_.mapping(),
              discretization_.finite_element(),
              discretization_.face_quadrature(),
              dealii::update_values | dealii::update_quadrature_points |
                  dealii::update_JxW_values | dealii::update_normal_vectors)
        , fe_face_values_nodal_(discretization_.mapping(),
                                discretization_.finite_element(),
                                discretization_.face_nodal_quadrature(),
                                dealii::update_values |
                                    dealii::update_quadrature_points)
        , fe_neighbor_face_values_(discretization_.mapping(),
                                   discretization_.finite_element(),
                                   discretization_.face_quadrature(),
                                   dealii::update_values)
        , fe_neighbor_face_values_nodal_(
              discretization_.mapping(),
              discretization_.finite_element(),
              discretization_.face_nodal_quadrature(),
              dealii::update_values)
    {
    }

    const Discretization<dim> &discretization_;
    dealii::FEValues<dim> fe_values_;
    dealii::FEFaceValues<dim> fe_face_values_;
    dealii::FEFaceValues<dim> fe_face_values_nodal_;
    dealii::FEFaceValues<dim> fe_neighbor_face_values_;
    dealii::FEFaceValues<dim> fe_neighbor_face_values_nodal_;
  };

  /**
   * Internal copy data for thread parallelized assembly. See the deal.II
   * Workstream documentation for details.
   */
  template <int dim, typename Number = double>
  class AssemblyCopyData
  {
  public:
    bool is_locally_owned_;
    std::vector<dealii::types::global_dof_index> local_dof_indices_;
    dealii::FullMatrix<Number> cell_mass_matrix_;
    dealii::FullMatrix<Number> cell_mass_matrix_inverse_;
    std::array<dealii::FullMatrix<Number>, dim> cell_cij_matrix_;
    Number cell_measure_;

    static constexpr unsigned int n_faces = 2 * dim;
    std::array<std::vector<dealii::types::global_dof_index>, n_faces>
        neighbor_local_dof_indices_;
    std::array<std::array<dealii::FullMatrix<Number>, dim>, n_faces>
        interface_cij_matrix_;
    std::array<dealii::FullMatrix<Number>, n_faces> interface_incidence_matrix_;
  };

} // namespace ryujin
