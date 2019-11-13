#include <compile_time_options.h>

#include "problem_description.h"
#include "sparse_matrix_simd.template.h"

namespace grendel
{
  /* instantiations */

  template class SparsityPatternSIMD<
      dealii::VectorizedArray<NUMBER>::n_array_elements>;

  template class SparseMatrixSIMD<
      dealii::VectorizedArray<NUMBER>::n_array_elements,
      NUMBER>;

  template class SparseMatrixSIMD<
      dealii::VectorizedArray<NUMBER>::n_array_elements,
      NUMBER,
      DIM>;

  template class SparseMatrixSIMD<
      dealii::VectorizedArray<NUMBER>::n_array_elements,
      NUMBER,
      ProblemDescription<DIM>::problem_dimension>;

} /* namespace grendel */
