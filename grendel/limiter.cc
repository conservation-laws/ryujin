#include "limiter.template.h"

using namespace dealii;

namespace grendel
{
  /* instantiations */

  template NUMBER
  Limiter<DIM, NUMBER>::limit<Limiter<DIM, NUMBER>::Limiters::specific_entropy>(
      const std::array<NUMBER, 3> &,
      const rank1_type &,
      const rank1_type &,
      const NUMBER,
      const NUMBER);

  template VectorizedArray<NUMBER> Limiter<DIM, VectorizedArray<NUMBER>>::limit<
      Limiter<DIM, VectorizedArray<NUMBER>>::Limiters::specific_entropy>(
      const std::array<VectorizedArray<NUMBER>, 3> &,
      const rank1_type &,
      const rank1_type &,
      const VectorizedArray<NUMBER>,
      const VectorizedArray<NUMBER>);

  template NUMBER Limiter<DIM, NUMBER>::limit<
      Limiter<DIM, NUMBER>::Limiters::entropy_inequality>(
      const std::array<NUMBER, 5> &,
      const rank1_type &,
      const rank1_type &,
      const NUMBER,
      const NUMBER);

  template VectorizedArray<NUMBER> Limiter<DIM, VectorizedArray<NUMBER>>::limit<
      Limiter<DIM, VectorizedArray<NUMBER>>::Limiters::entropy_inequality>(
      const std::array<VectorizedArray<NUMBER>, 5> &,
      const rank1_type &,
      const rank1_type &,
      const VectorizedArray<NUMBER>,
      const VectorizedArray<NUMBER>);

} // namespace grendel
