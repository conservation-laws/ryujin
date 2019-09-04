#ifndef SIMD_H
#define SIMD_H

#include <deal.II/base/vectorization.h>


namespace std
{
  //
  // FIXME: Refactor into library
  //

  template <typename Number, int width, typename Number2>
  inline ::dealii::VectorizedArray<Number, width>
  pow(const ::dealii::VectorizedArray<Number, width> &x, const Number2 p)
  {
    Number values[::dealii::VectorizedArray<Number, width>::n_array_elements];
    for (unsigned int i = 0;
         i < dealii::VectorizedArray<Number, width>::n_array_elements;
         ++i)
      values[i] = std::pow(x[i], p);
    ::dealii::VectorizedArray<Number, width> out;
    out.load(&values[0]);
    return out;
  }
}


namespace dealii
{
  //
  // FIXME: Refactor into library and document
  //

  DEAL_II_ALWAYS_INLINE inline VectorizedArray<float>
  ternary_gt(const VectorizedArray<float> &left,
             const VectorizedArray<float> &right,
             const VectorizedArray<float> &true_value,
             const VectorizedArray<float> &false_value)
  {
    const auto mask = _mm256_cmp_ps(left.data, right.data, _CMP_GT_OQ);
    VectorizedArray<float> result;
    result.data = _mm256_or_ps(_mm256_and_ps(mask, true_value.data),
                               _mm256_andnot_ps(mask, false_value.data));
    return result;
  }

  DEAL_II_ALWAYS_INLINE inline VectorizedArray<double>
  ternary_gt(const VectorizedArray<double> &left,
             const VectorizedArray<double> &right,
             const VectorizedArray<double> &true_value,
             const VectorizedArray<double> &false_value)
  {
    const auto mask = _mm256_cmp_pd(left.data, right.data, _CMP_GT_OQ);
    VectorizedArray<double> result;
    result.data = _mm256_or_pd(_mm256_and_pd(mask, true_value.data),
                               _mm256_andnot_pd(mask, false_value.data));
    return result;
  }


  DEAL_II_ALWAYS_INLINE inline double ternary_gt(const double left,
                                                 const double right,
                                                 const double true_value,
                                                 const double false_value)
  {
    return (left > right) ? true_value : false_value;
  }


  DEAL_II_ALWAYS_INLINE inline float ternary_gt(const float left,
                                                const float right,
                                                const float true_value,
                                                const float false_value)
  {
    return (left > right) ? true_value : false_value;
  }

} // namespace dealii

#endif /* SIMD_H */
