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
  enum SIMDComparison : int {
    equal = _CMP_EQ_OQ,
    not_equal = _CMP_NEQ_OQ,
    less_than = _CMP_LT_OQ,
    less_than_or_equal = _CMP_LE_OQ,
    greater_than = _CMP_GT_OQ,
    greater_than_or_equal = _CMP_GE_OQ
  };


  template <SIMDComparison predicate, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  compare_and_apply_mask(const Number &left,
                         const Number &right,
                         const Number &true_value,
                         const Number &false_value)
  {
    bool mask;

    switch (predicate) {
    case SIMDComparison::equal:
      mask = left == right;
    case SIMDComparison::not_equal:
      mask = left != right;
    case SIMDComparison::less_than:
      mask = left < right;
    case SIMDComparison::less_than_or_equal:
      mask = left <= right;
    case SIMDComparison::greater_than:
      mask = left > right;
    case SIMDComparison::greater_than_or_equal:
      mask = left >= right;
    }

    return mask ? true_value : false_value;
  }


#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 3 && defined(__AVX512F__)

  template <int predicate>
  DEAL_II_ALWAYS_INLINE inline VectorizedArray<float, 16>
  compare_and_apply_mask(
             const VectorizedArray<float, 16> &left,
             const VectorizedArray<float, 16> &right,
             const VectorizedArray<float, 16> &true_values,
             const VectorizedArray<float, 16> &false_values)
  {
    const auto mask = _mm512_cmp_ps(left.data, right.data, predicate);
    VectorizedArray<float, 16> result;
    result.data = _mm512_or_ps(_mm512_and_ps(mask, true_values.data),
                               _mm512_andnot_ps(mask, false_values.data));
    return result;
  }


  template <int predicate>
  DEAL_II_ALWAYS_INLINE inline VectorizedArray<double, 8>
  compare_and_apply_mask(
             const VectorizedArray<double, 8> &left,
             const VectorizedArray<double, 8> &right,
             const VectorizedArray<double, 8> &true_values,
             const VectorizedArray<double, 8> &false_values)
  {
    const auto mask = _mm512_cmp_pd(left.data, right.data, predicate);
    VectorizedArray<double, 8> result;
    result.data = _mm256_or_ps(_mm512_and_ps(mask, true_values.data),
                               _mm512_andnot_ps(mask, false_values.data));
    return result;
  }

#endif


#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 2 && defined(__AVX__)

  template <int predicate>
  DEAL_II_ALWAYS_INLINE inline VectorizedArray<float, 8>
  compare_and_apply_mask(const VectorizedArray<float, 8> &left,
                         const VectorizedArray<float, 8> &right,
                         const VectorizedArray<float, 8> &true_values,
                         const VectorizedArray<float, 8> &false_values)
  {
    const auto mask = _mm256_cmp_ps(left.data, right.data, predicate);

    VectorizedArray<float, 8> result;
    result.data = _mm256_or_ps(_mm256_and_ps(mask, true_values.data),
                               _mm256_andnot_ps(mask, false_values.data));
    return result;
  }


  template <int predicate>
  DEAL_II_ALWAYS_INLINE inline VectorizedArray<double, 4>
  compare_and_apply_mask(const VectorizedArray<double, 4> &left,
                         const VectorizedArray<double, 4> &right,
                         const VectorizedArray<double, 4> &true_values,
                         const VectorizedArray<double, 4> &false_values)
  {
    const auto mask = _mm256_cmp_pd(left.data, right.data, predicate);

    VectorizedArray<double, 4> result;
    result.data = _mm256_or_ps(_mm256_and_ps(mask, true_values.data),
                               _mm256_andnot_ps(mask, false_values.data));
    return result;
  }

#endif

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)

  template <int predicate>
  DEAL_II_ALWAYS_INLINE inline VectorizedArray<float, 4>
  compare_and_apply_mask(const VectorizedArray<float, 4> &left,
                         const VectorizedArray<float, 4> &right,
                         const VectorizedArray<float, 4> &true_values,
                         const VectorizedArray<float, 4> &false_values)
  {
    const auto mask = _mm_cmp_ps(left.data, right.data, predicate);

    VectorizedArray<float, 4> result;
    result.data = _mm_or_ps(_mm_and_ps(mask, true_values.data),
                            _mm_andnot_ps(mask, false_values.data));
    return result;
  }


  template <int predicate>
  DEAL_II_ALWAYS_INLINE inline VectorizedArray<double, 2>
  compare_and_apply_mask(const VectorizedArray<double, 2> &left,
                         const VectorizedArray<double, 2> &right,
                         const VectorizedArray<double, 2> &true_values,
                         const VectorizedArray<double, 2> &false_values)
  {
    const auto mask = _mm_cmp_pd(left.data, right.data, predicate);

    VectorizedArray<double, 2> result;
    result.data = _mm_or_ps(_mm_and_ps(mask, true_values.data),
                            _mm_andnot_ps(mask, false_values.data));
    return result;
  }

#endif


  //
  // FIXME: Refactor into library and document
  //

  DEAL_II_ALWAYS_INLINE inline VectorizedArray<float>
  ternary_gt(const VectorizedArray<float> &left,
             const VectorizedArray<float> &right,
             const VectorizedArray<float> &true_value,
             const VectorizedArray<float> &false_value)
  {
    return compare_and_apply_mask<SIMDComparison::greater_than>(
        left, right, true_value, false_value);
  }

  DEAL_II_ALWAYS_INLINE inline VectorizedArray<double>
  ternary_gt(const VectorizedArray<double> &left,
             const VectorizedArray<double> &right,
             const VectorizedArray<double> &true_value,
             const VectorizedArray<double> &false_value)
  {
    return compare_and_apply_mask<SIMDComparison::greater_than>(
        left, right, true_value, false_value);
  }


  DEAL_II_ALWAYS_INLINE inline double ternary_gt(const double left,
                                                 const double right,
                                                 const double true_value,
                                                 const double false_value)
  {
    return compare_and_apply_mask<SIMDComparison::greater_than>(
        left, right, true_value, false_value);
  }


  DEAL_II_ALWAYS_INLINE inline float ternary_gt(const float left,
                                                const float right,
                                                const float true_value,
                                                const float false_value)
  {
    return compare_and_apply_mask<SIMDComparison::greater_than>(
        left, right, true_value, false_value);
  }

} // namespace dealii

#endif /* SIMD_H */
