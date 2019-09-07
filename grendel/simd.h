#ifndef SIMD_H
#define SIMD_H

#include <deal.II/base/vectorization.h>

namespace dealii
{
/**
 * @name Ternary operations on VectorizedArray
 */
//@{

/**
 * enum class encoding binary operations for a component-wise comparison of
 * VectorizedArray data types.
 *
 * @note In case of SIMD vecorization (sse, avx, av512) we select the
 * corresponding ordered, non-signalling (<code>OQ</code>) variants.
 */
enum class SIMDComparison : int
{
#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 2 && defined(__AVX__)
  equal                 = _CMP_EQ_OQ,
  not_equal             = _CMP_NEQ_OQ,
  less_than             = _CMP_LT_OQ,
  less_than_or_equal    = _CMP_LE_OQ,
  greater_than          = _CMP_GT_OQ,
  greater_than_or_equal = _CMP_GE_OQ
#else
  equal,
  not_equal,
  less_than,
  less_than_or_equal,
  greater_than,
  greater_than_or_equal
#endif
};


/**
 * Computes the vectorized equivalent of the following ternary operation:
 * @code
 *   (left OP right) ? true_value : false_value
 * @endcode
 * where <code>OP</code> is a binary operator (such as <code>=</code>,
 * <code>!=</code>, <code><</code>, <code><=</code>, <code>></code>, and
 * <code>>=</code>).
 *
 * Such a computational idiom is useful as an alternative to branching
 * whenever the control flow itself would depend on (computed) data. For
 * example, in case of a scalar data type the statement
 * <code>(left < right) ? true_value : false_value</code>
 * could have been also implementd using an <code>if</code>-statement:
 * @code
 * if (left < right)
 *     result = true_value;
 * else
 *     result = false_value;
 * @endcode
 * This, however, is fundamentally impossible in case of vectorization
 * because different decisions will be necessary on different vector entries
 * (lanes) and
 * the first variant (based on a ternary operator) has to be used instead:
 * @code
 *   result = compare_and_apply_mask<SIMDComparison::less_than>
 *     (left, right, true_value, false_value);
 * @endcode
 * Some more illustrative examples (that are less efficient than the
 * dedicated <code>std::max</code> and <code>std::abs</code> overloads):
 * @code
 *   VectorizedArray<double> left;
 *   VectorizedArray<double> right;
 *
 *   // std::max
 *   const auto maximum = compare_and_apply_mask<SIMDComparison::greater_than>
 *     (left, right, left, right);
 *
 *   // std::abs
 *   const auto absolute = compare_and_apply_mask<SIMDComparison::less_than>
 *     (left, VectorizedArray<double>(0.), -left, left);
 * @endcode
 *
 * More precisely, this function first computes a (boolean) mask that is
 * the result of a binary operator <code>OP</code> applied to all elements
 * of the VectorizedArray arguments @p left and @p right. The mask is then
 * used to either select the corresponding component of @p true_value (if
 * the binary operation equates to true), or @p false_value. The binary
 * operator is encoded via the SIMDComparison template argument
 * @p predicate.
 *
 * In order to ease with generic programming approaches, the function
 * provides overloads for all VectorizedArray<Number> variants as well as
 * generic POD types such as double and float.
 *
 * @note For this function to work the binary operation has to be encoded
 * via a SIMDComparison template argument @p predicate. Depending on it
 * appropriate low-level machine instructions are generated replacing the
 * call to compare_and_apply_mask. This also explains why @p predicate is a
 * compile-time constant template parameter and not a constant function
 * argument. In order to be able to emit the correct low-level instruction,
 * the compiler has to know the comparison at compile time.
 */
template <SIMDComparison predicate, typename Number>
DEAL_II_ALWAYS_INLINE inline Number
compare_and_apply_mask(const Number &left,
                       const Number &right,
                       const Number &true_value,
                       const Number &false_value)
{
  bool mask;
  switch (predicate)
    {
      case SIMDComparison::equal:
        mask = (left == right);
        break;
      case SIMDComparison::not_equal:
        mask = (left != right);
        break;
      case SIMDComparison::less_than:
        mask = (left < right);
        break;
      case SIMDComparison::less_than_or_equal:
        mask = (left <= right);
        break;
      case SIMDComparison::greater_than:
        mask = (left > right);
        break;
      case SIMDComparison::greater_than_or_equal:
        mask = (left >= right);
        break;
    }

  return mask ? true_value : false_value;
}


/**
 * Specialization of above function for the non-vectorized
 * VectorizedArray<Number, 1> variant.
 */
template <SIMDComparison predicate, typename Number>
DEAL_II_ALWAYS_INLINE inline VectorizedArray<Number, 1>
compare_and_apply_mask(const VectorizedArray<Number, 1> &left,
                       const VectorizedArray<Number, 1> &right,
                       const VectorizedArray<Number, 1> &true_value,
                       const VectorizedArray<Number, 1> &false_value)
{
  VectorizedArray<Number, 1> result;
  result.data = compare_and_apply_mask<predicate, Number>(left.data,
                                                          right.data,
                                                          true_value.data,
                                                          false_value.data);
  return result;
}

//@}

#ifndef DOXYGEN
#  if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 3 && defined(__AVX512F__)

template <SIMDComparison predicate>
DEAL_II_ALWAYS_INLINE inline VectorizedArray<float, 16>
compare_and_apply_mask(const VectorizedArray<float, 16> &left,
                       const VectorizedArray<float, 16> &right,
                       const VectorizedArray<float, 16> &true_values,
                       const VectorizedArray<float, 16> &false_values)
{
  const __mmask16 mask =
    _mm512_cmp_ps_mask(left.data, right.data, static_cast<int>(predicate));
  VectorizedArray<float, 16> result;
  result.data = _mm512_mask_mov_ps(false_values.data, mask, true_values.data);
  return result;
}



template <SIMDComparison predicate>
DEAL_II_ALWAYS_INLINE inline VectorizedArray<double, 8>
compare_and_apply_mask(const VectorizedArray<double, 8> &left,
                       const VectorizedArray<double, 8> &right,
                       const VectorizedArray<double, 8> &true_values,
                       const VectorizedArray<double, 8> &false_values)
{
  const __mmask16 mask =
    _mm512_cmp_pd_mask(left.data, right.data, static_cast<int>(predicate));
  VectorizedArray<double, 8> result;
  result.data = _mm512_mask_mov_pd(false_values.data, mask, true_values.data);
  return result;
}

#  endif

#  if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 2 && defined(__AVX__)

template <SIMDComparison predicate>
DEAL_II_ALWAYS_INLINE inline VectorizedArray<float, 8>
compare_and_apply_mask(const VectorizedArray<float, 8> &left,
                       const VectorizedArray<float, 8> &right,
                       const VectorizedArray<float, 8> &true_values,
                       const VectorizedArray<float, 8> &false_values)
{
  const auto mask =
    _mm256_cmp_ps(left.data, right.data, static_cast<int>(predicate));

  VectorizedArray<float, 8> result;
  result.data = _mm256_or_ps(_mm256_and_ps(mask, true_values.data),
                             _mm256_andnot_ps(mask, false_values.data));
  return result;
}


template <SIMDComparison predicate>
DEAL_II_ALWAYS_INLINE inline VectorizedArray<double, 4>
compare_and_apply_mask(const VectorizedArray<double, 4> &left,
                       const VectorizedArray<double, 4> &right,
                       const VectorizedArray<double, 4> &true_values,
                       const VectorizedArray<double, 4> &false_values)
{
  const auto mask =
    _mm256_cmp_pd(left.data, right.data, static_cast<int>(predicate));

  VectorizedArray<double, 4> result;
  result.data = _mm256_or_pd(_mm256_and_pd(mask, true_values.data),
                             _mm256_andnot_pd(mask, false_values.data));
  return result;
}

#  endif

#  if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && defined(__SSE2__)

template <SIMDComparison predicate>
DEAL_II_ALWAYS_INLINE inline VectorizedArray<float, 4>
compare_and_apply_mask(const VectorizedArray<float, 4> &left,
                       const VectorizedArray<float, 4> &right,
                       const VectorizedArray<float, 4> &true_values,
                       const VectorizedArray<float, 4> &false_values)
{
  __m128 mask;
  switch (predicate)
    {
      case SIMDComparison::equal:
        mask = _mm_cmpeq_ps(left.data, right.data);
        break;
      case SIMDComparison::not_equal:
        mask = _mm_cmpneq_ps(left.data, right.data);
        break;
      case SIMDComparison::less_than:
        mask = _mm_cmplt_ps(left.data, right.data);
        break;
      case SIMDComparison::less_than_or_equal:
        mask = _mm_cmple_ps(left.data, right.data);
        break;
      case SIMDComparison::greater_than:
        mask = _mm_cmpgt_ps(left.data, right.data);
        break;
      case SIMDComparison::greater_than_or_equal:
        mask = _mm_cmpge_ps(left.data, right.data);
        break;
    }

  VectorizedArray<float, 4> result;
  result.data = _mm_or_ps(_mm_and_ps(mask, true_values.data),
                          _mm_andnot_ps(mask, false_values.data));

  return result;
}


template <SIMDComparison predicate>
DEAL_II_ALWAYS_INLINE inline VectorizedArray<double, 2>
compare_and_apply_mask(const VectorizedArray<double, 2> &left,
                       const VectorizedArray<double, 2> &right,
                       const VectorizedArray<double, 2> &true_values,
                       const VectorizedArray<double, 2> &false_values)
{
  __m128d mask;
  switch (predicate)
    {
      case SIMDComparison::equal:
        mask = _mm_cmpeq_pd(left.data, right.data);
        break;
      case SIMDComparison::not_equal:
        mask = _mm_cmpneq_pd(left.data, right.data);
        break;
      case SIMDComparison::less_than:
        mask = _mm_cmplt_pd(left.data, right.data);
        break;
      case SIMDComparison::less_than_or_equal:
        mask = _mm_cmple_pd(left.data, right.data);
        break;
      case SIMDComparison::greater_than:
        mask = _mm_cmpgt_pd(left.data, right.data);
        break;
      case SIMDComparison::greater_than_or_equal:
        mask = _mm_cmpge_pd(left.data, right.data);
        break;
    }

  VectorizedArray<double, 2> result;
  result.data = _mm_or_pd(_mm_and_pd(mask, true_values.data),
                          _mm_andnot_pd(mask, false_values.data));

  return result;
}

#  endif
#endif // DOXYGEN

} // namespace dealii

#endif /* SIMD_H */
