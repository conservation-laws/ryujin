#ifndef TO_VECTOR_FUNCTION_TEMPLATE_H
#define TO_VECTOR_FUNCTION_TEMPLATE_H

#include <deal.II/base/function.h>

namespace grendel
{
  namespace
  {
    template <int dim, typename Number, typename Callable>
    class ToVectorFunction : public dealii::Function<dim, Number>
    {
    public:

      ToVectorFunction(const Callable &callable)
          : dealii::Function<dim, Number>(dim)
          , callable_(callable)
      {
      }

      virtual Number value(const dealii::Point<dim> &point,
                           unsigned int component) const
      {
        return callable_(point)[component];
      }

      virtual void vector_value(const dealii::Point<dim> &point,
                                dealii::Vector<double> &v) const
      {
        for (unsigned int i = 0; i < dim; ++i)
          v(i) = callable_(point)[i];
      }

    private:
      const Callable callable_;
    };
  } // namespace


  template <int dim, typename Number, typename Callable>
  ToVectorFunction<dim, Number, Callable>
  to_vector_function(const Callable &callable)
  {
    return ToVectorFunction<dim, Number, Callable>(callable);
  }

} /* namespace grendel */

#endif /* TO_VECTOR_FUNCTION_TEMPLATE_H */
