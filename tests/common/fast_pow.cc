#include <simd.h>

#include <iomanip>
#include <iostream>

int main()
{
  std::cout << std::setprecision(16);
  std::cout << std::scientific;

  using VA = dealii::VectorizedArray<double>;

  auto test = [&](const VA a, const VA b) {
    std::cout << "a:        " << a << "\n";
    std::cout << "b:        " << b << "\n";
    std::cout << "pow:      " << ryujin::pow(a, b) << "\n";
    std::cout << "fast_pow: " << ryujin::fast_pow(a, b, ryujin::Bias::none)
              << "\n"
              << std::endl;
  };

  test(1.225, 2.3559);
  test(2.135, 1. / 3.);
}
