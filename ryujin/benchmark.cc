// DEBUG

using rank1_type = typename RiemannSolver<dim>::rank1_type;

#if 0
      rank1_type U_i{{2.21953,1.09817,0,5.09217}};
      rank1_type U_j{{2.21953,1.09817,0,5.09217}};
      Tensor<1, dim> n_ij{{0.948683,-0.316228}};
      // output 1.57222 NEW
#endif

#if 0
      rank1_type U_i{{2.18162, 1.06679, 5.52606e-06, 5.00393}};
      rank1_type U_j{{1.97325, 0.777591, -1.21599e-06, 4.33331}};
      Tensor<1, dim> n_ij{{0.83205, -0.5547}};
      // output 1.47181 NEW
#endif

rank1_type U_i{{2.21953, 1.09817, 0., 5.09217}};
rank1_type U_j{{1.4, 0., 0., 2.5}};
Tensor<1, dim> n_ij{{0.948683, -0.316228}};
// output 1.33017 NEW

// benchmarking:
constexpr unsigned int size = 10000000;
std::vector<double> scratch(size);
{
  TimerOutput::Scope t(computing_timer, "benchmark - compute lambda_max");

  const auto on_subranges = [&](auto it1, auto it2) {
    for (auto it = it1; it != it2; ++it) {
      const auto [lambda_max, n_iterations] =
          riemann_solver.lambda_max(U_i, U_j, n_ij);
      *it = lambda_max;
    }
  };
  parallel::apply_to_subranges(
      scratch.begin(), scratch.end(), on_subranges, 4096);
}

// Output result:
{
  const auto [lambda_max, n_iterations] =
      riemann_solver.lambda_max(U_i, U_j, n_ij);
  std::cout << "RESULT: " << lambda_max << " in n=" << n_iterations
            << " iterations <-----" << std::endl;
}
