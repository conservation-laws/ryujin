#ifndef TIMELOOP_H
#define TIMELOOP_H

#include <discretization.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{

  template <int dim>
  class TimeLoop : public dealii::ParameterAcceptor
  {
  public:
    TimeLoop();
    virtual ~TimeLoop() final = default;

    virtual void run();

  private:
    /* Private methods for run(): */

    virtual void initialize();
    virtual void compute_offline_data();

    /* Data: */

    std::string base_name_;

    grendel::Discretization<dim> discretization;

    std::unique_ptr<std::ofstream> filestream;

  };

} // namespace ryujin

#endif /* TIMELOOP_H */
