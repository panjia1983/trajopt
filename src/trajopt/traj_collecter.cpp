#include "traj_plotter.hpp"
#include "traj_feature.h"

using namespace OpenRAVE;

namespace trajopt
{


void TrajCollecter::Add(const std::vector<CostPtr>& costs)
{
  BOOST_FOREACH(const CostPtr& cost, costs)
  {
    if(PlotterPtr plotter = boost::dynamic_pointer_cast<Plotter>(cost))
    {
      m_plotters.push_back(plotter);
    }
  }
}

void TrajCollecter::Add(const std::vector<ConstraintPtr>& constraints)
{
  BOOST_FOREACH(const ConstraintPtr& cnt, constraints)
  {
    if(PlotterPtr plotter = boost::dynamic_pointer_cast<Plotter>(cnt))
    {
      m_plotters.push_back(plotter);
    }
  }
}

void TrajCollecter::Add(const std::vector<PlotterPtr>& plotters)
{
  BOOST_FOREACH(const PlotterPtr& plotter, plotters)
  {
    m_plotters.push_back(plotter);
  }
}


void TrajCollecter::OptimizerCallback(OptProb* prob, DblVec& x, OptStepResults& stepres)
{
  // get trajectory for current configuration x
  MatrixXd traj = getTraj(x, m_trajvars);

  // set the rich trajectory
  RichTrajectory rtraj;
  rtraj.traj = traj;

  // get optimization internal conditions
  BOOST_FOREACH(PlotterPtr& plotter, m_plotters)
  {
    plotter->Plot(x, *m_env, handles);
  }

  std::vector<double> costs;
  std::vector<std::vector<double> > constraints;

  BOOST_FOREACH(const PlotterPtr& plotter, m_plotters)
  {
    if(ConstraintPtr constraint = boost::dynamic_pointer_cast<Constraint>(plotter))
    {
      constraints.push_back(constraint->violations(x));
    }
    else if(CostPtr cost = boost::dynamic_pointer_cast<Cost>(plotter))
    {
      costs.push_back(cost->value(x));
    }
    else
    {
      // throw exception
    }
  }

  rtraj.costs = costs;
  rtraj.constraints = constraints;
  rtraj.time_to_convergence = stepres.time_since_start; // need post-processing
  rtraj.iter_id = stepres.step_id;

  if(stepres.is_converged != OPTS_UNKNOWN) // final step of one optimization
  {
    if(stepres.is_converged == OPTS_CONVERGED)
      rtraj.is_converged_eventually = true;
    else
      rtraj.is_converged_eventually = false;
  }

  m_trajs.push_back(rtraj);

  if(stepres.is_converged != OPTS_UNKNOWN) // final step of one optimization, some postprocessing
  {
    double entire_opt_time = stepres.time_since_start;
    for(std::size_t i = 0; i < m_trajs.size(); ++i)
    {
      m_trajs[i].time_to_convergence = entire_opt_time - m_trajs[i].time_to_convergence;
    }
  }
}




}
