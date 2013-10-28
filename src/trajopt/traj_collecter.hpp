#pragma once
#include "trajopt/common.hpp"
#include "trajopt/collision_checker.hpp"
#include "trajopt/collision_terms.hpp"
#include "utils/timer.hpp"
#include "sco/optimizers.hpp"

#include <iostream>
#include <vector>

namespace trajopt
{


struct RichTrajectory
{
  MatrixXd traj;
  
  double time_to_converge;
  bool is_converged_eventually;

  std::vector<double> costs;
  std::vector<std::vector<double> > constraints;
  std::size_t iter_id;
};

struct TRAJOPT_API TrajCollecter
{

  void OptimizerCallback(OptProb* prob, DblVec& x, OptStepResults& stepres);

  
  TrajCollecter(OR::EnvironmentBasePtr env, ConfigurationPtr config, const VarArray& trajvars)
    : m_env(env), m_config(config), m_trajvars(trajvars)
  {
  }


  
  void Add(const std::vector<CostPtr>& costs);
  void Add(const std::vector<ConstraintPtr>& constraints);
  void Add(const std::vector<PlotterPtr>& plotters);
  void Add(PlotterPtr plotter);
  
  std::vector<RichTrajectory> m_trajs;

  // benchmark environment
  OpenRAVE::EnvironmentBasePtr m_env;

  // configuration space setting
  ConfigurationPtr m_config;

  // optimization variable for trajectory
  VarArray m_trajvars;

  // plotters obtain internal information from optimization solver
  std::vector<PlotterPtr> m_plotters;
};

typedef boost::shared_ptr<TrajCollecter> TrajCollecterPtr;

}
