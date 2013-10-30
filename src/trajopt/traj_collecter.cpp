#include "traj_collecter.hpp"
#include "traj_feature.hpp"
#include <boost/foreach.hpp>

using namespace OpenRAVE;

namespace trajopt
{


bool CheckTraj(const MatrixXd& traj, ConfigurationPtr rad, int n = 100)
{
  CollisionCheckerPtr cc = CollisionChecker::GetOrCreate(*(rad->GetEnv()));
  double old_threshold = cc->GetContactDistance();
  cc->SetContactDistance(.01);
  
  int n_dof = traj.cols();
  for(int i = 0; i < traj.rows() - 1; ++i)
  {
    MatrixXd interp(n, n_dof);

    for(int j = 0; j < n_dof; ++j)
      interp.col(j) = VectorXd::LinSpaced(n, traj(i, j), traj(i+1, j));

    for(int j = 0; j < n; ++j)
    {
      vector<Collision> collisions;
      rad->SetDOFValues(trajToDblVec(interp.row(j)));
      cc->LinksVsAll(rad->GetAffectedLinks(), collisions, -1);
      if(collisions.size() > 0)
      {
          cc->SetContactDistance(old_threshold);
          return false;
      }
    }
  }

  cc->SetContactDistance(old_threshold);
  return true;
}



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

std::size_t TrajCollecter::GetNumCosts() const
{
  std::size_t num_costs = 0;
  BOOST_FOREACH(const PlotterPtr& plotter, m_plotters)
  {
    if(CostPtr cost = boost::dynamic_pointer_cast<Cost>(plotter))
    {
      num_costs ++;
    }
  }

  return num_costs;
}

std::size_t TrajCollecter::GetNumConstraints() const
{
  std::size_t num_constraints = 0;
  BOOST_FOREACH(const PlotterPtr& plotter, m_plotters)
  {
    if(ConstraintPtr constraint = boost::dynamic_pointer_cast<Constraint>(plotter))
    {
      num_constraints ++;
    }
  }

  return num_constraints;
}

std::vector<std::string> TrajCollecter::GetCostNames() const
{
  std::vector<std::string> cost_names;
  BOOST_FOREACH(const PlotterPtr& plotter, m_plotters)
  {
    if(CostPtr cost = boost::dynamic_pointer_cast<Cost>(plotter))
    {
      cost_names.push_back(cost->name());
    }
  }

  return cost_names;
}

std::vector<std::string> TrajCollecter::GetConstraintNames() const
{
  std::vector<std::string> constraint_names;
  BOOST_FOREACH(const PlotterPtr& plotter, m_plotters)
  {
    if(ConstraintPtr constraint = boost::dynamic_pointer_cast<Constraint>(plotter))
    {
      constraint_names.push_back(constraint->name());
    }
  }

  return constraint_names;
}


void TrajCollecter::OptimizerCallback(OptProb* prob, DblVec& x, OptStepResults& stepres)
{
  // get trajectory for current configuration x
  MatrixXd traj = getTraj(x, m_trajvars);

  // set the rich trajectory
  RichTrajectory rtraj;
  rtraj.traj = traj;
  rtraj.task_id = m_task_id;
  rtraj.perturb_id = m_perturb_id;

  // get optimization internal conditions
  std::vector<GraphHandlePtr> handles; // empty, not plotting anything
  BOOST_FOREACH(PlotterPtr& plotter, m_plotters)
  {
    plotter->Plot(x, *m_env, handles);
  }

  std::vector<double> costs;
  std::vector<double> constraints;

  BOOST_FOREACH(const PlotterPtr& plotter, m_plotters)
  {
    if(ConstraintPtr constraint = boost::dynamic_pointer_cast<Constraint>(plotter))
    {
      constraints.push_back(constraint->violation(x));
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
  rtraj.time_to_converge = stepres.time_since_start; // need post-processing
  rtraj.iter_id = stepres.step_id;

  if(stepres.is_converged != OPTS_UNKNOWN) // final step of one optimization
  {
    if(stepres.is_converged == OPTS_CONVERGED)
      rtraj.is_converged_eventually = true;
    else
      rtraj.is_converged_eventually = false;

    bool is_collision_free = CheckTraj(rtraj.traj, m_config);
    rtraj.is_collision_free_eventually = is_collision_free;
  }

  m_trajs.push_back(rtraj);

  if(stepres.is_converged != OPTS_UNKNOWN) // final step of one optimization, some postprocessing
  {
    double entire_opt_time = stepres.time_since_start;
    for(std::size_t i = 0; i < m_trajs.size(); ++i)
    {
      m_trajs[i].time_to_converge = entire_opt_time - m_trajs[i].time_to_converge;
      m_trajs[i].is_converged_eventually = rtraj.is_converged_eventually;
      m_trajs[i].is_collision_free_eventually = rtraj.is_collision_free_eventually;
    }
  }


  // if compute some default features during trajectory collection
  if(m_compute_default_features)
  {
    DirectionQuantizer quantizer(m_quantizer_direction_number);
    SphericalHarmonicsGrid shg(m_spherical_harmonics_grid_number);
    shg.setEffectiveBandWidth(m_spherical_harmonics_effective_bandwidth);

    for(std::size_t i = 0; i < m_trajs.size(); ++i)
    {
      std::vector<double> f_signed_distances = computeTrajectorySignedDistanceFeature(m_config, m_trajs[i].traj, quantizer, m_perturb_rotation_number);

      std::vector<double> f_signed_distances_dotproduct_links = computeTrajectorySignedDistanceDotProductBetweenLinksFeature(m_config, m_trajs[i].traj, quantizer);

      std::vector<double> f_signed_distances_dotproduct_adjacent_waypoints = computeTrajectorySignedDistanceDotProductBetweenAdjacentWaypointsFeature(m_config, m_trajs[i].traj, quantizer);


      std::vector<double> f_signed_distances_spherical_harmonics_short = computeTrajectorySphericalHarmonicsShortFeature(m_config, m_trajs[i].traj, shg);

      m_trajs[i].features.clear();

      m_trajs[i].features.push_back(f_signed_distances);
      m_trajs[i].features.push_back(f_signed_distances_dotproduct_links);
      m_trajs[i].features.push_back(f_signed_distances_dotproduct_adjacent_waypoints);
      m_trajs[i].features.push_back(f_signed_distances_spherical_harmonics_short);
    }
  }
}

void TrajCollecter::printTraj(std::ostream& os, const std::string& scene_filename) const
{
  for(std::size_t i = 0; i < m_trajs.size(); ++i)
  {
    os << scene_filename << " ";
    os << m_trajs[i].task_id << " " << m_trajs[i].perturb_id << " " << m_trajs[i].iter_id << " ";
    for(std::size_t j = 0; j < m_trajs[i].costs.size(); ++j)
      os << m_trajs[i].costs[j] << " ";
    for(std::size_t j = 0; j < m_trajs[i].constraints.size(); ++j)
      os << m_trajs[i].constraints[j] << " ";
    os << m_trajs[i].time_to_converge << " ";
    if(m_trajs[i].is_converged_eventually)
      os << "T ";
    else
      os << "F ";

    if(m_trajs[i].is_collision_free_eventually)
      os << "T ";
    else
      os << "F ";

    for(std::size_t j = 0; j < m_trajs[i].traj.rows(); ++j)
    {
      for(std::size_t k = 0; k < m_trajs[i].traj.cols(); ++k)
        os << m_trajs[i].traj(j, k) << " ";
    } 
    os << std::endl;
  }
}

void TrajCollecter::printFeature(std::ostream& os) const
{
  for(std::size_t i = 0; i < m_trajs.size(); ++i)
  {
    for(std::size_t j = 0; j < m_trajs[i].features.size(); ++j)
    {
      for(std::size_t k = 0; k < m_trajs[i].features[j].size(); ++k)
        os << m_trajs[i].features[j][k] << " ";
    }
    os << std::endl;
  }
}




}
