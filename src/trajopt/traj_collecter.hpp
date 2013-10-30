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
  bool is_collision_free_eventually;

  std::vector<double> costs;
  std::vector<double> constraints;
  std::size_t iter_id;
  std::size_t task_id;
  std::size_t perturb_id;

  std::vector<std::vector<double> > features; // maybe empty, according to setting
};

struct TRAJOPT_API TrajCollecter
{

  void OptimizerCallback(OptProb* prob, DblVec& x, OptStepResults& stepres);

  
  TrajCollecter(OR::EnvironmentBasePtr env, ConfigurationPtr config, const VarArray& trajvars, bool compute_default_features = false)
    : m_env(env), m_config(config), m_trajvars(trajvars), m_compute_default_features(compute_default_features)
  {
    m_quantizer_direction_number = 10;
    m_spherical_harmonics_grid_number = 10;
    m_spherical_harmonics_effective_bandwidth = m_spherical_harmonics_grid_number / 2;
    m_perturb_rotation_number = 1;
    m_task_id = 0;
    m_perturb_id = 0;
  }


  
  void Add(const std::vector<CostPtr>& costs);
  void Add(const std::vector<ConstraintPtr>& constraints);
  void Add(const std::vector<PlotterPtr>& plotters);
  void Add(PlotterPtr plotter);

  

  std::size_t GetNumCosts() const;
  std::size_t GetNumConstraints() const;
  std::vector<std::string> GetCostNames() const;
  std::vector<std::string> GetConstraintNames() const;

  const std::vector<RichTrajectory>& GetCollectedTrajs() const
  {
    return m_trajs;
  }
  
  std::vector<RichTrajectory> m_trajs;

  // benchmark environment
  OpenRAVE::EnvironmentBasePtr m_env;

  // configuration space setting
  ConfigurationPtr m_config;

  // optimization variable for trajectory
  VarArray m_trajvars;

  // plotters obtain internal information from optimization solver
  std::vector<PlotterPtr> m_plotters;

  // whether compute some default features during trajectory collection
  bool m_compute_default_features;

  // default selection for quantizer directions
  std::size_t m_quantizer_direction_number;
  // default spherical harmonics bandwidth
  std::size_t m_spherical_harmonics_grid_number;
  // default effective bandwidth for spherical harmonics
  std::size_t m_spherical_harmonics_effective_bandwidth;
  // default perturb_rotation_number for signed distance
  std::size_t m_perturb_rotation_number;


  // task_id and perturb_id
  std::size_t m_task_id;
  std::size_t m_perturb_id;

  void printTraj(std::ostream& os, const std::string& scene_filename) const;

  void printFeature(std::ostream& os) const;


  void printTrajConfig(std::ostream& os) const
  {
    os << "items: [scene_name, task_id, perturb_id, iter_id, costs, constraints, converge_time, is_converged_eventually, is_collision_free_eventually, trajectory]" << std::endl;
    os << "costs: [";
    std::vector<std::string> cost_names = GetCostNames();
    for(std::size_t i = 0; i < cost_names.size(); ++i)
    {
      os << cost_names[i];
      if (i != cost_names.size() - 1)
        os << ",";
    }
    os << "]" << std::endl;

    os << "constraints: [";
    std::vector<std::string> constraint_names = GetConstraintNames();
    for(std::size_t i = 0; i < constraint_names.size(); ++i)
    {
      os << constraint_names[i];
      if (i != constraint_names.size() - 1)
        os << ",";
    }
    os << "]" << std::endl;

    os << "trajectory: " << std::endl;
    os << "  num_waypoints: " << m_trajvars.rows() << std::endl;
    os << "  num_dof: " << m_trajvars.cols() << std::endl;
  }

  void printFeatureConfig(std::ostream& os) const
  {
    std::vector<OR::KinBody::LinkPtr> links;
    std::vector<int> inds;
    m_config->GetAffectedLinks(links, true, inds);
    int n_links = links.size();
    int n_waypoints = m_trajvars.rows();
    int n_linklinks = 0;
    for(std::vector<OR::KinBody::LinkPtr>::const_iterator it = links.begin();
        it != links.end();
        ++it)
    {
      const KinBody::Link* link = it->get();
      std::vector<KinBody::LinkPtr> parent_links;
      link->GetParentLinks(parent_links);
      for(std::size_t i = 0; i < parent_links.size(); ++i)
      {
        const KinBody::Link* p_link = parent_links[i].get();
        if(!p_link) continue;

        n_linklinks += 1;
      }
    }

    int id = -1;
    int f_dim = 0;
    f_dim = n_links * m_quantizer_direction_number * n_waypoints;
    if(f_dim > 0)
    {
      os << "f_signed_distances: " << std::endl;
      os << "  start: " << id + 1 << std::endl;
      id += f_dim;
      os << "  goal: " <<  id << std::endl;
    }

    f_dim = n_linklinks * 1 * n_waypoints;
    if(f_dim > 0)
    {
      os << "f_signed_distances_dotproduct_links: " << std::endl;
      os << "  start: " << id + 1 << std::endl;
      id += f_dim;
      os << "  goal: " << id << std::endl;
    }

    f_dim = n_links * (n_waypoints - 1);
    if(f_dim > 0)
    {
      os << "f_signed_distances_dotproduct_adjacent_waypoints: " << std::endl;
      os << "  start: " << id + 1 << std::endl;
      id += f_dim;
      os << "  goal: " << id << std::endl;
    }

    f_dim = m_spherical_harmonics_effective_bandwidth * n_links * n_waypoints;
    if(f_dim > 0)
    {
      os << "f_signed_distances_spherical_harmonics_short: " << std::endl;
      os << "  start: " << id + 1 << std::endl;
      id += f_dim;
      os << "  goal: " << id << std::endl;
    }
  }
};

typedef boost::shared_ptr<TrajCollecter> TrajCollecterPtr;

}
