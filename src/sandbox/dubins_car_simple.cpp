#include "osgviewer/osgviewer.hpp"
#include "sco/expr_ops.hpp"
#include "sco/modeling_utils.hpp"
#include "trajopt/collision_checker.hpp"
#include "trajopt/collision_terms.hpp"
#include "trajopt/common.hpp"
#include "trajopt/plot_callback.hpp"
#include "trajopt/problem_description.hpp"
#include "trajopt/rave_utils.hpp"
#include "trajopt/trajectory_costs.hpp"
#include "utils/clock.hpp"
#include "utils/config.hpp"
#include "utils/eigen_conversions.hpp"
#include "utils/stl_to_string.hpp"
#include "utils/random_utils.hpp"
#include "sco/expr_op_overloads.hpp"
#include <boost/assign.hpp>
#include <boost/foreach.hpp>
#include <ctime>
#include <openrave-core.h>
#include <openrave/openrave.h>
#include <boost/timer.hpp>

using namespace trajopt;
using namespace std;
using namespace OpenRAVE;
using namespace util;
using namespace boost::assign;
using namespace Eigen;


// this is slightly different from openrave's collision checking
// discrete checking the collision status of a trajectory, with n discrete steps between adjacent waypoints.
bool CheckTraj0(const MatrixXd& traj, ConfigurationPtr rad, int n = 100)
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



int main(int argc, char* argv[])
{
  bool plotting=true, verbose=false;
  string envfile="jagged_narrow.xml";
  int decimation=1;
  
  int trajectory_num = 1;
  int task_num = 1;
  
  double random_range = 0.5;
  
  std::string feature_filename = "feature";
  std::string traj_filename = "traj";

  {
    Config config;
    config.add(new Parameter<bool>("plotting", &plotting, "plotting"));
    config.add(new Parameter<bool>("verbose", &verbose, "verbose"));
    config.add(new Parameter<string>("envfile", &envfile, "jagged_narrow.xml"));
    config.add(new Parameter<int>("decimation", &decimation, "plot every n"));
    config.add(new Parameter<int>("tasknum", &task_num, "1"));
    config.add(new Parameter<int>("trajectorynum", &trajectory_num, "1"));
    config.add(new Parameter<double>("randomrange", &random_range, "0.5"));
    config.add(new Parameter<std::string>("featurefilename", &feature_filename, "feature"));
    config.add(new Parameter<std::string>("trajfilename", &traj_filename, "traj"));
    CommandParser parser(config);
    parser.read(argc, argv);
  }

  std::ofstream feature_os(feature_filename.c_str());
  std::ofstream feature_config_os((feature_filename + std::string("_config.yaml")).c_str());
  std::ofstream traj_os(traj_filename.c_str());
  std::ofstream traj_config_os((traj_filename + std::string("_config.yaml")).c_str());

  RaveInitialize(false, verbose ? Level_Debug : Level_Info);
  EnvironmentBasePtr env = RaveCreateEnvironment();
  env->StopSimulation();
  OSGViewerPtr viewer = OSGViewer::GetOrCreate(env);
  assert(viewer);

  env->Load(string(DATA_DIR) + "/" + envfile);
  RobotBasePtr robot = GetRobot(*env);
  RobotAndDOFPtr rad(new RobotAndDOF(robot, vector<int>(), OR::DOF_X | OR::DOF_Y | OR::DOF_RotationAxis, OR::Vector(0, 0, 1)));

  int n_dof = 3;
  int n_steps;


  RNG rng;

  for(int task_id = 0; task_id < task_num; ) // different init/goal
  {
    Vector3d start, goal;
    if(envfile == "jagged_narrow.xml")
    {
      Vector3d delta1(0, 0, 0); if(task_id > 0) delta1 = Vector3d(rng.uniformReal(-1, 1) * 6, rng.uniformReal(-1, 1) * 4, rng.uniformReal(-1, 1) * M_PI);
      Vector3d delta2(0, 0, 0); if(task_id > 0) delta2 = Vector3d(rng.uniformReal(-1, 1) * 6, rng.uniformReal(-1, 1) * 4, rng.uniformReal(-1, 1) * M_PI);
      start = Vector3d(-2,-2,0) + delta1;
      goal = Vector3d(4,2,0) + delta2;
      n_steps = 19;
    }
    else if(envfile == "cargapprob.env.xml")
    {
      Vector3d delta1(0, 0, 0); if(task_id > 0) delta1 = Vector3d(rng.uniformReal(-1, 1) * 9, rng.uniformReal(-1, 1) * 4, rng.uniformReal(-1, 1) * M_PI);
      Vector3d delta2(0, 0, 0); if(task_id > 0) delta2 = Vector3d(rng.uniformReal(-1, 1) * 9, rng.uniformReal(-1, 1) * 4, rng.uniformReal(-1, 1) * M_PI);
      start = Vector3d(-2.5,0,0) + delta1;
      goal = Vector3d(6.5,0,-M_PI) + delta2;
      n_steps = 19;
    }
    else if(envfile == "squared_narrow_9.xml")
    {
      Vector3d delta1(0, 0, 0); if(task_id > 0) delta1 = Vector3d(0, rng.uniformReal(-1, 1) * 2, rng.uniformReal(-1, 1) * M_PI);
      Vector3d delta2(0, 0, 0); if(task_id > 0) delta2 = Vector3d(0, rng.uniformReal(-1, 1) * 2, rng.uniformReal(-1, 1) * M_PI);
      start = Vector3d(-3.5, 2 ,0) + delta1;
      goal = Vector3d(3.5, 2, 0) + delta2;
      n_steps = 19;      
    }
    else if(envfile == "squared_easier_9.xml")
    {
      Vector3d delta1(0, 0, 0); if(task_id > 0) delta1 = Vector3d(0, rng.uniformReal(-1, 1) * 2.5, rng.uniformReal(-1, 1) * M_PI);
      Vector3d delta2(0, 0, 0); if(task_id > 0) delta2 = Vector3d(0, rng.uniformReal(-1, 1) * 2.5, rng.uniformReal(-1, 1) * M_PI);
      start = Vector3d(-4, 2.5 ,0) + delta1;
      goal = Vector3d(4, 2.5, 0) + delta2;
      n_steps = 19;      
    }
    else if(envfile == "squared_easy_9.xml")
    {
      Vector3d delta1(0, 0, 0); if(task_id > 0) delta1 = Vector3d(0, rng.uniformReal(-1, 1) * 2.25, rng.uniformReal(-1, 1) * M_PI);
      Vector3d delta2(0, 0, 0); if(task_id > 0) delta2 = Vector3d(0, rng.uniformReal(-1, 1) * 2.25, rng.uniformReal(-1, 1) * M_PI);
      start = Vector3d(-3.75, 2.25 ,0) + delta1;
      goal = Vector3d(3.75, 2.25, 0) + delta2;
      n_steps = 19;      
    }
    else
    {
      throw runtime_error("no envfile");
    }


    // set linear trajectory
    MatrixXd linearTraj(n_steps, n_dof);
    for(int i = 0; i < n_dof; ++i)
      linearTraj.col(i) = VectorXd::LinSpaced(n_steps, start[i], goal[i]);
    
    VectorXd range(3); range = (start - goal) * 0.1; range[0] = abs(range[0]); range[1] = abs(range[1]); range[2] = 2 * M_PI * 0.1;

    // set collision threshold
    CollisionCheckerPtr cc = CollisionChecker::GetOrCreate(*env);
    cc->SetContactDistance(.15);
    std::cout << "task " << task_id << std::endl;
    
    {
      // the initial and goal should be collision-free, otherwise retry.
      vector<Collision> collisions_init, collisions_goal;
      rad->SetDOFValues(trajToDblVec(linearTraj.row(0)));
      cc->LinksVsAll(rad->GetAffectedLinks(), collisions_init, -1);
      if(collisions_init.size() > 0) continue;
      rad->SetDOFValues(trajToDblVec(linearTraj.row(linearTraj.rows() - 1)));
      cc->LinksVsAll(rad->GetAffectedLinks(), collisions_goal, -1);
      if(collisions_goal.size() > 0) continue;

      //check whether the linear's guess is in-collision, otherwise too trivial
      bool is_collision = !CheckTraj0(linearTraj, rad);
      if(!is_collision) continue;
    }

    task_id++;

    cc->SetContactDistance(.15);
    for(int trajectory_id = 0; trajectory_id < trajectory_num; ) // different random init trajectories with different perturbations
    {
      MatrixXd initTraj(n_steps, n_dof);
      if(trajectory_id == 0) initTraj = linearTraj; // use linear initialization
      else // use random initialization near the linear
      {
        initTraj = randomTrajectory(linearTraj, range, rng);
      }

      OptProbPtr prob(new OptProb());
      VarArray trajvars;
      AddVarArray(*prob, n_steps, n_dof, "j", trajvars);

      double maxdtheta = .2;
      for(int i = 0; i < n_steps - 1; ++i)
      {
        AffExpr vel = trajvars(i+1,2) - trajvars(i,2);
        prob->addLinearConstraint(vel - maxdtheta, INEQ);
        prob->addLinearConstraint(-vel - maxdtheta, INEQ);
      }

      // lower bound on length per step
      double length_lb = (goal - start).norm() / (n_steps - 1);
      // length per step variable
      Var lengthvar = prob->createVariables(singleton<string>("speed"), singleton<double>(length_lb), singleton<double>(INFINITY))[0];

      // car dynamics constraints
      for(int i = 0; i < n_steps-1; ++i)
      {
        VarVector vars0 = trajvars.row(i), vars1 = trajvars.row(i+1);
        VectorXd coeffs = VectorXd::Ones(2);
        VarVector vars = concat(vars0, vars1); vars.push_back(lengthvar);

        // collision cost
        if(i > 0)
        {
          prob->addCost(CostPtr(new CollisionCost(.05, 10, rad, vars0, vars1))); //continuous one
        }

        
        //prob->addCost(costPtr(new CollisionCost(.1, 10, rad, vars0);
      }

      // start and goal constraints
      for(int i = 0; i < n_dof; ++i)
      {
        prob->addLinearConstraint(exprSub(AffExpr(trajvars(0, i)), start[i]), EQ);
        prob->addLinearConstraint(exprSub(AffExpr(trajvars(n_steps-1, i)), goal[i]), EQ);
      }

      std::cout << "trajectory " << trajectory_id << std::endl;
      trajectory_id++;

      
      // optimization
      BasicTrustRegionSQP opt(prob);  
      DblVec initVec = trajToDblVec(initTraj);
      
      // speed variable
      initVec.push_back(length_lb);
      opt.initialize(initVec);

      TrajPlotter plotter(env, rad, trajvars);
      plotter.Add(prob->getCosts());
      plotter.Add(prob->getConstraints());
      if(plotting) opt.addCallback(boost::bind(&TrajPlotter::OptimizerCallback, boost::ref(plotter), _1, _2, _3));
      plotter.SetDecimation(decimation);
      plotter.AddLink(rad->GetAffectedLinks()[0]);

      TrajCollecter collecter(env, rad, trajvars);
      collecter.Add(prob->getCosts());
      collecter.Add(prob->getConstraints());
      opt.addCallback(boost::bind(&TrajCollecter::OptimizerCallback, boost::ref(collecter), _1, _2, _3));

      collecter.m_task_id = task_id - 1;
      collecter.m_perturb_id = trajectory_id - 1;
      collecter.m_compute_default_features = true;
      
      if(task_id == 1 && trajectory_id == 1)
      {
        collecter.printTrajConfig(traj_config_os);
        collecter.printFeatureConfig(feature_config_os);
      }
      

      Timer timer;
      timer.start();
      try
      {
        opt.optimize();
      }
      catch(...)
      {
        std::cout << "exception happend" << std::endl;
      }

      collecter.printTraj(traj_os, envfile);
      collecter.printFeature(feature_os);

      std::cout << timer.getElapsedTime() << " elapsed" << std::endl;
      std::cout << "number of intermediate traj: " << collecter.m_trajs.size() << std::endl;
      for(int i = 0; i < collecter.m_trajs.size(); ++i)
      {
        std::cout << "iter:" << collecter.m_trajs[i].iter_id << std::endl;
        std::cout << "time to converge:" << collecter.m_trajs[i].time_to_converge << std::endl;
        if(collecter.m_trajs[i].is_converged_eventually)
          std::cout << "is_converged: true" << std::endl;
        else
          std::cout << "is_converged: false" << std::endl;
        std::cout << "costs:" << std::endl;
        for(int j = 0; j < collecter.m_trajs[i].costs.size(); ++j)
          std::cout << collecter.m_trajs[i].costs[j] << " ";
        std::cout << std::endl;

        for(int j = 0; j < collecter.m_trajs[i].traj.rows(); ++j)
        {
          for(int k = 0; k < collecter.m_trajs[i].traj.cols(); ++k)
          {
            std::cout << collecter.m_trajs[i].traj(j, k) << " ";
          }
          std::cout << std::endl;
        }
        
      }
    }      
  }

      

  feature_os.close();
  traj_os.close();

  RaveDestroy();
}
