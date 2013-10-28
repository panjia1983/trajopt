#pragma once
#include "trajopt/common.hpp"
#include "utils/eigen_conversions.hpp"
#include "traj_feature_utils.hpp"

#include <iostream>
#include <limits>
#include <boost/math/constants/constants.hpp>

namespace trajopt
{

struct DirectionQuantizer
{
private:
  std::vector<OR::Vector> directions;

public:
  DirectionQuantizer(std::size_t n)
  {
    std::vector<double> dirs = S2_sequence(n);

    for(std::size_t i = 0; i < n; ++i)
    {
      directions.push_back(OR::Vector(dirs[3*i], dirs[3*i+1], dirs[3*i+2]));
    }
  }

  const std::vector<OR::Vector>& GetDirections() const
  {
    return directions;
  }

  std::size_t Dim() const
  {
    return directions.size(); // quatization vector dimension
  }

  std::size_t QuantizationId(const OR::Vector& v) const
  {
    double max_v = -std::numeric_limits<double>::max();
    std::size_t max_id = 0;
    for(std::size_t i = 0; i < directions.size(); ++i)
    {
      double a = abs(directions[i].dot3(v));
      if(a > max_v)
      {
        max_v = a;
        max_id = i;
      }
    }

    return max_id;
  }

  std::size_t QuantizationId(const OR::Vector& v, double angle) const
  {
    OR::Vector v_rotated = OR::geometry::quatRotate(OR::geometry::quatFromAxisAngle(OR::Vector(0, 0, 1), angle), v);
    double max_v = -std::numeric_limits<double>::max();
    std::size_t max_id = 0;
    for(std::size_t i = 0; i < directions.size(); ++i)
    {
      double a = abs(directions[i].dot3(v_rotated));
      if(a > max_v)
      {
        max_v = a;
        max_id = i;
      }
    }

    return max_id;    
  }
};

inline std::ostream& operator << (std::ostream& os, const DirectionQuantizer& quantizer)
{
  const std::vector<OR::Vector>& directions = quantizer.GetDirections();
  for(std::size_t i = 0; i < directions.size(); ++i)
  {
    os << directions[i].x << " " << directions[i].y << " " << directions[i].z << std::endl;
  }
  return os;
}


struct SphericalHarmonicsGrid
{
private:
  std::vector<std::vector<OR::Vector> > grid_directions;
  
public:
  std::size_t getBandWidth() const { return grid_directions.size() / 2; }
  std::size_t getGridSize() const { return grid_directions.size(); }
  std::size_t getDim() const { return grid_directions.size(); }

  SphericalHarmonicsGrid(std::size_t n)
  {
    grid_directions.resize(2 * n);
    for(std::size_t i = 0; i < grid_directions.size(); ++i)
    {
      grid_directions[i].resize(2 * n);
    }
    
    for(std::size_t i = 0; i < 2 * n; ++i)
    {
      for(std::size_t j = 0; j < 2 * n; ++j)
      {
        double theta = boost::math::constants::pi<double>() * (2 * i + 1) / (4 * n);
        double phi = boost::math::constants::pi<double>() * j / n;
        grid_directions[i][j] = OR::Vector(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
      }
    }
  }

  const std::vector<std::vector<OR::Vector> >& getDirections() const
  {
    return grid_directions;
  }

  std::pair<std::size_t, std::size_t> QuantizationId(const OR::Vector& v) const
  {
    double max_v = -std::numeric_limits<double>::max();
    std::size_t max_id_x = 0, max_id_y = 0;
    for(std::size_t i = 0; i < grid_directions.size(); ++i)
    {
      for(std::size_t j = 0; j < grid_directions[i].size(); ++j)
      {
        double a = abs(grid_directions[i][j].dot3(v));
        if(a > max_v)
        {
          max_v = a;
          max_id_x = i;
          max_id_y = j;
        }
      }
    }

    return std::make_pair(max_id_x, max_id_y);
  }
};


std::map<const KinBody::Link*, int> mapLinkToId(const std::vector<OR::KinBody::LinkPtr>& links);


struct LinkSignedDistance
{
  struct SignedDistanceElement
  {
    OR::Vector normal; 
    double distance;
    const OR::KinBody::Link* link;
  };

  const OR::KinBody::Link* self_link;
  std::vector<SignedDistanceElement> signed_distances;
};

typedef std::vector<LinkSignedDistance> RobotSignedDistance;

RobotSignedDistance computeRobotSignedDistance(ConfigurationPtr config, const DblVec& x);

// signed distance features for the entire robot, for one waypoint x
// [features for link 1][features for link 2] ...
std::vector<double> computeRobotSignedDistanceFeature(ConfigurationPtr config, const DblVec& x, const DirectionQuantizer& quantizer, double perturb_rotation_angle);

// dot product between signed distance features for adjacent links for the entire robot, for one waypoint x
std::vector<double> computeRobotSignedDistanceDotProductBetweenLinksFeature(ConfigurationPtr config, const DblVec& x, const DirectionQuantizer& quantizer);

// dot product between signed distance features for the entire robot, for two waypoints x and y
std::vector<double> computeRobotSignedDistanceDotProductBetweenTwoWaypointsFeature(ConfigurationPtr config, const DblVec& x, const DblVec& y, const DirectionQuantizer& quantizer);

// spherical harmonics features (short version) for the entire robot, for one waypoint x
// [features for link 1][features for link 2] ...
std::vector<double> computeRobotSphericalHarmonicsShortFeature(ConfigurationPtr config, const DblVec& x, const SphericalHarmonicsGrid& shg);

// spherical harmonics features (long version) for the entire robot, for one waypoint x
// [features for link 1][features for link 2] ...
std::vector<double> computeRobotSphericalHarmonicsLongFeature(ConfigurationPtr config, const DblVec& x, const SphericalHarmonicsGrid& shg);





// signed distance features for all waypoints in a trajectory
// [features for waypoint 1][features for waypoint 2]...
std::vector<double> computeTrajectorySignedDistanceFeature(ConfigurationPtr config, const MatrixXd& traj, const DirectionQuantizer& quantizer, double perturb_rotation_angle);

/// signed distance features for all waypoints in a trajectory, with N perturbations
std::vector<double> computeTrajectorySignedDistanceFeature(ConfigurationPtr config, const MatrixXd& traj, const DirectionQuantizer& quantizer, std::size_t N);

// dot product between signed distance features for adjacent links, for the entire trajectory
std::vector<double> computeTrajectorySignedDistanceDotProductBetweenLinksFeature(ConfigurationPtr config, const MatrixXd& traj, const DirectionQuantizer& quantizer);

// dot product between signed distance features for the entire robot, for all adjacent waypoints in a trajectory
std::vector<double> computeTrajectorySignedDistanceDotProductBetweenAdjacentWaypointsFeature(ConfigurationPtr config, const MatrixXd& traj, const DirectionQuantizer& quantizer);

// spherical harmonics features (short version) for the entire trajectory
std::vector<double> computeTrajectorySphericalHarmonicsShortFeature(ConfigurationPtr config, const MatrixXd& traj, const SphericalHarmonicsGrid& shg);

// spherical harmonics features (long version) for the entire trajectory
std::vector<double> computeTrajectorySphericalHarmonicsLongFeature(ConfigurationPtr config, const MatrixXd& traj, const SphericalHarmonicsGrid& shg);

















}
