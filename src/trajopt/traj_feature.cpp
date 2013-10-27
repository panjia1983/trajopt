#include "traj_feature.hpp"
#include "trajopt/collision_checker.hpp"
#include "trajopt/collision_terms.hpp"
#include "trajopt/traj_feature_utils.hpp"

namespace trajopt
{

std::map<const KinBody::Link*, int> mapLinkToId(const std::vector<OR::KinBody::LinkPtr>& links)
{
  std::map<const KinBody::Link*, int> link_to_id_map;
  int id = 0;
  for(std::size_t i = 0; i < links.size(); ++i)
  {
    link_to_id_map.insert(std::make_pair(links[i].get(), id));
    id++;
  }

  return link_to_id_map;
}


RobotSignedDistance computeRobotSignedDistance(ConfigurationPtr config, DblVec& x)
{
  CollisionCheckerPtr cc = CollisionChecker::GetOrCreate(*(config->GetEnv()));
  std::vector<OR::KinBody::LinkPtr> links;
  std::vector<int> inds;
  config->GetAffectedLinks(links, true, inds);

  // perform collision checking for links
  double old_contactDistance = cc->GetContactDistance();
  cc->SetContactDistance(100);

  std::vector<Collision> collisions;
  config->SetDOFValues(x);
  cc->LinksVsAll(links, collisions, -1);
  
  cc->SetContactDistance(old_contactDistance);

  // reconstruct link to id map
  std::map<const KinBody::Link*, int> link_to_id_map = mapLinkToId(links);

  RobotSignedDistance rsd(links.size());

  for(std::size_t i = 0; i < links.size(); ++i)
  {
    std::map<const KinBody::Link*, int>::const_iterator search_it = link_to_id_map.find(links[i].get());
    int id = search_it->second;
    rsd[id].self_link = links[i].get();
  }

  for(std::size_t i = 0; i < collisions.size(); ++i)
  {
    const KinBody::Link* linkA = collisions[i].linkA;
    const KinBody::Link* linkB = collisions[i].linkB;

    int find_item = 0;

    std::map<const KinBody::Link*, int>::const_iterator it = link_to_id_map.find(linkA);
    if(it != link_to_id_map.end())
    {
      find_item++;
      LinkSignedDistance::SignedDistanceElement elem;
      elem.normal = (collisions[i].distance > 0)? -collisions[i].normalB2A : collisions[i].normalB2A;
      elem.distance = collisions[i].distance;
      elem.link = collisions[i].linkB;
      rsd[it->second].signed_distances.push_back(elem);
    }

    it = link_to_id_map.find(linkB);
    if(it != link_to_id_map.end())
    {
      find_item++;
      LinkSignedDistance::SignedDistanceElement elem;
      elem.normal = (collisions[i].distance > 0) ? collisions[i].normalB2A : -collisions[i].normalB2A;
      elem.distance = collisions[i].distance;
      elem.link = collisions[i].linkA;
      rsd[it->second].signed_distances.push_back(elem);
    }
  }

  // if(find_item == 0) std::cout << "Warning! Should never happen!" << std::endl;


  return rsd;
}


// [features for link 1][features for link 2] ...
std::vector<double> computeRobotSignedDistanceFeature(ConfigurationPtr config, DblVec& x, const DirectionQuantizer& quantizer, double perturb_rotation_angle)
{
  RobotSignedDistance rsd = computeRobotSignedDistance(config, x);
  int n_links = rsd.size();
  int n_quantize_step = quantizer.Dim();
  std::vector<double> features(n_links * n_quantize_step, 0);
  for(std::size_t i = 0; i < n_links; ++i)
  {
    const LinkSignedDistance& lsd = rsd[i];
    double R_max = sqrt(lsd.self_link->ComputeAABB().extents.lengthsqr2());
    for(std::size_t j = 0; j < n_quantize_step; ++j)
      features[i * n_quantize_step + j] = R_max; // default value is Rmax

    for(std::size_t j = 0; j < lsd.signed_distances.size(); ++j)
    {
      int quantize_id = quantizer.QuantizationId(lsd.signed_distances[j].normal, perturb_rotation_angle);

      if(lsd.signed_distances[j].distance < 0) // in collision
      {
        if(features[i * n_quantize_step + quantize_id] >= 0) // if no collision was detected along this direction previously
        {
          features[i * n_quantize_step + quantize_id] = lsd.signed_distances[j].distance;
        }
        else // if collision has already been detected along this direction
        {
          features[i * n_quantize_step + quantize_id] += lsd.signed_distances[j].distance;
        }
      }
      else // not in collision
      {
        features[i * n_quantize_step + quantize_id] = std::min(lsd.signed_distances[j].distance, features[i * n_quantize_step + quantize_id]);
      }
    }
  }

  return features;
}

std::vector<double> computeRobotSphericalHarmonicShortFeature(ConfigurationPtr config, DblVec& x, const SphericalHarmonicsGrid& shg, double perturb_rotation_angle)
{
  RobotSignedDistance rsd = computeRobotSignedDistance(config, x);
  int n_links = rsd.size();
  std::vector<double> features;

  int grid_size = shg.getGridSize();

  for(std::size_t i = 0; i < n_links; ++i)
  {
    const LinkSignedDistance& lsd = rsd[i];
    double R_max = sqrt(lsd.self_link->ComputeAABB().extents.lengthsqr2());

    std::vector<std::vector<double> > samples(grid_size);
    for(std::size_t j = 0; j < grid_size; ++j)
      samples[j].resize(grid_size, R_max);

    for(std::size_t j = 0; j < lsd.signed_distances.size(); ++j)
    {
      std::pair<std::size_t, std::size_t> quantize_id = shg.QuantizationId(lsd.signed_distances[j].normal);

      if(lsd.signed_distances[j].distance < 0) // in collision
      {
        if(samples[quantize_id.first][quantize_id.second] >= 0) // if not collision was detected along this direction
        {
          samples[quantize_id.first][quantize_id.second] = lsd.signed_distances[j].distance;
        }
        else // if collision has already been detected along this direction
        {
          samples[quantize_id.first][quantize_id.second] += lsd.signed_distances[j].distance;
        }
      }
      else // not in collision
      {
        samples[quantize_id.first][quantize_id.second] = std::min(lsd.signed_distances[j].distance, samples[quantize_id.first][quantize_id.second]);
      }
    }

    // compute spherical harmonics coeffcients
    std::vector<std::vector<std::complex<double> > > sht_coeffs = SHT(samples);
    int sht_bandwidth = sht_coeffs.size(); // can change to smaller value
    std::vector<double> sht_feature = collectGlobalSHTFeature(sht_coeffs, sht_bandwidth);

    std::copy(sht_feature.begin(), sht_feature.end(), std::back_inserter(features));
  }

  return features;
}

std::vector<double> computeRobotSphericalHarmonicLongFeature(ConfigurationPtr config, DblVec& x, const SphericalHarmonicsGrid& shg, double perturb_rotation_angle)
{
  RobotSignedDistance rsd = computeRobotSignedDistance(config, x);
  int n_links = rsd.size();
  std::vector<double> features;

  int grid_size = shg.getGridSize();

  for(std::size_t i = 0; i < n_links; ++i)
  {
    const LinkSignedDistance& lsd = rsd[i];
    double R_max = sqrt(lsd.self_link->ComputeAABB().extents.lengthsqr2());

    std::vector<std::vector<double> > samples(grid_size);
    for(std::size_t j = 0; j < grid_size; ++j)
      samples[j].resize(grid_size, R_max);

    for(std::size_t j = 0; j < lsd.signed_distances.size(); ++j)
    {
      std::pair<std::size_t, std::size_t> quantize_id = shg.QuantizationId(lsd.signed_distances[j].normal);

      if(lsd.signed_distances[j].distance < 0) // in collision
      {
        if(samples[quantize_id.first][quantize_id.second] >= 0) // if not collision was detected along this direction
        {
          samples[quantize_id.first][quantize_id.second] = lsd.signed_distances[j].distance;
        }
        else // if collision has already been detected along this direction
        {
          samples[quantize_id.first][quantize_id.second] += lsd.signed_distances[j].distance;
        }
      }
      else // not in collision
      {
        samples[quantize_id.first][quantize_id.second] = std::min(lsd.signed_distances[j].distance, samples[quantize_id.first][quantize_id.second]);
      }
    }

    // compute spherical harmonics coeffcients
    std::vector<std::vector<std::complex<double> > > sht_coeffs = SHT(samples);
    int sht_bandwidth = sht_coeffs.size(); // can change to smaller value
    std::vector<double> sht_feature = collectShapeSHTFeature(sht_coeffs, sht_bandwidth);

    std::copy(sht_feature.begin(), sht_feature.end(), std::back_inserter(features));
  }

  return features;
}




}
