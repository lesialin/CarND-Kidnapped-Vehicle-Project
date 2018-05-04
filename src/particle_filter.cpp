/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cfloat>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 50;
  default_random_engine gen;
  double std_x, std_y, std_theta;
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  for (int i = 0; i < num_particles ; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    weights.push_back(1.0);
    particles.push_back(p);

  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  double inverse_yaw_rate = 1 / yaw_rate;
  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) < DBL_MIN) {
      particles[i].x += velocity * delta_t*cos(particles[i].theta);
      particles[i].y += velocity * delta_t*sin(particles[i].theta);

    } else {
      particles[i].x += velocity * inverse_yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity * inverse_yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));

    }
    particles[i].theta += yaw_rate * delta_t;
    //add noise
    normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  //  Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  int min_id;
  double min_dist;
  for (unsigned int i = 0; i < observations.size(); i++) {
    min_dist = DBL_MAX;
    min_id = -1;
    for (unsigned int j = 0; j < predicted.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < min_dist) {
        min_id = predicted[j].id;
        min_dist = distance;
      }
    }
    observations[i].id =  min_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double inverse_double_std_x_square  = 1 / (2 * std_x * std_x);
  double inverse_double_std_y_square  = 1 / (2 * std_y * std_y);
  double gauss_norm = 1 / (2 * M_PI * std_x * std_y);

  for (int i = 0; i < num_particles; i++ ) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    
    //get the observation
    std::vector<LandmarkObs> map_observations;
    for (unsigned  int j = 0; j < observations.size(); j++) {
      int obs_id = observations[j].id;
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;
      double m_x = p_x + cos(p_theta) *  obs_x - sin(p_theta) * obs_y;
      double m_y = p_y + sin(p_theta) * obs_x + cos(p_theta) * obs_y;
      map_observations.push_back(LandmarkObs{obs_id, m_x, m_y});

    }
    //get the prediction
    std::vector<LandmarkObs> predictions;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      int map_id = map_landmarks.landmark_list[j].id_i;
      double map_x = map_landmarks.landmark_list[j].x_f;
      double map_y = map_landmarks.landmark_list[j].y_f;
      double distance = dist(map_x, map_y, p_x, p_y);
      if (distance <= sensor_range) {
        predictions.push_back(LandmarkObs{map_id, map_x, map_y});
      }
    }


    //assosication to landmark
    dataAssociation(predictions, map_observations);
    //Update weight
    double weight = 1.0;
    for (unsigned int j = 0; j < map_observations.size(); j++) {
      for (unsigned int k = 0; k < predictions.size(); k++) {
        if ( map_observations[j].id == predictions[k].id) {
          double dx = map_observations[j].x - predictions[k].x;
          double dy = map_observations[j].y - predictions[k].y;
          double exponent = (dx * dx) * inverse_double_std_x_square + (dy * dy) * inverse_double_std_y_square;
          weight *= gauss_norm * exp(-exponent);

        }
      }
    }

    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> resample_particles;
  discrete_distribution <int> idx(weights.begin(), weights.end());
  default_random_engine gen;
  for (int i = 0; i < num_particles; i++) {
    int index = idx(gen);
    Particle p;
    p.id = index;
    p.x = particles[index].x;
    p.y = particles[index].y;
    p.theta = particles[index].theta;
    p.weight = 1.0;
    resample_particles.push_back(p);

  }
  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
    const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
