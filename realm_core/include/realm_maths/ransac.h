#ifndef PROJECT_RANSAC_H
#define PROJECT_RANSAC_H

#include <iostream>
#include <vector>
#include <realm_types/map_point.h>
#include <realm_common/loguru.hpp>

#include <eigen3/Eigen/Eigen>
#include <opencv2/core.hpp>

#include <map>
#include <math.h>
#include <list>
#include <set>
#include <algorithm>
#include <stdexcept> // std::out_of_range
#include <vector>
#include <cstdlib>
#include <iostream>

namespace realm
{

class RANSAC
{
public:
  RANSAC();
  void setData(std::vector<MapPoint> &dataset_1, std::vector<MapPoint> &dataset_2);
  void optimal_ransac(double general_tolerance, double final_tolerance);
  void better_ransac(int n, int k, double threshold1, double threshold2, int d);
  void normal_ransac(int n, int k, double threshold, int d);
  void getResult(std::vector<int> &out_inliers, int &out_num_of_inliers, cv::Mat &out_R, cv::Mat &out_t, double &out_error);

private:
  std::vector<realm::MapPoint> _in_dataset_1;
  std::vector<realm::MapPoint> _in_dataset_2;
  std::vector<int> _index_all;
  std::vector<int> _out_inliers;
  int _out_num_of_inliers;
  cv::Mat _out_R;
  cv::Mat _out_t;
  double _out_error;

  void rand_sample(int n, int x, std::vector<int> &samples);
  void model(std::vector<int> &index, cv::Mat &R, cv::Mat &t);
  void score(std::vector<int> &index, cv::Mat &R, cv::Mat &t, double tolerance, std::vector<int> &inliers ,std::vector<double> &dist);

  void resample(cv::Mat &R, cv::Mat &t, double tolerance, std::vector<int> &inliers);
  void rescore(double tolerance, std::vector<int> inliers, cv::Mat &R, cv::Mat &t, std::vector<int> &new_inliers);
  void pruneset(std::vector<int> &inliers, cv::Mat &R, cv::Mat &t, double tolerance);

  void create_Point3f_from_MapPoint(const std::vector<MapPoint> &mps, std::vector<cv::Point3f> &pts);
  void create_Point3f_from_MapPoint_visual(const std::vector<MapPoint> &mps, std::vector<cv::Point3f> &pts);
  void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R, cv::Mat &t);
};

} // namespace realm

#endif //PROJECT_RANSAC_H
