/**
* This file is part of OpenREALM.
*
* Copyright (C) 2018 Alexander Kern <laxnpander at gmail dot com> (Braunschweig University of Technology)
* For more information see <https://github.com/laxnpander/OpenREALM>
*
* OpenREALM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* OpenREALM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OpenREALM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PROJECT_COMAPPING_H
#define PROJECT_COMAPPING_H

#include <deque>
#include <chrono>

#include <realm_stages/stage_base.h>
#include "conversions.h"
#include <realm_stages/stage_settings.h>
#include <realm_types/frame.h>
#include <realm_types/cv_grid_map.h>
#include <realm_cv/analysis.h>
#include <realm_io/cv_export.h>
#include <realm_io/pcl_export.h>
#include "../../../realm_io/include/realm_io/gis_export.h"
#include <realm_io/utilities.h>
#include <realm_ortho/delaunay_2d.h>
#include <realm_maths/ransac.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <map>
#include <math.h>
#include <list>
#include <set>
#include <algorithm>
#include <stdexcept>      // std::out_of_range
#include <vector>
#include <cstdlib>
#include <iostream>
#include <iomanip> 
namespace realm
{
namespace stages
{

class Comapping : public StageBase
{
public:
  using Ptr = std::shared_ptr<Comapping>;
  using ConstPtr = std::shared_ptr<const Comapping>;

  struct Map
  {
  public:
    using Ptr = std::shared_ptr<Map>;
    using ConstPtr = std::shared_ptr<const Map>;

    CvGridMap::Ptr map;
    CvGridMap::Ptr map_update;
    UTMPose::Ptr utm_reference;
    std::string creater;
    std::string co_observer;
    //std::set<std::string> agent_names;
    std::list<Frame::Ptr> key_frame_list;
    std::map<std::string,cv::Mat> error_correction;
  };

  struct SaveSettings
  {
    bool save_valid;
    bool save_ortho_rgb_one;
    bool save_ortho_rgb_all;
    bool save_ortho_gtiff_one;
    bool save_ortho_gtiff_all;
    bool save_elevation_one;
    bool save_elevation_all;
    bool save_elevation_var_one;
    bool save_elevation_var_all;
    bool save_elevation_obs_angle_one;
    bool save_elevation_obs_angle_all;
    bool save_elevation_mesh_one;
    bool save_num_obs_one;
    bool save_num_obs_all;
    bool save_dense_ply;
  };

  struct GridQuickAccess
  {
  public:
    using Ptr = std::shared_ptr<GridQuickAccess>;
    using ConstPtr = std::shared_ptr<const GridQuickAccess>;

  public:
    GridQuickAccess(const std::vector<std::string> &layer_names, const CvGridMap &map);
    void move(int row, int col);

    float *ele;        // elevation at row, col
    float *var;        // elevation variance at row, col
    float *hyp;        // elevation hypothesis at row, col
    float *angle;      // elevation observation angle at row, col
    uint16_t *nobs;    // number of observations at row, col
    cv::Vec3f *normal; // surface normal
    cv::Vec4b *rgb;    // color at row, col
    uchar *elevated;   // elevation computed at row, col
    uchar *valid;      // valid at row, col
  private:
    cv::Mat _elevation;
    cv::Mat _elevation_normal;
    cv::Mat _elevation_var;
    cv::Mat _elevation_hyp;
    cv::Mat _elevation_angle;
    cv::Mat _elevated;
    cv::Mat _num_observations;
    cv::Mat _color_rgb;
    cv::Mat _valid;
  };

public:
  explicit Comapping(const StageSettings::Ptr &stage_set);
  void addFrame(const Frame::Ptr &frame) override;
  bool process() override;
  void runPostProcessing();
  void saveAll();

private:
  std::deque<Frame::Ptr> _buffer;
  std::mutex _mutex_buffer;

  //! Publish of mesh is optional. Set >0 if should be published. Additionally it can be downsampled.
  int _publish_mesh_nth_iter;
  int _publish_mesh_every_nth_kf;
  bool _do_publish_mesh_at_finish;
  double _downsample_publish_mesh; // [m/pix]

  bool _use_surface_normals;

  int _th_elevation_min_nobs;
  float _th_elevation_var;

  SaveSettings _settings_save;

  UTMPose::Ptr _utm_reference;
  Delaunay2D::Ptr _mesher;

  void finishCallback() override;
  void printSettingsToLog() override;

  CvGridMap blend(CvGridMap::Overlap *overlap);

  void setGridElement(const GridQuickAccess::Ptr &ref, const GridQuickAccess::Ptr &inp);
  void updateGridElement(const GridQuickAccess::Ptr &ref, const GridQuickAccess::Ptr &inp);

  void reset() override;
  void initStageCallback() override;
  std::vector<Face> createMeshFaces(const CvGridMap::Ptr &map);

  void publish(const Frame::Ptr &frame, const Map::Ptr &map, uint64_t timestamp);

  void saveIter(uint32_t id);
  Frame::Ptr getNewFrame();



  bool _global_map_initialized;
  Map _global_map;
  std::vector<Map::Ptr> _map_stack;


  bool merged_flag;
  void checkMapStack(const Frame::Ptr &frame, Map::Ptr &found_by_agentName, std::vector<Map::Ptr> &only_in_overlap);
  Comapping::Map::Ptr createNewMap(const Frame::Ptr &frame);
  void addFrametoMap(const Frame::Ptr &frame, Map::Ptr &map);
  void mergeMaps(const Frame::Ptr &frame,std::vector<Map::Ptr> &overlapd_maps, Map::Ptr &agent_map);

  void compute_error_correction(Frame::Ptr &frame, Map::Ptr agent_map, std::vector<Map::Ptr> &overlapd_maps);
  void searchNearestFrame(Frame::Ptr &frame, Map::Ptr map, Frame::Ptr &f2,double &min_UTM_dst);

  double distance_of_two_UTM(const UTMPose pos1,const UTMPose pos2);
  double distance_of_two_MPs(const MapPoint &p1, const MapPoint &p2);
  void create_Point3f_from_MapPoint(const std::vector<MapPoint>& mps, std::vector<cv::Point3f> &pts);
  void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1,const std::vector<cv::Point3f> &pts2,cv::Mat &R, cv::Mat &t);
  void changeROI(cv::Mat T, CvGridMap::Ptr &map);

  void match_two_frames_by_cv(Frame::Ptr &f1, Frame::Ptr &f2);
  void match_two_frames_by_ORB_SLAM_descriptor(Frame::Ptr &f1, Frame::Ptr &f2, std::vector<MapPoint> &mps1, std::vector<MapPoint> &mps2);
  void match_two_frames_by_ORB_SLAM_keypoint(Frame::Ptr &f1, Frame::Ptr &f2);
  void match_two_frames_by_ORB_SLAM_descriptor_without_goodmatch(Frame::Ptr &f1, Frame::Ptr &f2, std::vector<MapPoint> &mps1, std::vector<MapPoint> &mps2);
  void compute_error_correction_with_RANSAC(Frame::Ptr &frame, Map::Ptr found_by_agentName,std::vector<Map::Ptr> &only_by_overlap);
  void RANSAC_(std::vector<MapPoint> &mps1_before, std::vector<MapPoint> &mps2_before, std::vector<int> &choosed_index, cv::Mat &best_rot, cv::Mat &best_trans);
  void match_two_frames_by_ransac(Frame::Ptr &f1, Frame::Ptr &f2, int &num, std::vector<int> &best_inliers, cv::Mat &R, cv::Mat &t,double &error);
};

} // namespace stages
} // namespace realm

#endif //PROJECT_Comapping_H