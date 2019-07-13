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

#include <realm_stages/comapping.h>

using namespace realm;
using namespace stages;

Comapping::Comapping(const StageSettings::Ptr &stage_set)
    : StageBase("comapping", stage_set->get<std::string>("path_output"), stage_set->get<int>("queue_size")),
      _utm_reference(nullptr),
      _publish_mesh_nth_iter(0),
      _publish_mesh_every_nth_kf(stage_set->get<int>("publish_mesh_every_nth_kf")),
      _do_publish_mesh_at_finish(stage_set->get<int>("publish_mesh_at_finish") > 0),
      _downsample_publish_mesh(stage_set->get<double>("downsample_publish_mesh")),
      _use_surface_normals(true),
      _th_elevation_min_nobs(stage_set->get<int>("th_elevation_min_nobs")),
      _th_elevation_var((float)stage_set->get<double>("th_elevation_variance")),
      _settings_save({stage_set->get<int>("save_valid") > 0,
                      stage_set->get<int>("save_ortho_rgb_one") > 0,
                      stage_set->get<int>("save_ortho_rgb_all") > 0,
                      stage_set->get<int>("save_ortho_gtiff_one") > 0,
                      stage_set->get<int>("save_ortho_gtiff_all") > 0,
                      stage_set->get<int>("save_elevation_one") > 0,
                      stage_set->get<int>("save_elevation_all") > 0,
                      stage_set->get<int>("save_elevation_var_one") > 0,
                      stage_set->get<int>("save_elevation_var_all") > 0,
                      stage_set->get<int>("save_elevation_obs_angle_one") > 0,
                      stage_set->get<int>("save_elevation_obs_angle_all") > 0,
                      stage_set->get<int>("save_elevation_mesh_one") > 0,
                      stage_set->get<int>("save_num_obs_one") > 0,
                      stage_set->get<int>("save_num_obs_all") > 0,
                      stage_set->get<int>("save_dense_ply") > 0}),
      _global_map_initialized(false),
      merged_flag(false)
{
  std::cout << "Stage [" << _stage_name << "]: Created Stage with Settings: " << std::endl;
  stage_set->print();
}

void Comapping::addFrame(const Frame::Ptr &frame)
{
  if (frame->getObservedMap()->empty())
  {
    LOG_F(INFO, "Input frame missing observed map. Dropping!");
    return;
  }
  std::unique_lock<std::mutex> lock(_mutex_buffer);
  _buffer.push_back(frame);

  // Ringbuffer implementation for buffer with no pose
  if (_buffer.size() > _queue_size)
    _buffer.pop_front();
}

bool Comapping::process()
{
  bool has_processed = false;
  if (!_buffer.empty())
  {

    Frame::Ptr frame = getNewFrame();
    int id = frame->getFrameId();
    std::string agent = frame->getCameraId();
    if (merged_flag && (agent == "agent0"))
      return true;

    _use_surface_normals = (_use_surface_normals && frame->getObservedMap()->exists("elevation_normal"));

    Map::Ptr found_by_agentName;
    std::vector<Map::Ptr> only_by_overlap;

    checkMapStack(frame, found_by_agentName, only_by_overlap);

    if (found_by_agentName == nullptr)
    {
      found_by_agentName = createNewMap(frame);

      _map_stack.push_back(found_by_agentName);
    }
    else
    {
      addFrametoMap(frame, found_by_agentName);
    }

    if (only_by_overlap.size() != 0 && found_by_agentName != nullptr)
    {
      compute_error_correction(frame, found_by_agentName, only_by_overlap);
    }

    if (_map_stack.size() > 0)
    {
      if (_map_stack[0] != nullptr)
      {
        publish(frame, _map_stack[0], frame->getTimestamp());
      }
    }

    has_processed = true;
  }

  return has_processed;
}

CvGridMap Comapping::blend(CvGridMap::Overlap *overlap)
{
  // Overlap between global mosaic (ref) and new data (inp)
  CvGridMap ref = *overlap->first;
  CvGridMap inp = *overlap->second;

  // Data layers to grab from reference
  std::vector<std::string> ref_layers;
  // Data layers to grab from input map
  std::vector<std::string> inp_layers;

  // Surface normal computation is optional, therefore use only if set
  if (_use_surface_normals)
  {
    ref_layers = {"elevation", "elevation_normal", "elevation_var", "elevation_hyp", "elevation_angle", "elevated", "color_rgb", "num_observations", "valid"};
    inp_layers = {"elevation", "elevation_normal", "elevation_angle", "elevated", "color_rgb", "valid"};
  }
  else
  {
    ref_layers = {"elevation", "elevation_var", "elevation_hyp", "elevation_angle", "elevated", "color_rgb", "num_observations", "valid"};
    inp_layers = {"elevation", "elevation_angle", "elevated", "color_rgb", "valid"};
  }

  GridQuickAccess::Ptr ref_grid_element = std::make_shared<GridQuickAccess>(ref_layers, ref);
  GridQuickAccess::Ptr inp_grid_element = std::make_shared<GridQuickAccess>(inp_layers, inp);

  cv::Size size = ref.size();
  for (int r = 0; r < size.height; ++r)
    for (int c = 0; c < size.width; ++c)
    {
      // Move the quick access element to current position
      ref_grid_element->move(r, c);
      inp_grid_element->move(r, c);

      // Check cases for input
      if (*inp_grid_element->valid == 0)
        continue;
      if (*ref_grid_element->elevated && !*inp_grid_element->elevated)
        continue;

      if (*ref_grid_element->nobs == 0 || (*inp_grid_element->elevated && !*ref_grid_element->elevated))
        setGridElement(ref_grid_element, inp_grid_element);
      else
        updateGridElement(ref_grid_element, inp_grid_element);
    }

  return ref;
}

void Comapping::updateGridElement(const GridQuickAccess::Ptr &ref, const GridQuickAccess::Ptr &inp)
{
  // Formulas avr+std_dev: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
  assert(*inp->valid == 255);

  if (!*inp->elevated)
  {
    *ref->ele = *inp->ele;
    *ref->elevated = 0;
    *ref->valid = 255;
    *ref->nobs = (*ref->nobs) + (uint16_t)1;
    if (fabsf(*inp->angle - 90) < fabsf(*ref->angle - 90))
    {
      *ref->angle = *inp->angle;
      *ref->rgb = *inp->rgb;
    }
    return;
  }

  // First assume grid element is not valid and set only if legitimate values were computed.
  *ref->valid = 0;

  // First compute new variance of elevation WITH input elevation and check if it is below threshold
  // If yes, update average elevation for grid element and return
  // If no, check if hypothesis is valid
  float variance_new = ((*ref->nobs) + 1 - 2) / (float)((*ref->nobs) + 1 - 1) * (*ref->var) + (*inp->ele - *ref->ele) * (*inp->ele - *ref->ele) / (float)((*ref->nobs) + 1);
  if (variance_new < _th_elevation_var)
  {
    *ref->ele = (*ref->ele) + ((*inp->ele) - (*ref->ele)) / (float)(*ref->nobs + 1);
    *ref->var = variance_new;
    *ref->nobs = (*ref->nobs) + (uint16_t)1;
    if (_use_surface_normals)
      *ref->normal = (*ref->normal) + ((*inp->normal) - (*ref->normal)) / (float)(*ref->nobs + 1);
    if (*ref->nobs >= _th_elevation_min_nobs)
      *ref->valid = 255;
    // Color blending
    if (fabsf(*inp->angle - 90) < fabsf(*ref->angle - 90))
    {
      *ref->angle = *inp->angle;
      *ref->rgb = *inp->rgb;
    }
    return;
  }

  // Compute std deviation of elevation hypothesis WITH input elevation. Check then if below threshold.
  // If yes OR hypothesis is better than set elevation, switch hypothesis and elevation, update std deviation
  // If no, go on
  if (*ref->hyp > consts::getNoValue<float>())
  {
    float variance_hyp = ((*inp->ele) - (*ref->hyp)) * ((*inp->ele) - (*ref->hyp)) / 2.0f;
    if (variance_hyp < _th_elevation_var || variance_hyp < variance_new)
    {
      float elevation = (*ref->ele);
      *ref->var = variance_hyp;
      *ref->ele = ((*ref->hyp) + (*inp->ele)) / 2.0f;
      *ref->hyp = elevation;
      *ref->nobs = 2;
      if (_use_surface_normals)
        *ref->normal = ((*ref->normal) + (*inp->normal)) / 2.0f;
      if (*ref->nobs >= _th_elevation_min_nobs)
        *ref->valid = 255;
      *ref->angle = *inp->angle;
      *ref->rgb = *inp->rgb;
    }
  }

  // No valid assumption of grid element can be identified.
  // Set current input as new hypothesis and choose the one with the lowest variance.
  *ref->hyp = *inp->ele;
}

void Comapping::setGridElement(const GridQuickAccess::Ptr &ref, const GridQuickAccess::Ptr &inp)
{
  assert(*inp->valid == 255);
  *ref->ele = *inp->ele;
  *ref->elevated = *inp->elevated;
  *ref->var = 0.0f;
  *ref->hyp = 0.0f;
  *ref->angle = *inp->angle;
  *ref->rgb = *inp->rgb;
  *ref->nobs = 1;
  *ref->valid = 255;
  if (_use_surface_normals)
    *ref->normal = *inp->normal;
}

void Comapping::saveIter(uint32_t id)
{
  if (_settings_save.save_valid)
    io::saveImage((*(_map_stack[0]->map))["valid"], _stage_path + "/valid", "valid", id);
  if (_settings_save.save_ortho_rgb_all)
    io::saveImage((*(_map_stack[0]->map))["color_rgb"], _stage_path + "/ortho", "ortho", id);
  if (_settings_save.save_elevation_all)
    io::saveImageColorMap((*(_map_stack[0]->map))["elevation"], (*(_map_stack[0]->map))["valid"], _stage_path + "/elevation/color_map", "elevation", id, io::ColormapType::ELEVATION);
  if (_settings_save.save_elevation_var_all)
    io::saveImageColorMap((*(_map_stack[0]->map))["elevation_var"], (*(_map_stack[0]->map))["valid"], _stage_path + "/variance", "variance", id, io::ColormapType::ELEVATION);
  if (_settings_save.save_elevation_obs_angle_all)
    io::saveImageColorMap((*(_map_stack[0]->map))["elevation_angle"], (*(_map_stack[0]->map))["valid"], _stage_path + "/obs_angle", "angle", id, io::ColormapType::ELEVATION);
  if (_settings_save.save_num_obs_all)
    io::saveImageColorMap((*(_map_stack[0]->map))["num_observations"], (*(_map_stack[0]->map))["valid"], _stage_path + "/nobs", "nobs", id, io::ColormapType::ELEVATION);
  if (_settings_save.save_ortho_gtiff_all)
    io::saveGeoTIFF(*(_map_stack[0]->map), "color_rgb", _utm_reference->zone, _stage_path + "/ortho", "ortho", id);
}

void Comapping::saveAll()
{
  // 2D map output
  //if (_settings_save.save_ortho_rgb_one)
  io::saveImage((*(_map_stack[0]->map))["color_rgb"], _stage_path + "/ortho", "ortho1");
  //if (_settings_save.save_elevation_one)
  // io::saveImageColorMap((*(_map_stack[0]->map))["elevation"], (*(_map_stack[0]->map))["valid"], _stage_path + "/elevation/color_map", "elevation", io::ColormapType::ELEVATION);
  //if (_settings_save.save_elevation_var_one)
  // io::saveImageColorMap((*(_map_stack[0]->map))["elevation_var"], (*(_map_stack[0]->map))["valid"], _stage_path + "/variance", "variance", io::ColormapType::ELEVATION);
  //if (_settings_save.save_elevation_obs_angle_one)
  //  io::saveImageColorMap((*(_map_stack[0]->map))["elevation_angle"], (*(_map_stack[0]->map))["valid"], _stage_path + "/obs_angle", "angle", io::ColormapType::ELEVATION);
  //if (_settings_save.save_num_obs_one)
  //  io::saveImageColorMap((*(_map_stack[0]->map))["num_observations"], (*(_map_stack[0]->map))["valid"], _stage_path + "/nobs", "nobs", io::ColormapType::ELEVATION);
  //if (_settings_save.save_num_obs_one)
  // io::saveGeoTIFF(*(_map_stack[0]->map), "num_observations", _map_stack[0]->utm_reference->zone, _stage_path + "/nobs", "nobs");
  //if (_settings_save.save_ortho_gtiff_one)
  io::saveGeoTIFF(*(_map_stack[0]->map), "color_rgb", _map_stack[0]->utm_reference->zone, _stage_path + "/ortho", "ortho2");
  //if (_settings_save.save_elevation_one)
  // io::saveGeoTIFF(*(_map_stack[0]->map), "elevation", _map_stack[0]->utm_reference->zone, _stage_path + "/elevation/gtiff", "elevation");

  // 3D Point cloud output
  if (_settings_save.save_dense_ply)
  {
    if ((_map_stack[0]->map)->exists("elevation_normal"))
      io::saveElevationPointsToPLY(*(_map_stack[0]->map), "elevation", "elevation_normal", "color_rgb", "valid", _stage_path + "/elevation/ply", "elevation");
    else
      io::saveElevationPointsToPLY(*(_map_stack[0]->map), "elevation", "", "color_rgb", "valid", _stage_path + "/elevation/ply", "elevation");
  }

  // 3D Mesh output
  if (_settings_save.save_elevation_mesh_one)
  {
    std::vector<cv::Point2i> vertex_ids = _mesher->buildMesh(*(_map_stack[0]->map), "valid");
    if ((_map_stack[0]->map)->exists("elevation_normal"))
      io::saveElevationMeshToPLY(*(_map_stack[0]->map), vertex_ids, "elevation", "elevation_normal", "color_rgb", "valid", _stage_path + "/elevation/mesh", "elevation");
    else
      io::saveElevationMeshToPLY(*(_map_stack[0]->map), vertex_ids, "elevation", "", "color_rgb", "valid", _stage_path + "/elevation/mesh", "elevation");
  }
}

void Comapping::reset()
{
  LOG_F(INFO, "Reseted!");
}

void Comapping::finishCallback()
{
  // First polish results
  runPostProcessing();
  // Publish final mesh at the end
  if (_do_publish_mesh_at_finish)
    _transport_mesh(createMeshFaces((_map_stack[0]->map)), "output/mesh");
}

void Comapping::runPostProcessing()
{
}

Frame::Ptr Comapping::getNewFrame()
{
  std::unique_lock<std::mutex> lock(_mutex_buffer);
  Frame::Ptr frame = _buffer.front();
  _buffer.pop_front();
  return (std::move(frame));
}

void Comapping::initStageCallback()
{
  // Stage directory first
  if (!io::dirExists(_stage_path))
    io::createDir(_stage_path);

  // Then sub directories
  if (!io::dirExists(_stage_path + "/elevation"))
    io::createDir(_stage_path + "/elevation");
  if (!io::dirExists(_stage_path + "/elevation/color_map"))
    io::createDir(_stage_path + "/elevation/color_map");
  if (!io::dirExists(_stage_path + "/elevation/ply"))
    io::createDir(_stage_path + "/elevation/ply");
  if (!io::dirExists(_stage_path + "/elevation/pcd"))
    io::createDir(_stage_path + "/elevation/pcd");
  if (!io::dirExists(_stage_path + "/elevation/mesh"))
    io::createDir(_stage_path + "/elevation/mesh");
  if (!io::dirExists(_stage_path + "/elevation/gtiff"))
    io::createDir(_stage_path + "/elevation/gtiff");
  if (!io::dirExists(_stage_path + "/obs_angle"))
    io::createDir(_stage_path + "/obs_angle");
  if (!io::dirExists(_stage_path + "/variance"))
    io::createDir(_stage_path + "/variance");
  if (!io::dirExists(_stage_path + "/ortho"))
    io::createDir(_stage_path + "/ortho");
  if (!io::dirExists(_stage_path + "/nobs"))
    io::createDir(_stage_path + "/nobs");
  if (!io::dirExists(_stage_path + "/valid"))
    io::createDir(_stage_path + "/valid");
}

void Comapping::printSettingsToLog()
{
  LOG_F(INFO, "### Stage process settings ###");
  LOG_F(INFO, "- publish_mesh_nth_iter: %i", _publish_mesh_nth_iter);
  LOG_F(INFO, "- publish_mesh_every_nth_kf: %i", _publish_mesh_every_nth_kf);
  LOG_F(INFO, "- do_publish_mesh_at_finish: %i", _do_publish_mesh_at_finish);
  LOG_F(INFO, "- downsample_publish_mesh: %4.2f", _downsample_publish_mesh);
  LOG_F(INFO, "- use_surface_normals: %i", _use_surface_normals);
  LOG_F(INFO, "- th_elevation_min_nobs: %i", _th_elevation_min_nobs);
  LOG_F(INFO, "- th_elevation_var: %4.2f", _th_elevation_var);

  LOG_F(INFO, "### Stage save settings ###");
  LOG_F(INFO, "- save_valid: %i", _settings_save.save_valid);
  LOG_F(INFO, "- save_ortho_rgb_one: %i", _settings_save.save_ortho_rgb_one);
  LOG_F(INFO, "- save_ortho_rgb_all: %i", _settings_save.save_ortho_rgb_all);
  LOG_F(INFO, "- save_ortho_gtiff_one: %i", _settings_save.save_ortho_gtiff_one);
  LOG_F(INFO, "- save_ortho_gtiff_all: %i", _settings_save.save_ortho_gtiff_all);
  LOG_F(INFO, "- save_elevation_one: %i", _settings_save.save_elevation_one);
  LOG_F(INFO, "- save_elevation_all: %i", _settings_save.save_elevation_all);
  LOG_F(INFO, "- save_elevation_var_one: %i", _settings_save.save_elevation_var_one);
  LOG_F(INFO, "- save_elevation_var_all: %i", _settings_save.save_elevation_var_all);
  LOG_F(INFO, "- save_elevation_obs_angle_one: %i", _settings_save.save_elevation_obs_angle_one);
  LOG_F(INFO, "- save_elevation_obs_angle_all: %i", _settings_save.save_elevation_obs_angle_all);
  LOG_F(INFO, "- save_elevation_mesh_one: %i", _settings_save.save_elevation_mesh_one);
  LOG_F(INFO, "- save_num_obs_one: %i", _settings_save.save_num_obs_one);
  LOG_F(INFO, "- save_num_obs_all: %i", _settings_save.save_num_obs_all);
  LOG_F(INFO, "- save_dense_ply: %i", _settings_save.save_dense_ply);
}

std::vector<Face> Comapping::createMeshFaces(const CvGridMap::Ptr &map)
{
  CvGridMap::Ptr mesh_sampled;
  if (_downsample_publish_mesh > 10e-6)
  {
    // Downsampling was set by the user in settings
    LOG_F(INFO, "Downsampling mesh publish to %4.2f [m/gridcell]...", _downsample_publish_mesh);
    mesh_sampled = std::make_shared<CvGridMap>(map->cloneSubmap({"elevation", "color_rgb", "valid"}));

    // TODO: Change resolution correction is not cool -> same in ortho rectification
    // Check ranges of input elevation, this is necessary to correct resizing interpolation errors
    double ele_min, ele_max;
    cv::Point2i min_loc, max_loc;
    cv::minMaxLoc((*mesh_sampled)["elevation"], &ele_min, &ele_max, &min_loc, &max_loc, (*mesh_sampled)["valid"]);

    mesh_sampled->changeResolution(_downsample_publish_mesh);

    // After resizing through bilinear interpolation there can occure bad elevation values at the border
    cv::Mat mask_low = ((*mesh_sampled)["elevation"] < ele_min);
    cv::Mat mask_high = ((*mesh_sampled)["elevation"] > ele_max);
    (*mesh_sampled)["elevation"].setTo(consts::getNoValue<float>(), mask_low);
    (*mesh_sampled)["elevation"].setTo(consts::getNoValue<float>(), mask_high);
    (*mesh_sampled)["valid"].setTo(0, mask_low);
    (*mesh_sampled)["valid"].setTo(0, mask_high);
  }
  else
  {
    LOG_F(INFO, "No downsampling of mesh publish...");
    // No downsampling was set
    mesh_sampled = map;
  }

  std::vector<cv::Point2i> vertex_ids = _mesher->buildMesh(*mesh_sampled, "valid");
  std::vector<Face> faces = cvtToMesh((*mesh_sampled), "elevation", "color_rgb", vertex_ids);
  return faces;
}

void Comapping::publish(const Frame::Ptr &frame, const Map::Ptr &map, uint64_t timestamp)
{
  _transport_img((*(map->map))["color_rgb"], "output/rgb");
  _transport_img(analysis::convertToColorMapFromCVFC1((*(map->map))["elevation"],
                                                      (*(map->map))["valid"],
                                                      cv::COLORMAP_JET),
                 "output/elevation");
  _transport_cvgridmap(map->map_update->getSubmap({"color_rgb"}), map->utm_reference->zone, map->utm_reference->band, "output/update/ortho");

  if (_publish_mesh_every_nth_kf > 0 && _publish_mesh_every_nth_kf == _publish_mesh_nth_iter)
  {
    std::vector<Face> faces = createMeshFaces(map->map);
    std::thread t(_transport_mesh, faces, "output/mesh");
   t.detach();
    _publish_mesh_nth_iter = 0;
  }
  else if (_publish_mesh_every_nth_kf > 0)
  {
    _publish_mesh_nth_iter++;
  }
}

Comapping::GridQuickAccess::GridQuickAccess(const std::vector<std::string> &layer_names, const CvGridMap &map)
    : ele(nullptr),
      var(nullptr),
      hyp(nullptr),
      nobs(nullptr),
      rgb(nullptr),
      valid(nullptr)
{
  for (const auto &layer_name : layer_names)
    if (layer_name == "elevation")
      _elevation = map["elevation"];
    else if (layer_name == "elevation_normal")
      _elevation_normal = map["elevation_normal"];
    else if (layer_name == "elevation_var")
      _elevation_var = map["elevation_var"];
    else if (layer_name == "elevation_hyp")
      _elevation_hyp = map["elevation_hyp"];
    else if (layer_name == "elevation_angle")
      _elevation_angle = map["elevation_angle"];
    else if (layer_name == "color_rgb")
      _color_rgb = map["color_rgb"];
    else if (layer_name == "num_observations")
      _num_observations = map["num_observations"];
    else if (layer_name == "elevated")
      _elevated = map["elevated"];
    else if (layer_name == "valid")
      _valid = map["valid"];
    else
      throw(std::out_of_range("Error creating GridQuickAccess object. Demanded layer name does not exist!"));

  assert(!_elevation.empty() && _elevation.type() == CV_32F);
  assert(!_elevation_angle.empty() && _elevation_angle.type() == CV_32F);
  assert(!_color_rgb.empty() && _color_rgb.type() == CV_8UC4);
  assert(!_elevated.empty() && _elevated.type() == CV_8UC1);
  assert(!_valid.empty() && _valid.type() == CV_8UC1);

  move(0, 0);
}

void Comapping::GridQuickAccess::move(int row, int col)
{
  ele = &_elevation.ptr<float>(row)[col];
  var = &_elevation_var.ptr<float>(row)[col];
  hyp = &_elevation_hyp.ptr<float>(row)[col];
  angle = &_elevation_angle.ptr<float>(row)[col];
  nobs = &_num_observations.ptr<uint16_t>(row)[col];
  rgb = &_color_rgb.ptr<cv::Vec4b>(row)[col];
  elevated = &_elevated.ptr<uchar>(row)[col];
  valid = &_valid.ptr<uchar>(row)[col];
  if (!_elevation_normal.empty())
    normal = &_elevation_normal.ptr<cv::Vec3f>(row)[col];
}

void Comapping::checkMapStack(const Frame::Ptr &frame, Map::Ptr &found_by_agentName, std::vector<Map::Ptr> &only_by_overlap)
{
  CvGridMap::Ptr observed_frame = frame->getObservedMap();
  std::string agent_of_frame = frame->getCameraId();
  std::vector<Map::Ptr> found_by_overlap;

  std::set<std::string>::iterator iter;
  std::set<std::string> names;

  for (Map::Ptr &map_in_stack : _map_stack)
  {

    if ((agent_of_frame == map_in_stack->creater) || (agent_of_frame == map_in_stack->co_observer))
    {
      found_by_agentName = map_in_stack;
    }
    else if ((agent_of_frame != map_in_stack->creater) && (agent_of_frame != map_in_stack->co_observer))
    {
      cv::Rect2d roi_1 = observed_frame->roi();
      cv::Rect2d roi_2 = map_in_stack->map->roi();

      if (((roi_1.x > roi_2.x && roi_1.x < roi_2.x + roi_2.width) || (roi_1.x + roi_1.width > roi_2.x && roi_1.x + roi_1.width < roi_2.x + roi_2.width)) && ((roi_1.y < roi_2.y && roi_1.y > roi_2.y - roi_2.height) || (roi_1.y - roi_1.height < roi_2.y && roi_1.y - roi_1.height > roi_2.y - roi_2.height)))
      {
        only_by_overlap.push_back(map_in_stack);
      }
    }
  }
}

Comapping::Map::Ptr Comapping::createNewMap(const Frame::Ptr &frame)
{
  LOG_F(INFO, "Initializing local map of agent %s...", frame->getCameraId().c_str());
  Map::Ptr newMap = std::make_shared<Map>();

  newMap->map = frame->getObservedMap();

  newMap->utm_reference = std::make_shared<UTMPose>(frame->getGnssUtm());

  newMap->creater = frame->getCameraId();

  newMap->key_frame_list.push_back(frame);

  (*newMap->map).add("elevation_var", cv::Mat::ones(newMap->map->size(), CV_32F) * consts::getNoValue<float>());
  (*newMap->map).add("elevation_hyp", cv::Mat::ones(newMap->map->size(), CV_32F) * consts::getNoValue<float>());
  newMap->map_update = newMap->map;

  return newMap;
}

void Comapping::addFrametoMap(const Frame::Ptr &frame, Map::Ptr &map)
{
  CvGridMap::Ptr observed_map = frame->getObservedMap();
  std::string name = frame->getCameraId();
  cv::Mat T;
  try
  {

    T = map->error_correction.at(name);
    changeROI(T, observed_map);
  }
  catch (const std::out_of_range &oor)
  {
    LOG_F(INFO, "addFrametoMap:  without error correction");
  }

  (*(map->map)).add(*observed_map, REALM_OVERWRITE_ZERO, true);

  CvGridMap::Overlap overlap = map->map->getOverlap(*observed_map);

  if (overlap.first == nullptr && overlap.second == nullptr)
  {
    LOG_F(INFO, "No overlap detected. Add without blending...");
  }
  else
  {
    LOG_F(INFO, "Overlap detected. Add with blending...");
    CvGridMap overlap_blended = blend(&overlap);
    (*(map->map)).add(overlap_blended, REALM_OVERWRITE_ALL, false);
    cv::Rect2d roi = overlap_blended.roi();
    LOG_F(INFO, "Overlap region: [%4.2f, %4.2f] [%4.2f x %4.2f]", roi.x, roi.y, roi.width, roi.height);
    LOG_F(INFO, "Overlap area: %6.2f", roi.area());
  }

  map->map_update = std::make_shared<CvGridMap>(map->map->getSubmap({"color_rgb", "elevation", "valid"}, observed_map->roi()));

  std::string s = map->creater;

  if (s == "agent0")
  {
    while (map->key_frame_list.size() > 2)
    {
      map->key_frame_list.pop_front();
    }
    map->key_frame_list.push_back(frame);
  }

  if (s == "agent1")
  {
    while (map->key_frame_list.size() > 8)
    {
      map->key_frame_list.pop_back();
    }
    if (map->key_frame_list.size() < 20)
    {
      map->key_frame_list.push_back(frame);
    }
  }
}

void Comapping::mergeMaps(const Frame::Ptr &frame, std::vector<Map::Ptr> &overlapd_maps, Map::Ptr &agent_map)
{
  std::string agent_of_frame = frame->getCameraId();

  std::string co_observer = overlapd_maps[0]->creater;
  io::saveImage((*(_map_stack[0]->map))["color_rgb"], _stage_path + "/ortho", "agent1_color");
  io::saveGeoTIFF(*(_map_stack[0]->map), "color_rgb", _map_stack[0]->utm_reference->zone, _stage_path + "/ortho", "agent1_ortho");
  io::saveImage((*(_map_stack[1]->map))["color_rgb"], _stage_path + "/ortho", "agent2_color");
  io::saveGeoTIFF(*(_map_stack[1]->map), "color_rgb", _map_stack[1]->utm_reference->zone, _stage_path + "/ortho", "agent2_ortho");

  try
  {
    cv::Mat T = agent_map->error_correction.at(co_observer);
    changeROI(T, overlapd_maps[0]->map);
  }
  catch (const std::out_of_range &oor)
  {
    return;
  }

  CvGridMap::Overlap overlap = agent_map->map->getOverlap(*(overlapd_maps[0]->map));

  (*(agent_map->map)).add(*(overlapd_maps[0]->map), REALM_OVERWRITE_ZERO, true);

  CvGridMap overlap_blended = blend(&overlap);

  (*(agent_map->map)).add(overlap_blended, REALM_OVERWRITE_ALL, false);

  (*(overlapd_maps[0]->map)).add(overlap_blended, REALM_OVERWRITE_ALL, false);

  agent_map->map_update = std::make_shared<CvGridMap>(overlapd_maps[0]->map->getSubmap({"color_rgb", "elevation", "valid"}, overlapd_maps[0]->map->roi()));

  agent_map->co_observer = co_observer;

  //delete  merged map from map stack
  for (int i = 0; i < _map_stack.size(); i++)
  {
    if (_map_stack[i]->creater == co_observer)
    {
      _map_stack.erase(_map_stack.begin() + i);
    }
  }
  merged_flag = true;
  saveAll();
}

void Comapping::compute_error_correction(Frame::Ptr &f1, Map::Ptr found_by_agentName, std::vector<Map::Ptr> &only_by_overlap)
{
  Frame::Ptr f2;
  double min_UTM_dst;

  for (Map::Ptr map : only_by_overlap)
  {
    searchNearestFrame(f1, map, f2, min_UTM_dst);

    if (min_UTM_dst < 20)
    {
      int num = 0;
      std::vector<int> best_inliers;
      cv::Mat R, t;
      double error;
      match_two_frames_by_ransac(f1, f2, num, best_inliers, R, t, error);

      if (num > 30 && error < 2 && error > 0.1)
      {
        //computer error between coordinates of two maps
        cv::Mat T = R.t();
        T.push_back(t.t());
        T = T.t();
        cv::Mat m = cv::Mat(1, 4, CV_64F);
        m = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
        T.push_back(m);

        found_by_agentName->error_correction.insert(std::pair<std::string, cv::Mat>(map->creater, T));

        mergeMaps(f1, only_by_overlap, found_by_agentName);
      }
    }
  }
}

void Comapping::changeROI(cv::Mat T, CvGridMap::Ptr &map)
{
  double width = map->roi().width;
  double height = map->roi().height;
  double x = map->roi().x;
  double y = map->roi().y;

  cv::Mat new_x_y = (cv::Mat_<double>(4, 1) << x, y, 0, 1);
  cv::Mat new_x_width = (cv::Mat_<double>(4, 1) << x + width, y, 0, 1);
  cv::Mat new_y_height = (cv::Mat_<double>(4, 1) << x, y + height, 0, 1);

  new_x_y = T * new_x_y;
  new_x_width = T * new_x_width;
  new_y_height = T * new_y_height;

  double new_x = new_x_y.at<double>(0, 0);
  double new_y = new_x_y.at<double>(1, 0);
  double new_width = new_x_width.at<double>(0, 0) - new_x;
  double new_height = new_y_height.at<double>(1, 0) - new_y;

  map->setroi(cv::Rect2d(new_x, new_y, new_width, new_height));
}

void Comapping::searchNearestFrame(Frame::Ptr &frame, Map::Ptr map, Frame::Ptr &f2, double &min_UTM_dst)
{

  min_UTM_dst = 1000;
  double dst = 0.0;

  std::list<Frame::Ptr>::iterator it;
  for (it = map->key_frame_list.begin(); it != map->key_frame_list.end(); it++)
  {
    dst = distance_of_two_UTM(frame->getGnssUtm(), (*it)->getGnssUtm());
    if (dst < min_UTM_dst)
    {
      min_UTM_dst = dst;
      f2 = *it;
    }
  }
}

double Comapping::distance_of_two_UTM(const UTMPose pos1, const UTMPose pos2)
{
  return sqrt(pow(pos1.easting - pos2.easting, 2) + pow(pos1.northing - pos2.northing, 2));
}

double Comapping::distance_of_two_MPs(const MapPoint &p1, const MapPoint &p2)
{
  cv::Mat pos_1, pos_2;
  pos_1 = p1._mGeoPos;
  pos_2 = p2._mGeoPos;
  return sqrt(
      pow(pos_1.at<double>(0, 0) - pos_2.at<double>(0, 0), 2) + pow(pos_1.at<double>(1, 0) - pos_2.at<double>(1, 0), 2) + pow(pos_1.at<double>(2, 0) - pos_2.at<double>(2, 0), 2));
}

void Comapping::create_Point3f_from_MapPoint(const std::vector<MapPoint> &mps, std::vector<cv::Point3f> &pts)
{
  double x, y, z;
  for (MapPoint mp : mps)
  {
    x = mp._mGeoPos.at<double>(0, 0);
    y = mp._mGeoPos.at<double>(1, 0);
    z = mp._mGeoPos.at<double>(2, 0);
    pts.push_back(cv::Point3f(x, y, z));
  }
}

void Comapping::pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R, cv::Mat &t)
{
  // center of mass
  cv::Point3f p1, p2;
  int N = pts1.size();
  for (int i = 0; i < N; i++)
  {
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = cv::Point3f(cv::Vec3f(p1) / N);
  p2 = cv::Point3f(cv::Vec3f(p2) / N);
  // remove the center
  std::vector<cv::Point3f> q1(N), q2(N);
  for (int i = 0; i < N; i++)
  {
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }
  // compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++)
  {
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }
  // SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  if (U.determinant() * V.determinant() < 0)
  {
    for (int x = 0; x < 3; ++x)
    {
      U(x, 2) *= -1;
    }
  }

  Eigen::Matrix3d R_ = U * (V.transpose());
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

  // convert to cv::Mat
  R = (cv::Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2),
       R_(1, 0), R_(1, 1), R_(1, 2),
       R_(2, 0), R_(2, 1), R_(2, 2));
  t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void Comapping::match_two_frames_by_ransac(Frame::Ptr &f1, Frame::Ptr &f2, int &num, std::vector<int> &best_inliers, cv::Mat &R, cv::Mat &t, double &error)
{

  cv::Mat img1_raw = f1->getImageUndistorted();
  cv::Mat img2_raw = f2->getImageUndistorted();
  cv::Mat img_1_new;
  cv::Mat img_2_new;
  cv::cvtColor(img1_raw, img_1_new, CV_BGRA2BGR);
  cv::cvtColor(img2_raw, img_2_new, CV_BGRA2BGR);

  f1->compute_GeoPos_of_MapPoint();
  f2->compute_GeoPos_of_MapPoint();
  std::vector<MapPoint> MPs_1 = f1->getMapPoint();
  std::vector<MapPoint> MPs_2 = f2->getMapPoint();

  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  cv::Mat descriptors_1, descriptors_2;

  for (MapPoint &p_1 : MPs_1)
  {
    descriptors_1.push_back(p_1._mDescriptor);
    keypoints_1.push_back(cv::KeyPoint(cv::Point2f(p_1._x, p_1._y), 1));
  }

  for (MapPoint &p_2 : MPs_2)
  {
    descriptors_2.push_back(p_2._mDescriptor);
    keypoints_2.push_back(cv::KeyPoint(cv::Point2f(p_2._x, p_2._y), 1));
  }

  std::vector<cv::DMatch> matches;
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  matcher.match(descriptors_1, descriptors_2, matches);

  if (matches.size() > 20)
  {
    std::vector<MapPoint> mps1, mps2;
    for (int i = 0; i < matches.size(); i++)
    {
      mps1.push_back(MPs_1[matches[i].queryIdx]);
      mps2.push_back(MPs_2[matches[i].trainIdx]);
    }

    RANSAC ransac;
    ransac.setData(mps1, mps2);

    num = 0;
    error = 0;
    ransac.better_ransac(6, 200, 3.0, 1.0, 7);
    ransac.getResult(best_inliers, num, R, t, error);

    std::vector<cv::DMatch> better_ransac;
    for (int i = 0; i < best_inliers.size(); i++)
    {
      better_ransac.push_back(matches[best_inliers[i]]);
    }

    /*
    //save matching pairs for debug
    int name1, name2;
    cv::Mat img_1;

    name1 = f1->getFrameId();
    name2 = f2->getFrameId();
    std::string dest1 = "../temp_data/betterRANSAC/better_" + std::to_string(name1) + "_" + std::to_string(name2) + ".jpg";
    cv::drawMatches(img_1_new, keypoints_1, img_2_new, keypoints_2, better_ransac, img_1, cv::Scalar(0,0,255));
    */
  }
}

void Comapping::match_two_frames_by_ORB_SLAM_descriptor(Frame::Ptr &f1, Frame::Ptr &f2, std::vector<MapPoint> &mps1, std::vector<MapPoint> &mps2)
{

  cv::Mat img1_raw = f1->getImageUndistorted();
  cv::Mat img2_raw = f2->getImageUndistorted();
  cv::Mat img_1_new;
  cv::Mat img_2_new;
  cv::cvtColor(img1_raw, img_1_new, CV_BGRA2BGR);
  cv::cvtColor(img2_raw, img_2_new, CV_BGRA2BGR);

  f1->compute_GeoPos_of_MapPoint();
  f2->compute_GeoPos_of_MapPoint();
  std::vector<MapPoint> MPs_1 = f1->getMapPoint();
  std::vector<MapPoint> MPs_2 = f2->getMapPoint();

  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  cv::Mat descriptors_1, descriptors_2;

  for (MapPoint &p_1 : MPs_1)
  {
    descriptors_1.push_back(p_1._mDescriptor);
    keypoints_1.push_back(cv::KeyPoint(cv::Point2f(p_1._x, p_1._y), 1));
  }

  for (MapPoint &p_2 : MPs_2)
  {
    descriptors_2.push_back(p_2._mDescriptor);
    keypoints_2.push_back(cv::KeyPoint(cv::Point2f(p_2._x, p_2._y), 1));
  }

  std::vector<cv::DMatch> matches;
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  matcher.match(descriptors_1, descriptors_2, matches);

  double min_dist = 10000, max_dist = 0;
  for (int i = 0; i < matches.size(); i++)
  {
    double dist = matches[i].distance;
    if (dist < min_dist)
      min_dist = dist;
    if (dist > max_dist)
      max_dist = dist;
  }

  std::vector<cv::DMatch> good_matches;
  std::vector<cv::DMatch> RANSAC_matches;
  for (int i = 0; i < matches.size(); i++)
  {
    if (matches[i].distance <= std::max(2 * min_dist, 60.0))
    {
      good_matches.push_back(matches[i]);
    }
  }

  double dist = 0;
  double total_dist = 0;
  double average_dist = 0;

  for (int i = 0; i < good_matches.size(); i++)
  {
    dist = distance_of_two_MPs(MPs_1[good_matches[i].queryIdx], MPs_2[good_matches[i].trainIdx]);
    total_dist += dist;
  }

  average_dist = total_dist / good_matches.size();

  for (int i = 0; i < good_matches.size(); i++)
  {
    dist = distance_of_two_MPs(MPs_1[good_matches[i].queryIdx], MPs_2[good_matches[i].trainIdx]);
    if (dist < 2 * average_dist)
    {
      LOG_S(WARNING) << " good matches dist:" << dist;
      mps1.push_back(MPs_1[good_matches[i].queryIdx]);
      mps2.push_back(MPs_2[good_matches[i].trainIdx]);
    }
  }

  // display for debugging
  int name1, name2;
  name1 = f1->getFrameId();
  name2 = f2->getFrameId();
  std::string dest1 = "../temp_data/noRANSAC/img_goodmatch_" + std::to_string(name1) + "_" + std::to_string(name2) + ".jpg";
  std::string dest2 = "../temp_data/noRANSAC/img_match_" + std::to_string(name1) + "_" + std::to_string(name2) + ".jpg";
  cv::Mat img_match;
  cv::Mat img_goodmatch;
  cv::drawMatches(img_1_new, keypoints_1, img_2_new, keypoints_2, matches, img_match);
  cv::drawMatches(img_1_new, keypoints_1, img_2_new, keypoints_2, good_matches, img_goodmatch);

  uint32_t id_1 = f1->getFrameId();
  uint32_t id_2 = f2->getFrameId();
  std::string window_name = "ORB_SLAM_descriptor_img_goodmatch_of_";
  window_name = window_name + std::to_string(id_1) + "_" + std::to_string(id_2);

  LOG_S(WARNING) << " average dist of " << std::to_string(id_1) << " and " << std::to_string(id_2) << ":" << total_dist / good_matches.size();
  LOG_S(WARNING) << " totally found :" << matches.size();
  LOG_S(WARNING) << " totally good found :" << good_matches.size();
  //LOG_S(WARNING) << " distance of MPs:";

  //cv::imshow("ORB_SLAM_descriptor_img_matchby", img_match);
  //cv::imshow(window_name, img_goodmatch);
  cv::imwrite(dest1, img_goodmatch);
  //cv::imwrite(dest2, img_match);
  cv::waitKey(0);
}
