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

#include <realm_stages/stage_base.h>

using namespace realm;

StageBase::StageBase(const std::string &name, const std::string &path, int queue_size)
: WorkerThreadBase("Stage [" + name + "]", true),
  _stage_name(name),
  _stage_path(path),
  _queue_size(queue_size),
  _is_output_dir_initialized(false)
{
}

bool StageBase::changeParam(const std::string &name, const std::string &val)
{
  LOG_F(WARNING, "Changing parameter not implemented for this stage!");
  return false;
}

void StageBase::initStagePath(const std::string &abs_path)
{
  // Set and create output directory
  _stage_path = abs_path + "/" + _stage_name;
  initStageCallback();
  _is_output_dir_initialized = true;

  // Init logging
  loguru::add_file((_stage_path + "/stage.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
  LOG_F(INFO, "Successfully initialized!", _stage_name.c_str());
  LOG_F(INFO, "Stage path set to: %s", _stage_path.c_str());
  printSettingsToLog();
}

void StageBase::registerFrameTransport(const std::function<void(const Frame::Ptr&, const std::string&)> &func)
{
  _transport_frame = func;
}

void StageBase::registerPoseTransport(const std::function<void(const cv::Mat &, uint8_t zone, char band, const std::string &)> &func)
{
  _transport_pose = func;
}

void StageBase::registerDepthMapTransport(const std::function<void(const cv::Mat&, const std::string&)> &func)
{
  _transport_depth_map = func;
}

void StageBase::registerPointCloudTransport(const std::function<void(const cv::Mat&, const std::string&)> &func)
{
  _transport_pointcloud = func;
}

void StageBase::registerImageTransport(const std::function<void(const cv::Mat&, const std::string&)> &func)
{
  _transport_img = func;
}

void StageBase::registerMeshTransport(const std::function<void(const std::vector<Face>&, const std::string&)> &func)
{
  _transport_mesh = func;
}

void StageBase::registerCvGridMapTransport(const std::function<void(const CvGridMap &, uint8_t zone, char band, const std::string&)> &func)
{
  _transport_cvgridmap = func;
}