#include "stella_vslam/system.h"
#include "stella_vslam/config.h"
#include "stella_vslam/type.h"
#include "stella_vslam/publish/map_publisher.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/util/stereo_rectifier.h"


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "jlcxx/stl.hpp"


struct stella_vslam_jll
{
  stella_vslam_jll();
  ~stella_vslam_jll();
  bool initialize(std::string configFile,std::string orbFile);
  void shutdown();
  void load_map_database(std::string path);
  void save_map_database(std::string path);
  void enable_mapping_module();
  void disable_mapping_module();
  void enable_loop_detector();
  void disable_loop_detector();
  bool loop_BA_is_running();
  void reset();
  void pause_tracker();
  void resume_tracker();
  bool tracker_is_paused();
  void request_terminate();
  bool relocalize_by_pose(std::vector<double> cam_pose_wc);
  std::vector<double> processStereo(jlcxx::ArrayRef<uint8_t> left, jlcxx::ArrayRef<uint8_t> right, int width, int height, double timestamp);
private:
    std::shared_ptr<stella_vslam::system> system;
    std::shared_ptr<stella_vslam::util::stereo_rectifier> rectifier;
};

stella_vslam_jll::stella_vslam_jll()
: system{nullptr},rectifier{nullptr}
{}

stella_vslam_jll::~stella_vslam_jll()
{}

bool stella_vslam_jll::initialize(std::string configFile,std::string orbFile)
{
    std::shared_ptr<stella_vslam::config> cfg;
    cfg = std::make_shared<stella_vslam::config>(configFile);

    // build a SLAM system
    // FW:
    // - the 3rd parameter: b_seg_or_not = false,
    // - the 4th parameter: b_use_line_tracking = true
    system = std::make_shared<stella_vslam::system>(cfg, orbFile);
    system->startup();
    rectifier = std::make_shared<stella_vslam::util::stereo_rectifier>(cfg, system->get_camera()) ;

    return true;
}

void stella_vslam_jll::shutdown()
{
    system->shutdown();
}

void stella_vslam_jll::load_map_database(std::string path)
{
    system->load_map_database(path);
}

void stella_vslam_jll::save_map_database(std::string path)
{
    system->save_map_database(path);
}

void stella_vslam_jll::enable_mapping_module()
{
    system->enable_mapping_module();
}

void stella_vslam_jll::disable_mapping_module()
{
    system->disable_mapping_module();
}

void stella_vslam_jll::enable_loop_detector()
{
    system->enable_loop_detector();
}

void stella_vslam_jll::disable_loop_detector()
{
    system->disable_loop_detector();
}

bool stella_vslam_jll::loop_BA_is_running()
{
    return system->loop_BA_is_running();
}

void stella_vslam_jll::reset()
{
    system->request_reset();
}

bool stella_vslam_jll::tracker_is_paused()
{
    return system->tracker_is_paused();
}

void stella_vslam_jll::pause_tracker()
{
    system->pause_tracker();
}

void stella_vslam_jll::resume_tracker()
{
    system->resume_tracker();
}

void stella_vslam_jll::request_terminate()
{
    system->request_terminate();
}

bool stella_vslam_jll::relocalize_by_pose(std::vector<double> cam_pose_wc)
{
    stella_vslam::Mat44_t pose;
    for(size_t i = 0; i < pose.rows(); i++)
    {
		for(size_t j = 0; j < pose.cols(); j++)
        {
			pose(i,j) = cam_pose_wc[4*j+i];
        }
    }
    return system->relocalize_by_pose(pose);
}

std::vector<double> stella_vslam_jll::processStereo(jlcxx::ArrayRef<uint8_t> left, jlcxx::ArrayRef<uint8_t> right, int width, int height, double timestamp)
{
    cv::Mat leftCV=cv::Mat(cv::Size(width, height), CV_8UC1, left.data());
    cv::Mat rightCV=cv::Mat(cv::Size(width, height), CV_8UC1, right.data());
    
    rectifier->rectify(leftCV, rightCV, leftCV, rightCV);

    //cv::Mat flat = leftCV.reshape(1, leftCV.total()*leftCV.channels());
    //std::vector<uint8_t> vec = leftCV.isContinuous()? flat : leftCV.clone();
    // input the current frame and estimate the camera pose
    std::shared_ptr<stella_vslam::Mat44_t> pose = (system->feed_stereo_frame(leftCV, rightCV, timestamp));

    std::vector<double> out;
    
    out.resize(16);
    if (pose)
    {

        if (pose->rows() == 4 && pose->cols() == 4)
        {
            for(size_t i = 0; i < 4; i++)
            {
                for(size_t j = 0; j < 4; j++)
                {
                    out[4*j+i] = pose->coeff(i,j);
                }
            }
        }

    }

    leftCV.release();
    rightCV.release();
    return out;
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  mod.add_type<stella_vslam_jll>("stella_vslam_jll")
  .constructor(false)
  .method("initialize", &stella_vslam_jll::initialize)
  .method("shutdown", &stella_vslam_jll::shutdown)
  .method("load_map_database", &stella_vslam_jll::load_map_database)
  .method("save_map_database", &stella_vslam_jll::save_map_database)
  .method("enable_mapping_module", &stella_vslam_jll::enable_mapping_module)
  .method("disable_mapping_module", &stella_vslam_jll::disable_mapping_module)
  .method("enable_loop_detector", &stella_vslam_jll::enable_loop_detector)
  .method("disable_loop_detector", &stella_vslam_jll::disable_loop_detector)
  .method("loop_BA_is_running", &stella_vslam_jll::loop_BA_is_running)
  .method("reset", &stella_vslam_jll::reset)
  .method("pause_tracker", &stella_vslam_jll::pause_tracker)
  .method("resume_tracker", &stella_vslam_jll::resume_tracker)
  .method("tracker_is_paused", &stella_vslam_jll::tracker_is_paused)
  .method("request_terminate", &stella_vslam_jll::request_terminate)
  .method("relocalize_by_pose", &stella_vslam_jll::relocalize_by_pose)
  .method("processStereo", &stella_vslam_jll::processStereo);
};

