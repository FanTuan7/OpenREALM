#ifndef PROJECT_MAP_POINT_H
#define PROJECT_MAP_POINT_H

//#include <cstdint>
#include <memory>
//#include <mutex>
#include <string>

#include <opencv2/core.hpp>

//#include <realm_types/frame.h>

namespace realm
{

class MapPoint
{
    public:
    using Ptr = std::shared_ptr<MapPoint>;
    using ConstPtr = std::shared_ptr<const MapPoint>;

    MapPoint(const cv::Mat P,const cv::Mat D,const double x,const double y);
    MapPoint();
    cv::Mat _mWorldPos;
    cv::Mat _mGeoPos;
    cv::Mat _mDescriptor;
    
    //Pos in 2D image
    double _x;
    double _y;
};

}

#endif