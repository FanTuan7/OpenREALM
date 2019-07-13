#include <realm_types/map_point.h>

namespace realm
{
    MapPoint::MapPoint(const cv::Mat P,const cv::Mat D,const double x,const double y)
    :_mWorldPos(P),
     _mDescriptor(D),
     _x(x),
     _y(y)
     {

     }

    MapPoint::MapPoint()
     {

     }

}