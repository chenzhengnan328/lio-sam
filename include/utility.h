#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

// CXX standard
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

// special IO
#include <yaml-cpp/yaml.h>
#include <json.hpp>

#include <opencv/cv.h>
#include <eigen3/Eigen/Dense>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h> 

using namespace std;
using json = nlohmann::json;

typedef pcl::PointXYZI PointType;

static const double GPS_X_OFFSET = 441000.0;
static const double GPS_Y_OFFSET = 4426000.0;
static const double GPS_Z_OFFSET = 0.0;

typedef struct {
    double r,g,b;
} COLOUR;

// oriented 3D bounding box
typedef struct {
    int type_id_;
    float yaw_;
    Eigen::Vector3f center_;
    Eigen::Vector3f size_;  // len, width, height
} OBBox;

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    std::uint16_t ring;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint16_t, ring, ring) (float, time, time)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

class ImuMsg{
public:
    double timestamp;
    Eigen::Vector3d acc; //linear acceleration in m/s/s
    Eigen::Vector3d gyr; //gyroscope rotation speed in rad/s
    Eigen::Quaterniond orientation; //orientation in quaternion
};

class OdomMsg{
public:
    double timestamp;
    Eigen::Vector3d position; //position in meters
    Eigen::Quaterniond orientation; //orientation in quaternion
    std::vector<float> covariance;
};

class cloud_info{
public:
    vector<int> startRingIndex;
    vector<int> endRingIndex;

    vector<int>  pointColInd;
    vector<float> pointRange;

    bool imuAvailable;
    bool odomAvailable;

    Eigen::Quaternionf imu_quaternion_init;

    Eigen::Affine3f intialGuessAffine;

    double timestamp;

    pcl::PointCloud<PointType> cloud_deskewed;
    pcl::PointCloud<PointType> cloud_corner;
    pcl::PointCloud<PointType> cloud_surface;

};

class ParamServer
{
public:

    int dynamicRemoveMode;
    bool bboxremovalflag;

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Lidar Sensor Configuration
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;
    float lidarMinZ;
    float boxSizeDilate;

    float mapLidarMinRange;
    float mapLidarMaxRange;
    float mapLidarMinZ;

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;


    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance; 
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;
    
    // Loop closure
    bool  loopClosureEnableFlag;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;
    float staticEgoMotionThreshold;


    ParamServer(const string& param_path);

};


void imuAngular2rosAngular(ImuMsg *thisImuMsg, double *angular_x, double *angular_y, double *angular_z);

float pointDistance(PointType p);

float pointDistance(PointType p1, PointType p2);

Eigen::Matrix3d eulerAnglesToRotationMatrix(const double& rotX, const double& rotY, const double& rotZ);


Eigen::Affine3f QuatTran2Affine(const Eigen::Quaterniond& quat, const Eigen::Vector3d& trans);

Eigen::Affine3f EulerTran2Affine(const double& x, const double& y, const double& z,
                                 const double& rotX, const double& rotY, const double& rotZ);
// void Affine2EulerTran(const Eigen::Affine3f& affine, float& x, float& y, float& z,
//                                  float& rotX, float& rotY, float& rotZ);

ImuMsg json2imu(const json& imu_json);

OdomMsg json2gps(const json& gps_json);

void odom2char(const OdomMsg& odom_msg, char output[]);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorizeCloud(const pcl::PointCloud<PointXYZIRT>& pc);

void OctreeDownSample(pcl::PointCloud<PointType>::Ptr& pointcloud, const float& leaf_size);

void FilterPointCloudByDistance(pcl::PointCloud<PointXYZIRT>::Ptr& pointclouds, const float& min_range, const float& max_range, const float& min_z);

void FilterPointCloudByDistance(pcl::PointCloud<PointType>::Ptr& pointclouds, const float& min_range, const float& max_range, const float& min_z);

#endif
