#include "utility.h"
#include <yaml-cpp/yaml.h>
#include <pcl/octree/octree_pointcloud_density.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>

ParamServer::ParamServer(const std::string& param_path)
{

    YAML::Node config = YAML::LoadFile(param_path);

    dynamicRemoveMode = config["lio_sam"]["dynamicRemoveMode"].as<int>();
    bboxremovalflag = config["lio_sam"]["bboxremovalflag"].as<bool>();

    N_SCAN = config["lio_sam"]["N_SCAN"].as<int>();
    Horizon_SCAN = config["lio_sam"]["Horizon_SCAN"].as<int>();
    downsampleRate = config["lio_sam"]["downsampleRate"].as<int>();
    lidarMinRange = config["lio_sam"]["lidarMinRange"].as<float>();
    lidarMaxRange = config["lio_sam"]["lidarMaxRange"].as<float>();
    lidarMinZ = config["lio_sam"]["lidarMinZ"].as<float>();
    boxSizeDilate = config["lio_sam"]["boxSizeDilate"].as<float>();

    mapLidarMinRange = config["lio_sam"]["mapLidarMinRange"].as<float>();
    mapLidarMaxRange = config["lio_sam"]["mapLidarMaxRange"].as<float>();
    mapLidarMinZ = config["lio_sam"]["mapLidarMinZ"].as<float>();



    imuAccNoise = config["lio_sam"]["imuAccNoise"].as<float>();
    imuGyrNoise = config["lio_sam"]["imuGyrNoise"].as<float>();
    imuAccBiasN = config["lio_sam"]["imuAccBiasN"].as<float>();
    imuGyrBiasN = config["lio_sam"]["imuGyrBiasN"].as<float>();
    imuGravity = config["lio_sam"]["imuGravity"].as<float>();


    edgeThreshold = config["lio_sam"]["edgeThreshold"].as<float>();
    surfThreshold = config["lio_sam"]["surfThreshold"].as<float>();

    edgeFeatureMinValidNum = config["lio_sam"]["edgeFeatureMinValidNum"].as<int>();
    surfFeatureMinValidNum = config["lio_sam"]["surfFeatureMinValidNum"].as<int>();

    odometrySurfLeafSize = config["lio_sam"]["odometrySurfLeafSize"].as<float>();
    mappingCornerLeafSize = config["lio_sam"]["mappingCornerLeafSize"].as<float>();
    mappingSurfLeafSize = config["lio_sam"]["mappingSurfLeafSize"].as<float>();

    z_tollerance = config["lio_sam"]["z_tollerance"].as<float>();
    rotation_tollerance = config["lio_sam"]["rotation_tollerance"].as<float>();

    numberOfCores = config["lio_sam"]["numberOfCores"].as<int>();

    mappingProcessInterval = config["lio_sam"]["mappingProcessInterval"].as<double>();

    surroundingkeyframeAddingDistThreshold = config["lio_sam"]["surroundingkeyframeAddingDistThreshold"].as<float>();
    surroundingkeyframeAddingAngleThreshold = config["lio_sam"]["surroundingkeyframeAddingAngleThreshold"].as<float>();
    surroundingKeyframeDensity = config["lio_sam"]["surroundingKeyframeDensity"].as<float>();
    surroundingKeyframeSearchRadius = config["lio_sam"]["surroundingKeyframeSearchRadius"].as<float>();
    surroundingKeyframeSize = config["lio_sam"]["surroundingKeyframeSize"].as<int>();

    historyKeyframeSearchRadius = config["lio_sam"]["historyKeyframeSearchRadius"].as<float>();
    historyKeyframeSearchTimeDiff = config["lio_sam"]["historyKeyframeSearchTimeDiff"].as<float>();
    historyKeyframeFitnessScore = config["lio_sam"]["historyKeyframeFitnessScore"].as<float>();
    historyKeyframeSearchNum = config["lio_sam"]["historyKeyframeSearchNum"].as<int>();
    staticEgoMotionThreshold = config["lio_sam"]["staticEgoMotionThreshold"].as<float>();



    loopClosureEnableFlag = config["lio_sam"]["loopClosureEnableFlag"].as<bool>();
    useGpsElevation = config["lio_sam"]["useGpsElevation"].as<bool>();
    useImuHeadingInitialization = config["lio_sam"]["useImuHeadingInitialization"].as<bool>();
    gpsCovThreshold = config["lio_sam"]["gpsCovThreshold"].as<float>();
    poseCovThreshold = config["lio_sam"]["poseCovThreshold"].as<float>();



}

void imuAngular2rosAngular(ImuMsg *thisImuMsg, double *angular_x, double *angular_y, double *angular_z)
{
    *angular_x = thisImuMsg->gyr.x();
    *angular_y = thisImuMsg->gyr.y();
    *angular_z = thisImuMsg->gyr.z();
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

Eigen::Affine3f QuatTran2Affine(const Eigen::Quaterniond& quat, const Eigen::Vector3d& trans){
    Eigen::Affine3d affine;
    affine = quat.toRotationMatrix();
    affine.translation() = trans;
    return affine.cast<float>();
}


 

// Calculates rotation matrix given euler angles.
Eigen::Matrix3d eulerAnglesToRotationMatrix(const double& rotX, const double& rotY, const double& rotZ)
{
    // Calculate rotation about x axis
    Eigen::Matrix3d R_x;
    R_x <<     1,       0,              0,
               0,       cos(rotX),   -sin(rotX),
               0,       sin(rotX),   cos(rotX);
    
    // Calculate rotation about y axis
    Eigen::Matrix3d R_y;
    R_y <<     cos(rotY),    0,      sin(rotY),
               0,               1,      0,
               -sin(rotY),   0,      cos(rotY);
    
    // Calculate rotation about z axis
    Eigen::Matrix3d R_z;
    R_z <<     cos(rotZ),    -sin(rotZ),      0,
               sin(rotZ),    cos(rotZ),       0,
               0,               0,                  1;
    return R_z * R_y * R_x;
}


Eigen::Affine3f EulerTran2Affine(const double& x, const double& y, const double& z,
                                 const double& rotX, const double& rotY, const double& rotZ)
{
    Eigen::Vector3d trans(x, y, z);

    Eigen::Matrix3d R = eulerAnglesToRotationMatrix(rotX, rotY, rotZ);

    Eigen::Affine3d affine(R);
    affine.translation() = trans;
    return affine.cast<float>();
}

// void Affine2EulerTran(const Eigen::Affine3f& affine, float& x, float& y, float& z,
//                                  float& rotX, float& rotY, float& rotZ)
// {
//     Eigen::Vector3f euler = affine.rotation().eulerAngles(0, 1, 2);
//     rotX = euler[0];
//     rotY = euler[1];
//     rotZ = euler[2];
//     Eigen::Vector3f translation = affine.translation();
//     x = translation.x();
//     y = translation.y();
//     z = translation.z();
// }

ImuMsg json2imu(const json& imu_json)
{
    ImuMsg imu_msg;


    // note that quaternion order is different from that of GPS
    // Data extracted by TAT
    if (imu_json.contains("quaterntion")){
        std::string timestamp_long = imu_json["sampleStamp"];
        double timestamp = std::stod(timestamp_long) / 1000.0;
        imu_msg.timestamp = timestamp;


        imu_msg.acc.x() = imu_json["acc"]["x"];
        imu_msg.acc.y() = imu_json["acc"]["y"];
        imu_msg.acc.z() = imu_json["acc"]["z"];

        imu_msg.gyr.x() = imu_json["gyro"]["x"];
        imu_msg.gyr.y() = imu_json["gyro"]["y"];
        imu_msg.gyr.z() = imu_json["gyro"]["z"];

        imu_msg.orientation.w() = imu_json["quaterntion"]["x"];
        imu_msg.orientation.x() = imu_json["quaterntion"]["y"];
        imu_msg.orientation.y() = imu_json["quaterntion"]["z"];
        imu_msg.orientation.z() = imu_json["quaterntion"]["w"];
    }
    // Data extracted by pack_streamer
    else{
        long timestamp_long = imu_json["time_stamp"];
        double timestamp = double(timestamp_long)/1000.0;
        imu_msg.timestamp = timestamp;

        imu_msg.acc.x() = imu_json["accleration"][0];
        imu_msg.acc.y() = imu_json["accleration"][1];
        imu_msg.acc.z() = imu_json["accleration"][2];

        imu_msg.gyr.x() = imu_json["gyro"][0];
        imu_msg.gyr.y() = imu_json["gyro"][1];
        imu_msg.gyr.z() = imu_json["gyro"][2];

        imu_msg.orientation.w() = imu_json["quaternion"][0];
        imu_msg.orientation.x() = imu_json["quaternion"][1];
        imu_msg.orientation.y() = imu_json["quaternion"][2];
        imu_msg.orientation.z() = imu_json["quaternion"][3];

    }

    return imu_msg;
}

OdomMsg json2gps(const json& gps_json){
    OdomMsg odom_msg;
    std::string timestamp_long = gps_json["sensorStamp"];
    double timestamp = std::stod(timestamp_long) / 1000.0;

    odom_msg.timestamp = timestamp;


    // woops! something ugly happens here:
    // the pcl library usually deal with numeric number in float
    // while gps UTM location is too large for float to represent ideal precision
    // thus the UTM location is move by GPS_X_OFFSET to make it has mean near 0 in beijing area
    // after all calculation, it will be moved back to original UTM coordinates

    double x_temp = gps_json["position"]["x"];
    double y_temp = gps_json["position"]["y"];
    double z_temp = gps_json["position"]["z"];
    odom_msg.position.x() = float(x_temp - GPS_X_OFFSET);
    odom_msg.position.y() = float(y_temp - GPS_Y_OFFSET);
    odom_msg.position.z() = float(z_temp - GPS_Z_OFFSET);



    // note that quaternion order is different from that of IMU
    odom_msg.orientation.x() = gps_json["orientation"]["x"];
    odom_msg.orientation.y() = gps_json["orientation"]["y"];
    odom_msg.orientation.z() = gps_json["orientation"]["z"];
    odom_msg.orientation.w() = gps_json["orientation"]["w"];

    int num_stat = gps_json["numSates"];
    float pseudo_covariance = 15 * exp(-float(num_stat)/10);
    odom_msg.covariance = std::vector<float>(21, 0);
    odom_msg.covariance[0] = pseudo_covariance;
    odom_msg.covariance[7] = pseudo_covariance;
    odom_msg.covariance[14] = pseudo_covariance;
    return odom_msg;
}

void odom2char(const OdomMsg& odom_msg, char output[]){
    sprintf(output, "%f %f %f %f %f %f %f %f\n",
        odom_msg.timestamp,
        odom_msg.position.x(),
        odom_msg.position.y(),
        odom_msg.position.z(),
        odom_msg.orientation.x(),
        odom_msg.orientation.y(),
        odom_msg.orientation.z(),
        odom_msg.orientation.w());
}


COLOUR GetColour(float v,float vmin,float vmax)
{
   COLOUR c = {1.0,1.0,1.0}; // white
   float dv;
   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c.r = 0;
      c.g = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      c.r = 0;
      c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      c.r = 4 * (v - vmin - 0.5 * dv) / dv;
      c.b = 0;
   } else {
      c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      c.b = 0;
   }

   return(c);
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorizeCloud(const pcl::PointCloud<PointXYZIRT>& pc){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    int num_points = pc.points.size();
    cloud->width=num_points;
    cloud->height=1;
    cloud->is_dense = false;
    cloud->points.resize(num_points);
    for(int i=0;i<num_points; i++){
        cloud->points[i].x = pc.points[i].x;
        cloud->points[i].y = pc.points[i].y;
        cloud->points[i].z = pc.points[i].z;
        float relatvie_time = (pc.points[i].time - pc.points[0].time);
        COLOUR rgb = GetColour(relatvie_time, 0.0, 0.1);
        cloud->points[i].r = rgb.r*255;
        cloud->points[i].g = rgb.g*255;
        cloud->points[i].b = rgb.b*255;
    }
    return  cloud;
}


void OctreeDownSample(pcl::PointCloud<PointType>::Ptr& pointcloud, const float& leaf_size)
{
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    pcl::octree::OctreePointCloudVoxelCentroid<PointType> sor(leaf_size);
    sor.setInputCloud(pointcloud);
    sor.addPointsFromInputCloud();
    std::vector<PointType, Eigen::aligned_allocator<PointType> > voxelCentroids;
    sor.getVoxelCentroids(voxelCentroids);
    pointcloud->clear();
    for(PointType point: voxelCentroids){
        pointcloud->push_back(point);
    }
}

void FilterPointCloudByDistance(pcl::PointCloud<PointXYZIRT>::Ptr& pointclouds, const float& min_range, const float& max_range, const float& min_z){
    pcl::PointCloud<PointXYZIRT>::Ptr pointcloud_filtered(new pcl::PointCloud<PointXYZIRT>);
    for(int i=0; i<pointclouds->points.size(); i++){
        PointXYZIRT point = pointclouds->points[i];
        float range = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
        if (range < min_range || range > max_range)
            continue;
        if (point.z < min_z)
            continue;
        pointcloud_filtered->points.push_back(point);
    }
    pointclouds.swap(pointcloud_filtered);
    pointcloud_filtered.reset(new pcl::PointCloud<PointXYZIRT>());
}

void FilterPointCloudByDistance(pcl::PointCloud<PointType>::Ptr& pointclouds, const float& min_range, const float& max_range, const float& min_z){
    pcl::PointCloud<PointType>::Ptr pointcloud_filtered(new pcl::PointCloud<PointType>);
    for(int i=0; i<pointclouds->points.size(); i++){
        PointType point = pointclouds->points[i];
        float range = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
        if (range < min_range || range > max_range)
            continue;
        if (point.z < min_z)
            continue;
        pointcloud_filtered->points.push_back(point);
    }
    pointclouds.swap(pointcloud_filtered);
    pointcloud_filtered.reset(new pcl::PointCloud<PointType>());
}
