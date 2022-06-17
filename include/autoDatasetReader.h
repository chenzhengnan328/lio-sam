#include "json.hpp"
#include "utility.h"
#include <opencv2/opencv.hpp>

using json = nlohmann::json;
/**
 * @brief Dataset Reader for HDE generated data
 * 
 */
class autoDatasetReader: public ParamServer
{
public:
    int num_lidar;

    json attribute;

    Eigen::Matrix3d extRotGNSS2LiDAR;
    Eigen::Vector3d extTransGNSS2LiDAR;

    Eigen::Matrix3d extRotIMU2LiDAR;
    Eigen::Matrix3d extRPYIMU2LiDA;
    Eigen::Vector3d extTransIMU2LiDAR;

    /**
     * @brief Construction from only parameter file
     * 
     * @param param_path config/params_horizon.yaml
     */
    autoDatasetReader(const std::string& param_path);

    /**
     * @brief Construction from data dir and parameter file
     * 
     * @param data_dir FSD_Site_v1/Site_20211230_116_15472_40_06718/H3165_20211230_132320
     * @param param_path config/params_horizon.yaml
     */
    autoDatasetReader(const std::string &data_dir, const std::string& param_path);


    /**
     * @brief load a new clip or pack
     * 
     * @param data_dir FSD_Site_v1/Site_20211230_116_15472_40_06718/H3165_20211230_132320
     */
    void load(const std::string &data_dir);

    /**
     * @brief helper function to read json file
     * 
     * @param file_path absolute path to a json file
     * @return json 
     */
    json readJSON(const std::string &file_path);

    /**
     * @brief read imu_json file timestamp
     * 
     */
    bool read_TimeStamp(const char* data_dir, std::vector<long>& timestamp_array);


    /**
     * @brief Read a pointcloud bin file
     *        assuming data in double with shape (Nx6)
     *        x y z intensity ringID point_timestamp
     * @param file_path absolute path to a .bin file 
     * @return pcl::PointCloud<PointXYZIRT>::Ptr load pointcloud
     */
    pcl::PointCloud<PointXYZIRT>::Ptr readLiDAR_with_pointtime(const std::string &file_path);

    /**
     * @brief Read a pointcloud bin file
     *        assuming data in double with shape (Nx6)
     *        x y z intensity ringID point_timestamp
     *        This function does not rely on point_timestamp,
     *        instead we calculate each point timestamp
     * @param file_path absolute path to a .bin file 
     * @return pcl::PointCloud<PointXYZIRT>::Ptr load pointcloud
     */
    pcl::PointCloud<PointXYZIRT>::Ptr readLiDAR_calculate_pointtime(const std::string &file_path);

    /**
     * @brief Wrapper function to read LiDAR frame and corrsponding IMU frames
     * 
     * @param frame_idx input: LiDAR frame index according to attribute["sync"]
     * @param return_pointcloud output: pointcloud frame
     * @param return_imu output: IMU frames in vector<json>
     * @param step input: If > 1, skip every step frame
     * @return double LiDAR frame timestamp in secs
     */
    double get_LiDAR_IMU(const int &frame_idx, pcl::PointCloud<PointXYZIRT>::Ptr &return_pointcloud, std::vector<json> &return_imu, const int &step = 1);
    
    /**
     * @brief Wrapper function to read LiDAR frame and corrsponding IMU frames
     * 
     * @param frame_idx input: LiDAR frame index according to attribute["sync"]
     * @param return_pointcloud output: pointcloud frame
     * @param return_imu output: IMU frames in vector<json>
     * @param return_ub482: output: GPS frame
     * @param step input: If > 1, skip every step frame
     * @return double LiDAR frame timestamp in secs
     */
    double get_LiDAR_IMU_GPS(const int &frame_idx, pcl::PointCloud<PointXYZIRT>::Ptr &return_pointcloud, std::vector<json> &return_imu, std::vector<json> &return_ub482, const int &step = 1);

    /**
     * @brief wrapper function to load gps by frame_idx
     * 
     * @param frame_idx input: UB482 frame index according to attribute["sync"]
     * @param return_ub482 output: GPS frame
     * @return double GPS frame timestamp in secs
     */
    double get_GPS(const int &frame_idx, json &return_ub482);

    /**
     * @brief Read semantic image of a camera
     * 
     * @param camera camera name, e.g. camera_rear_left
     * @param frame_idx camera frame idx according to attribute["sync"]
     * @return cv::Mat one channel semantic label image
     */
    cv::Mat get_semantic(const std::string& camera, const int& frame_idx);

    /**
     * @brief Load camera intrinsic
     * 
     * @param camera camera name, e.g. camera_rear_left
     * @param K camera intrinsic matrix
     * @param d camera distortion coefficient
     * @return true load success
     * @return false load fail
     */
    bool get_intrinsic(const std::string& camera, cv::Mat& K, cv::Mat&d);

    /**
     * @brief Get the extrinsic between to sensor
     * 
     * @param from_sensor sensor_name, e.g. lidar_top
     * @param to_sensor sensor_name, e.g. camera_front
     * @param R rotation matrix
     * @param t translation vector
     * @return true load success
     * @return false load fail
     */
    bool get_extrinsic(const std::string &from_sensor, const std::string &to_sensor, cv::Mat& R, cv::Mat&t);

    /**
     * @brief Get the extrinsic between to sensor
     * 
     * @param from_sensor sensor_name, e.g. lidar_top
     * @param to_sensor sensor_name, e.g. camera_front
     * @param output 4x4 transformation matrix
     * @return true load success
     * @return false load fail
     */
    bool get_extrinsic(const std::string &from_sensor, const std::string &to_sensor, Eigen::Matrix4d &output);

    /**
     * @brief Get the extrinsic between to sensor
     * 
     * @param from_sensor sensor_name, e.g. lidar_top
     * @param to_sensor sensor_name, e.g. camera_front
     * @param output_R rotation matrix
     * @param output_t translation vector
     * @return true load success
     * @return false load fail
     */
    bool get_extrinsic(const std::string &from_sensor, const std::string &to_sensor,
                       Eigen::Matrix3d &output_R, Eigen::Vector3d &output_t);

    /**
     * @brief Convert GPS pose into lidar frame
     * 
     * @param gps_in gps message
     * @return OdomMsg gps pose in lidar frame
     */
    OdomMsg gpsConverter(const OdomMsg &gps_in);

    /**
     * @brief Convert IMU pose into lidar frame
     * 
     * @param imu_in IMU message
     * @return ImuMsg IMU message in lidar frame
     */
    ImuMsg imuConverter(const ImuMsg &imu_in);

    /**
     * @brief Split 4x4 transformation matrix into 3x3 rotation matrix and 3x1 translation vector
     * 
     * @param input 4xe transformation matrix
     * @param output_R rotation matrix
     * @param output_t translation vector
     * @return true convert success
     * @return false convert fail
     */
    bool Transform2RotT(const Eigen::Matrix4d &input, Eigen::Matrix3d &output_R, Eigen::Vector3d &output_t);

    /**
     * @brief remove points belongs to dynamic objects from cloud_in
     *        according to semantic info: label_img
     * @param label_img input semantic label image
     * @param cloud_in input and output for pointclouds before and after culling
     * @param K camera intrinsic matrix of label_img
     * @param R rotation extrinsic from lidar to camera
     * @param t translation extrinsic from lidar to camera
     * @param dist_coeffs distort coefficients for camera
     * @param imgH image height of label_img
     * @param imgW image width of label_img
     */
    void dynamics_culling(const cv::Mat &label_img,
                          pcl::PointCloud<PointXYZIRT>::Ptr cloud_in,
                          const cv::Mat &K,
                          const cv::Mat &R,
                          const cv::Mat &t,
                          const cv::Mat &dist_coeffs,
                          const int &imgH,
                          const int &imgW);

    /**
     * @brief Filter out points belong to dynamic objects by
     * projecting LiDAR points into camera and check with camera semantic label
     * 
     * @param pointclouds Input/Output, pointcloud
     * @param frame_idx frame index according to attribute["sync"]
     */
    void FilterPointCloudBySemanticImage(pcl::PointCloud<PointXYZIRT>::Ptr& pointclouds, const int& frame_idx);

    /**
     * @brief Filter out points belong to dynamic objects by
     * LiDAR 3D detection model results
     * 
     * @param lidar_timestamp LiDAR timestamp in microseconds
     * @param pointclouds Input/Output, pointcloud
     */
    void FilterPointCloudByLiDARDet(const std::string& lidar_timestamp, pcl::PointCloud<PointXYZIRT>::Ptr& pointclouds);

    /**
     * @brief Filter out points belong to dynamic objects by bbox
     * 
     * @param boxes list of bbox in LiDAR frame
     * @param pointclouds Input/Output, poincloud
     */
    void FilterPointCLoudByBox(const std::vector<OBBox>& boxes, pcl::PointCloud<PointXYZIRT>::Ptr& pointclouds);

    /**
     * @brief Check a if a point is in a box
     * 
     * @param pt single point
     * @param box_point bbox 8 edge point
     * @return true InBox
     * @return false OutofBox
     */
    bool IsInbox(PointXYZIRT pt, std::vector<Eigen::Vector3f> box_point);

    /**
     * @brief Convert BBox object into 8 edge points
     * 
     * @param bbox BBox object
     * @return std::vector<Eigen::Vector3f> 8 edge points
     */
    std::vector<Eigen::Vector3f> OBBox2EigenPoints(const OBBox& bbox);

    /**
     * @brief Read all BBox for one pointcloud 
     * 
     * @param lidar_time_str LiDAR frame timestamp in microseconds
     * @return std::vector<OBBox> All bbox for this pointcloud
     */
    std::vector<OBBox> ReadBoxPerFrame(const std::string& lidar_time_str);

private:
    std::string attribute_path;
    std::string data_dir;
    json lidar_det_json;
    std::vector<json> imu_json_frames;
    std::vector<long> imu_timestamps;
    const float score_threshold = 0.6;
    const static int num_camera = 6;
    const std::string CAMERA_NAME[num_camera] = {
        "camera_front",
        "camera_front_left",
        "camera_front_right",
        "camera_rear",
        "camera_rear_left",
        "camera_rear_right",
    };
    std::vector<long> lidar_timeStamp;
    std::vector<long> gps_timeStamp;
};