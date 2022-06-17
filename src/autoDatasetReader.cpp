#include "autoDatasetReader.h"

template <typename T>
int binary_search_find_index(std::vector<T> v, T data)
{
    auto it = std::lower_bound(v.begin(), v.end(), data);
    if (it == v.end())
    {
        return -1;
    }
    else
    {
        std::size_t index = std::distance(v.begin(), it);
        return index;
    }
}

autoDatasetReader::autoDatasetReader(const std::string& param_path):ParamServer(param_path){};

autoDatasetReader::autoDatasetReader(const std::string &data_dir, const std::string& param_path):ParamServer(param_path)
{
    //read lidar time-stamp
    std::string lidar_timestamp_path = data_dir + "/lidar_top";
    read_TimeStamp(lidar_timestamp_path.data(), lidar_timeStamp);
    if(lidar_timeStamp.size() < 10)
    {
        std::cerr << "can not load Lidar frames! abandon this clip: " << data_dir << std::endl;
        exit(0);
    }

    load(data_dir);
    // prepare extrinsic
    Eigen::Matrix4d lidar2chassis;
    this->get_extrinsic("lidar_top", "chassis", lidar2chassis);

    // prepare gnss 2 lidar

    Eigen::Matrix4d gnss2chassis;
    this->get_extrinsic("UB482", "chassis", gnss2chassis);
    Eigen::Matrix4d gnss2lidar = lidar2chassis.inverse() * gnss2chassis;
    Transform2RotT(gnss2lidar, extRotGNSS2LiDAR, extTransGNSS2LiDAR);

    // prepare imu 2 lidar
    Eigen::Matrix4d imu2chassis;
    this->get_extrinsic("IMU", "chassis", imu2chassis);
    Eigen::Matrix4d imu2lidar = lidar2chassis.inverse() * imu2chassis;
    Transform2RotT(imu2lidar, extRotIMU2LiDAR, extTransIMU2LiDAR);
    extRPYIMU2LiDA = extRotIMU2LiDAR;
    std::cout << "extRotIMU2LiDAR = \n" << extRotIMU2LiDAR << std::endl;
    std::cout << "extTransIMU2LiDAR = \n" << extTransIMU2LiDAR << std::endl;
    std::cout << "imu2chassis = \n" << imu2chassis << std::endl;
    std::cout << "gnss2chassis = \n" << gnss2chassis << std::endl;

    // read lidar detection results

    std::string lidar_det_path = data_dir + "/detection_tracking_lidar_top/labels.json";
    if (dynamicRemoveMode==2 && bboxremovalflag){
        std::cout << "loading " << lidar_det_path << std::endl;
        lidar_det_json = readJSON(lidar_det_path);
        if(lidar_det_json != NULL)
            std::cout << "load success " <<  std::endl;
        else
            std::cout << "file does not exist" << std::endl;
    }
    // else if(dynamicRemoveMode==1){
    //     std::cout << "Project lidar points to semantic camera images for dynamic points removal" << std::endl;
    // }
    // else{
    //     std::cout << "Dynamic points removal disabled, use all points for LIO-SAM" << std::endl;
    // }
    

    std::cout << "preloading all IMU frames and performing timestamp checking and reparing" << std::endl;
    imu_json_frames.clear();
    imu_timestamps.clear();
    int imu_error_count = 0;
    
    // using different Attribute data structure
    //std::vector<long> imu_timestamps_json = attribute["unsync"]["IMU"];
    std::vector<long> imu_timestamps_json;
    std::string imu_timestamp_path = data_dir + "/IMU";
    read_TimeStamp(imu_timestamp_path.data(), imu_timestamps_json);
    if(imu_timestamps_json.size() < 100){
        std::cerr << "can not load IMU frames! abandon this clip: " << data_dir << std::endl;
        exit(0);
    }
    imu_timestamps = imu_timestamps_json;
    long last_imu_timestamp = 0;

    // check which extraction pipeline is used, they have different format
    long imu_timestamp = imu_timestamps[0];
    std::string imu_path = data_dir + "/IMU/" + std::to_string(imu_timestamp) + ".json";
    json imu_message = readJSON(imu_path);
    if (imu_message.contains("quaterntion")){
        // extracted by TAT
        for(int imu_idx=0; imu_idx<imu_timestamps.size(); imu_idx++){
            long imu_timestamp = imu_timestamps[imu_idx];
            std::string imu_path = data_dir + "/IMU/" + std::to_string(imu_timestamp) + ".json";
            json imu_message = readJSON(imu_path);
            std::string imu_timestamp_str;
            if(last_imu_timestamp != 0){
                long diff = imu_timestamp - last_imu_timestamp;
                if(diff > 15 || diff < 5){
                    imu_error_count ++;
                    imu_message["sampleStamp"] = std::to_string(last_imu_timestamp + 10);
                    imu_timestamp_str = imu_message["sampleStamp"];
                    imu_timestamps[imu_idx] = std::stol(imu_timestamp_str);
                }
            }
            imu_timestamp_str = imu_message["sampleStamp"];
            last_imu_timestamp = std::stol(imu_timestamp_str);
            imu_json_frames.push_back(imu_message);
        }
    }
    else{
        // extracted by pack streamer
        for(int imu_idx=0; imu_idx<imu_timestamps.size(); imu_idx++){
            long imu_timestamp = imu_timestamps[imu_idx];
            std::string imu_path = data_dir + "/IMU/" + std::to_string(imu_timestamp) + ".json";
            json imu_message = readJSON(imu_path);
            if(last_imu_timestamp != 0){
                long diff = imu_timestamp - last_imu_timestamp;
                if(diff > 15 || diff < 5){
                    imu_error_count ++;
                    imu_message["time_stamp"] = last_imu_timestamp + 10;
                    imu_timestamps[imu_idx] = long(imu_message["time_stamp"]);
                }
            }
            last_imu_timestamp = long(imu_message["time_stamp"]);
            imu_json_frames.push_back(imu_message);
        }
    }

    std::string output_string;
    float abnormal_rate = float(imu_error_count) / float(imu_timestamps.size());
    if(abnormal_rate > 0.5){
        std::cerr << "IMU abnormal rate = " << std::setprecision(2) << abnormal_rate*100 << "%" 
        << " abandon this clip: "<< data_dir << std::endl;
        exit(0);
    }
    std::cout << "Loaded " << imu_timestamps.size() << " IMU frames with " << imu_error_count
              << " abnormal frames, abnormal rate = " << std::setprecision(2) << abnormal_rate*100 << "%" << std::endl;
}

void autoDatasetReader::load(const std::string &data_dir_)
{
    data_dir = data_dir_;
    attribute_path = data_dir + '/' + "attribute.json";
    std::cout << "loading " << attribute_path << std::endl;
    std::ifstream attribute_reader(attribute_path);
    attribute_reader >> attribute;
    
    //num_lidar = attribute["sync"]["lidar_top"].size();
    
    //Read Folder
    num_lidar = lidar_timeStamp.size();
    //loading GPS data
    std::string gps_timestamp_path = data_dir + "/UB482";
    read_TimeStamp(gps_timestamp_path.data(), gps_timeStamp);
    
    std::cout << "get " << gps_timeStamp.size() << " GPS data from this clip" <<std::endl;
}

json autoDatasetReader::readJSON(const std::string &file_path)
{
    json json_frame;
    std::ifstream json_reader(file_path);
    if(!json_reader.is_open())
        return NULL;
    json_reader >> json_frame;
    return json_frame;
}

pcl::PointCloud<PointXYZIRT>::Ptr autoDatasetReader::readLiDAR_with_pointtime(const std::string &file_path)
{
    const int len = file_path.size();
    std::string timestamp_str = file_path.substr(len - 17, 13);
    std::stringstream stream;
    stream << timestamp_str;
    double timestamp;
    stream >> timestamp;
    timestamp /= 1000.0;

    pcl::PointCloud<PointXYZIRT>::Ptr pointcloud(new pcl::PointCloud<PointXYZIRT>);

    std::ifstream fin(file_path.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!fin)
    {
        std::cerr << "Error to load " << file_path << std::endl;
        return pointcloud;
    }

    // 读取点的个数
    std::ifstream in(file_path.c_str());
    in.seekg(0, std::ios::end);
    std::streampos size = in.tellg();
    in.close();
    uint npts = size / sizeof(double) / 6;

    // init pcl pointcloud ptr
    pointcloud->clear();
    pointcloud->width = npts;
    pointcloud->height = 1;
    pointcloud->is_dense = false;
    pointcloud->points.resize(npts);

    for (int i = 0; i < npts; i++)
    {
        double x, y, z, intensity, ring, local_timestamp;
        fin.read((char *)&x, sizeof(double));
        fin.read((char *)&y, sizeof(double));
        fin.read((char *)&z, sizeof(double));
        fin.read((char *)&intensity, sizeof(double));
        fin.read((char *)&ring, sizeof(double));
        fin.read((char *)&local_timestamp, sizeof(double));
        local_timestamp /= 1000.0; // not very reliable
        pointcloud->points[i].x = (float)x;
        pointcloud->points[i].y = (float)y;
        pointcloud->points[i].z = (float)z;
        pointcloud->points[i].intensity = (float)intensity;
        pointcloud->points[i].ring = (int)ring;
        pointcloud->points[i].time = local_timestamp;
    }
    return pointcloud;
}


pcl::PointCloud<PointXYZIRT>::Ptr autoDatasetReader::readLiDAR_calculate_pointtime(const std::string &file_path)
{
    const int len = file_path.size();
    std::string timestamp_str = file_path.substr(len - 17, 13);
    std::stringstream stream;
    stream << timestamp_str;
    double timestamp;
    stream >> timestamp;
    timestamp /= 1000.0;

    pcl::PointCloud<PointXYZIRT>::Ptr pointcloud(new pcl::PointCloud<PointXYZIRT>);

    std::ifstream fin(file_path.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!fin)
    {
        std::cerr << "Error to load " << file_path << std::endl;
        return pointcloud;
    }

    if (!(N_SCAN == 128 || N_SCAN == 64)){
        std::cerr<< "unsupported scan number: " << N_SCAN << std::endl;
        return pointcloud;
    }

    // 读取点的个数
    std::ifstream in(file_path.c_str());
    in.seekg(0, std::ios::end);
    std::streampos size = in.tellg();
    in.close();
    uint npts = size / sizeof(double) / 6;

    // init pcl pointcloud ptr
    pointcloud->clear();
    pointcloud->width = npts;
    pointcloud->height = 1;
    pointcloud->is_dense = false;
    pointcloud->points.resize(npts);

    int column_index = 0;
    int last_ring_id = -1;
    float relTime = 0;
    bool relTime_normal = true;
    for (int i = 0; i < npts; i++)
    {
        double x, y, z, intensity, ring, local_timestamp;
        fin.read((char *)&x, sizeof(double));
        fin.read((char *)&y, sizeof(double));
        fin.read((char *)&z, sizeof(double));
        fin.read((char *)&intensity, sizeof(double));
        fin.read((char *)&ring, sizeof(double));
        fin.read((char *)&local_timestamp, sizeof(double));
        local_timestamp /= 1000.0; // not very reliable
        pointcloud->points[i].x = (float)x;
        pointcloud->points[i].y = (float)y;
        pointcloud->points[i].z = (float)z;
        pointcloud->points[i].intensity = (float)intensity;
        pointcloud->points[i].ring = (int)ring;
        if (pointcloud->points[i].ring > last_ring_id){
            last_ring_id = pointcloud->points[i].ring;
        }
        else{
            column_index += 1;
            last_ring_id = -1;
        }
        relTime = float(column_index) / Horizon_SCAN * 0.1;
        if (relTime < 0){
            relTime_normal = false;
            //std::cout << "relTime = " << relTime << "corrected to 0" << std::endl;
            relTime = 0;
        }
        else if(relTime > 0.1){
            relTime_normal = false;
            //std::cout << "relTime = " << relTime << "corrected to 0.1" << std::endl;
            relTime = 0.1;
        }
        pointcloud->points[i].time = timestamp + relTime;
    }
    if(!relTime_normal){
        std::cerr << "WARNING: calucation of lidar point time abnormal: " << file_path << std::endl;
    }
    return pointcloud;
}

double autoDatasetReader::get_GPS(const int &frame_idx, json &return_ub482)
{
    long ub482_timestamp = attribute["unsync"]["UB482"][frame_idx];
    std::string ub482_path = data_dir + "/UB482/" + std::to_string(ub482_timestamp) + ".json";
    return_ub482 = readJSON(ub482_path);
    return (double)ub482_timestamp / 1000.0;
}

double autoDatasetReader::get_LiDAR_IMU_GPS(const int &frame_idx, pcl::PointCloud<PointXYZIRT>::Ptr &return_pointcloud_ptr, std::vector<json> &return_imu, std::vector<json> &return_ub482, const int &step)
{
    /*
    double timestamp_sec = get_LiDAR_IMU(frame_idx, return_pointcloud_ptr, return_imu, step);
    long next_lidar_timestamp = attribute["sync"]["lidar_top"][frame_idx + step];

    std::vector<long> ub482_timestamps = attribute["unsync"]["UB482"];
    int ub482_index = binary_search_find_index(ub482_timestamps, next_lidar_timestamp);
    if (ub482_index >= 0)
    {
        get_GPS(ub482_index, return_ub482);
    }
    
    return timestamp_sec;
    */

    long current_lidar_timestamp = lidar_timeStamp[frame_idx];
    long next_lidar_timestamp = lidar_timeStamp[frame_idx + step];
    
    std::string current_lidar_frame = std::to_string(current_lidar_timestamp);

    std::string lidar_path = data_dir + "/lidar_top/" + current_lidar_frame + ".bin";
    return_pointcloud_ptr = readLiDAR_calculate_pointtime(lidar_path);
    
    FilterPointCloudByDistance(return_pointcloud_ptr, lidarMinRange, lidarMaxRange, lidarMinZ);
    
    // if(dynamicRemoveMode==2){
    //     FilterPointCloudByLiDARDet(current_lidar_frame, return_pointcloud_ptr);
    // }
    // else if(dynamicRemoveMode==1){
    //     FilterPointCloudBySemanticImage(return_pointcloud_ptr, frame_idx);
    // }

    return_imu.clear();
    return_ub482.clear();

    int imu_start_index = binary_search_find_index(imu_timestamps, current_lidar_timestamp)-1;
    int imu_end_index = binary_search_find_index(imu_timestamps, next_lidar_timestamp)+1;

    int gps_start_index = binary_search_find_index(gps_timeStamp, current_lidar_timestamp)-1;
    int gps_end_index = binary_search_find_index(gps_timeStamp, next_lidar_timestamp)+1;
    // make sure index is with vector size
    if(imu_start_index <0){
        imu_start_index = 0;
    }
    if(imu_end_index >=imu_timestamps.size()){
        imu_end_index = imu_timestamps.size();
    }

    gps_start_index = gps_start_index < 0 ? 0 : gps_start_index;
    gps_end_index = gps_end_index >= gps_timeStamp.size() ? gps_timeStamp.size():gps_end_index;

    for(int imu_idx=imu_start_index; imu_idx<imu_end_index; imu_idx++){
        return_imu.push_back(imu_json_frames[imu_idx]);
    }

    for(int gps_idx=gps_start_index; gps_idx<gps_end_index; gps_idx++){
        std::string ub482_path = data_dir + "/UB482/" + std::to_string(gps_timeStamp[gps_idx]) + ".json";
        return_ub482.push_back(readJSON(ub482_path));
    }

    std::cout << "get " << return_imu.size() << " imu  " ;
    std::cout << "get " << return_ub482.size() << " gps  " ;
    return (double)current_lidar_timestamp / 1000.0;

}


double autoDatasetReader::get_LiDAR_IMU(const int &frame_idx, pcl::PointCloud<PointXYZIRT>::Ptr &return_pointcloud_ptr, std::vector<json> &return_imu, const int &step)
{
    //long current_lidar_timestamp = attribute["sync"]["lidar_top"][frame_idx];
    //long next_lidar_timestamp = attribute["sync"]["lidar_top"][frame_idx + step];

    //using lidar_timeStamp
    long current_lidar_timestamp = lidar_timeStamp[frame_idx];
    long next_lidar_timestamp = lidar_timeStamp[frame_idx + step];
    
    std::string current_lidar_frame = std::to_string(current_lidar_timestamp);

    std::string lidar_path = data_dir + "/lidar_top/" + current_lidar_frame + ".bin";
    return_pointcloud_ptr = readLiDAR_calculate_pointtime(lidar_path);
    
    FilterPointCloudByDistance(return_pointcloud_ptr, lidarMinRange, lidarMaxRange, lidarMinZ);
    
    if(dynamicRemoveMode==2){
        FilterPointCloudByLiDARDet(current_lidar_frame, return_pointcloud_ptr);
    }
    else if(dynamicRemoveMode==1){
        FilterPointCloudBySemanticImage(return_pointcloud_ptr, frame_idx);
    }

    return_imu.clear();

    int imu_start_index = binary_search_find_index(imu_timestamps, current_lidar_timestamp)-1;
    int imu_end_index = binary_search_find_index(imu_timestamps, next_lidar_timestamp)+1;

    // make sure index is with vector size
    if(imu_start_index <0){
        imu_start_index = 0;
    }
    if(imu_end_index >=imu_timestamps.size()){
        imu_end_index = imu_timestamps.size();
    }

    for(int imu_idx=imu_start_index; imu_idx<imu_end_index; imu_idx++){
        return_imu.push_back(imu_json_frames[imu_idx]);
    }
    std::cout << "get " << return_imu.size() << " imu  " ;
    return (double)current_lidar_timestamp / 1000.0;
}

bool autoDatasetReader::get_extrinsic(const std::string &from_sensor, const std::string &to_sensor, Eigen::Matrix4d &output)
{
    json T = attribute["calibration"][from_sensor + "_2_" + to_sensor];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            output(i, j) = T[i][j];
        }
    }
    return true;
}

bool autoDatasetReader::get_extrinsic(
    const std::string &from_sensor,
    const std::string &to_sensor,
    Eigen::Matrix3d &output_R,
    Eigen::Vector3d &output_t)
{
    json T = attribute["calibration"][from_sensor + "_2_" + to_sensor];
    for (int i = 0; i < 3; i++)
    {
        output_t(i) = T[i][3];
        for (int j = 0; j < 3; j++)
        {
            output_R(i, j) = T[i][j];
        }
    }
    return true;
}

bool autoDatasetReader::Transform2RotT(const Eigen::Matrix4d &input, Eigen::Matrix3d &output_R, Eigen::Vector3d &output_t)
{
    output_R = input.topLeftCorner(3, 3);
    output_t = input.topRightCorner(3, 1);
}

OdomMsg autoDatasetReader::gpsConverter(const OdomMsg &gps_in)
{
    OdomMsg gps_out = gps_in;
    // position
    gps_out.position = gps_in.position + extTransGNSS2LiDAR;
    // orientation
    Eigen::Matrix3d gps_in_orientation_mat = gps_in.orientation.toRotationMatrix();
    Eigen::Matrix3d gps_out_orientation_mat = extRotGNSS2LiDAR.inverse() * gps_in_orientation_mat;
    gps_out.orientation = Eigen::Quaterniond(gps_out_orientation_mat);
    return gps_out;
}

ImuMsg autoDatasetReader::imuConverter(const ImuMsg &imu_in)
{
    ImuMsg imu_out = imu_in;
    // rotate acceleration
    imu_out.acc = extRotIMU2LiDAR * imu_in.acc;
    // rotate gyroscope
    imu_out.gyr = extRotIMU2LiDAR * imu_in.gyr;
    // prepare quaternion
    Eigen::Matrix3d imu_in_orientation_mat = imu_in.orientation.toRotationMatrix();
    Eigen::Matrix3d imu_out_orientation_mat = extRPYIMU2LiDA * imu_in_orientation_mat;

    // rotate roll pitch yaw
    imu_out.orientation = Eigen::Quaterniond(imu_out_orientation_mat);
    return imu_out;
}

cv::Mat autoDatasetReader::get_semantic(const std::string &camera, const int &frame_idx)
{
    long camera_timestamp = attribute["sync"][camera][frame_idx];
    std::string camera_frame = std::to_string(camera_timestamp);
    std::string semantic_path = data_dir + "/seg_" + camera + "/" + camera_frame + ".png";
    cv::Mat label_img = cv::imread(semantic_path, cv::IMREAD_UNCHANGED);
    if (label_img.empty())
    {
        std::cout << "Warning: failed to load " << label_img << std::endl;
    }
    else
    {
        // here we assume segmentic label is 1/2 of original image size
        cv::resize(label_img, label_img, cv::Size(0, 0), 2, 2, 0);
    }
    return label_img;
}

bool autoDatasetReader::get_intrinsic(const std::string &camera, cv::Mat &K, cv::Mat &d)
{
    json K_json = attribute["calibration"][camera]["K"];
    json d_json = attribute["calibration"][camera]["d"];

    K = cv::Mat::eye(3, 3, CV_32FC1);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            K.at<float>(i, j) = (float)K_json[i][j];
        }
    }
    d = cv::Mat(d_json.size(), 1, CV_32FC1);
    for (int i = 0; i < d_json.size(); i++)
    {
        d.at<float>(i) = d_json[i];
    }
    return true;
}


bool autoDatasetReader::get_extrinsic(const std::string &from_sensor, const std::string &to_sensor, cv::Mat& R, cv::Mat&t){
    json T = attribute["calibration"][from_sensor + "_2_" + to_sensor];
    R = cv::Mat::eye(3, 3, CV_32FC1);
    t = cv::Mat(3,1, CV_32FC1);;
    for(int i=0; i<3; i++){
        t.at<float>(i) = T[i][3];
        for(int j=0; j<3;j++){
            R.at<float>(i,j) = T[i][j];
        }
    }
    return true;
}

void autoDatasetReader::FilterPointCloudByLiDARDet(const std::string& lidar_timestamp, pcl::PointCloud<PointXYZIRT>::Ptr& pointclouds){
    int num_points_before = pointclouds->points.size();

    std::vector<OBBox> bboxes = ReadBoxPerFrame(lidar_timestamp);

    FilterPointCLoudByBox(bboxes, pointclouds);
    std::cout << "Filtered by Box: [remain/original] = ["
              << pointclouds->points.size() << "/"
              << num_points_before << "] " << std::endl;
}

std::vector<Eigen::Vector3f> autoDatasetReader::OBBox2EigenPoints(const OBBox& bbox){
    static float sqrt2 = 0.7;
    std::vector<Eigen::Vector3f> box_point;
    Eigen::AngleAxisf box_alx(-bbox.yaw_, Eigen::Vector3f::UnitZ());
    Eigen::Matrix3f boxR = box_alx.toRotationMatrix();
    box_point.resize(8);
    box_point[0] = Eigen::Vector3f(-bbox.size_.x()*sqrt2,-bbox.size_.y()*sqrt2,-bbox.size_.z()*0.5);
    box_point[1] = Eigen::Vector3f( bbox.size_.x()*sqrt2,-bbox.size_.y()*sqrt2,-bbox.size_.z()*0.5);
    box_point[2] = Eigen::Vector3f( bbox.size_.x()*sqrt2, bbox.size_.y()*sqrt2,-bbox.size_.z()*0.5);
    box_point[3] = Eigen::Vector3f(-bbox.size_.x()*sqrt2, bbox.size_.y()*sqrt2,-bbox.size_.z()*0.5);
    box_point[4] = Eigen::Vector3f(-bbox.size_.x()*sqrt2,-bbox.size_.y()*sqrt2, bbox.size_.z()*0.75);
    box_point[5] = Eigen::Vector3f( bbox.size_.x()*sqrt2,-bbox.size_.y()*sqrt2, bbox.size_.z()*0.75);
    box_point[6] = Eigen::Vector3f( bbox.size_.x()*sqrt2, bbox.size_.y()*sqrt2, bbox.size_.z()*0.75);
    box_point[7] = Eigen::Vector3f(-bbox.size_.x()*sqrt2, bbox.size_.y()*sqrt2, bbox.size_.z()*0.75);
    
    for(int j=0;j<8;++j)
    {
        box_point[j] = boxR * box_point[j] + bbox.center_; 
    }
    return box_point;
}


bool autoDatasetReader::IsInbox(PointXYZIRT pt, std::vector<Eigen::Vector3f> box_point)
{
    Eigen::Vector3f pointO(pt.x, pt.y, pt.z);
    Eigen::Vector3f AB = box_point[1] - box_point[0];
    Eigen::Vector3f BC = box_point[2] - box_point[1];
    Eigen::Vector3f OA = box_point[0] - pointO;
    Eigen::Vector3f OB = box_point[1] - pointO;
    Eigen::Vector3f OC = box_point[2] - pointO;
    float cosAOB = OA.dot(AB) * OB.dot(AB);
    float cosBOC = OB.dot(BC) * OC.dot(BC);
    return cosAOB<0 && cosBOC<0 && pt.z>box_point[0](2) && pt.z<box_point[6](2);
}


void autoDatasetReader::FilterPointCLoudByBox(const std::vector<OBBox>& boxes, pcl::PointCloud<PointXYZIRT>::Ptr& pointclouds){
    if (boxes.size()==0){
        return;
    }
    // prepare box edges for inside-box-judging for static objects
    std::vector<std::vector<Eigen::Vector3f>> all_box_edges;
    for(OBBox box: boxes){
        OBBox loose_box = box;
        // move box center up such that box bottom remain unchanged after box dilation
        // this will filter out as much dynamic points as possible while keep ground points
        loose_box.center_.z() += (loose_box.size_.z() * (boxSizeDilate-1));
        loose_box.size_ *= boxSizeDilate;
        all_box_edges.push_back(OBBox2EigenPoints(loose_box));
    }
    pcl::PointCloud<PointXYZIRT>::Ptr pointcloud_filtered(new pcl::PointCloud<PointXYZIRT>);
    // loop over every point in pointcloud
    for(int i=0; i<pointclouds->points.size(); i++){
        PointXYZIRT point = pointclouds->points[i];
        bool is_in_box = false;
        // for each point, loop over every box and check if the point is in that box
        // judge if the point is in the loose box first
        for(int j=0; j<all_box_edges.size(); j++){
            // if the point in current box,
            // it can not belong to other box, thus break the loop
            if(IsInbox(point, all_box_edges[j])){
                is_in_box = true;
                break;
            }
        }
        // if the point does not belong to any box
        // save to static pointcloud
        if (!is_in_box){
            pointcloud_filtered->points.push_back(point);
        }
    }
    pointclouds.swap(pointcloud_filtered);
    pointcloud_filtered.reset(new pcl::PointCloud<PointXYZIRT>());
}

std::vector<OBBox> autoDatasetReader::ReadBoxPerFrame(const std::string& lidar_time_str){
    std::vector<OBBox> boxes_per_frame;
    json boxes_per_frame_json;
    if (lidar_det_json.contains(lidar_time_str)){
        boxes_per_frame_json = lidar_det_json[lidar_time_str];
    }
    else{
        std::cout << "label does not exist for " << lidar_time_str << std::endl;
        return boxes_per_frame;
    }
    std::vector<int> label_pred = boxes_per_frame_json["label_preds"];
    std::vector<float> scores = boxes_per_frame_json["scores"];
    std::vector<std::vector<float>> coordinate = boxes_per_frame_json["box3d_lidar"];

    if(label_pred.size() != scores.size() || scores.size() != coordinate.size() || coordinate.size() != label_pred.size())
        return boxes_per_frame;

    

    // for(auto box_json:boxes_per_frame_json){
    //     std::string box_label = box_json["label"];
    //     int track_id = int(box_json["track_id"]);
    //     // make sure track id is between 0 and 10000
    //     // Here we assume that every category's track id is below 1000
    //     // Track id is encode with type id
    //     if(track_id < 0 || track_id >= 10000){
    //         std::cout << "WARNING: track id overflow, skip this box" << std::endl;
    //         continue;
    //     }
    //     int type_id = 0;
    //     if(box_label == "Car"){
    //         type_id = int(track_id);
    //     }
    //     else if(box_label == "Cyclist"){
    //         type_id = int(track_id) + 10000;
    //     }
    //     else if(box_label == "Tricycle"){
    //         type_id = int(track_id) + 20000;
    //     }
    //     else if(box_label == "Pedestrian"){
    //         type_id = int(track_id) + 30000;
    //     }
    //     else if(box_label == "Truck"){
    //         type_id = int(track_id) + 40000;
    //     }
    //     else if(box_label == "Bus"){
    //         type_id = int(track_id) + 50000;
    //     }
    //     else{
    //         type_id = -1;
    //     }
    //     if(box_json["score"] < score_threshold){
    //         type_id = -1;
    //     };
    //     // type id = -1 means this box is invalid for dynamic object clustering
    //     // but maybe useful for other purpose like dynamic points culling
    //     OBBox box;
    //     box.yaw_ = 3.14 -  float(box_json["yaw"]);
    //     box.type_id_ = type_id;
    //     Eigen::Vector3f center_(box_json["location"][0], box_json["location"][1], box_json["location"][2]);
    //     box.center_ = center_;
    //     Eigen::Vector3f size_(box_json["dimension"][0], box_json["dimension"][1], box_json["dimension"][2]);
    //     box.size_ = size_;
    //     boxes_per_frame.push_back(box);
    //}


    for(int i=0; i<scores.size(); i++){
        int box_label = label_pred[i];
        if(box_label < 0 || box_label >= 10000){
            std::cout << "WARNING: track id overflow, skip this box" << std::endl;
            continue;
        }
        if((float)scores[i] < score_threshold)
            continue;
        // type id = -1 means this box is invalid for dynamic object clustering
        // but maybe useful for other purpose like dynamic points culling
        OBBox box;
        box.yaw_ = 3.14 - coordinate[i][6];
        Eigen::Vector3f center_(coordinate[i][0], coordinate[i][1], coordinate[i][2]);
        box.center_ = center_;
        Eigen::Vector3f size_(coordinate[i][3], coordinate[i][4], coordinate[i][5]);
        box.size_ = size_;
        boxes_per_frame.push_back(box);
    }
    
    return boxes_per_frame;
}

void autoDatasetReader::FilterPointCloudBySemanticImage(pcl::PointCloud<PointXYZIRT>::Ptr& pointclouds, const int &frame_idx)
{
    int num_semantics = 0;
    int num_points_before = pointclouds->points.size();
    for (int i = 0; i < num_camera; i++)
    {
        std::string camera = CAMERA_NAME[i];
        cv::Mat label_img = this->get_semantic(camera, frame_idx);
        if (label_img.empty())
            continue;
        else
            num_semantics++;

        cv::Mat K, d;
        this->get_intrinsic(camera, K, d);
        cv::Mat R, t;
        this->get_extrinsic("lidar_top", camera, R, t);
        this->dynamics_culling(label_img, pointclouds, K, R, t, d, label_img.rows, label_img.cols);
    }
    std::cout << "load " << num_semantics << " sementic, "
              << "[remain/original] = ["
              << pointclouds->points.size() << "/"
              << num_points_before << "] ";
}

void autoDatasetReader::dynamics_culling(const cv::Mat &label_img,
                                         pcl::PointCloud<PointXYZIRT>::Ptr cloud_in,
                                         const cv::Mat &K,
                                         const cv::Mat &R,
                                         const cv::Mat &t,
                                         const cv::Mat &dist_coeffs,
                                         const int &imgH,
                                         const int &imgW)
{

    std::vector<cv::Point3f> points3D; // points in lidar
    std::vector<cv::Point2f> points2D; // projected coordinates
    for (int i = 0; i < cloud_in->points.size(); i++)
    {
        cv::Mat pt_3d(3, 1, CV_32FC1);
        pt_3d.at<float>(0) = cloud_in->points[i].x;
        pt_3d.at<float>(1) = cloud_in->points[i].y;
        pt_3d.at<float>(2) = cloud_in->points[i].z;

        points3D.push_back(cv::Point3f(pt_3d.at<float>(0),
                                       pt_3d.at<float>(1),
                                       pt_3d.at<float>(2)));
    }

    if (points3D.size() == 0)
    {
        std::cerr << "Error: empty points after culling!" << std::endl;
    }

    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    cv::Mat tvec = t;
    cv::projectPoints(points3D,
                      rvec,
                      tvec,
                      K,
                      dist_coeffs,
                      points2D);

    // mask background image
    cv::Mat foreground(imgH, imgW, CV_8UC1);
    foreground.setTo(0);
    for (int i = 0; i < imgH; i++)
    {
        for (int j = 0; j < imgW; j++)
        {

            int label = -1;
            if (label_img.channels() == 1)
            {
                label = label_img.at<uchar>(i, j);
            }
            else if (label_img.channels() == 3)
            {
                label = label_img.at<cv::Vec3b>(i, j)[0];
            }

            if (label == 9      // persion
                || label == 10  // rider
                || label == 11  // bicycle
                || label == 12  // moto bicycle
                || label == 13  // tricycle
                || label == 14  // car
                || label == 15  // truck
                || label == 16  // bus
                || label == 17) //train
            {
                foreground.at<uchar>(i, j) = 255;
            }
        }
    }
    cv::Mat kernel = cv::getStructuringElement(0 /*Rect*/, cv::Size(15, 15));
    cv::dilate(foreground, foreground, kernel, cv::Point2i(-1, -1), 2);

    std::vector<cv::Point3f> points3D_filtered;
    std::vector<float> intensity;
    std::vector<std::uint16_t> ring;
    std::vector<double> timestamp;

    for (int i = 0; i < points2D.size(); i++)
    {
        int x = int(points2D[i].x + 0.5);
        int y = int(points2D[i].y + 0.5);

        // if points fall outside of image, save for other camera
        if (x < 0 || x > imgW || y < 0 || y > imgH)
        {
            points3D_filtered.push_back(points3D[i]);
            intensity.push_back(cloud_in->points[i].intensity);
            ring.push_back(cloud_in->points[i].ring);
            timestamp.push_back(cloud_in->points[i].time);
            continue;
        };
        int flag = foreground.at<uchar>(y, x);
        // flag = 255 if is foreground, flag = 0 if background
        if (flag < 1)
        {
            points3D_filtered.push_back(points3D[i]);
            intensity.push_back(cloud_in->points[i].intensity);
            ring.push_back(cloud_in->points[i].ring);
            timestamp.push_back(cloud_in->points[i].time);
        }
    }
    points3D.swap(points3D_filtered);

    cloud_in->points.clear();
    for (int i = 0; i < points3D.size(); i++)
    {
        PointXYZIRT pt;
        pt.x = points3D[i].x;
        pt.y = points3D[i].y;
        pt.z = points3D[i].z;
        pt.intensity = intensity[i];
        pt.ring = ring[i];
        pt.time = timestamp[i];
        cloud_in->points.push_back(pt);
    }
    cloud_in->height = 1;
    cloud_in->width = cloud_in->points.size();
}

bool autoDatasetReader::read_TimeStamp(const char* dir_name, std::vector<long>& files)
{
	if( NULL == dir_name )
	{
		cout<<" dir_name is null ! "<<endl;
		return false;
	}
 
	// check if dir_name is a valid dir
	struct stat s;
	lstat( dir_name , &s );
	if( ! S_ISDIR( s.st_mode ) )
	{
		std::cout<<"dir_name is not a valid directory !"<<std::endl;
		return false;
	}
	
	struct dirent * filename;    // return value for readdir()
 	DIR * dir;                   // return value for opendir()
	dir = opendir( dir_name );

	
	/* read all the files in the dir ~ */
	while( ( filename = readdir(dir) ) != NULL )
	{
		// get rid of "." and ".."
		if( strcmp( filename->d_name , "." ) == 0 || 
			strcmp( filename->d_name , "..") == 0    )
			continue;
		std::string file_name = std::string(filename->d_name);
        long time = stol( file_name.substr(0, file_name.find('.')) );
        files.push_back(time);
	}
    sort(files.begin(), files.end());
    return true;
}
