#include "utility.h"
#include "featureExtraction.h"
#include "imageProjection.h"
#include "imuPreintegration.h"
#include "mapOptmization.h"
#include "autoDatasetReader.h"
#include <chrono>

void run_one_frame(autoDatasetReader &adr,
                   FeatureExtraction &fe,
                   ImageProjection &ip,
                   IMUPreintegration &imu_pre,
                   mapOptimization &mo,
                   std::ofstream &lidar_odom_writer,
                   std::ofstream &lidar_imu_writer,
                   std::string &lio_sam_global_dir,
                   const std::string& deskewed_dir,
                   const int &i)
{
    std::cout << "LIO-SAM-OFFLINE: [" << i << "/" << adr.num_lidar << "] ";
    pcl::PointCloud<PointXYZIRT>::Ptr pointcloud_ptr;
    std::vector<json> imu_messages;
    std::vector<json> gps_message;
    double timestamp = adr.get_LiDAR_IMU(i, pointcloud_ptr, imu_messages);

    for (json imu_json : imu_messages)
    {
        ImuMsg imu_msg = json2imu(imu_json);
        ImuMsg imu_msg_lidar_frame = adr.imuConverter(imu_msg);

        ip.imuHandler(imu_msg_lidar_frame);
        OdomMsg imu_odom;
        bool success = imu_pre.imuHandler(imu_msg_lidar_frame, imu_odom);
        if (success)
        {
            ip.odometryHandler(imu_odom);
            char lidar_imu_msg[255];
            odom2char(imu_odom, lidar_imu_msg);
            lidar_imu_writer << lidar_imu_msg;
        }
    }
    for(json gps_jason : gps_message)
    {
        OdomMsg gps_msg = json2gps(gps_jason);
        OdomMsg gps_msg_lidar_frame = adr.gpsConverter(gps_msg);
        if( mo.gpsQueue.empty() || gps_msg_lidar_frame.timestamp > mo.gpsQueue.back().timestamp)
            mo.gpsHandler(gps_msg_lidar_frame);
    }

    // image projection
    cloud_info cloud_info_deskewed = ip.cloudHandler(pointcloud_ptr);
    cloud_info_deskewed.timestamp = timestamp;
    ip.resetParameters();

    if(!deskewed_dir.empty() && cloud_info_deskewed.cloud_deskewed.points.size() > 0){
        std::string timestamp_str = std::to_string(long(timestamp * 1000.0));
        std::string deskewed_path = deskewed_dir + "/" + timestamp_str + ".pcd";
        pcl::io::savePCDFileBinary(deskewed_path, cloud_info_deskewed.cloud_deskewed);
    }
    // std::string origin_path = "/mnt/alpha/jiaxin02.zhang/Code/lio_sam_offline/tmp/" + timestamp_str + "_origin.ply";
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr rendered_pc_ptr = colorizeCloud(*pointcloud_ptr);
    // pcl::io::savePLYFileBinary(origin_path, *rendered_pc_ptr);

    // feature extraction
    cloud_info cloud_info_featured;
    fe.laserCloudInfoHandler(cloud_info_deskewed, cloud_info_featured);


    // mapping
    mo.laserCloudInfoHandler(cloud_info_featured);
    OdomMsg lidar_odom = mo.publishOdometry();
    imu_pre.odometryHandler(lidar_odom);

    // write path to odometry/
    if(cloud_info_deskewed.cloud_deskewed.points.size() > 0){
        char lidar_odom_msg[255];
        odom2char(lidar_odom, lidar_odom_msg);
        lidar_odom_writer << lidar_odom_msg;
    }

}

std::string getfoldername(const std::string& filepath)
{
    std::string foldername;
    if(filepath.empty())
    {
        return "";
    }

    int index = filepath.size() - 1;
    if(filepath[index] == '/')
        index--;
    
    for(int i=index; i>=0; i--)
    {
        if(filepath[i] == '/')
            break;
        foldername.push_back(filepath[i]);
    }

    reverse(foldername.begin(), foldername.end());
    return foldername;
}

int main(int argc, char *argv[])
{

    if (argc < 5)
    {
        std::cout << "Usage: ./liosam config_path clip_path save_deskewed save_map_resolution(-1 to disable)" << std::endl;
        return -1;
    }

    // load config from command line
    std::string param_path = std::string(argv[1]);
    std::string clip_path = std::string(argv[2]);
    const int save_deskewed = atoi(argv[3]);
    const float save_map_resolution = atof(argv[4]);
    std::string foldername = getfoldername(clip_path);

    std::cout << "LIO-SAM-OFFLINE: running for = " << clip_path << std::endl;
    // initialize class with config yaml and data dir
    autoDatasetReader adr(clip_path, param_path);
    FeatureExtraction fe(param_path);
    ImageProjection ip(param_path);
    IMUPreintegration imu_pre(param_path, adr.extTransIMU2LiDAR);
    mapOptimization mo(param_path);

    // prepare output dir
    std::string outputstr = "result/" + foldername;
    std::string lio_sam_global_dir = "result/" + foldername + "/odometry";
    std::string command = "mkdir -p " + lio_sam_global_dir;
    int success = system(command.c_str());
    std::string deskewed_dir = "";
    if (save_deskewed){
        deskewed_dir = outputstr + "/liosam_deskewed_lidar_top";
        command = "mkdir -p " + deskewed_dir;
        int success = system(command.c_str());
    }
    // prepare output file writer
    std::string global_maps_path = lio_sam_global_dir + "/liosam_offline_map.pcd";
    std::ofstream lidar_odom_writer(lio_sam_global_dir + "/liosam_offline_10HZ.txt");
    std::ofstream lidar_imu_writer(lio_sam_global_dir + "/liosam_offline_100HZ.txt");
    std::cout << "finish perparing running LIO_SAM" << std::endl;
    
    // skip some head and tail frames for data stability
    for (int i = 2; i < adr.num_lidar - 3; i++){
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        run_one_frame(adr, fe, ip, imu_pre, mo,
                        lidar_odom_writer,
                        lidar_imu_writer,
                        lio_sam_global_dir,
                        deskewed_dir,
                        i);
        
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double us_per_frame = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        double fps = 1e6/us_per_frame;
        std::cout<< std::setprecision(2) << std::fixed << "FPS = " << fps << std::endl;
        // if(i % 100 == 0){
        //     std::cout << "LIO-SAM-OFFLINE saving global maps to " << global_maps_path << std::endl;
        //     pcl::PointCloud<PointType>::Ptr global_maps = mo.publishGlobalMap();
        //     if (save_map_resolution >0){
        //         OctreeDownSample(global_maps, save_map_resolution);
        //     }
        //     pcl::io::savePCDFileBinary(global_maps_path, *global_maps);
        // }
    }

    // save optimized global keyframe path
    std::ofstream lidar_path_writer(lio_sam_global_dir + "/liosam_offline_keyframe_10HZ.txt");
    std::vector<OdomMsg> global_paths;
    mo.publishGlobalPose(global_paths);
    for(OdomMsg global_path: global_paths){
        char lidar_global_char[255];
        odom2char(global_path, lidar_global_char);
        lidar_path_writer << lidar_global_char;
    }

    // save whole global maps into a single pcd file
    if(save_map_resolution >= 0){
        std::cout << "LIO-SAM-OFFLINE saving global maps to " << global_maps_path << std::endl;
        pcl::PointCloud<PointType>::Ptr global_maps = mo.publishGlobalMap();
        // if save_map_resolution set to 0, skip downsample
        if (save_map_resolution >0){
            std::cout << "downsample with resolution " << save_map_resolution << std::endl;
            OctreeDownSample(global_maps, save_map_resolution);
        }
        pcl::io::savePCDFileBinary(global_maps_path, *global_maps);
        std::cout << "LIO-SAM-OFFLINE complete!" << std::endl;
    }

    return 0;
}