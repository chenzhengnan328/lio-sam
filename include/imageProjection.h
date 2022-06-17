#include "utility.h"


const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:
    std::deque<pcl::PointCloud<PointXYZIRT>> cloudQueue;
    std::deque<ImuMsg> imuQueue;
    std::deque<OdomMsg> odomQueue;


    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;

    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;


public:
    ImageProjection(const std::string& param_path);

    void allocateMemory();

    void resetParameters();

    void imuHandler(const ImuMsg& imuMsg);

    void odometryHandler(const OdomMsg& odometryMsg);

    cloud_info cloudHandler(const pcl::PointCloud<PointXYZIRT>::Ptr& laserCloudIn);

    bool cachePointCloud(const pcl::PointCloud<PointXYZIRT>::Ptr& laserCloudIn);

    bool deskewInfo();

    void imuDeskewInfo();

    void odomDeskewInfo();

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur);

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur);

    PointType deskewPoint(PointType *point, double pointTime);

    void projectPointCloud();

    void cloudExtraction();

    void publishClouds();
    
};