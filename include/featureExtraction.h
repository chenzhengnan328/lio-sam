#include "utility.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    cloud_info cloudInfo;

    FeatureExtraction(const std::string& param_path);

    void initializationValue();

    void laserCloudInfoHandler(const cloud_info& cloudIn, cloud_info& cloudOut);

    void calculateSmoothness();

    void markOccludedPoints();

    void extractFeatures();

    void freeCloudInfoMemory();

    void publishFeatureCloud();
};

