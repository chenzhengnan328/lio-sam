#include "utility.h"
#include "imageProjection.h"


ImageProjection::ImageProjection(const std::string& param_path):ParamServer(param_path)
{
    allocateMemory();
    resetParameters();

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
}

void ImageProjection::allocateMemory()
{
    laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
    fullCloud.reset(new pcl::PointCloud<PointType>());
    extractedCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN*Horizon_SCAN);

    cloudInfo.startRingIndex.assign(N_SCAN, 0);
    cloudInfo.endRingIndex.assign(N_SCAN, 0);

    cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
    cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

    resetParameters();
}

void ImageProjection::resetParameters()
{
    laserCloudIn->clear();
    extractedCloud->clear();
    // reset range matrix for range image projection
    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

    imuPointerCur = 0;
    firstPointFlag = true;
    odomDeskewFlag = false;
    deskewFlag = 1;

    for (int i = 0; i < queueLength; ++i)
    {
        imuTime[i] = 0;
        imuRotX[i] = 0;
        imuRotY[i] = 0;
        imuRotZ[i] = 0;
    }
}

void ImageProjection::imuHandler(const ImuMsg& imuMsg)
{
    ImuMsg thisImu = imuMsg;

    if(!imuQueue.empty() && imuQueue.back().timestamp >= thisImu.timestamp){
        // old reprecate IMU message
        return;
    }
    imuQueue.push_back(thisImu);

    // debug IMU data
    // cout << std::setprecision(6);
    // cout << "IMU acc: " << endl;
    // cout << "x: " << thisImu.linear_acceleration.x << 
    //       ", y: " << thisImu.linear_acceleration.y << 
    //       ", z: " << thisImu.linear_acceleration.z << endl;
    // cout << "IMU gyro: " << endl;
    // cout << "x: " << thisImu.angular_velocity.x << 
    //       ", y: " << thisImu.angular_velocity.y << 
    //       ", z: " << thisImu.angular_velocity.z << endl;
    // double imuRoll, imuPitch, imuYaw;
    // tf::Quaternion orientation;
    // tf::quaternionMsgToTF(thisImu.orientation, orientation);
    // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
    // cout << "IMU roll pitch yaw: " << endl;
    // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
}

void ImageProjection::odometryHandler(const OdomMsg& odometryMsg)
{
    odomQueue.push_back(odometryMsg);
}

cloud_info ImageProjection::cloudHandler(const pcl::PointCloud<PointXYZIRT>::Ptr& laserCloudIn)
{
    cloudInfo.cloud_deskewed.clear();
    cloudInfo.cloud_corner.clear();
    cloudInfo.cloud_surface.clear();
    if (!cachePointCloud(laserCloudIn))
        return cloudInfo;

    if (!deskewInfo())
        return cloudInfo;

    projectPointCloud();

    cloudExtraction();

    publishClouds();

    return cloudInfo;

}

bool ImageProjection::cachePointCloud(const pcl::PointCloud<PointXYZIRT>::Ptr& laserCloudInPtr)
{
    laserCloudIn = laserCloudInPtr;

    // get timestamp
    timeScanCur = laserCloudIn->points.front().time;
    timeScanEnd = laserCloudIn->points.back().time;


    return true;
}

bool ImageProjection::deskewInfo()
{

    // make sure IMU data available for the scan
    if (imuQueue.empty() || imuQueue.front().timestamp > timeScanCur || imuQueue.back().timestamp < timeScanEnd)
    {
        std::cout << "Waiting for IMU data ..." << std::endl;
        return false;
    }

    imuDeskewInfo();

    odomDeskewInfo();

    return true;
}

void ImageProjection::imuDeskewInfo()
{
    cloudInfo.imuAvailable = false;

    while (!imuQueue.empty())
    {
        if (imuQueue.front().timestamp < timeScanCur - 0.01)
            imuQueue.pop_front();
        else
            break;
    }

    if (imuQueue.empty())
        return;

    imuPointerCur = 0;

    for (int i = 0; i < (int)imuQueue.size(); ++i)
    {
        ImuMsg thisImuMsg = imuQueue[i];
        double currentImuTime = thisImuMsg.timestamp;

        // get roll, pitch, and yaw estimation for this scan
        if (currentImuTime <= timeScanCur)
            cloudInfo.imu_quaternion_init = thisImuMsg.orientation.cast<float>();

        if (currentImuTime > timeScanEnd + 0.01)
            break;

        if (imuPointerCur == 0){
            imuRotX[0] = 0;
            imuRotY[0] = 0;
            imuRotZ[0] = 0;
            imuTime[0] = currentImuTime;
            ++imuPointerCur;
            continue;
        }

        // get angular velocity
        double angular_x, angular_y, angular_z;
        imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

        // integrate rotation
        double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
        imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
        imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
        imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
        imuTime[imuPointerCur] = currentImuTime;
        ++imuPointerCur;
    }

    --imuPointerCur;

    if (imuPointerCur <= 0)
        return;

    cloudInfo.imuAvailable = true;
}

void ImageProjection::odomDeskewInfo()
{
    cloudInfo.odomAvailable = false;

    while (!odomQueue.empty())
    {
        if (odomQueue.front().timestamp < timeScanCur - 0.01)
            odomQueue.pop_front();
        else
            break;
    }

    if (odomQueue.empty())
        return;

    if (odomQueue.front().timestamp > timeScanCur)
        return;

    // get start odometry at the beinning of the scan
    OdomMsg startOdomMsg;

    for (int i = 0; i < (int)odomQueue.size(); ++i)
    {
        startOdomMsg = odomQueue[i];

        if (startOdomMsg.timestamp < timeScanCur)
            continue;
        else
            break;
    }


    Eigen::Affine3d affine_3d;
    affine_3d = startOdomMsg.orientation.toRotationMatrix();
    affine_3d.translation() = startOdomMsg.position;
    // // Initial guess used in mapOptimization
    cloudInfo.intialGuessAffine = affine_3d.cast<float>();

    cloudInfo.odomAvailable = true;

    // get end odometry at the end of the scan
    odomDeskewFlag = false;

    if (odomQueue.back().timestamp < timeScanEnd)
        return;

    OdomMsg endOdomMsg;

    for (int i = 0; i < (int)odomQueue.size(); ++i)
    {
        endOdomMsg = odomQueue[i];

        if (endOdomMsg.timestamp < timeScanEnd)
            continue;
        else
            break;
    }

    // if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
    //     return;

    Eigen::Affine3f transBegin = QuatTran2Affine(startOdomMsg.orientation, startOdomMsg.position);

    Eigen::Affine3f transEnd = QuatTran2Affine(endOdomMsg.orientation, endOdomMsg.position);

    Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

    odomIncreX = transBt.translation().x();
    odomIncreY = transBt.translation().y();
    odomIncreZ = transBt.translation().z();

    odomDeskewFlag = true;
}

void ImageProjection::findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
{
    *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

    int imuPointerFront = 0;
    while (imuPointerFront < imuPointerCur)
    {
        if (pointTime < imuTime[imuPointerFront])
            break;
        ++imuPointerFront;
    }

    if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
    {
        *rotXCur = imuRotX[imuPointerFront];
        *rotYCur = imuRotY[imuPointerFront];
        *rotZCur = imuRotZ[imuPointerFront];
    } else {
        int imuPointerBack = imuPointerFront - 1;
        double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
        *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
        *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
    }
}

void ImageProjection::findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
{
    *posXCur = 0; *posYCur = 0; *posZCur = 0;

    // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

    if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        return;

    float ratio = relTime / (timeScanEnd - timeScanCur);

    *posXCur = ratio * odomIncreX;
    *posYCur = ratio * odomIncreY;
    *posZCur = ratio * odomIncreZ;
}

PointType ImageProjection::deskewPoint(PointType *point, double pointTime)
{
    if (deskewFlag == -1 || cloudInfo.imuAvailable == false){
        return *point;
    }
    double relTime = pointTime - timeScanCur;

    float rotXCur, rotYCur, rotZCur;
    findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

    float posXCur, posYCur, posZCur;
    findPosition(relTime, &posXCur, &posYCur, &posZCur);

    if (firstPointFlag == true)
    {
        transStartInverse = EulerTran2Affine(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur).inverse();
        firstPointFlag = false;
    }

    // transform points to start
    Eigen::Affine3f transFinal = EulerTran2Affine(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
    Eigen::Affine3f transBt = transStartInverse * transFinal;

    PointType newPoint;
    newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
    newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
    newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
    newPoint.intensity = point->intensity;

    return newPoint;
}

void ImageProjection::projectPointCloud()
{
    int cloudSize = laserCloudIn->points.size();
    // range image projection
    for (int i = 0; i < cloudSize; ++i)
    {
        PointType thisPoint;
        thisPoint.x = laserCloudIn->points[i].x;
        thisPoint.y = laserCloudIn->points[i].y;
        thisPoint.z = laserCloudIn->points[i].z;
        thisPoint.intensity = laserCloudIn->points[i].intensity;

        float range = pointDistance(thisPoint);
        if (range < lidarMinRange || range > lidarMaxRange)
            continue;

        int rowIdn = laserCloudIn->points[i].ring;
        if (rowIdn < 0 || rowIdn >= N_SCAN)
            continue;

        if (rowIdn % downsampleRate != 0)
            continue;

        float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

        static float ang_res_x = 360.0/float(Horizon_SCAN);
        int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
        if (columnIdn >= Horizon_SCAN)
            columnIdn -= Horizon_SCAN;

        if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
            continue;

        if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
            continue;

        thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

        rangeMat.at<float>(rowIdn, columnIdn) = range;

        int index = columnIdn + rowIdn * Horizon_SCAN;
        fullCloud->points[index] = thisPoint;
    }
}

void ImageProjection::cloudExtraction()
{
    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < N_SCAN; ++i)
    {
        cloudInfo.startRingIndex[i] = count - 1 + 5;

        for (int j = 0; j < Horizon_SCAN; ++j)
        {
            if (rangeMat.at<float>(i,j) != FLT_MAX)
            {
                // mark the points' column index for marking occlusion later
                cloudInfo.pointColInd[count] = j;
                // save range info
                cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                // save extracted cloud
                extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                // size of extracted cloud
                ++count;
            }
        }
        cloudInfo.endRingIndex[i] = count -1 - 5;
    }
}

void ImageProjection::publishClouds()
{
    cloudInfo.cloud_deskewed = *extractedCloud;
}

