//
// Created by suiwei on 20-9-15.
//

#include "timepose.h"
#include <fstream>

TimestampedPose interploate(const TimestampedPose & pose1, const TimestampedPose & pose2, float ratio){

    TimestampedPose pose3;
    pose3.q_ = pose1.q_.slerp(ratio, pose2.q_);
    pose3.t_ = (1-ratio)*pose1.t_ + ratio * pose2.t_;
    return pose3;
}

std::vector<TimestampedPose> loadPoses(const std::string & pose_file){

    std::vector<TimestampedPose> poses;

    std::ifstream fin(pose_file);
    assert(fin.is_open());

    std::string line;
    while(getline(fin, line)){

        std::stringstream stream(line);
        double ts;
        double tx, ty, tz, qx, qy, qz, qw;
        stream>>ts>>tx>>ty>>tz>>qx>>qy>>qz>>qw;

        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Vector3d t(tx, ty, tz);

        TimestampedPose pose(ts, q, t);
        poses.push_back(pose);
    }

    return poses;
}