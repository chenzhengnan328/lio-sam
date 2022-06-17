//
// Created by suiwei on 20-9-15.
//

#ifndef TIME_POSE_H
#define TIME_POSE_H


#include <Eigen/Dense>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>

class TimestampedPose{

public:
    TimestampedPose(){
    }
    TimestampedPose(const double & timestamp, const Eigen::Quaterniond & q, const Eigen::Vector3d & t): timestamp_(timestamp), q_(q), t_(t){}

    TimestampedPose (const TimestampedPose & pose){
        *this= pose;
    }

    TimestampedPose (const gtsam::Pose3& pose3){
        Eigen::Vector3d t(pose3.translation());
        Eigen::Quaterniond q(pose3.rotation().toQuaternion().w(),
                             pose3.rotation().toQuaternion().x(),
                             pose3.rotation().toQuaternion().y(),
                             pose3.rotation().toQuaternion().z());
        timestamp_ = 0;
        q_ = q;
        t_ = t;
    }

    TimestampedPose (const double& time, const gtsam::Pose3& pose3){
        Eigen::Vector3d t(pose3.translation());
        Eigen::Quaterniond q(pose3.rotation().toQuaternion().w(),
                             pose3.rotation().toQuaternion().x(),
                             pose3.rotation().toQuaternion().y(),
                             pose3.rotation().toQuaternion().z());
        timestamp_ = time;
        q_ = q;
        t_ = t;
    }

    TimestampedPose &operator=(const TimestampedPose& pose){
        if(this != &pose){
            q_ = pose.q_;
            t_ = pose.t_;
            timestamp_ = pose.timestamp_;
        }
        return  *this;
    }

    friend std::ostream & operator << (std::ostream &out, const TimestampedPose &tp){
        out << tp.timestamp_ << " " << tp.t_.x() << " " << tp.t_.y() << " " << tp.t_.z() << " "
            << tp.q_.x() << " " << tp.q_.y() << " " << tp.q_.z() << " " << tp.q_.w();
    }

    Eigen::Matrix4d toMat(){

        Eigen::Matrix4d mat = Eigen::Matrix4d::Identity(4, 4);

        Eigen::Matrix3d rot = q_.toRotationMatrix();
        mat.block(0, 0, 3, 3) = rot;
        mat.block(0, 3, 3, 1) = t_;

        return mat;
    }

    gtsam::Pose3 toGTSAM(){
        gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(q_.w(), q_.x(), q_.y(), q_.z());
        gtsam::Point3 position = gtsam::Point3(t_);

        return gtsam::Pose3(rotation, position);
    }


    const Eigen::Matrix4d toMat() const {

        Eigen::Matrix4d mat = Eigen::Matrix4d::Identity(4, 4);

        Eigen::Matrix3d rot = q_.toRotationMatrix();
        mat.block(0, 0, 3, 3) = rot;
        mat.block(0, 3, 3, 1) = t_;

        return mat;
    }

    Eigen::Quaterniond q_;
    Eigen::Vector3d t_;
    double timestamp_;
};

TimestampedPose interploate(const TimestampedPose & pose1, const TimestampedPose & pose2, float ratio);

std::vector<TimestampedPose> loadPoses(const std::string & pose_file);

#endif //TIME_POSE_H
