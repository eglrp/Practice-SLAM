#ifndef BASICVO_VISUALODOMETRY_H
#define BASICVO_VISUALODOMETRY_H

#include <vector>
#include <opencv2/core.hpp>
#include "PinholeCamera.h"

// 简单视觉里程计类定义
class VisualOdometry
{
public:
    // 定义帧类型枚举
    enum FrameStage
    {
        STAGE_FIRST_FRAME,  // 第一帧
        STAGE_SECOND_FRAME, // 第二帧
        STAGE_DEFAULT_FRAME // 第三帧
    };

    VisualOdometry(PinholeCamera *camera);

    ~VisualOdometry();

    void AddImage(const cv::Mat &frame, int frame_id);

    void SetGroundTruthPath(std::string ground_truth_path);

    cv::Mat getCurrentR() { return cur_R_; }

    cv::Mat getCurrentT() { return cur_t_; }

private:
    bool processFirstFrame();

    bool processSecondFrame();

    bool processFrame(int frame_id);

    double getAbsoluteScale(int frame_id);

    void featureDetection(cv::Mat &image, std::vector<cv::Point2f> &keypoints);

    void featureTracking(cv::Mat &image_previous, cv::Mat &image_current, std::vector<cv::Point2f> &keyPoint_previous,
                         std::vector<cv::Point2f> &keyPoint_current, std::vector<double> &disparities);

private:
    FrameStage frameStage_;
    PinholeCamera *pcamera_;
    cv::Mat current_frame_;
    cv::Mat previous_frame_;

    cv::Mat cur_R_;
    cv::Mat cur_t_;

    std::vector<cv::Point2f> px_current_;
    std::vector<cv::Point2f> px_previous_;
    std::vector<double> disparities_;

    double focal_;
    cv::Point2d pp_;

    std::string ground_truth_path_;
};


#endif //BASICVO_VISUALODOMETRY_H
