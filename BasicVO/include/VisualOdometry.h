#ifndef BASICVO_VISUALODOMETRY_H
#define BASICVO_VISUALODOMETRY_H

#include <vector>
#include <opencv2/core.hpp>
#include "PinholeCamera.h"

using namespace cv;
using namespace std;

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

    void addImage(const Mat &img, int frame_id);

    Mat getCurrentR() { return cur_R; }

    Mat getCurrentT() { return cur_t; }

private:
    bool processFirstFrame();

    bool processSecondFrame();

    bool processFrame(int frame_id);

    double getAbsoluteScale(int frame_id);

    void featureDetection(Mat &image, vector<Point2f>& keypoints);

    void featureTracking(Mat &image_previous, Mat &image_current, vector<Point2f> &keyPoint_previous,
                         vector<Point2f> &keyPoint_current, vector<double>& disparities);

private:
    FrameStage frameStage_;
    PinholeCamera *pcamera_;
    Mat current_frame_;
    Mat previous_frame_;

    Mat cur_R;
    Mat cur_t;

    vector<Point2f> px_current_;
    vector<Point2f> px_previous_;
    vector<double> disparities;

    double focal;
    Point2d pp_;

};


#endif //BASICVO_VISUALODOMETRY_H
