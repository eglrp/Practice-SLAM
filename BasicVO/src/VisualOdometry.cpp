#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include "VisualOdometry.h"
#include <string>

int kMinNumFeature = 2000;

#define ERROR_FIXED

VisualOdometry::VisualOdometry(PinholeCamera *camera) :
        pcamera_(camera),
        x_pre_(0.0),
        y_pre_(0.0),
        z_pre_(0.0)
{
    focal_ = camera->fx();
    pp_ = cv::Point2d(camera->cx(), camera->cy());
    frameStage_ = FrameStage::STAGE_FIRST_FRAME;
}

VisualOdometry::~VisualOdometry()
{
    if (this->ground_truth_stream_.is_open())
        this->ground_truth_stream_.close();
}

bool VisualOdometry::AddImage(const cv::Mat &frame)
{
    if (frame.empty() || frame.type() != CV_8UC1 || frame.rows != pcamera_->height() || frame.cols != pcamera_->width())
        throw std::runtime_error(
                "Frame: provide frame has not the same size of camera model or frame is not grayscale");
    current_frame_ = frame;

    bool res;
    switch (frameStage_)
    {
        case STAGE_DEFAULT_FRAME:
            res = processFrame();
            break;
        case STAGE_FIRST_FRAME:
            res = processFirstFrame();
            break;
        case STAGE_SECOND_FRAME :
            res = processSecondFrame();
            break;
        default:
            res = false;
            break;
    }
    previous_frame_ = current_frame_;
    return res;
}

bool VisualOdometry::processFirstFrame()
{
    featureDetection(current_frame_, px_previous_);

    GetGroundTruth();

    frameStage_ = FrameStage::STAGE_SECOND_FRAME;
    return true;
}

bool VisualOdometry::processSecondFrame()
{
    featureTracking(previous_frame_, current_frame_, px_previous_, px_current_, disparities_);

    cv::Mat E, R, T;
    cv::Mat mask;
    E = findEssentialMat(px_current_, px_previous_, focal_, pp_, cv::RANSAC, 0.999, 1.0, mask);
    recoverPose(E, px_current_, px_previous_, R, T, focal_, pp_, mask);
    cur_R_ = R.clone();
    cur_t_ = T.clone();
    frameStage_ = FrameStage::STAGE_DEFAULT_FRAME;
    px_previous_ = px_current_;

    GetGroundTruth();
    return true;
}

bool VisualOdometry::processFrame()
{
    double scale;
    featureTracking(previous_frame_, current_frame_, px_previous_, px_current_, disparities_);
    cv::Mat E, R, T, mask;

    E = cv::findEssentialMat(px_current_, px_previous_, focal_, pp_, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, px_current_, px_previous_, R, T, focal_, pp_, mask);
    scale = getAbsoluteScale();
    if (scale > 0.1)
    {
        cur_t_ = cur_t_ + scale * (cur_R_ * T);
        cur_R_ = R * cur_R_;
    }
    if (px_previous_.size() < kMinNumFeature)
    {
        featureDetection(current_frame_, px_previous_);
        featureTracking(previous_frame_, current_frame_, px_previous_, px_current_, disparities_);
    }
#ifdef ERROR_FIXED
    else
    {
#endif
        px_previous_ = px_current_;
#ifdef ERROR_FIXED
    }
#endif
    return true;
}

void VisualOdometry::GetGroundTruth()
{
    std::string line;
    getline(ground_truth_stream_, line);
    std::stringstream ss;
    ss << line;
    double r1, r2, r3, r4, r5, r6, r7, r8, r9;
    ss >> r1 >> r2 >> r3 >> x_pre_ >> r4 >> r5 >> r6 >> y_pre_ >> r7 >> r8 >> r9 >> z_pre_;
}

double VisualOdometry::getAbsoluteScale()
{
    std::string line;

    if (std::getline(this->ground_truth_stream_, line))
    {
        double r1, r2, r3, r4, r5, r6, r7, r8, r9;
        double x = 0, y = 0, z = 0;
        std::istringstream in(line);
        in >> r1 >> r2 >> r3 >> x >> r4 >> r5 >> r6 >> y >> r7 >> r8 >> r9 >> z;
        double scale = sqrt((x - x_pre_) * (x - x_pre_) + (y - y_pre_) * (y - y_pre_) + (z - z_pre_) * (z - z_pre_));
        x_pre_ = x;
        y_pre_ = y;
        z_pre_ = z;
        return scale;
    } else
    {
        return 0;
    }
}

void VisualOdometry::featureDetection(cv::Mat &frame, std::vector<cv::Point2f> &keypoints)
{
    std::vector<cv::KeyPoint> key_points;
    int fast_threshold = 20;
    bool non_max_suppression = true;
    FAST(frame, key_points, fast_threshold, non_max_suppression);
    cv::KeyPoint::convert(key_points, keypoints);
}

void VisualOdometry::featureTracking(cv::Mat &image_previous, cv::Mat &image_current,
                                     std::vector<cv::Point2f> &keyPoint_previous,
                                     std::vector<cv::Point2f> &keyPoint_current, std::vector<double> &disparities)
{
    const double klt_win_size = 21.0;
    const int klt_max_iter = 30;
    const double kly_eps = 0.001;

    std::vector<uchar> status;
    std::vector<float> error;
    std::vector<float> min_eig_vec;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, klt_max_iter, kly_eps);
    calcOpticalFlowPyrLK(previous_frame_, current_frame_,
                         px_previous_, px_current_,
                         status, error,
                         cv::Size2i(klt_win_size, klt_win_size),
                         4, criteria, 0);
    disparities.clear();
    disparities.reserve(px_current_.size());

    std::vector<cv::Point2f>::iterator px_prev_it = px_previous_.begin();
    std::vector<cv::Point2f>::iterator px_curr_it = px_current_.begin();
    for (size_t i = 0; px_prev_it != px_previous_.end(); ++i)
    {
        if (!status[i])
        {
            px_curr_it = px_current_.erase(px_curr_it);
            px_prev_it = px_previous_.erase(px_prev_it);
            continue;
        }
        disparities.push_back(cv::norm(cv::Point2d(px_prev_it->x - px_curr_it->x, px_prev_it->y - px_curr_it->y)));
        ++px_curr_it;
        ++px_prev_it;
    }
}

bool VisualOdometry::SetGroundTruth(std::string ground_truth_path)
{
    this->ground_truth_path_ = ground_truth_path;
    this->ground_truth_stream_.open(this->ground_truth_path_);
    if (!this->ground_truth_stream_.is_open())
    {
        return false;
    } else
    {
        return true;
    }
}