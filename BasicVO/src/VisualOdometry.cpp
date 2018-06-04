#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include "VisualOdometry.h"
#include <string>

using namespace std;

int kMinNumFeature = 2000;

VisualOdometry::VisualOdometry(PinholeCamera *camera) :
        pcamera_(camera)
{
    focal = camera->fx();
    pp_ = Point2d(camera->cx(), camera->cy());
    frameStage_ = FrameStage::STAGE_FIRST_FRAME;
}

VisualOdometry::~VisualOdometry() {}

void VisualOdometry::addImage(const Mat &img, int frame_id)
{
    if (img.empty() || img.type() != CV_8UC1 || img.rows != pcamera_->height() || img.cols != pcamera_->width())
        throw std::runtime_error(
                "Frame: provide image has not the same size of camera model or image is not grayscale");
    current_frame_ = img;

    bool res = true;
    if (frameStage_ == FrameStage::STAGE_DEFAULT_FRAME)
        res = processFrame(frame_id);
    else if (frameStage_ == FrameStage::STAGE_FIRST_FRAME)
        res = processFirstFrame();
    else if (frameStage_ == FrameStage::STAGE_SECOND_FRAME)
        res = processSecondFrame();

    previous_frame_ = current_frame_;
}

bool VisualOdometry::processFirstFrame()
{
    featureDetection(current_frame_, px_previous_);
    frameStage_ = FrameStage::STAGE_SECOND_FRAME;
    return true;
}

bool VisualOdometry::processSecondFrame()
{
    featureTracking(previous_frame_, current_frame_, px_previous_, px_current_,disparities);
    Mat E, R, T;
    Mat mask;
    E = findEssentialMat(px_current_, px_previous_, focal, pp_, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, px_current_, px_previous_, R, T, focal, pp_, mask);
    cur_R = R.clone();
    cur_t = T.clone();
    frameStage_ = FrameStage::STAGE_DEFAULT_FRAME;
    px_previous_ = px_current_;
    return true;
}

bool VisualOdometry::processFrame(int frame_id)
{
    double scale = 1.00;
    featureTracking(previous_frame_, current_frame_,px_previous_, px_current_, disparities);
    Mat E, R, T, mask;

    E = findEssentialMat(px_current_, px_previous_, focal, pp_, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, px_current_, px_previous_, R, T, focal, pp_, mask);
    scale = getAbsoluteScale(frame_id);
    if (scale > 0.1)
    {
        cur_t = cur_t + scale * (cur_R * T);
        cur_R == R * cur_R;
    }
    if (px_previous_.size() < kMinNumFeature)
    {
        featureDetection(current_frame_, px_previous_);
        featureTracking(previous_frame_, current_frame_, px_previous_, px_current_, disparities);
    }
    px_previous_ = px_current_;
    return true;
}

double VisualOdometry::getAbsoluteScale(int frame_id)
{
    string line;
    int i = 0;
    ifstream groundTruth("/home/ynzhang/data/KITTI/dataset/poses/00.txt");
    double x = 0, y = 0, z = 0;
    double x_pre = 0, y_pre = 0, z_pre = 0;
    if (groundTruth.is_open())
    {
        while ((std::getline(groundTruth, line)) && (i <= frame_id))
        {
            z_pre = z;
            x_pre = x;
            y_pre = y;
            istringstream in(line);
            for (int j = 0; j < 12; j++)
            {
                in >> z;
                if (j == 7) y = z;
                if (j == 3) x = z;
            }
            i++;
        }
        groundTruth.close();
    } else
    {
        cout << "Unable open file" << endl;
    }
    return sqrt((x - x_pre) * (x - x_pre) + (y - y_pre) * (y - y_pre) + (z - z_pre) * (z - z_pre));
}

void VisualOdometry::featureDetection(Mat &image, vector<Point2f>& keypoints)
{
    vector<KeyPoint> key_points;
    int fast_threshold = 20;
    bool non_max_suppression = true;
    FAST(image, key_points, fast_threshold, non_max_suppression);
    KeyPoint::convert(key_points, keypoints);
}

void VisualOdometry::featureTracking(Mat &image_previous, Mat &image_current, vector<Point2f> &keyPoint_previous,
                                     vector<Point2f> &keyPoint_current, vector<double>& disparities)
{
    const double klt_win_size = 21.0;
    const int klt_max_iter = 30;
    const double kly_eps = 0.001;
    vector<uchar> status;
    vector<float> error;
    vector<float> min_eig_vec;
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, klt_max_iter, kly_eps);
    calcOpticalFlowPyrLK(previous_frame_, current_frame_,
                         px_previous_, px_current_,
                         status, error,
                         Size2i(klt_win_size, klt_win_size),
                         4, criteria, 0);
    disparities.clear();
    disparities.reserve(px_current_.size());

    vector<Point2f>::iterator px_prev_it = px_previous_.begin();
    vector<Point2f>::iterator px_curr_it = px_current_.begin();
    for (size_t i = 0; px_prev_it != px_previous_.end(); ++i)
    {
        if (!status[i])
        {
            px_curr_it = px_current_.erase(px_curr_it);
            px_prev_it = px_previous_.erase(px_prev_it);
            continue;
        }
        disparities.push_back(norm(Point2d(px_prev_it->x - px_curr_it->x, px_prev_it->y - px_curr_it->y)));
        ++px_curr_it;
        ++px_prev_it;
    }
}