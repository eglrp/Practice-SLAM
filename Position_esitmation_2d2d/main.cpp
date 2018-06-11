#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

void find_feature_matches(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches);

void position_estimation_2d2d(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                              std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t);

cv::Point2d pixel2cam ( const cv::Point2d& p, const cv::Mat& K )
{
    return cv::Point2d
            (
                    ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
                    ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
            );
}

int main()
{
    cv::Mat img1 = cv::imread("../1.png");
    cv::Mat img2 = cv::imread("../2.png");
    if (img1.empty() || img2.empty())
    {
        std::cout << "Read image file failed!" << std::endl;
        return 1;
    }

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img1, img2, keypoints1, keypoints2, matches);

    std::cout << "Find matched keypoints count is " << matches.size() << std::endl;

    cv::Mat R;
    cv::Mat t;
    position_estimation_2d2d(keypoints1, keypoints2, matches, R, t);


    cv::Mat t_x = (cv::Mat_<double>(3, 3) <<
            0, -t.at<double>(2, 0), t.at<double>(1, 0),
            t.at<double>(2, 0), 0, -t.at<double>(0, 0),
            -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    std::cout << "t^R = " << std::endl << t_x * R << std::endl;

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 326.1, 0, 521.0, 249.7, 0, 0, 1);

    for(cv::DMatch m : matches)
    {
        cv::Point2d pt1 = pixel2cam ( keypoints1[ m.queryIdx ].pt, K );
        cv::Mat y1 = ( cv::Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
        cv::Point2d pt2 = pixel2cam ( keypoints2[ m.trainIdx ].pt, K );
        cv::Mat y2 = ( cv::Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
        cv::Mat d = y2.t() * t_x * R * y1;
        std::cout << "epipolar constraint = " << d << std::endl;
    }
    return 0;
}

void position_estimation_2d2d(std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                              std::vector<cv::DMatch> &matches, cv::Mat &R, cv::Mat &t)
{


    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
//    cv::KeyPoint::convert(keypoints1, points1);
//    cv::KeyPoint::convert(keypoints2, points2);

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints2[matches[i].trainIdx].pt );
    }

    cv::Mat F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    std::cout << "Fundamental matrix is " << std::endl << F << std::endl;

    cv::Point2f pricipal_point(326.1, 249.7);
    int focal_length = 521;

    cv::Mat E = cv::findEssentialMat(points1, points2, focal_length, pricipal_point, CV_FM_RANSAC);
    std::cout << "Essential matrix is " << std::endl << E << std::endl;

    cv::Mat H = cv::findHomography(points1, points2, CV_FM_RANSAC, 3);

    std::cout << "Homography matrix is " << std::endl << H << std::endl;

    cv::recoverPose(E, points1, points2, R, t, focal_length, pricipal_point);
    std::cout << "Rotation matrix is " << std::endl << R << std::endl;
    std::cout << "transform matrix is " << std::endl << t << std::endl;

}

void find_feature_matches(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches)
{
    cv::Mat descriptor1;
    cv::Mat descriptor2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    orb->detect(img1, keypoints1);
    orb->detect(img2, keypoints2);

    orb->compute(img1, keypoints1, descriptor1);
    orb->compute(img2, keypoints2, descriptor2);

    std::vector<cv::DMatch> basic_matches;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptor1, descriptor2, basic_matches);

    double min_dist = 10000;
    double max_dist = 0;

    for (int i = 0; i < descriptor1.rows; ++i)
    {
        double dist = basic_matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::vector<cv::KeyPoint>::iterator it1 = keypoints1.begin();
    std::vector<cv::KeyPoint>::iterator it2 = keypoints2.begin();
    for (int i = 0; i < descriptor1.rows; ++i)
    {
        if (basic_matches[i].distance <= cv::max(2 * min_dist, 30.0))
        {
            matches.push_back(basic_matches[i]);
        }
        else
        {
            it1 = keypoints1.erase(it1);
            it2 = keypoints2.erase(it2);
        }
    }
}