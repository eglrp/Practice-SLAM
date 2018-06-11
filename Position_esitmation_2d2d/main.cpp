#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

void find_feature_matches(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1,
                          std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches);

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
    find_feature_matches(img1, img2, keypoints1, keypoints2,matches);

    std::cout << matches.size() << std::endl;

    return 0;
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

    for (int i = 0; i < descriptor1.rows; ++i)
    {
        if(basic_matches[i].distance <= cv::max(2 * min_dist, 30.0))
        {
            matches.push_back(basic_matches[i]);
        }
    }
}