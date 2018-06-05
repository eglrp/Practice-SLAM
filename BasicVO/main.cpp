#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>

#include "PinholeCamera.h"
#include "VisualOdometry.h"

using namespace std;
using namespace cv;

int main()
{
    PinholeCamera *pcamera = new PinholeCamera(1241, 376, 718.8560, 718.8560, 607.1928, 185.2157);
    VisualOdometry vo(pcamera);

    fstream out("position.txt");
    char text[200];
    int font_face = cv::FONT_HERSHEY_PLAIN;
    double font_scale = 1;
    int thickness = 1;
    Point text_org(10, 50);
    namedWindow("Road facing camera", cv::WINDOW_AUTOSIZE);
    namedWindow("Trajectory", WINDOW_AUTOSIZE);
    Mat traj = Mat::zeros(600, 600, CV_8UC3);

    double x = 0, y = 0, z = 0;
    for (int im_id = 0; im_id < 2000; ++im_id)
    {
        stringstream ss;
        ss << "/home/runisys/Desktop/data/slamDataset/sequences/00/image_0/" << setw(6) << setfill('0') << im_id
           << ".png";

        Mat img = imread(ss.str().c_str(), 0);
        assert(!img.empty());

        vo.addImage(img, im_id);
        cv::Mat cur_t = vo.getCurrentT();
        if (cur_t.rows != 0)
        {
            x = cur_t.at<double>(0);
            y = cur_t.at<double>(1);
            z = cur_t.at<double>(2);
        }
        out << x << " " << y << " " << z << std::endl;

        int draw_x = int(x) + 300;
        int draw_y = int(z) + 100;
        cv::circle(traj, cv::Point(draw_x, draw_y), 1, CV_RGB(255, 0, 0), 2);
        std::cout << im_id << " " << draw_x << " " << draw_y << std::endl;

        cv::rectangle(traj, cv::Point(10, 30), cv::Point(580, 60), CV_RGB(0, 0, 0), CV_FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", x, y, z);
        cv::putText(traj, text, text_org, font_face, font_scale, cv::Scalar::all(255), thickness, 8);

        cv::imshow("Road facing camera", img);
        cv::imshow("Trajectory", traj);

        cv::waitKey(1);
    }
    delete pcamera;
    out.close();

    return 0;
}