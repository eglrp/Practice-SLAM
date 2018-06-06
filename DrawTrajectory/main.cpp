#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

int main()
{
    std::ifstream trajectory_1_in("../data/trajectory1.txt");
    std::ifstream trajectory_2_in("../data/trajectory2.txt");
    std::ifstream trajectory_3_in("../data/ground_truth.txt");

    if (!trajectory_1_in.is_open() || !trajectory_2_in.is_open() || !trajectory_3_in.is_open())
    {
        std::cout << "Unable to read trajectorys" << std::endl;
        return 1;
    }

    std::string line1;
    std::string line2;
    std::string line3;

    cv::Mat traj_view = cv::Mat::zeros(600, 600, CV_8UC3);

    bool show_trajectory1 = true;
    bool show_trajectory2 = true;
    bool show_gound_truth = true;

    while (getline(trajectory_1_in, line1) && getline(trajectory_2_in, line2) && getline(trajectory_3_in, line3))
    {
        if (show_trajectory1)
        {
            std::stringstream ss1;
            ss1 << line1;
            double x1, y1, z1;
            ss1 >> x1 >> y1 >> z1;
            int drawx1 = int(x1) + 300;
            int drawy1 = int(z1) + 100;
            cv::circle(traj_view, cv::Point(drawx1, drawy1), 1, CV_RGB(255, 255, 0), 2);
        }


        if (show_trajectory2)
        {
            std::stringstream ss2;
            ss2 << line2;
            double x2, y2, z2;
            ss2 >> x2 >> y2 >> z2;
            int drawx2 = int(x2) + 300;
            int drawy2 = int(z2) + 100;
            cv::circle(traj_view, cv::Point(drawx2, drawy2), 1, CV_RGB(0, 255, 255), 2);
        }
        if (show_gound_truth)
        {
            std::stringstream ss3;
            ss3 << line3;
            double r1, r2, r3, x, r4, r5, r6, y, r7, r8, r9, z;
            ss3 >> r1 >> r2 >> r3 >> x >> r4 >> r5 >> r6 >> y >> r7 >> r8 >> r9 >> z;
            int drawx = int(x) + 300;
            int drawy = int(z) + 100;
            cv::circle(traj_view, cv::Point(drawx, drawy), 1, CV_RGB(255, 0, 255), 2);
        }
    }
    cv::imshow("Trajectory", traj_view);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}