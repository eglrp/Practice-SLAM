#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("/home/runisys/Desktop/data/KITTI/object/training/image_2/000000.png");
    if (img.empty())
    {
        cout << "Open Image failed!" << endl;
        return -1;
    }
    FastFeatureDetector detector(40);
    vector<KeyPoint> keyPoints;
    detector.detect(img, keyPoints);
    drawKeypoints(img, keyPoints, img);

    imshow("Detected KeyPoints", img);
    waitKey(0);
    return 0;
}