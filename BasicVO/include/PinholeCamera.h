
#ifndef BASICVO_PINHOLECAMERA_H
#define BASICVO_PINHOLECAMERA_H

#include <opencv2/core.hpp>

class PinholeCamera
{
public:
    PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
                  double k1 = 0,
                  double k2 = 0,
                  double p1 = 0,
                  double p2 = 0,
                  double k3 = 0);

    ~PinholeCamera();

    inline int width() { return this->width_; }

    inline int height() { return this->height_; }

    inline double fx() { return this->fx_; }

    inline double fy() { return this->fy_; }

    inline double cx() { return this->cx_; }

    inline double cy() { return this->cx_; }

    inline bool distortion() { return this->distortion_; }

    inline double k1() { return this->d_[0]; }

    inline double k2() { return this->d_[1]; }

    inline double p1() { return this->d_[2]; }

    inline double p2() { return this->d_[3]; }

    inline double k3() { return this->d_[4]; }

private:
    int width_;
    int height_;
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    bool distortion_;
    double d_[5];
};


#endif //BASICVO_PINHOLECAMERA_H
