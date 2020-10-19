#pragma once

#include <Eigen/Dense>
#include "opencv/cv.hpp"

template<typename T> void free_array_allocator(T * p)
{
    if (nullptr != p)
    {
        delete []p;
    }
}

class HoughLineExtrator
{
public:
    HoughLineExtrator();
    HoughLineExtrator(float r_resolution, float ang_resolution);
    ~HoughLineExtrator();
    HoughLineExtrator(const HoughLineExtrator &) = delete;
    HoughLineExtrator & operator = (const HoughLineExtrator &) = delete;
    void init(float r_resolution, float ang_resolution);

    void line_extract(cv::Mat& image, int threshold,
        int lineLength, int lineGap, std::vector<cv::Vec4i>& lines);
    void line_extract(int rows, int cols, 
        const std::vector<cv::Point>& non_zero_points, int pts_threshold, int line_length,  
        int line_gap, std::vector<cv::Vec4i> & lines) const;
private:
    void nonzero_points_extract(const cv::Mat& image, std::vector<cv::Point>& nzloc) const;
    void line_extract_from_points(float angle, int pts_threshold, float line_lenthgh, float gap, 
        const std::vector<cv::Point> & points, std::vector<cv::Vec4i> & lines) const;
    void hough_transform(int rows, int cols,  
        const std::vector<cv::Point>& nzloc, cv::Mat & accum) const;
    float _r_resolution;
    float _ang_resolution;
    int _num_ang;
    float * _cos_tab;
    float * _sin_tab;
};

/****************************************************************************************\
*                              Probabilistic Hough Transform                             *
\****************************************************************************************/

void hough_lines_probabilistic(cv::Mat& image,
                        float rho, float theta, int threshold,
                        int lineLength, int lineGap,
                        std::vector<cv::Vec4i>& lines, int linesMax);

