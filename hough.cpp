#include "hough.h"

#include <iostream>
#include <algorithm>


HoughLineExtrator::HoughLineExtrator()
{
    _ang_resolution = 1.0f / 180.0f * M_PI;
    _cos_tab = nullptr;
    _sin_tab = nullptr;
}

HoughLineExtrator::HoughLineExtrator(float r_resolution, float ang_resolution)
{
    init(r_resolution, ang_resolution);
}

HoughLineExtrator::~HoughLineExtrator()
{
    free_array_allocator(_cos_tab);
    free_array_allocator(_sin_tab);
}

void HoughLineExtrator::init(float r_resolution, float ang_resolution)
{
    _r_resolution = r_resolution;
    _ang_resolution = ang_resolution;
    _num_ang = cvRound(CV_PI / _ang_resolution);

    free_array_allocator(_cos_tab);
    free_array_allocator(_sin_tab);
    _cos_tab = new float[_num_ang];
    _sin_tab = new float[_num_ang];
    if (nullptr == _cos_tab || nullptr == _sin_tab)
    {
        std::cerr << "Memory allocator error in HoughLineExtrator::init(...)!" << std::endl;
    }
    float inverse_r_resolution = 1.0f / _r_resolution;
    for (int n = 0; n < _num_ang; n++)
    {
        double angle = static_cast<double>(n) * _ang_resolution;
        _cos_tab[n] = static_cast<float>(std::cos(angle) * inverse_r_resolution);
        _sin_tab[n] = static_cast<float>(std::sin(angle) * inverse_r_resolution);
    }
}

void HoughLineExtrator::nonzero_points_extract(const cv::Mat& image, 
    std::vector<cv::Point>& nzloc) const
{
    nzloc.clear();
    CV_Assert(image.type() == CV_8UC1);
    int width = image.cols;
    int height = image.rows;

    cv::Point pt;
    for (pt.y = 0; pt.y < height; pt.y++)
    {
        const uchar* data = image.ptr(pt.y);
        for (pt.x = 0; pt.x < width; pt.x++)
        {
            if (data[pt.x] > 100)
            {
                nzloc.push_back(pt);
            }
        }
    }
}

void HoughLineExtrator::line_extract_from_points(float angle, int pts_threshold, float line_lenthgh, float gap, 
    const std::vector<cv::Point> & points, std::vector<cv::Vec4i> & lines) const
{
    float sin_angle = sin(angle);
    float cos_angle = cos(angle);
    bool x_dominate = (fabs(sin_angle) >= fabs(cos_angle)); 

    float eta = x_dominate ? fabs(sin_angle) : fabs(cos_angle);
    gap *= eta;
    line_lenthgh *= eta;
    int pixel_gap = cvRound(gap);

    std::vector<int> idxs(points.size());
    for (size_t i = 0; i != points.size(); ++i)
    {
        idxs[i] = i;
    }
    auto lmd = [&x_dominate](const cv::Point & point) -> int 
        {return (x_dominate ? point.x : point.y);};
    auto gma = [&lmd, &points](int i, int j) -> bool 
        {
            return lmd(points[i]) < lmd(points[j]);
        };
    std::sort(idxs.begin(), idxs.end(), gma);
    size_t i_0 = 0;
    cv::Point pa = points[idxs[i_0]];
    cv::Point pb_1 = pa;
    cv::Point pb = pb_1;
    Eigen::Vector2d sum_point(0.0, 0.0);
    for (size_t i = 0; i != idxs.size(); ++i)
    {
        pb = points[idxs[i]];
        if (lmd(pb) - lmd(pb_1) >= pixel_gap
         || i == idxs.size() - 1)
        {
            if (lmd(pb) - lmd(pb_1) < pixel_gap)
            {
                pb_1 = pb;
            }
            if (lmd(pb_1) - lmd(pa) >= line_lenthgh && (i - i_0) > pts_threshold)
            {
                Eigen::Vector2d center_point = sum_point / (i - i_0);
                Eigen::Vector2d vec_a(pa.x, pa.y);
                Eigen::Vector2d vec_b(pb_1.x, pb_1.y);
                Eigen::Vector2d unit_vertic_line(cos_angle, sin_angle);
                vec_a -= (vec_a - center_point).dot(unit_vertic_line) * unit_vertic_line;
                vec_b -= (vec_b - center_point).dot(unit_vertic_line) * unit_vertic_line;
                
                cv::Vec4i line(cvRound(vec_a.x()), cvRound(vec_a.y()), cvRound(vec_b.x()), cvRound(vec_b.y()));
                lines.push_back(line);

                // // for debug
                // cv::Mat img(500, 500, CV_8UC3, cv::Scalar(0,0,0));
                // for (auto point : points) {
                //     cv::circle(img, point, 0, cv::Scalar(255,0,0));
                // }
                // cv::line(img, cv::Point(line[0], line[1]),
                //     cv::Point(line[2], line[3]), cv::Scalar(255,255,255), 1, 1);

                // cv::namedWindow("img", 1);
                // cv::imshow("img", img);
                // cv::waitKey(0);
            }
            i_0 = i;
            pa = pb;
            sum_point.setZero();
        }
        sum_point.x() += pb.x;
        sum_point.y() += pb.y;
        pb_1 = pb;
    }
}

void HoughLineExtrator::hough_transform(int rows, int cols, 
    const std::vector<cv::Point>& nzloc, cv::Mat & accum) const
{
    Eigen::Vector2d rows_cols(rows, cols);
    int num_r = 2 * cvRound((rows_cols.norm() + 0.1) / _r_resolution) + 1;
    accum = cv::Mat::zeros(_num_ang, num_r, CV_32SC1);
    
    int bias_r = (num_r - 1) / 2;
    for (const auto & point : nzloc)
    {
        float px = point.x;
        float py = point.y;
        int* adata = accum.ptr<int>();
        for (int ang_idx = 0; ang_idx < _num_ang; ++ang_idx, adata += num_r)
        {
            int r = cvRound(px * _cos_tab[ang_idx] + py * _sin_tab[ang_idx]);
            r += bias_r;
            ++adata[r];
        }
    }
}

void HoughLineExtrator::line_extract(int rows, int cols, 
    const std::vector<cv::Point>& non_zero_points, int pts_threshold, int line_length,  
    int line_gap, std::vector<cv::Vec4i> & lines) const
{
    std::vector<cv::Point> nzloc = non_zero_points;
    lines.clear();
    cv::Mat accum;
    hough_transform(rows, cols, nzloc, accum);
    int num_r = accum.cols;
    int bias_r = (num_r - 1) / 2;
    
    std::vector<int> aviable_idxs;
    int* adata = accum.ptr<int>();
    size_t max_accum_idx = _num_ang * num_r;
    for (size_t accum_idx = 0; accum_idx < max_accum_idx; ++accum_idx)
    {
        if (adata[accum_idx] >= pts_threshold)
        {
            aviable_idxs.push_back(accum_idx);
        }
    }

    auto lmd = [&adata](const int & a, const int & b) -> bool {return adata[a] < adata[b];};
    
    auto max_aviable_iter = std::max_element(aviable_idxs.begin(), aviable_idxs.end(), lmd);
    // std::vector<cv::Vec4i> lines;
    size_t residual_counter_nzloc = nzloc.size();
    while (max_aviable_iter != aviable_idxs.end()
     && adata[*max_aviable_iter] >= pts_threshold)
    {
        int most_avilable_idx = *max_aviable_iter;
        
        int ang_idx = most_avilable_idx / num_r;
        int r_idx = most_avilable_idx - ang_idx * num_r;
        int r0 = r_idx - bias_r;
        std::vector<cv::Point> line_points;
        cv::Point point;
        for (int pt_idx = 0; pt_idx < residual_counter_nzloc; ++pt_idx)
        {
            point = nzloc[pt_idx];
            float px = point.x;
            float py = point.y;
            int r = cvRound(px * _cos_tab[ang_idx] + py * _sin_tab[ang_idx]);
            if (abs(r - r0) <= 0)
            {
                line_points.push_back(point);
                for (auto aviable_idx : aviable_idxs)
                {
                    int ang_idx = aviable_idx / num_r;
                    int r_idx = aviable_idx - ang_idx * num_r;
                    int r = cvRound(px * _cos_tab[ang_idx] + py * _sin_tab[ang_idx]);
                    if (r == r_idx - bias_r)
                    {
                        --adata[aviable_idx];
                    }
                }
                nzloc[pt_idx] = nzloc[residual_counter_nzloc - 1];
                --residual_counter_nzloc;
                --pt_idx;
            }
        }
        if (line_points.size() >= pts_threshold)
        {
            // exract_line_from_points();
            float angle = ang_idx * _ang_resolution;
            line_extract_from_points(angle, pts_threshold, line_length, line_gap, line_points, lines);
        }
        max_aviable_iter = std::max_element(aviable_idxs.begin(), aviable_idxs.end(), lmd);
    }
}

void HoughLineExtrator::line_extract(cv::Mat& image, int threshold,
    int lineLength, int lineGap, std::vector<cv::Vec4i>& lines)
{
    std::vector<cv::Point> nzloc_c;
    nonzero_points_extract(image, nzloc_c);
    line_extract(image.rows, image.cols, nzloc_c, threshold, lineLength, lineGap, lines);
    return;
}

/****************************************************************************************\
*                              Probabilistic Hough Transform                             *
\****************************************************************************************/

void hough_lines_probabilistic(cv::Mat& image,
                        float rho, float theta, int threshold,
                        int lineLength, int lineGap,
                        std::vector<cv::Vec4i>& lines, int linesMax)
{
    cv::Point pt;
    float irho = 1 / rho;
    cv::RNG rng((uint64)-1);

    CV_Assert(image.type() == CV_8UC1);

    int width = image.cols;
    int height = image.rows;

    int numangle = cvRound(CV_PI / theta);
    int numrho = cvRound(((width + height) * 2 + 1) / rho);

    cv::Mat accum = cv::Mat::zeros(numangle, numrho, CV_32SC1);
    cv::Mat mask(height, width, CV_8UC1);
    std::vector<float> trigtab(numangle*2);

    for (int n = 0; n < numangle; n++)
    {
        trigtab[n*2] = (float)(std::cos((double)n*theta) * irho);
        trigtab[n*2+1] = (float)(std::sin((double)n*theta) * irho);
    }
    const float* ttab = &trigtab[0];
    uchar* mdata0 = mask.ptr();
    std::vector<cv::Point> nzloc;

    // stage 1. collect non-zero image points
    for (pt.y = 0; pt.y < height; pt.y++)
    {
        const uchar* data = image.ptr(pt.y);
        uchar* mdata = mask.ptr(pt.y);
        for (pt.x = 0; pt.x < width; pt.x++)
        {
            if (data[pt.x] > 100)
            {
                mdata[pt.x] = (uchar)1;
                nzloc.push_back(pt);
            }
            else
                mdata[pt.x] = 0;
        }
    }

    int count = (int)nzloc.size();

    // stage 2. process all the points in random order
    for (; count > 0; count--)
    {
        // choose random point out of the remaining ones
        int idx = rng.uniform(0, count);
        int max_val = threshold-1;
        int max_n = 0;
        cv::Point point = nzloc[idx];
        cv::Point line_end[2];
        float a = 0.0;
        float b = 0.0;
        int* adata = accum.ptr<int>();
        int i = point.y;
        int j = point.x;
        int k = 0;
        int x0 = 0;
        int y0 = 0;
        int dx0 = 0;
        int dy0 = 0;
        int xflag = 0;
        int good_line = 0;
        const int shift = 16;

        // "remove" it by overriding it with the last element
        nzloc[idx] = nzloc[count-1];

        // check if it has been excluded already (i.e. belongs to some other line)
        if (!mdata0[i*width + j])
            continue;

        // update accumulator, find the most probable line
        for (int n = 0; n < numangle; n++, adata += numrho)
        {
            int r = cvRound(j * ttab[n*2] + i * ttab[n*2+1]);
            r += (numrho - 1) / 2;
            int val = ++adata[r];
            if (max_val < val)
            {
                max_val = val;
                max_n = n;
            }
        }

        // if it is too "weak" candidate, continue with another point
        if (max_val < threshold)
            continue;

        // from the current point walk in each direction
        // along the found line and extract the line segment
        a = -ttab[max_n*2+1];
        b = ttab[max_n*2];
        x0 = j;
        y0 = i;
        if (std::fabs(a) > std::fabs(b))
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = cvRound(b*(1 << shift)/std::fabs(a));
            y0 = (y0 << shift) + (1 << (shift-1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = cvRound(a*(1 << shift)/std::fabs(b));
            x0 = (x0 << shift) + (1 << (shift-1));
        }

        for (k = 0; k < 2; k++)
        {
            int gap = 0;
            int x = x0;
            int y = y0;
            int dx = dx0;
            int dy = dy0;

            if (k > 0) {
                dx = -dx;
                dy = -dy;
            }

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            for (;; x += dx, y += dy)
            {
                int i1 = 0;
                int j1 = 0;

                if (xflag)
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
                    break;

                uchar* mdata = mdata0 + i1*width + j1;

                // for  each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if (*mdata)
                {
                    gap = 0;
                    line_end[k].y = i1;
                    line_end[k].x = j1;
                }
                else if (++gap > lineGap)
                    break;
            }
        }

        good_line = std::abs(line_end[1].x - line_end[0].x) >= lineLength ||
                    std::abs(line_end[1].y - line_end[0].y) >= lineLength;

        for (k = 0; k < 2; k++)
        {
            int x = x0;
            int y = y0;
            int dx = dx0;
            int dy = dy0;

            if (k > 0) {
                dx = -dx;
                dy = -dy;
            }

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            for (;; x += dx, y += dy)
            {
                int i1 = 0;
                int j1 = 0;

                if (xflag)
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                uchar* mdata = mdata0 + i1*width + j1;

                // for  each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if (*mdata)
                {
                    if (good_line)
                    {
                        adata = accum.ptr<int>();
                        for (int n = 0; n < numangle; n++, adata += numrho)
                        {
                            int r = cvRound(j1 * ttab[n*2] + i1 * ttab[n*2+1]);
                            r += (numrho - 1) / 2;
                            adata[r]--;
                        }
                    }
                    *mdata = 0;
                }

                if (i1 == line_end[k].y && j1 == line_end[k].x)
                    break;
            }
        }

        if (good_line)
        {
            cv::Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
            lines.push_back(lr);
            if ((int)lines.size() >= linesMax)
                return;
        }
    }
}

