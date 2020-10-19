#include <math.h>
#include <Eigen/Core>
#include <vector>
#include <algorithm>

bool fitting_line_ls(const std::vector<Eigen::Vector3d> &ref_pts, double &ref_ang, Eigen::VectorXd &err_vec)
{
    int dim = ref_pts.size();
    if (dim < 2) {     // give enough points
        return false;
    }

    Eigen::MatrixXd A_l(dim, 2);
    Eigen::Vector2d x_l;
    Eigen::VectorXd b_l(dim);

    // x is dominant, y = k * x + b
    for (size_t i = 0; i < dim; i++) {
        A_l.row(i) << ref_pts[i].x(), 1;
        b_l.row(i) << ref_pts[i].y();
    }
    x_l = A_l.colPivHouseholderQr().solve(b_l);
    err_vec = b_l - A_l * x_l;
    double error = err_vec.norm() / sqrt(dim - 1);
    ref_ang = atan2(x_l(0), 1);

    // y is dominant, x = k * y + b
    for (size_t i = 0; i < dim; i++) {
        A_l.row(i) << ref_pts[i].y(), 1;
        b_l.row(i) << ref_pts[i].x();
    }
    x_l = A_l.colPivHouseholderQr().solve(b_l);
    Eigen::VectorXd tmp_err_vec = b_l - A_l * x_l;
    double tmp_error_y = tmp_err_vec.norm() / sqrt(dim - 1);

    if (error > tmp_error_y) {
        err_vec = tmp_err_vec;
        error = tmp_error_y;
        ref_ang = atan2(1, x_l(0));
    }

    return true;
}

bool fitting_line_ls_recur(const std::vector<Eigen::Vector3d> &ref_pts, double &ref_ang, double thre, int max_iter_cnt)
{
    int dim = ref_pts.size();
    if (dim < 2) {     // give enough points
        return false;
    }

    Eigen::VectorXd err_vec;
    fitting_line_ls(ref_pts, ref_ang, err_vec);
    double error = err_vec.norm() / sqrt(dim - 1);

    if (error < thre || 0 == max_iter_cnt) {
        return true;
    } else {
        std::vector<Eigen::Vector3d> pts;
        for (size_t i = 0; i < dim; i++) {
            if (fabs(err_vec(i)) < error) {
                pts.push_back(ref_pts[i]);
            }
        }
        if (pts.size() < dim) {
            max_iter_cnt--;
            return fitting_line_ls_recur(pts, ref_ang, max_iter_cnt);
        } else {
            return true;
        }
    }
}

bool fitting_line_ransac(const std::vector<Eigen::Vector3d> &ref_pts, double &ref_ang, double thre, int max_iter_cnt)
{
    int dim = ref_pts.size();

    if (dim < 2) {     // give enough points
        return false;
    }

    int max_inlier_cnt = 0;
    std::vector<Eigen::Vector3d> max_inliers;

    std::srand(std::time(0));
    for (size_t iter_cnt = 0; iter_cnt < max_iter_cnt; iter_cnt++) {
        std::vector<Eigen::Vector3d> sam_pts;
        sam_pts.resize(2);
        for (size_t i = 0; i < 2; i++) {
            sam_pts[i] = ref_pts[(int)((double)std::rand() / (double)RAND_MAX * (dim - 1))];
        }

        // don't calculate when two points are too close
        double sam_pts_dis = (sam_pts[1] - sam_pts[0]).norm();
        if (sam_pts_dis < 0.6) {
            continue;
        }

        // fitting line with two sample points
        double sam_ang = 0.0;
        Eigen::VectorXd err_vec;
        fitting_line_ls(sam_pts, sam_ang, err_vec);

        // calculate inliers
        int inlier_cnt = 0;
        std::vector<Eigen::Vector3d> inliers;
        Eigen::Vector3d sam_ang_unit_vec(cos(sam_ang), sin(sam_ang), 0.0);
        for (size_t i = 0; i < dim; i++) {
            Eigen::Vector3d pt_vec = ref_pts[i] - sam_pts[0];
            double dis = fabs(sam_ang_unit_vec.cross(pt_vec).z());
            if (dis < thre) {
                inliers.emplace_back(ref_pts[i]);
                inlier_cnt++;
            }
        }

        // update the line
        if (inlier_cnt > max_inlier_cnt) {
            max_inlier_cnt = inlier_cnt;
            max_inliers = inliers;
        }

        // update the parameters of ransac
        double inlier_ratio = (double)max_inlier_cnt / (double)dim;
        max_iter_cnt = std::min(max_iter_cnt, (int)(log(0.003) / log(1 - pow(inlier_ratio, 2))));
    }

    if (max_inliers.size() < (double)dim * 2.0 / 3.0) {
        return false;
    } else {
        // fitting line with inliers
        Eigen::VectorXd err_vec;
        return fitting_line_ls(max_inliers, ref_ang, err_vec);
    }
}

