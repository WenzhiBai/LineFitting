#include "kalmanfilter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;


void kalman_filter_correct(Eigen::VectorXd & x, Eigen::MatrixXd & P,
        const Eigen::VectorXd & res_error,const Eigen::MatrixXd & H, 
        const Eigen::MatrixXd & R,
        const Eigen::MatrixXd & proj_mat,
        double chi_square_threshold)
{
    MatrixXd residule_cov_inverse =  (H * P * H.transpose() + R).inverse();
    double chi_square = (res_error.transpose() * residule_cov_inverse * res_error)(0);
    if (chi_square < chi_square_threshold)
    {
        MatrixXd gain = P * H.transpose() * residule_cov_inverse;
        if (proj_mat.rows() == proj_mat.cols() && proj_mat.cols() == x.rows())
        {
            gain = proj_mat * gain;
        }
        x += gain * res_error;
        MatrixXd i_kh = MatrixXd::Identity(P.rows(), P.cols()) - gain * H;
        P = i_kh * P * i_kh.transpose() + gain * R * gain.transpose();
    }
}

void KalmanFilterCorrector::basic_correct(Eigen::VectorXd & x, Eigen::MatrixXd & P,
                   const Eigen::VectorXd & res_error,
                   const Eigen::MatrixXd & H, const Eigen::MatrixXd & R)
{
    MatrixXd gain = P * H.transpose() * (H * P * H.transpose() + R).inverse();
    if (_gain_coef.rows() == gain.rows())
    {
        for (unsigned int i = 0; i != gain.rows(); ++i)
        {
            gain.row(i) *= _gain_coef(i);
        }
    }
    VectorXd cali = gain * res_error;
    x += cali;
    MatrixXd i_kh = MatrixXd::Identity(P.rows(), P.cols()) - gain * H;
    P = i_kh * P * i_kh.transpose() + gain * R * gain.transpose();
}

void KalmanFilterCorrector::adaptive_correct(Eigen::VectorXd & x,
       Eigen::MatrixXd & P, const Eigen::VectorXd & res_error,
       const Eigen::MatrixXd & H, const Eigen::MatrixXd & R)
{
    MatrixXd r_est = R;
    if (0 != _len_sw)
    {
        if (_hist_res_error.empty())
        {
            _hist_res_error = std::vector<VectorXd>(_len_sw, res_error);
            _est_sum_res_var = double(_len_sw) * res_error * res_error.transpose();
            _cur_idx = 0;
        }
        else
        {
            _est_sum_res_var += res_error * res_error.transpose()
                    - _hist_res_error[_cur_idx]
                        * _hist_res_error[_cur_idx].transpose();
            _hist_res_error[_cur_idx] = res_error;
            _cur_idx = (_cur_idx + 1) % _len_sw;
        }
        double error_mag = res_error.transpose() * R.inverse() * res_error;
        if (error_mag > _chi_thr)
        {
            MatrixXd scale = (1.0 / double(_len_sw) * _est_sum_res_var
                    - (H * P * H.transpose())) * R.inverse();
            for (unsigned int i = 0; i != R.rows(); ++i)
            {
                r_est(i, i) = R(i, i) * std::max(scale(i, i), 1.0);
            }
        }
    }
    basic_correct(x, P, res_error, H, r_est);
}

