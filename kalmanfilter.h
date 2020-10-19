#pragma once

#include "structs.h"


void kalman_filter_correct(Eigen::VectorXd & x, Eigen::MatrixXd & P,
        const Eigen::VectorXd & res_error,const Eigen::MatrixXd & H, 
        const Eigen::MatrixXd & R,
        const Eigen::MatrixXd & proj_mat = Eigen::MatrixXd(),
        double chi_square_threshold = std::numeric_limits<double>::max());

class KalmanFilterCorrector
{
public:
    unsigned int _len_sw = 0; // sliding window length of history residual error
    double _chi_thr = 1.0; // chi-square threshold
    Eigen::VectorXd _gain_coef; // gain coef

    void basic_correct(Eigen::VectorXd & x, Eigen::MatrixXd & P,
                       const Eigen::VectorXd & res_error,
                       const Eigen::MatrixXd & H, const Eigen::MatrixXd & R);
	void adaptive_correct(Eigen::VectorXd & x, Eigen::MatrixXd & P,
                       const Eigen::VectorXd & res_error,
                       const Eigen::MatrixXd & H, const Eigen::MatrixXd & R);

private:
    std::vector<Eigen::VectorXd> _hist_res_error; // history residual error
    Eigen::MatrixXd _est_sum_res_var; // estimated sum of residual variance
    unsigned int _cur_idx; // current index of residual error
};

