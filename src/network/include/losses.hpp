#ifndef LOSSES_HPP
#define LOSSES_HPP

#include <dlib/matrix/matrix.h>
#include <dlib/matrix/matrix_math_functions.h>
#include <dlib/matrix/matrix_utilities.h>

template <typename T>
inline T mse(dlib::matrix<T> y_true, dlib::matrix<T> y_pred){
    return dlib::mean(dlib::pow(2, y_true - y_pred));
};

template <typename T>
inline dlib::matrix<T> mse_prime(dlib::matrix<T> y_true, dlib::matrix<T> y_pred){
    return 2 * (y_pred - y_true) / y_true.nr();
}

template <typename T>
inline T cross_entropy(dlib::matrix<T> y_true, dlib::matrix<T> y_pred) {
    return dlib::sum(
        dlib::pointwise_multiply(-y_true, dlib::log(y_pred)) - 
        dlib::pointwise_multiply((1 - y_true), dlib::log(1 -y_pred))
    );
}

template <typename T>
inline dlib::matrix<T> cross_entropy_prime(dlib::matrix<T> y_true, dlib::matrix<T> y_pred) {
    return ((1 - y_true)/(1 - y_pred) - (y_true/y_pred) ) / y_true.nr();
}

#endif //LOSSES_HPP
