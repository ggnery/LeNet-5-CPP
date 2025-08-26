#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <dlib/matrix/matrix.h>
#include <dlib/matrix/matrix_utilities.h>
#include <dlib/matrix/matrix_math_functions.h>

template <typename  T>
inline dlib::matrix<T> tanh_prime(dlib::matrix<T> x){
    return 1 - dlib::pow(2, dlib::tanh(x));
}

template <typename  T>
inline dlib::matrix<T> sigmoid_prime(dlib::matrix<T> x){
    dlib::matrix<T> s = dlib::sigmoid(x);
    return dlib::pointwise_multiply(s, (1 - s));
}

#endif // MATH_UTILS_HPP