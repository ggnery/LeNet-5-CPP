#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <dlib/matrix/matrix.h>
#include <dlib/matrix/matrix_utilities.h>
#include <dlib/matrix/matrix_math_functions.h>

template <typename  T>
dlib::matrix<T> tanh_prime(dlib::matrix<T> x);

template <typename  T>
dlib::matrix<T> sigmoid_prime(dlib::matrix<T> x);

#endif // MATH_UTILS_HPP