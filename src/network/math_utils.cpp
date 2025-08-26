#include "include/math_utils.hpp"

template dlib::matrix<float> tanh_prime<float>(dlib::matrix<float>);
template dlib::matrix<double> tanh_prime<double>(dlib::matrix<double>);
template dlib::matrix<long double> tanh_prime<long double>(dlib::matrix<long double>);

template dlib::matrix<float> sigmoid_prime<float>(dlib::matrix<float>);
template dlib::matrix<double> sigmoid_prime<double>(dlib::matrix<double>);
template dlib::matrix<long double> sigmoid_prime<long double>(dlib::matrix<long double>);

template <typename T>
dlib::matrix<T> tanh_prime(dlib::matrix<T> x) {
    return 1 - dlib::pow(2, dlib::tanh(x));
}

template <typename  T>
dlib::matrix<T> sigmoid_prime(dlib::matrix<T> x) {
    dlib::matrix<T> s = dlib::sigmoid(x);
    return dlib::pointwise_multiply(s, (1 - s));
}