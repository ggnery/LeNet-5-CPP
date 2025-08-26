#include "dense.hpp"
#include <dlib/matrix/matrix_utilities.h>

template class Dense<double>;
template class Dense<float>;

// Constructor
template <typename T>
Dense<T>::Dense(long input_size, long output_size){
    this->weights = dlib::matrix_cast<T>(dlib::randm(output_size, input_size) / std::sqrt(static_cast<T>(input_size))); // W ~ N(0,1) 
    this->bias = dlib::matrix_cast<T>(dlib::randm(output_size, 1)); // b ~ N(0,1)
}

//Forward
template <typename T>
dlib::matrix<T> Dense<T>::forward(dlib::matrix<T> input) {
    this->input = input;
    return this->weights * input + this->bias; // W*x + b
}

// Backward
template <typename T>
dlib::matrix<T> Dense<T>::backward(dlib::matrix<T> output_gradient, double eta) {
    dlib::matrix<T> weight_gradient = output_gradient * dlib::trans(this->input); // ∂E/∂W = ∂E/∂Y * X^T
    dlib::matrix<T> bias_gradient = output_gradient; // ∂E/∂B = ∂E/∂Y 
    dlib::matrix<T> input_gradient =  dlib::trans(this->weights) * output_gradient; // ∂E/∂X = W^T * ∂E/∂Y 

    // SGD
    this->weights -= eta * weight_gradient;
    this-> bias -= eta * bias_gradient;

    return input_gradient;
}


