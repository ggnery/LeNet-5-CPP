#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "layer.hpp"
#include <dlib/matrix/matrix_utilities.h>
#include <functional>

template <typename T>
class Activation: public Layer<T> {
    public:
    //Calback function that computes the activation function f(X): R^n->R^n 
    std::function<dlib::matrix<T>(dlib::matrix<T>)> f; 
    
    //Calback function that computes the derivative of activation function f'(X): R^n->R^n 
    std::function<dlib::matrix<T>(dlib::matrix<T>)> f_prime; 

    // Constructor
    Activation(std::function<dlib::matrix<T>(dlib::matrix<T>)> f, std::function<dlib::matrix<T>(dlib::matrix<T>)> f_prime);

    dlib::matrix<T> forward(dlib::matrix<T> input);
    dlib::matrix<T> backward(dlib::matrix<T> output_gradient, double eta);
};

#endif // ACTIVATION_HPP