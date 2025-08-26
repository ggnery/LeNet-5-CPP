#ifndef DENSE_HPP
#define DENSE_HPP

#include "layer.hpp"

template <typename T>
class Dense: public Layer<T>{
    public:
        dlib::matrix<T> weights;
        dlib::matrix<T> bias;

        // Constructor
        Dense(long input_size, long output_size);
    
        dlib::matrix<T> forward(dlib::matrix<T> input);
        dlib::matrix<T> backward(dlib::matrix<T> output_gradient, double eta);
};

#endif // DENSE_HPP