#ifndef LAYER_HPP
#define LAYER_HPP

#include <dlib/matrix.h>

template <typename T>
class Layer{
    public:
        dlib::matrix<T> input;
        dlib::matrix<T> output;

        virtual dlib::matrix<T> forward(dlib::matrix<T> input) = 0;
        virtual dlib::matrix<T> backward(dlib::matrix<T> output_gradient, double eta) = 0;
};

#endif // LAYER_HPP