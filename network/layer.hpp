#ifndef LAYER_HPP
#define LAYER_HPP

#include <dlib/matrix.h>

template <typename T>
class Layer{
    public:
        dlib::matrix<T>* input;
        dlib::matrix<T>* output;

        // Constructor
        Layer(): input(nullptr), output(nullptr) {}

        // Destructor
        ~Layer(){
            delete input;
            delete output;
        }

        virtual dlib::vector<T> forward(dlib::vector<T> input) = 0;
        virtual dlib::vector<T> backward(dlib::vector<T> output_gradient) = 0;
};

#endif // LAYER_HPP