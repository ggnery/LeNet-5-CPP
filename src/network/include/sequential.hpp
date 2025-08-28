#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "layer.hpp"

class Sequential: public Layer {
private:
    std::vector<std::unique_ptr<Layer>> layers;
public:
    // Delete copy constructor and copy assignment
    Sequential(const Sequential&) = delete;
    Sequential& operator=(const Sequential&) = delete;
    
    // Add move constructor and move assignment
    Sequential(Sequential&&) = default;
    Sequential& operator=(Sequential&&) = default;

    torch::Tensor forward(torch::Tensor input);
    torch::Tensor backward(torch::Tensor output_gradient, double eta);
    
    // Constructor
    Sequential() = default;

    class SequentialBuilder {
    private:
        std::vector<std::unique_ptr<Layer>> layers_;
    public:
        
        template<typename T, typename ...Args>
        SequentialBuilder& add(Args&&... args) {
            this->layers_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
            return *this;
        }

        Sequential build() {
            Sequential sequential;
            sequential.layers = std::move(this->layers_); 
            return sequential;
        }

    };

    static SequentialBuilder builder() {return SequentialBuilder();};

    friend class SequentialBuilder;
};

#endif // SEQUENTIAL_HPP